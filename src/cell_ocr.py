"""
OCR module for ML Timemaster with Word-Level Correction.
Compatible with PaddleOCR 3.3.2 and PaddlePaddle 3.2.2
"""

import logging
import cv2
import numpy as np
import re
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from difflib import SequenceMatcher
import threading
import time
from collections import defaultdict


@dataclass
class OCRResult:
    """Holds the result of an OCR attempt."""
    text: str
    confidence: float
    quality_score: float
    preprocessing: str
    rotation: int = 0
    corrected: bool = False
    original_text: str = ""
    corrections_applied: List[Tuple[str, str]] = field(default_factory=list)


class ScheduleDictionary:
    """
    Dictionary manager with word-level correction support.
    
    Supports:
    - Word-level corrections: "DeBlgn" -> "Design"
    - Phrase-level corrections (optional, for special cases): "et al" -> "et al."
    - Dictionary terms for validation
    """
    
    def __init__(self, dictionary_path: Optional[str] = None, verbose: bool = False):
        """Initialize dictionary with word-level correction support."""
        self.logger = logging.getLogger(__name__ + ".Dictionary")
        self.verbose = verbose
        
        # Default terms
        self.default_terms = {
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
                    "Saturday", "Sunday", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "months": ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"],
            "schedule_terms": ["Lecture", "Lab", "Tutorial", "Seminar", "Workshop", 
                             "Class", "Break", "Lunch", "Exam", "Test"],
            "time_terms": ["AM", "PM", "Morning", "Afternoon", "Evening"],
            "locations": ["Room", "Hall", "Building", "Floor", "Lab", "Office"],
        }
        
        # Word-level corrections: incorrect_word -> correct_word
        self.word_corrections: Dict[str, str] = {}  # lowercase incorrect -> correct (with case)
        self.word_corrections_by_correct: Dict[str, List[str]] = {}  # correct -> [incorrect variations]
        
        # Phrase-level corrections (optional, for special multi-word cases)
        self.phrase_corrections: Dict[str, str] = {}  # lowercase incorrect phrase -> correct phrase
        self.phrase_corrections_by_correct: Dict[str, List[str]] = {}
        
        # All valid terms for validation
        self.all_terms: Set[str] = set()
        self.terms_lower_map: Dict[str, str] = {}
        
        # Custom terms loaded from file
        self.custom_terms: Dict[str, List[str]] = {}
        self.custom_word_corrections: Dict[str, List[str]] = {}  # correct -> [variations]
        self.custom_phrase_corrections: Dict[str, List[str]] = {}  # correct -> [variations]
        
        # File path
        self.dictionary_path = dictionary_path or "schedule_dictionary.json"
        
        # Load and build
        self._load_dictionary()
        self._build_lookup()
        
        # Character substitutions for suggestions
        self.char_substitutions = {
            '0': ['O', 'o'], 'O': ['0'], 'o': ['0'],
            '1': ['I', 'l', 'i'], 'I': ['1', 'l'], 'l': ['1', 'I', 'i'],
            '5': ['S', 's'], 'S': ['5'], 's': ['5'],
            '8': ['B'], 'B': ['8'],
            'rn': ['m'], 'm': ['rn'],
            'vv': ['w'], 'w': ['vv'],
            'cl': ['d'], 'd': ['cl'],
            'n': ['ri'], 'ri': ['n'],
        }
        
        stats = self.get_stats()
        self.logger.info(f"Dictionary initialized: {stats['word_corrections']} word corrections, "
                        f"{stats['phrase_corrections']} phrase corrections, "
                        f"{stats['total_terms']} terms")

    def _load_dictionary(self):
        """Load dictionary from JSON file."""
        if not os.path.exists(self.dictionary_path):
            self.logger.info(f"No dictionary at {self.dictionary_path}, using defaults")
            return
        
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load terms
            self.custom_terms = data.get("terms", {})
            
            # Load word corrections (new format)
            self.custom_word_corrections = data.get("word_corrections", {})
            
            # Load phrase corrections (optional)
            self.custom_phrase_corrections = data.get("phrase_corrections", {})
            
            # Backward compatibility: convert old "corrections" format
            if "corrections" in data and not self.custom_word_corrections:
                self.logger.info("Converting old corrections format to word-level...")
                self._convert_old_format(data["corrections"])
            
            self.logger.info(f"Loaded dictionary from {self.dictionary_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load dictionary: {e}")

    def _convert_old_format(self, old_corrections: Dict[str, List[str]]):
        """Convert old phrase-based corrections to word-level where possible."""
        for correct_phrase, variations in old_corrections.items():
            correct_words = correct_phrase.split()
            
            if len(correct_words) == 1:
                # Single word - add as word correction
                if correct_phrase not in self.custom_word_corrections:
                    self.custom_word_corrections[correct_phrase] = []
                for var in variations:
                    var_words = var.split()
                    if len(var_words) == 1:
                        self.custom_word_corrections[correct_phrase].append(var)
            else:
                # Multi-word - try to extract word-level corrections
                for variation in variations:
                    var_words = variation.split()
                    if len(var_words) == len(correct_words):
                        # Same word count - extract individual word corrections
                        for correct_word, var_word in zip(correct_words, var_words):
                            if correct_word.lower() != var_word.lower():
                                if correct_word not in self.custom_word_corrections:
                                    self.custom_word_corrections[correct_word] = []
                                if var_word not in self.custom_word_corrections[correct_word]:
                                    self.custom_word_corrections[correct_word].append(var_word)

    def _build_lookup(self):
        """Build lookup structures."""
        self.word_corrections.clear()
        self.word_corrections_by_correct.clear()
        self.phrase_corrections.clear()
        self.phrase_corrections_by_correct.clear()
        self.all_terms.clear()
        self.terms_lower_map.clear()
        
        # Build terms lookup
        for category, terms in {**self.default_terms, **self.custom_terms}.items():
            for term in terms:
                self.all_terms.add(term)
                self.terms_lower_map[term.lower()] = term
                # Also add individual words from multi-word terms
                for word in term.split():
                    if len(word) > 1:
                        self.all_terms.add(word)
                        self.terms_lower_map[word.lower()] = word
        
        # Build word corrections lookup
        for correct_word, variations in self.custom_word_corrections.items():
            self.word_corrections_by_correct[correct_word] = variations
            # Add correct word to terms
            self.all_terms.add(correct_word)
            self.terms_lower_map[correct_word.lower()] = correct_word
            
            for variation in variations:
                self.word_corrections[variation.lower()] = correct_word
                if self.verbose:
                    self.logger.debug(f"Word correction: '{variation}' -> '{correct_word}'")
        
        # Build phrase corrections lookup
        for correct_phrase, variations in self.custom_phrase_corrections.items():
            self.phrase_corrections_by_correct[correct_phrase] = variations
            for variation in variations:
                self.phrase_corrections[variation.lower()] = correct_phrase

    def save_dictionary(self) -> bool:
        """Save dictionary to JSON file."""
        try:
            data = {
                "terms": self.custom_terms,
                "word_corrections": self.custom_word_corrections,
                "phrase_corrections": self.custom_phrase_corrections
            }
            
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved dictionary to {self.dictionary_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save: {e}")
            return False

    # =========================================================================
    # Word-Level Correction Methods
    # =========================================================================
    
    def add_word_correction(self, incorrect: str, correct: str) -> bool:
        """
        Add a word-level correction.
        
        Args:
            incorrect: The incorrect OCR word (e.g., "DeBlgn")
            correct: The correct word (e.g., "Design")
            
        Returns:
            True if added successfully
            
        Example:
            dictionary.add_word_correction("DeBlgn", "Design")
            dictionary.add_word_correction("Learnlng", "Learning")
        """
        if not incorrect or not correct:
            return False
        
        incorrect = incorrect.strip()
        correct = correct.strip()
        
        # Don't add if same
        if incorrect.lower() == correct.lower():
            return False
        
        # Don't add multi-word as word correction
        if ' ' in incorrect or ' ' in correct:
            self.logger.warning(f"Use add_phrase_correction for multi-word: '{incorrect}' -> '{correct}'")
            return False
        
        # Add to custom corrections
        if correct not in self.custom_word_corrections:
            self.custom_word_corrections[correct] = []
        
        if incorrect not in self.custom_word_corrections[correct]:
            self.custom_word_corrections[correct].append(incorrect)
            
            # Update lookup
            self.word_corrections[incorrect.lower()] = correct
            self.word_corrections_by_correct[correct] = self.custom_word_corrections[correct]
            
            # Add correct word to terms
            self.all_terms.add(correct)
            self.terms_lower_map[correct.lower()] = correct
            
            self.logger.info(f"Added word correction: '{incorrect}' -> '{correct}'")
            return True
        
        return False

    def add_word_corrections_batch(self, correct_word: str, variations: List[str]) -> int:
        """
        Add multiple incorrect variations for a word.
        
        Args:
            correct_word: The correct word
            variations: List of incorrect variations
            
        Returns:
            Number of corrections added
            
        Example:
            dictionary.add_word_corrections_batch("Design", [
                "DeBlgn", "Deslgn", "Des1gn", "Desiqn"
            ])
        """
        added = 0
        for variation in variations:
            if self.add_word_correction(variation, correct_word):
                added += 1
        return added

    def get_word_correction(self, word: str) -> Optional[str]:
        """
        Get correction for a single word (exact match).
        
        Args:
            word: The word to look up
            
        Returns:
            Corrected word or None
        """
        return self.word_corrections.get(word.lower())

    def find_similar_word_correction(self, word: str, threshold: float = 0.80) -> Optional[Tuple[str, float]]:
        """
        Find similar word correction using fuzzy matching.
        
        Args:
            word: The word to match
            threshold: Minimum similarity (0-1)
            
        Returns:
            Tuple of (corrected_word, similarity) or None
        """
        if not word or len(word) < 2:
            return None
        
        word_lower = word.lower()
        
        # Exact match first
        if word_lower in self.word_corrections:
            return (self.word_corrections[word_lower], 1.0)
        
        # Fuzzy match against all incorrect variations
        best_match = None
        best_ratio = threshold
        
        for incorrect_lower, correct in self.word_corrections.items():
            # Skip if length difference too large
            if abs(len(word) - len(incorrect_lower)) > max(2, len(word) * 0.3):
                continue
            
            ratio = SequenceMatcher(None, word_lower, incorrect_lower).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = correct
        
        if best_match:
            return (best_match, best_ratio)
        
        return None

    # =========================================================================
    # Phrase-Level Correction Methods (for special cases)
    # =========================================================================
    
    def add_phrase_correction(self, incorrect: str, correct: str) -> bool:
        """
        Add a phrase-level correction (for special multi-word cases).
        
        Use this sparingly - prefer word-level corrections.
        
        Args:
            incorrect: The incorrect phrase
            correct: The correct phrase
        """
        if not incorrect or not correct:
            return False
        
        incorrect = incorrect.strip()
        correct = correct.strip()
        
        if incorrect.lower() == correct.lower():
            return False
        
        if correct not in self.custom_phrase_corrections:
            self.custom_phrase_corrections[correct] = []
        
        if incorrect not in self.custom_phrase_corrections[correct]:
            self.custom_phrase_corrections[correct].append(incorrect)
            self.phrase_corrections[incorrect.lower()] = correct
            self.phrase_corrections_by_correct[correct] = self.custom_phrase_corrections[correct]
            
            self.logger.info(f"Added phrase correction: '{incorrect}' -> '{correct}'")
            return True
        
        return False

    def get_phrase_correction(self, phrase: str) -> Optional[str]:
        """Get correction for a phrase (exact match)."""
        return self.phrase_corrections.get(phrase.lower())

    # =========================================================================
    # Term Validation Methods
    # =========================================================================
    
    def add_term(self, term: str, category: str = "custom"):
        """Add a term to the dictionary."""
        if category not in self.custom_terms:
            self.custom_terms[category] = []
        
        if term not in self.custom_terms[category]:
            self.custom_terms[category].append(term)
            self.all_terms.add(term)
            self.terms_lower_map[term.lower()] = term

    def is_valid_term(self, text: str) -> bool:
        """Check if text matches a dictionary term."""
        return text.lower() in self.terms_lower_map

    def is_valid_word(self, word: str) -> bool:
        """Check if a single word is valid/known."""
        word_lower = word.lower()
        # Valid if it's a term or a correct word in corrections
        return (word_lower in self.terms_lower_map or 
                any(w.lower() == word_lower for w in self.word_corrections_by_correct.keys()))

    def find_similar_term(self, text: str, threshold: float = 0.80) -> Optional[Tuple[str, float]]:
        """Find similar term using fuzzy matching."""
        if not text or len(text) < 2:
            return None
        
        text_lower = text.lower()
        
        if text_lower in self.terms_lower_map:
            return (self.terms_lower_map[text_lower], 1.0)
        
        best_match = None
        best_ratio = threshold
        
        for term_lower, term in self.terms_lower_map.items():
            if abs(len(text) - len(term)) > 3:
                continue
            ratio = SequenceMatcher(None, text_lower, term_lower).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = term
        
        return (best_match, best_ratio) if best_match else None

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def remove_word_correction(self, incorrect: str) -> bool:
        """Remove a word correction."""
        incorrect_lower = incorrect.lower()
        
        if incorrect_lower in self.word_corrections:
            correct = self.word_corrections[incorrect_lower]
            del self.word_corrections[incorrect_lower]
            
            if correct in self.custom_word_corrections:
                self.custom_word_corrections[correct] = [
                    v for v in self.custom_word_corrections[correct]
                    if v.lower() != incorrect_lower
                ]
                if not self.custom_word_corrections[correct]:
                    del self.custom_word_corrections[correct]
            
            return True
        return False

    def get_all_word_corrections(self) -> Dict[str, List[str]]:
        """Get all word corrections grouped by correct word."""
        return self.custom_word_corrections.copy()

    def get_corrections_for_word(self, correct_word: str) -> List[str]:
        """Get all registered variations for a correct word."""
        return self.custom_word_corrections.get(correct_word, [])

    def suggest_word_variations(self, word: str) -> List[str]:
        """Suggest possible OCR variations of a word."""
        suggestions = set()
        
        for i, char in enumerate(word):
            if char in self.char_substitutions:
                for replacement in self.char_substitutions[char]:
                    suggestions.add(word[:i] + replacement + word[i+1:])
        
        for pattern, replacements in self.char_substitutions.items():
            if len(pattern) > 1 and pattern in word:
                for replacement in replacements:
                    suggestions.add(word.replace(pattern, replacement))
        
        # Filter existing corrections
        suggestions = {s for s in suggestions 
                      if s.lower() not in self.word_corrections and s.lower() != word.lower()}
        
        return sorted(suggestions)[:10]

    def get_stats(self) -> Dict[str, Any]:
        """Get dictionary statistics."""
        return {
            "total_terms": len(self.all_terms),
            "word_corrections": len(self.word_corrections),
            "phrase_corrections": len(self.phrase_corrections),
            "unique_correct_words": len(self.word_corrections_by_correct),
            "categories": len(self.custom_terms)
        }


class WordLevelCorrector:
    """
    Corrects OCR text word by word.
    """
    
    def __init__(self, dictionary: ScheduleDictionary, verbose: bool = False):
        """Initialize corrector."""
        self.dictionary = dictionary
        self.verbose = verbose
        self.logger = logging.getLogger(__name__ + ".Corrector")
        
        # Pattern to split text into words while preserving punctuation
        self.word_pattern = re.compile(r'(\s+|[^\w\s]+)')
        
        # Patterns for validation
        self.time_pattern = re.compile(r'^([0-9]{1,2}):([0-9]{2})$')
        self.number_pattern = re.compile(r'^[0-9]+$')
        
        # Track corrections and unrecognized words
        self.unrecognized_words: Dict[str, int] = defaultdict(int)
        
        # Statistics
        self.stats = {
            'texts_processed': 0,
            'words_processed': 0,
            'words_corrected': 0,
            'exact_corrections': 0,
            'fuzzy_corrections': 0,
            'phrases_corrected': 0,
            'unrecognized': 0
        }

    def correct_text(self, text: str, confidence: float = 50.0) -> Tuple[str, float, bool, List[Tuple[str, str]]]:
        """
        Correct OCR text word by word.
        
        Args:
            text: The OCR text to correct
            confidence: OCR confidence (0-100)
            
        Returns:
            Tuple of (corrected_text, new_confidence, was_corrected, list of (original, corrected) pairs)
        """
        self.stats['texts_processed'] += 1
        
        if not text or not text.strip():
            return text, confidence, False, []
        
        original_text = text
        corrections_made = []
        
        # Step 1: Check for phrase-level corrections first (exact match)
        phrase_correction = self.dictionary.get_phrase_correction(text)
        if phrase_correction:
            self.stats['phrases_corrected'] += 1
            if self.verbose:
                self.logger.debug(f"Phrase correction: '{text}' -> '{phrase_correction}'")
            return phrase_correction, min(confidence + 15, 98), True, [(text, phrase_correction)]
        
        # Step 2: Split into tokens (words and separators)
        tokens = self._tokenize(text)
        corrected_tokens = []
        
        for token, is_word in tokens:
            if not is_word:
                # Separator (space, punctuation) - keep as is
                corrected_tokens.append(token)
                continue
            
            self.stats['words_processed'] += 1
            
            # Try to correct the word
            corrected_word, correction_type = self._correct_word(token, confidence)
            
            if corrected_word != token:
                corrections_made.append((token, corrected_word))
                self.stats['words_corrected'] += 1
                
                if correction_type == 'exact':
                    self.stats['exact_corrections'] += 1
                elif correction_type == 'fuzzy':
                    self.stats['fuzzy_corrections'] += 1
                
                if self.verbose:
                    self.logger.debug(f"Word correction ({correction_type}): '{token}' -> '{corrected_word}'")
            
            corrected_tokens.append(corrected_word)
        
        # Reconstruct text
        corrected_text = ''.join(corrected_tokens)
        
        # Calculate new confidence
        was_corrected = len(corrections_made) > 0
        if was_corrected:
            # Boost confidence based on number of corrections
            conf_boost = min(len(corrections_made) * 5, 15)
            new_confidence = min(confidence + conf_boost, 95)
        else:
            new_confidence = confidence
        
        return corrected_text, new_confidence, was_corrected, corrections_made

    def _tokenize(self, text: str) -> List[Tuple[str, bool]]:
        """
        Split text into tokens, preserving separators.
        
        Returns:
            List of (token, is_word) tuples
        """
        tokens = []
        parts = self.word_pattern.split(text)
        
        for part in parts:
            if not part:
                continue
            
            # Check if it's a word (alphanumeric) or separator
            is_word = bool(re.match(r'^\w+$', part))
            tokens.append((part, is_word))
        
        return tokens

    def _correct_word(self, word: str, confidence: float) -> Tuple[str, str]:
        """
        Correct a single word.
        
        Returns:
            Tuple of (corrected_word, correction_type)
            correction_type: 'exact', 'fuzzy', 'none'
        """
        # Skip short words
        if len(word) < 2:
            return word, 'none'
        
        # Skip numbers and times
        if self.number_pattern.match(word) or self.time_pattern.match(word):
            return word, 'none'
        
        # Step 1: Exact word correction
        exact_correction = self.dictionary.get_word_correction(word)
        if exact_correction:
            # Preserve original case pattern if possible
            return self._apply_case(word, exact_correction), 'exact'
        
        # Step 2: Check if word is already valid
        if self.dictionary.is_valid_word(word):
            return word, 'none'
        
        # Step 3: Fuzzy word correction (for lower confidence)
        if confidence < 85:
            threshold = 0.75 if confidence < 60 else 0.82
            fuzzy_result = self.dictionary.find_similar_word_correction(word, threshold)
            if fuzzy_result:
                return self._apply_case(word, fuzzy_result[0]), 'fuzzy'
        
        # Step 4: Fuzzy term matching (for lower confidence)
        if confidence < 80:
            fuzzy_term = self.dictionary.find_similar_term(word, threshold=0.80)
            if fuzzy_term:
                return self._apply_case(word, fuzzy_term[0]), 'fuzzy'
        
        # Track unrecognized word
        if confidence < 75 and len(word) > 2:
            self._track_unrecognized(word)
        
        return word, 'none'

    def _apply_case(self, original: str, corrected: str) -> str:
        """
        Apply the case pattern from original to corrected.
        
        Examples:
            original="DESIGN", corrected="Design" -> "DESIGN"
            original="design", corrected="Design" -> "design"
            original="Design", corrected="design" -> "Design"
        """
        if not original or not corrected:
            return corrected
        
        # All uppercase
        if original.isupper():
            return corrected.upper()
        
        # All lowercase
        if original.islower():
            return corrected.lower()
        
        # Title case (first letter uppercase)
        if original[0].isupper() and (len(original) == 1 or original[1:].islower()):
            return corrected.capitalize()
        
        # Keep corrected word's case (it's the "proper" form)
        return corrected

    def _track_unrecognized(self, word: str):
        """Track unrecognized words for review."""
        normalized = word.lower()
        self.unrecognized_words[normalized] += 1
        self.stats['unrecognized'] += 1

    def get_unrecognized_words(self, min_occurrences: int = 1) -> List[Tuple[str, int]]:
        """
        Get unrecognized words sorted by frequency.
        
        Returns:
            List of (word, count) tuples
        """
        filtered = [(w, c) for w, c in self.unrecognized_words.items() if c >= min_occurrences]
        return sorted(filtered, key=lambda x: x[1], reverse=True)

    def clear_unrecognized(self):
        """Clear unrecognized words tracking."""
        self.unrecognized_words.clear()

    def get_stats(self) -> dict:
        """Get correction statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'texts_processed': 0,
            'words_processed': 0,
            'words_corrected': 0,
            'exact_corrections': 0,
            'fuzzy_corrections': 0,
            'phrases_corrected': 0,
            'unrecognized': 0
        }


class CellOCR:
    """
    Fast OCR for table cells with word-level correction.
    """
    
    _ocr_instance = None
    _ocr_init_lock = threading.Lock()
    
    MIN_OCR_SIZE = 32
    MAX_OCR_SIZE = 2000
    OPTIMAL_HEIGHT = 48
    
    def __init__(
        self,
        minimum_confidence_threshold: float = 40.0,
        high_confidence_threshold: float = 80.0,
        verbose_logging: bool = False,
        empty_cell_variance_threshold: float = 50.0,
        languages: str = "en",
        enable_rotation: bool = True,
        max_retries: int = 2,
        dictionary_path: Optional[str] = None,
        enable_validation: bool = True,
    ):
        """Initialize CellOCR with word-level correction."""
        self.min_conf = minimum_confidence_threshold
        self.high_conf = high_confidence_threshold
        self.verbose = verbose_logging
        self.empty_variance = empty_cell_variance_threshold
        self.enable_rotation = enable_rotation
        self.max_retries = max_retries
        self.enable_validation = enable_validation
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose_logging else logging.INFO)
        
        self.lang = self._parse_lang(languages)
        
        # Initialize dictionary and corrector
        if self.enable_validation:
            self.dictionary = ScheduleDictionary(dictionary_path, verbose=verbose_logging)
            self.corrector = WordLevelCorrector(self.dictionary, verbose=verbose_logging)
            self.logger.info("Word-level correction enabled")
        else:
            self.dictionary = None
            self.corrector = None
        
        # Stats
        self.stats = {
            'cells': 0, 'ocr_calls': 0, 'empty': 0,
            'success': 0, 'time': 0, 'corrected': 0
        }
        
        self.logger.info(f"CellOCR initialized (lang={self.lang}, validation={enable_validation})")

    def _parse_lang(self, lang: str) -> str:
        """Parse language to PaddleOCR format."""
        mapping = {
            "eng": "en", "en": "en", "ron": "latin", "latin": "latin",
            "fra": "fr", "fr": "fr", "deu": "german", "german": "german",
            "ch": "ch", "chinese": "ch",
        }
        return mapping.get(lang.lower().split('+')[0].strip(), "en")

    def _get_ocr(self):
        """Get or create OCR instance."""
        if CellOCR._ocr_instance is not None:
            return CellOCR._ocr_instance
        
        with CellOCR._ocr_init_lock:
            if CellOCR._ocr_instance is not None:
                return CellOCR._ocr_instance
            
            self.logger.info("Loading PaddleOCR...")
            start = time.time()
            
            from paddleocr import PaddleOCR
            
            try:
                CellOCR._ocr_instance = PaddleOCR(lang=self.lang, show_log=self.verbose)
            except Exception as e:
                self.logger.warning(f"Init failed: {e}")
                CellOCR._ocr_instance = PaddleOCR(lang=self.lang)
            
            self.logger.info(f"PaddleOCR loaded in {time.time()-start:.1f}s")
        
        return CellOCR._ocr_instance

    @property
    def ocr(self):
        return self._get_ocr()

    def _prepare_image_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Prepare image for OCR."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        h, w = img.shape[:2]
        scale = 1.0
        
        if h < self.MIN_OCR_SIZE or w < self.MIN_OCR_SIZE:
            scale = max(self.MIN_OCR_SIZE / min(h, w), 1.0)
            if h < self.OPTIMAL_HEIGHT:
                scale = max(scale, self.OPTIMAL_HEIGHT / h)
        
        scale = min(scale, 4.0)
        if max(h, w) * scale > self.MAX_OCR_SIZE:
            scale = self.MAX_OCR_SIZE / max(h, w)
        
        if abs(scale - 1.0) > 0.01:
            new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        
        return img

    def _is_empty(self, img: np.ndarray) -> bool:
        """Check if cell is empty."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return np.var(gray) < self.empty_variance

    def _parse_paddle_result(self, result) -> List[Tuple[str, float]]:
        """Parse PaddleOCR result."""
        parsed = []
        
        if result is None:
            return parsed
        
        if isinstance(result, dict):
            if 'res' in result:
                result = result['res']
            elif 'texts' in result and 'scores' in result:
                for t, s in zip(result['texts'], result['scores']):
                    if t:
                        parsed.append((str(t), float(s)))
                return parsed
        
        if not isinstance(result, list) or len(result) == 0:
            return parsed
        
        first = result[0]
        if first is None:
            return parsed
        
        if isinstance(first, dict):
            texts = first.get('texts', first.get('rec_texts', []))
            scores = first.get('scores', first.get('rec_scores', []))
            for t, s in zip(texts, scores):
                if t:
                    parsed.append((str(t), float(s)))
            return parsed
        
        def extract(item):
            if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
                return None
            last = item[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 2:
                try:
                    t, c = str(last[0]).strip(), float(last[1])
                    if t and 0 <= c <= 1:
                        return (t, c)
                except:
                    pass
            return None
        
        if isinstance(first, list) and first:
            for det in first:
                e = extract(det)
                if e:
                    parsed.append(e)
        
        if not parsed:
            for item in result:
                if isinstance(item, list):
                    e = extract(item)
                    if e:
                        parsed.append(e)
        
        return parsed

    def _run_ocr(self, img: np.ndarray) -> Tuple[str, float]:
        """Run OCR on image."""
        self.stats['ocr_calls'] += 1
        
        try:
            prepared = self._prepare_image_for_ocr(img)
            
            try:
                result = self.ocr.ocr(prepared)
            except TypeError:
                try:
                    result = self.ocr.ocr(prepared, det=True, rec=True, cls=False)
                except:
                    return "", 0.0
            
            parsed = self._parse_paddle_result(result)
            if not parsed:
                return "", 0.0
            
            texts = [t for t, c in parsed]
            confs = [c for t, c in parsed]
            combined = " ".join(texts)
            
            total_len = sum(len(t) for t in texts)
            avg_conf = sum(c * len(t) for c, t in zip(confs, texts)) / total_len if total_len else 0
            
            return combined, avg_conf * 100
            
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return "", 0.0

    def _preprocess(self, img: np.ndarray, method: int = 0) -> np.ndarray:
        """Apply preprocessing."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        if method == 0:
            return img
        elif method == 1:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        elif method == 2:
            gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) < 127:
                binary = cv2.bitwise_not(binary)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return img

    def _is_valid_text(self, text: str, conf: float) -> bool:
        """Check if text is valid."""
        if not text or not text.strip():
            return False
        stripped = text.strip()
        if not any(c.isalnum() for c in stripped):
            return False
        unique = set(stripped.replace(" ", ""))
        if len(unique) == 1 and len(stripped) > 2:
            return False
        return True

    def _calculate_score(self, text: str, conf: float, rotated: bool = False) -> float:
        """Calculate quality score."""
        if not text:
            return 0.0
        score = conf
        length = len(text.strip())
        if length >= 2:
            score *= 1.1
        if length >= 5:
            score *= 1.1
        if length <= 1 and conf < 70:
            score *= 0.5
        if rotated and length >= 2:
            score *= 1.15
        return score

    def extract_cell_text(self, img: np.ndarray, cell: dict) -> str:
        """Extract text from a cell with word-level correction."""
        self.stats['cells'] += 1
        start = time.time()
        
        pad = 3
        y1 = max(0, cell["y1"] - pad)
        y2 = min(img.shape[0], cell["y2"] + pad)
        x1 = max(0, cell["x1"] - pad)
        x2 = min(img.shape[1], cell["x2"] + pad)
        
        cell_img = img[y1:y2, x1:x2]
        h, w = cell_img.shape[:2]
        
        if h < 3 or w < 3:
            return ""
        
        if self._is_empty(cell_img):
            self.stats['empty'] += 1
            return ""
        
        try_rotation = self.enable_rotation and cell.get("rowspan", 1) > 1
        best_text, best_score, best_conf = "", 0.0, 0.0
        
        for method in range(self.max_retries + 1):
            preprocessed = self._preprocess(cell_img, method)
            text, conf = self._run_ocr(preprocessed)
            
            if text and self._is_valid_text(text, conf):
                score = self._calculate_score(text, conf)
                if score > best_score:
                    best_text, best_score, best_conf = text, score, conf
                if score >= self.high_conf:
                    break
        
        if try_rotation and best_score < self.min_conf:
            for angle in [90, 270]:
                rotated = cv2.rotate(cell_img,
                    cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
                text, conf = self._run_ocr(rotated)
                if text and self._is_valid_text(text, conf):
                    score = self._calculate_score(text, conf, rotated=True)
                    if score > best_score:
                        best_text, best_score, best_conf = text, score, conf
                    if score >= self.high_conf:
                        break
        
        self.stats['time'] += time.time() - start
        
        if best_score >= self.min_conf or (best_text and best_score >= 20):
            self.stats['success'] += 1
            
            # Apply word-level correction
            if self.enable_validation and self.corrector:
                corrected_text, new_conf, was_corrected, corrections = \
                    self.corrector.correct_text(best_text, best_conf)
                
                if was_corrected:
                    self.stats['corrected'] += 1
                    if self.verbose:
                        self.logger.debug(f"Corrected: '{best_text}' -> '{corrected_text}'")
                        for orig, corr in corrections:
                            self.logger.debug(f"  Word: '{orig}' -> '{corr}'")
                    return self._clean_text(corrected_text)
            
            return self._clean_text(best_text)
        
        return ""

    def _clean_text(self, text: str) -> str:
        """Clean text."""
        if not text:
            return ""
        text = " ".join(text.split())
        text = text.replace("|", "I")
        return text.strip()

    def extract_batch(self, img: np.ndarray, cells: List[dict]) -> List[str]:
        """Extract text from multiple cells."""
        total = len(cells)
        self.logger.info(f"Processing {total} cells...")
        start = time.time()
        
        results = []
        for i, cell in enumerate(cells):
            if i > 0 and i % 50 == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                self.logger.info(f"Progress: {i}/{total} ({rate:.1f} cells/sec)")
            results.append(self.extract_cell_text(img, cell))
        
        elapsed = time.time() - start
        success = sum(1 for r in results if r)
        self.logger.info(f"Completed: {success}/{total} in {elapsed:.1f}s")
        
        if self.corrector:
            cs = self.corrector.get_stats()
            self.logger.info(f"Words corrected: {cs['words_corrected']} "
                           f"({cs['exact_corrections']} exact, {cs['fuzzy_corrections']} fuzzy)")
        
        return results

    # =========================================================================
    # Dictionary and Correction API
    # =========================================================================
    
    def add_word_correction(self, incorrect: str, correct: str) -> bool:
        """
        Add a word-level correction.
        
        Args:
            incorrect: The incorrect OCR word (e.g., "DeBlgn")
            correct: The correct word (e.g., "Design")
            
        Example:
            ocr.add_word_correction("DeBlgn", "Design")
            ocr.add_word_correction("Learnlng", "Learning")
            
            # Now "Web DeBlgn" will be corrected to "Web Design"
            # And "Graphic DeBlgn" will also be corrected to "Graphic Design"
        """
        if self.dictionary:
            return self.dictionary.add_word_correction(incorrect, correct)
        return False

    def add_word_corrections_batch(self, correct_word: str, variations: List[str]) -> int:
        """
        Add multiple incorrect variations for a word.
        
        Example:
            ocr.add_word_corrections_batch("Design", [
                "DeBlgn", "Deslgn", "Des1gn", "Desiqn", "Dcsign"
            ])
        """
        if self.dictionary:
            return self.dictionary.add_word_corrections_batch(correct_word, variations)
        return 0

    def add_phrase_correction(self, incorrect: str, correct: str) -> bool:
        """
        Add a phrase-level correction (for special multi-word cases).
        
        Use sparingly - prefer word corrections.
        
        Example:
            ocr.add_phrase_correction("et al", "et al.")
        """
        if self.dictionary:
            return self.dictionary.add_phrase_correction(incorrect, correct)
        return False

    def add_term(self, term: str, category: str = "custom"):
        """Add a valid term to the dictionary."""
        if self.dictionary:
            self.dictionary.add_term(term, category)

    def save_dictionary(self) -> bool:
        """Save dictionary to file."""
        if self.dictionary:
            return self.dictionary.save_dictionary()
        return False

    def get_word_corrections(self) -> Dict[str, List[str]]:
        """Get all word corrections."""
        if self.dictionary:
            return self.dictionary.get_all_word_corrections()
        return {}

    def get_unrecognized_words(self, min_occurrences: int = 2) -> List[Tuple[str, int]]:
        """
        Get words that weren't corrected, sorted by frequency.
        Useful for finding new corrections to add.
        """
        if self.corrector:
            return self.corrector.get_unrecognized_words(min_occurrences)
        return []

    def suggest_word_variations(self, word: str) -> List[str]:
        """Suggest possible OCR variations of a word."""
        if self.dictionary:
            return self.dictionary.suggest_word_variations(word)
        return []

    def get_stats(self) -> dict:
        """Get statistics."""
        stats = self.stats.copy()
        if stats['cells'] > 0:
            stats['success_rate'] = stats['success'] / stats['cells'] * 100
            stats['correction_rate'] = stats['corrected'] / stats['cells'] * 100
        if self.corrector:
            stats['corrector'] = self.corrector.get_stats()
        if self.dictionary:
            stats['dictionary'] = self.dictionary.get_stats()
        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'cells': 0, 'ocr_calls': 0, 'empty': 0,
            'success': 0, 'time': 0, 'corrected': 0
        }
        if self.corrector:
            self.corrector.reset_stats()

    def cleanup(self):
        """Cleanup resources."""
        CellOCR._ocr_instance = None


# =============================================================================
# Utility Functions
# =============================================================================

def create_word_level_dictionary(output_path: str = "schedule_dictionary.json"):
    """Create a template dictionary with word-level corrections."""
    dictionary = {
        "terms": {
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
                    "Saturday", "Sunday"],
            "subjects": ["Mathematics", "Physics", "Chemistry", "Biology", 
                        "Computer", "Science", "Engineering", "Design", 
                        "Machine", "Learning", "Software", "Database"],
            "schedule_terms": ["Lecture", "Tutorial", "Lab", "Seminar", 
                             "Workshop", "Exam", "Break"],
            "locations": ["Room", "Hall", "Building", "Floor", "Lab"],
            "custom": []
        },
        "word_corrections": {
            # Days - common OCR errors
            "Monday": ["Mond4y", "M0nday", "Mondav", "Mcnday"],
            "Tuesday": ["Tuesd4y", "Tu3sday", "Tucsday"],
            "Wednesday": ["Wednesd4y", "Wedn3sday", "Wcdnesday"],
            "Thursday": ["Thursd4y", "Th ursday", "Thursdav"],
            "Friday": ["Frid4y", "Fr1day", "Fridav"],
            
            # Common words
            "Design": ["DeBlgn", "Deslgn", "Des1gn", "Desiqn", "Dcsign"],
            "Machine": ["Machlne", "Mach1ne", "Machirie", "Machinc"],
            "Learning": ["Learnlng", "Learn1ng", "Learninq", "Lcarning"],
            "Computer": ["Cornputer", "C0mputer", "Computcr", "Cornputer"],
            "Science": ["Sclence", "Sc1ence", "Sciencc", "Scicnce"],
            "Software": ["Softwarc", "S0ftware", "Softwaro"],
            "Engineering": ["Englneering", "Engineer1ng", "Enginecring"],
            "Database": ["Databas3", "Databasc", "Datbase"],
            "Systems": ["Systerns", "Syst3ms", "Systcms"],
            
            # Schedule terms
            "Lecture": ["Lectur3", "Lcture", "Lecturc", "Lecturo"],
            "Tutorial": ["Tut0rial", "Tutoria1", "Tutorlal"],
            "Seminar": ["Semlnar", "Serninar", "Scminar"],
            "Workshop": ["W0rkshop", "Worksho p", "Vvorkshop"],
            
            # Locations
            "Room": ["R0om", "Roorn", "Rcom"],
            "Building": ["Bu1lding", "Buildlng", "Buildinq"],
            "Floor": ["Fl0or", "Fioor", "F1oor"],
            
            # Common instructor titles
            "Prof": ["Pr0f", "Prcf"],
            "Professor": ["Profcssor", "Profess0r"],
        },
        "phrase_corrections": {
            # Only for special cases that can't be handled word-by-word
            # Keep this minimal!
        }
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2, ensure_ascii=False)
        print(f"Created word-level dictionary at: {output_path}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False


def interactive_word_correction_builder(ocr: CellOCR):
    """Interactive tool to build word corrections."""
    print("\n" + "=" * 60)
    print("Interactive Word Correction Builder")
    print("=" * 60)
    
    unrecognized = ocr.get_unrecognized_words(min_occurrences=1)
    
    if not unrecognized:
        print("No unrecognized words to review!")
        return
    
    print(f"\nFound {len(unrecognized)} unrecognized word(s):")
    print("-" * 40)
    
    for i, (word, count) in enumerate(unrecognized[:20], 1):
        print(f"{i}. '{word}' (appeared {count}x)")
    
    print("\n" + "-" * 40)
    print("Commands:")
    print("  <number> <correct_word> - Add word correction")
    print("  s <word> - Show suggested variations for a word")
    print("  q - Quit and save")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nCommand: ").strip()
            
            if user_input.lower() == 'q':
                break
            
            if user_input.lower().startswith('s '):
                word = user_input[2:].strip()
                suggestions = ocr.suggest_word_variations(word)
                print(f"Suggested variations for '{word}': {suggestions}")
                continue
            
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                idx = int(parts[0]) - 1
                correct_word = parts[1]
                
                if 0 <= idx < len(unrecognized):
                    incorrect_word = unrecognized[idx][0]
                    
                    if ocr.add_word_correction(incorrect_word, correct_word):
                        print(f"✓ Added: '{incorrect_word}' -> '{correct_word}'")
                        
                        # Show suggestions
                        suggestions = ocr.suggest_word_variations(correct_word)
                        if suggestions:
                            print(f"  Also consider adding: {suggestions[:5]}")
                    else:
                        print("✗ Failed to add")
                else:
                    print("Invalid index")
            else:
                print("Invalid format. Use: <number> <correct_word>")
                
        except (ValueError, KeyboardInterrupt):
            break
    
    if ocr.save_dictionary():
        print("\n✓ Dictionary saved!")


# =============================================================================
# Tests
# =============================================================================

def test_word_level_correction():
    """Test word-level correction functionality."""
    print("\n" + "=" * 60)
    print("Testing Word-Level Correction")
    print("=" * 60)
    
    # Create dictionary
    dictionary = ScheduleDictionary(verbose=True)
    
    # Add word corrections
    print("\n--- Adding word corrections ---")
    dictionary.add_word_correction("DeBlgn", "Design")
    dictionary.add_word_correction("Deslgn", "Design")
    dictionary.add_word_correction("Learnlng", "Learning")
    dictionary.add_word_correction("Machlne", "Machine")
    dictionary.add_word_correction("Mond4y", "Monday")
    
    print(f"Total word corrections: {len(dictionary.word_corrections)}")
    
    # Create corrector
    corrector = WordLevelCorrector(dictionary, verbose=True)
    
    # Test cases
    test_cases = [
        # (input, expected_output)
        ("Web DeBlgn", "Web Design"),
        ("Graphic Deslgn", "Graphic Design"),
        ("Machlne Learnlng", "Machine Learning"),
        ("Deep Learnlng Course", "Deep Learning Course"),
        ("Mond4y Lecture", "Monday Lecture"),
        ("Web DeBlgn and Graphic Deslgn", "Web Design and Graphic Design"),
        ("Normal Text", "Normal Text"),  # No correction needed
        ("10:30 AM", "10:30 AM"),  # Time - no correction
    ]
    
    print("\n--- Testing corrections ---")
    for input_text, expected in test_cases:
        result, conf, corrected, corrections = corrector.correct_text(input_text, 65.0)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")
        if corrections:
            print(f"    Corrections: {corrections}")
    
    print("\n--- Statistics ---")
    for k, v in corrector.get_stats().items():
        print(f"  {k}: {v}")


def test_case_preservation():
    """Test that case is preserved correctly."""
    print("\n" + "=" * 60)
    print("Testing Case Preservation")
    print("=" * 60)
    
    dictionary = ScheduleDictionary()
    dictionary.add_word_correction("deslgn", "Design")
    
    corrector = WordLevelCorrector(dictionary, verbose=True)
    
    test_cases = [
        ("DESLGN", "DESIGN"),  # All caps
        ("deslgn", "design"),  # All lower
        ("Deslgn", "Design"),  # Title case
        ("DeSLGN", "Design"),  # Mixed - uses correct word's case
    ]
    
    print("\n--- Testing case preservation ---")
    for input_text, expected in test_cases:
        result, _, _, _ = corrector.correct_text(input_text, 65.0)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")


def test_punctuation_handling():
    """Test that punctuation is preserved."""
    print("\n" + "=" * 60)
    print("Testing Punctuation Handling")
    print("=" * 60)
    
    dictionary = ScheduleDictionary()
    dictionary.add_word_correction("DeBlgn", "Design")
    
    corrector = WordLevelCorrector(dictionary, verbose=True)
    
    test_cases = [
        ("Web DeBlgn.", "Web Design."),
        ("Web DeBlgn, Graphic DeBlgn", "Web Design, Graphic Design"),
        ("(DeBlgn)", "(Design)"),
        ("DeBlgn: Introduction", "Design: Introduction"),
        ('"DeBlgn"', '"Design"'),
    ]
    
    print("\n--- Testing punctuation ---")
    for input_text, expected in test_cases:
        result, _, _, _ = corrector.correct_text(input_text, 65.0)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")


def test_dictionary_efficiency():
    """Show how word-level corrections are more efficient."""
    print("\n" + "=" * 60)
    print("Dictionary Efficiency Comparison")
    print("=" * 60)
    
    # Old way: phrase-level
    phrase_corrections = {
        "Web Design": ["Web DeBlgn", "Web Deslgn"],
        "Graphic Design": ["Graphic DeBlgn", "Graphic Deslgn"],
        "Machine Learning": ["Machlne Learnlng", "Machine Learnlng"],
        "Deep Learning": ["Deep Learnlng"],
    }
    
    # New way: word-level
    word_corrections = {
        "Design": ["DeBlgn", "Deslgn"],
        "Machine": ["Machlne"],
        "Learning": ["Learnlng"],
    }
    
    print("\nOld phrase-level approach:")
    print(f"  Entries: {sum(len(v) for v in phrase_corrections.values())}")
    print(f"  Covers: {len(phrase_corrections)} phrases")
    
    print("\nNew word-level approach:")
    print(f"  Entries: {sum(len(v) for v in word_corrections.values())}")
    print(f"  Covers: ANY combination of these words!")
    
    print("\n  Examples of what word-level handles:")
    dictionary = ScheduleDictionary()
    for word, variations in word_corrections.items():
        for var in variations:
            dictionary.add_word_correction(var, word)
    
    corrector = WordLevelCorrector(dictionary)
    
    examples = [
        "Web DeBlgn",
        "Graphic Deslgn",
        "UI DeBlgn",
        "DeBlgn Patterns",
        "Machlne Learnlng",
        "Deep Learnlng",
        "Reinforcement Learnlng",
    ]
    
    for ex in examples:
        result, _, _, _ = corrector.correct_text(ex, 65.0)
        print(f"    '{ex}' -> '{result}'")


if __name__ == "__main__":
    # Create template dictionary
    print("Creating word-level dictionary template...")
    create_word_level_dictionary("schedule_dictionary.json")
    
    # Run tests
    test_word_level_correction()
    test_case_preservation()
    test_punctuation_handling()
    test_dictionary_efficiency()