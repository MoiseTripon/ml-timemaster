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

# Use the same default path as dictionary_optimizer
DEFAULT_DICTIONARY_PATH = "ocr_dictionary.json"


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
    Dictionary manager with word-level and phrase-level correction support.
    
    Loads corrections from the shared dictionary file and provides lookup methods.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None, verbose: bool = False):
        """Initialize dictionary with correction support."""
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
        
        # Word-level corrections: incorrect_word (lowercase) -> correct_word
        self.word_corrections: Dict[str, str] = {}
        self.word_corrections_by_correct: Dict[str, List[str]] = {}
        
        # Phrase-level corrections: incorrect_phrase (lowercase) -> correct_phrase
        self.phrase_corrections: Dict[str, str] = {}
        self.phrase_corrections_by_correct: Dict[str, List[str]] = {}
        
        # All valid terms for validation
        self.all_terms: Set[str] = set()
        self.terms_lower_map: Dict[str, str] = {}
        
        # Custom terms loaded from file
        self.custom_terms: Dict[str, List[str]] = {}
        self.custom_word_corrections: Dict[str, List[str]] = {}
        self.custom_phrase_corrections: Dict[str, List[str]] = {}
        
        # File path - use shared default
        self.dictionary_path = self._find_dictionary_path(dictionary_path or DEFAULT_DICTIONARY_PATH)
        
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
        
        # Load and build
        self._load_dictionary()
        self._build_lookup()
        
        stats = self.get_stats()
        self.logger.info(f"Dictionary initialized from {self.dictionary_path}: "
                        f"{stats['word_corrections']} word corrections, "
                        f"{stats['phrase_corrections']} phrase corrections, "
                        f"{stats['total_terms']} terms")

    def _find_dictionary_path(self, path: str) -> str:
        """Find the dictionary file in various locations."""
        paths_to_check = [
            path,
            os.path.join("src", path),
            os.path.join("src", "ocr_dictionary.json"),
            "ocr_dictionary.json",
            os.path.join(os.path.dirname(__file__), path),
            os.path.join(os.path.dirname(__file__), "ocr_dictionary.json"),
        ]
        
        for p in paths_to_check:
            if os.path.exists(p):
                self.logger.info(f"Found dictionary at: {p}")
                return p
        
        self.logger.warning(f"Dictionary not found, checked: {paths_to_check}")
        return path

    def _load_dictionary(self):
        """Load dictionary from JSON file."""
        if not os.path.exists(self.dictionary_path):
            self.logger.info(f"No dictionary at {self.dictionary_path}, using defaults")
            return
        
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Loading dictionary from {self.dictionary_path}")
            self.logger.info(f"Dictionary keys: {list(data.keys())}")
            
            # Load terms
            self.custom_terms = data.get("terms", {})
            
            # Load word corrections
            self.custom_word_corrections = data.get("word_corrections", {})
            self.logger.info(f"Loaded {len(self.custom_word_corrections)} word correction entries")
            
            # Load phrase corrections
            self.custom_phrase_corrections = data.get("phrase_corrections", {})
            self.logger.info(f"Loaded {len(self.custom_phrase_corrections)} phrase correction entries")
            
            # Also load from typed_corrections if present
            typed_corrections = data.get("typed_corrections", {})
            if typed_corrections:
                self.logger.info(f"Found typed_corrections with types: {list(typed_corrections.keys())}")
                for type_name, corrections in typed_corrections.items():
                    for correct_val, variations in corrections.items():
                        # Determine if word or phrase based on spaces
                        is_word = ' ' not in correct_val
                        
                        for variation in variations:
                            var_is_word = ' ' not in variation
                            
                            if is_word and var_is_word:
                                # Both are single words - add to word corrections
                                if correct_val not in self.custom_word_corrections:
                                    self.custom_word_corrections[correct_val] = []
                                if variation not in self.custom_word_corrections[correct_val]:
                                    self.custom_word_corrections[correct_val].append(variation)
                            else:
                                # One or both are phrases - add to phrase corrections
                                if correct_val not in self.custom_phrase_corrections:
                                    self.custom_phrase_corrections[correct_val] = []
                                if variation not in self.custom_phrase_corrections[correct_val]:
                                    self.custom_phrase_corrections[correct_val].append(variation)
            
            self.logger.info(f"After merging typed: {len(self.custom_word_corrections)} word, "
                           f"{len(self.custom_phrase_corrections)} phrase corrections")
            
        except Exception as e:
            self.logger.warning(f"Failed to load dictionary: {e}")
            import traceback
            traceback.print_exc()

    def _build_lookup(self):
        """Build lookup structures for fast correction lookup."""
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
                for word in term.split():
                    if len(word) > 1:
                        self.all_terms.add(word)
                        self.terms_lower_map[word.lower()] = word
        
        # Build word corrections lookup
        # Format: correct_word -> [list of incorrect variations]
        # We need: incorrect_word (lowercase) -> correct_word
        for correct_word, variations in self.custom_word_corrections.items():
            self.word_corrections_by_correct[correct_word] = variations
            self.all_terms.add(correct_word)
            self.terms_lower_map[correct_word.lower()] = correct_word
            
            for variation in variations:
                # Map lowercase variation to correct word
                self.word_corrections[variation.lower()] = correct_word
                if self.verbose:
                    self.logger.debug(f"Word correction: '{variation}' -> '{correct_word}'")
        
        # Build phrase corrections lookup
        for correct_phrase, variations in self.custom_phrase_corrections.items():
            self.phrase_corrections_by_correct[correct_phrase] = variations
            for variation in variations:
                self.phrase_corrections[variation.lower()] = correct_phrase
                if self.verbose:
                    self.logger.debug(f"Phrase correction: '{variation}' -> '{correct_phrase}'")
        
        self.logger.info(f"Built lookup: {len(self.word_corrections)} word mappings, "
                        f"{len(self.phrase_corrections)} phrase mappings")

    def reload(self):
        """Reload dictionary from file."""
        self.logger.info("Reloading dictionary...")
        self._load_dictionary()
        self._build_lookup()

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
        """Add a word-level correction."""
        if not incorrect or not correct:
            return False
        
        incorrect = incorrect.strip()
        correct = correct.strip()
        
        if incorrect.lower() == correct.lower():
            return False
        
        if ' ' in incorrect or ' ' in correct:
            self.logger.warning(f"Use add_phrase_correction for multi-word: '{incorrect}' -> '{correct}'")
            return False
        
        if correct not in self.custom_word_corrections:
            self.custom_word_corrections[correct] = []
        
        if incorrect not in self.custom_word_corrections[correct]:
            self.custom_word_corrections[correct].append(incorrect)
            self.word_corrections[incorrect.lower()] = correct
            self.word_corrections_by_correct[correct] = self.custom_word_corrections[correct]
            self.all_terms.add(correct)
            self.terms_lower_map[correct.lower()] = correct
            
            self.logger.info(f"Added word correction: '{incorrect}' -> '{correct}'")
            return True
        
        return False

    def add_word_corrections_batch(self, correct_word: str, variations: List[str]) -> int:
        """Add multiple incorrect variations for a word."""
        added = 0
        for variation in variations:
            if self.add_word_correction(variation, correct_word):
                added += 1
        return added

    def get_word_correction(self, word: str) -> Optional[str]:
        """
        Get correction for a single word (exact match on lowercase).
        
        Args:
            word: The word to look up
            
        Returns:
            Corrected word or None if no correction found
        """
        if not word:
            return None
        
        correction = self.word_corrections.get(word.lower())
        if correction and self.verbose:
            self.logger.debug(f"Found word correction: '{word}' -> '{correction}'")
        return correction

    def find_similar_word_correction(self, word: str, threshold: float = 0.80) -> Optional[Tuple[str, float]]:
        """Find similar word correction using fuzzy matching."""
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
    # Phrase-Level Correction Methods
    # =========================================================================
    
    def add_phrase_correction(self, incorrect: str, correct: str) -> bool:
        """Add a phrase-level correction."""
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
        """Get correction for a phrase (exact match on lowercase)."""
        if not phrase:
            return None
        
        correction = self.phrase_corrections.get(phrase.lower())
        if correction and self.verbose:
            self.logger.debug(f"Found phrase correction: '{phrase}' -> '{correction}'")
        return correction

    def find_similar_phrase_correction(self, phrase: str, threshold: float = 0.85) -> Optional[Tuple[str, float]]:
        """Find similar phrase correction using fuzzy matching."""
        if not phrase or len(phrase) < 3:
            return None
        
        phrase_lower = phrase.lower()
        
        # Exact match first
        if phrase_lower in self.phrase_corrections:
            return (self.phrase_corrections[phrase_lower], 1.0)
        
        # Fuzzy match
        best_match = None
        best_ratio = threshold
        
        for incorrect_lower, correct in self.phrase_corrections.items():
            if abs(len(phrase) - len(incorrect_lower)) > max(5, len(phrase) * 0.3):
                continue
            
            ratio = SequenceMatcher(None, phrase_lower, incorrect_lower).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = correct
        
        if best_match:
            return (best_match, best_ratio)
        
        return None

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
            "unique_correct_phrases": len(self.phrase_corrections_by_correct),
            "categories": len(self.custom_terms)
        }


class WordLevelCorrector:
    """
    Corrects OCR text word by word and phrase by phrase.
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
        Correct OCR text using word and phrase corrections.
        
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
        
        # Step 1: Check for full phrase correction (exact match)
        phrase_correction = self.dictionary.get_phrase_correction(text)
        if phrase_correction:
            self.stats['phrases_corrected'] += 1
            if self.verbose:
                self.logger.debug(f"Full phrase correction: '{text}' -> '{phrase_correction}'")
            return phrase_correction, min(confidence + 15, 98), True, [(text, phrase_correction)]
        
        # Step 2: Try fuzzy phrase matching for longer texts with low confidence
        if len(text) > 10 and confidence < 70:
            fuzzy_phrase = self.dictionary.find_similar_phrase_correction(text, threshold=0.85)
            if fuzzy_phrase:
                self.stats['phrases_corrected'] += 1
                if self.verbose:
                    self.logger.debug(f"Fuzzy phrase correction ({fuzzy_phrase[1]:.2f}): '{text}' -> '{fuzzy_phrase[0]}'")
                return fuzzy_phrase[0], min(confidence + 12, 95), True, [(text, fuzzy_phrase[0])]
        
        # Step 3: Check for partial phrase corrections within the text
        corrected_text = text
        for incorrect_phrase, correct_phrase in self.dictionary.phrase_corrections.items():
            if incorrect_phrase in corrected_text.lower():
                # Find the actual case in the original text
                start_idx = corrected_text.lower().find(incorrect_phrase)
                if start_idx >= 0:
                    original_segment = corrected_text[start_idx:start_idx + len(incorrect_phrase)]
                    corrected_text = (corrected_text[:start_idx] + 
                                     correct_phrase + 
                                     corrected_text[start_idx + len(incorrect_phrase):])
                    corrections_made.append((original_segment, correct_phrase))
                    self.stats['phrases_corrected'] += 1
                    if self.verbose:
                        self.logger.debug(f"Partial phrase correction: '{original_segment}' -> '{correct_phrase}'")
        
        # Step 4: Word-by-word correction
        tokens = self._tokenize(corrected_text)
        corrected_tokens = []
        
        for token, is_word in tokens:
            if not is_word:
                corrected_tokens.append(token)
                continue
            
            self.stats['words_processed'] += 1
            
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
        
        corrected_text = ''.join(corrected_tokens)
        
        was_corrected = len(corrections_made) > 0
        if was_corrected:
            conf_boost = min(len(corrections_made) * 5, 15)
            new_confidence = min(confidence + conf_boost, 95)
        else:
            new_confidence = confidence
        
        return corrected_text, new_confidence, was_corrected, corrections_made

    def _tokenize(self, text: str) -> List[Tuple[str, bool]]:
        """Split text into tokens, preserving separators."""
        tokens = []
        parts = self.word_pattern.split(text)
        
        for part in parts:
            if not part:
                continue
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
        if len(word) < 2:
            return word, 'none'
        
        if self.number_pattern.match(word) or self.time_pattern.match(word):
            return word, 'none'
        
        # Step 1: Exact word correction
        exact_correction = self.dictionary.get_word_correction(word)
        if exact_correction:
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
        
        # Step 4: Fuzzy term matching
        if confidence < 80:
            fuzzy_term = self.dictionary.find_similar_term(word, threshold=0.80)
            if fuzzy_term:
                return self._apply_case(word, fuzzy_term[0]), 'fuzzy'
        
        # Track unrecognized word
        if confidence < 75 and len(word) > 2:
            self._track_unrecognized(word)
        
        return word, 'none'

    def _apply_case(self, original: str, corrected: str) -> str:
        """Apply the case pattern from original to corrected."""
        if not original or not corrected:
            return corrected
        
        if original.isupper():
            return corrected.upper()
        if original.islower():
            return corrected.lower()
        if original[0].isupper() and (len(original) == 1 or original[1:].islower()):
            return corrected.capitalize()
        
        return corrected

    def _track_unrecognized(self, word: str):
        """Track unrecognized words for review."""
        normalized = word.lower()
        self.unrecognized_words[normalized] += 1
        self.stats['unrecognized'] += 1

    def get_unrecognized_words(self, min_occurrences: int = 1) -> List[Tuple[str, int]]:
        """Get unrecognized words sorted by frequency."""
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
        
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose_logging else logging.INFO)
        
        self.lang = self._parse_lang(languages)
        
        # Initialize dictionary and corrector - use shared default path
        if self.enable_validation:
            dict_path = dictionary_path or DEFAULT_DICTIONARY_PATH
            self.dictionary = ScheduleDictionary(dict_path, verbose=verbose_logging)
            self.corrector = WordLevelCorrector(self.dictionary, verbose=verbose_logging)
            self.logger.info(f"Word-level correction enabled with dictionary: {self.dictionary.dictionary_path}")
        else:
            self.dictionary = None
            self.corrector = None
        
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
                CellOCR._ocr_instance = PaddleOCR(
                    lang="latin",
                    show_log=False,

                    # Detection (good for small printed text)
                    text_det_limit_type="max",
                    text_det_limit_side_len=1536,
                    text_det_thresh=0.25,
                    text_det_box_thresh=0.55,
                    text_det_unclip_ratio=1.5,

                    # Recognition
                    text_rec_score_thresh=0.5,
                    # Wider recognition input to handle long single-line text without truncation
                    text_rec_input_shape="3,48,1280",
                    text_recognition_batch_size=1,
                )
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
        """Run OCR on image with sliding window for long text."""
        self.stats['ocr_calls'] += 1

        try:
            prepared = self._prepare_image_for_ocr(img)
            h, w = prepared.shape[:2]

            # For wide images (long text), use sliding window recognition
            # The recognizer degrades badly beyond ~30-40 chars worth of width
            # Threshold: if width > height * 6 and width > 400px, use sliding window
            if w > h * 6 and w > 400:
                sw_text, sw_conf = self._run_ocr_sliding_window(prepared)
                if sw_text and sw_conf > 20:
                    # Also try full rec-only for comparison
                    full_text, full_conf = self._run_ocr_rec_only(prepared)

                    # Also try detection-based
                    det_text, det_conf = self._run_ocr_with_det(prepared)

                    # Pick the best result based on confidence and text quality
                    candidates = [
                        (sw_text, sw_conf, "sliding_window"),
                        (full_text, full_conf, "rec_only"),
                        (det_text, det_conf, "det"),
                    ]

                    best_text, best_conf, best_method = max(
                        [(t, c, m) for t, c, m in candidates if t],
                        key=lambda x: self._score_long_text(x[0], x[1]),
                        default=("", 0.0, "none")
                    )

                    if self.verbose:
                        self.logger.debug(f"Long text candidates:")
                        for t, c, m in candidates:
                            if t:
                                self.logger.debug(f"  [{m}] conf={c:.1f} '{t[:80]}...'")
                        self.logger.debug(f"  Selected: [{best_method}]")

                    return best_text, best_conf

            # Standard path for normal-width cells
            text, conf = self._run_ocr_with_det(prepared)

            # Fallback for medium-length text with poor results
            if text and conf < 50 and len(text) > 20:
                rec_text, rec_conf = self._run_ocr_rec_only(prepared)
                if rec_text and rec_conf > conf:
                    return rec_text, rec_conf

            return text, conf

        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return "", 0.0

    def _run_ocr_sliding_window(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR using a sliding window approach for long text.
        
        Splits the image into overlapping windows, runs recognition on each,
        then stitches results together by matching overlapping text regions.
        
        Args:
            img: Prepared BGR image (already scaled appropriately)
            
        Returns:
            Tuple of (combined_text, average_confidence)
        """
        h, w = img.shape[:2]

        # Window width: aim for ~30 characters worth of pixels
        # Estimate char width from image height (for typical fonts, char width ≈ 0.5-0.7 * height)
        estimated_char_width = max(int(h * 0.55), 10)
        target_chars_per_window = 30
        window_width = min(estimated_char_width * target_chars_per_window, w)

        # Overlap: ~40% of window width to ensure sufficient text overlap for matching
        overlap = int(window_width * 0.4)

        # Minimum window width sanity check
        if window_width < 100:
            window_width = min(400, w)
            overlap = int(window_width * 0.4)

        # If image isn't much wider than one window, just do full rec
        if w <= window_width * 1.3:
            return self._run_ocr_rec_only(img)

        # Generate window positions
        stride = window_width - overlap
        windows = []
        x = 0
        while x < w:
            x_end = min(x + window_width, w)
            # Don't create tiny trailing windows
            if x > 0 and (x_end - x) < window_width * 0.4:
                # Extend previous window or adjust this one
                x = max(0, w - window_width)
                x_end = w
                windows.append((x, x_end))
                break
            windows.append((x, x_end))
            if x_end >= w:
                break
            x += stride

        if self.verbose:
            self.logger.debug(
                f"Sliding window: img={w}x{h}, window={window_width}, "
                f"overlap={overlap}, stride={stride}, n_windows={len(windows)}"
            )

        # Run recognition on each window
        window_results = []
        for i, (x_start, x_end) in enumerate(windows):
            window_img = img[:, x_start:x_end].copy()

            # Ensure the window image is contiguous
            if not window_img.flags['C_CONTIGUOUS']:
                window_img = np.ascontiguousarray(window_img)

            text, conf = self._run_ocr_rec_only(window_img)

            if not text:
                # Try with detection as fallback for this window
                text, conf = self._run_ocr_with_det(window_img)

            window_results.append({
                'text': text or "",
                'conf': conf,
                'x_start': x_start,
                'x_end': x_end,
                'index': i,
            })

            if self.verbose:
                self.logger.debug(f"  Window {i} [{x_start}:{x_end}]: conf={conf:.1f} '{text}'")

        # Stitch windows together using overlap matching
        combined_text, avg_conf = self._stitch_window_results(window_results, overlap, window_width)

        return combined_text, avg_conf

    def _stitch_window_results(
        self,
        window_results: List[Dict[str, Any]],
        overlap_pixels: int,
        window_width: int
    ) -> Tuple[str, float]:
        """
        Stitch sliding window OCR results by matching overlapping text.
        
        For each pair of adjacent windows, the overlap region produces text
        in both windows. We find the best alignment between the suffix of
        window N and the prefix of window N+1, then merge accordingly.
        
        Args:
            window_results: List of window OCR results
            overlap_pixels: Pixel overlap between windows
            window_width: Total window width in pixels
            
        Returns:
            Tuple of (stitched_text, average_confidence)
        """
        if not window_results:
            return "", 0.0

        # Filter out empty results but keep position info
        valid_results = [r for r in window_results if r['text'].strip()]

        if not valid_results:
            return "", 0.0

        if len(valid_results) == 1:
            return valid_results[0]['text'], valid_results[0]['conf']

        # Estimate what fraction of the text comes from the overlap region
        overlap_fraction = overlap_pixels / window_width if window_width > 0 else 0.4

        combined_text = valid_results[0]['text']
        total_conf = valid_results[0]['conf']
        conf_count = 1

        for i in range(1, len(valid_results)):
            prev_text = combined_text
            curr_text = valid_results[i]['text']
            curr_conf = valid_results[i]['conf']

            if not curr_text.strip():
                continue

            # Estimate overlap in characters
            # The overlap region appears in both the end of prev_text and start of curr_text
            estimated_overlap_chars = max(
                int(len(curr_text) * overlap_fraction),
                int(len(prev_text) * overlap_fraction),
                3  # minimum overlap to search for
            )

            # Find the best merge point
            merged = self._find_overlap_and_merge(prev_text, curr_text, estimated_overlap_chars)

            combined_text = merged
            total_conf += curr_conf
            conf_count += 1

        avg_conf = total_conf / conf_count if conf_count > 0 else 0.0

        return combined_text.strip(), avg_conf

    def _find_overlap_and_merge(
        self,
        text_a: str,
        text_b: str,
        estimated_overlap_chars: int
    ) -> str:
        """
        Find overlapping text between the suffix of text_a and prefix of text_b,
        then merge them.
        
        Uses multiple strategies:
        1. Exact substring matching
        2. Fuzzy matching with SequenceMatcher
        3. Character-level alignment
        
        Args:
            text_a: First text (we keep the beginning)
            text_b: Second text (we keep the end)
            estimated_overlap_chars: Estimated number of overlapping characters
            
        Returns:
            Merged text
        """
        if not text_a:
            return text_b
        if not text_b:
            return text_a

        # Search range: look at more than the estimated overlap to be safe
        search_len_a = min(len(text_a), int(estimated_overlap_chars * 1.8) + 10)
        search_len_b = min(len(text_b), int(estimated_overlap_chars * 1.8) + 10)

        suffix_a = text_a[-search_len_a:]  # End of first window
        prefix_b = text_b[:search_len_b]   # Start of second window

        # Strategy 1: Find longest exact substring match
        best_match_len = 0
        best_a_end = len(text_a)
        best_b_start = 0

        # Try matching suffix of A with prefix of B
        min_match = max(3, estimated_overlap_chars // 3)

        for match_len in range(min(len(suffix_a), len(prefix_b)), min_match - 1, -1):
            suffix_end = suffix_a[-match_len:]
            prefix_start = prefix_b[:match_len]

            if suffix_end == prefix_start:
                best_match_len = match_len
                best_a_end = len(text_a)  # Keep all of A
                best_b_start = match_len  # Skip matched part of B
                break

        if best_match_len >= min_match:
            merged = text_a[:best_a_end] + text_b[best_b_start:]
            if self.verbose:
                self.logger.debug(
                    f"  Exact overlap match ({best_match_len} chars): "
                    f"'{suffix_a[-best_match_len:]}'"
                )
            return merged

        # Strategy 2: Fuzzy matching - find best alignment
        best_ratio = 0.0
        best_split_a = len(text_a)
        best_split_b = 0

        # Try different split points
        min_overlap = max(3, estimated_overlap_chars // 4)
        max_overlap = min(len(suffix_a), len(prefix_b), estimated_overlap_chars * 2)

        for overlap_len in range(min_overlap, max_overlap + 1):
            # Compare end of A with start of B
            a_segment = text_a[-(overlap_len):]
            b_segment = text_b[:overlap_len]

            # Quick length filter
            if abs(len(a_segment) - len(b_segment)) > max(3, overlap_len * 0.3):
                continue

            ratio = SequenceMatcher(None, a_segment.lower(), b_segment.lower()).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_split_a = len(text_a) - overlap_len
                best_split_b = overlap_len

        # Accept fuzzy match if similarity is high enough
        if best_ratio >= 0.55:
            # Decide which version of the overlap to keep based on confidence
            # Generally prefer the version from the window with higher confidence
            # For simplicity, keep text_a's version up to the split point
            merged = text_a[:len(text_a)] + text_b[best_split_b:]

            if self.verbose:
                overlap_a = text_a[best_split_a:]
                overlap_b = text_b[:best_split_b]
                self.logger.debug(
                    f"  Fuzzy overlap match (ratio={best_ratio:.2f}): "
                    f"A='{overlap_a}' vs B='{overlap_b}'"
                )

            # If the overlap from A and B are very different, try character-level merge
            if best_ratio < 0.75:
                overlap_a = text_a[best_split_a:]
                overlap_b = text_b[:best_split_b]
                better_overlap = self._pick_better_overlap(overlap_a, overlap_b)
                merged = text_a[:best_split_a] + better_overlap + text_b[best_split_b:]

            return merged

        # Strategy 3: No good overlap found
        # This might mean the windows captured completely different text regions
        # or the overlap estimation was wrong. 
        # Try a smaller overlap search with word boundaries
        word_merge = self._try_word_boundary_merge(text_a, text_b, estimated_overlap_chars)
        if word_merge:
            return word_merge

        # Last resort: just concatenate with a space, removing likely duplicated end/start
        # Trim a small portion that's likely garbled overlap
        trim_chars = max(2, estimated_overlap_chars // 3)
        if len(text_a) > trim_chars:
            trimmed = text_a[:-trim_chars] + " " + text_b[trim_chars:]
        else:
            trimmed = text_a + " " + text_b

        if self.verbose:
            self.logger.debug(
                f"  No overlap found, concatenating with trim ({trim_chars} chars)"
            )

        return trimmed

    def _pick_better_overlap(self, overlap_a: str, overlap_b: str) -> str:
        """
        Pick the better version of overlapping text based on text quality heuristics.
        
        Prefers text with:
        - More alphabetic characters
        - Fewer repeated characters
        - More spaces (word separators)
        - Fewer suspicious patterns
        """
        def score_text(t: str) -> float:
            if not t:
                return 0.0
            s = 0.0
            # Prefer more alpha chars
            alpha_ratio = sum(1 for c in t if c.isalpha()) / len(t)
            s += alpha_ratio * 10

            # Prefer spaces (indicates word separation)
            space_ratio = t.count(' ') / max(len(t), 1)
            s += space_ratio * 5

            # Penalize repeated characters (sign of garbled text)
            for i in range(len(t) - 2):
                if t[i] == t[i+1] == t[i+2]:
                    s -= 3

            # Penalize all-caps runs > 10 chars without spaces (likely garbled)
            caps_run = 0
            for c in t:
                if c.isupper():
                    caps_run += 1
                    if caps_run > 10:
                        s -= 1
                else:
                    caps_run = 0

            return s

        score_a = score_text(overlap_a)
        score_b = score_text(overlap_b)

        if self.verbose:
            self.logger.debug(
                f"  Overlap quality: A({score_a:.1f})='{overlap_a}' vs B({score_b:.1f})='{overlap_b}'"
            )

        return overlap_a if score_a >= score_b else overlap_b

    def _try_word_boundary_merge(
        self,
        text_a: str,
        text_b: str,
        estimated_overlap_chars: int
    ) -> Optional[str]:
        """
        Try to merge texts by finding matching words at the boundary.
        
        Looks for common words between the end of text_a and start of text_b.
        """
        words_a = text_a.split()
        words_b = text_b.split()

        if not words_a or not words_b:
            return None

        # Look for matching words in the overlap region
        # Check last N words of A against first N words of B
        max_words_to_check = max(2, estimated_overlap_chars // 4)
        search_words_a = words_a[-max_words_to_check:]
        search_words_b = words_b[:max_words_to_check]

        best_word_match = 0
        best_a_word_idx = len(words_a)
        best_b_word_idx = 0

        for i, word_a in enumerate(search_words_a):
            for j, word_b in enumerate(search_words_b):
                # Check for exact word match
                if word_a.lower() == word_b.lower() and len(word_a) >= 3:
                    # Found matching word - check if subsequent words also match
                    match_count = 1
                    ai = len(words_a) - len(search_words_a) + i
                    bi = j

                    while (ai + match_count < len(words_a) and
                        bi + match_count < len(words_b)):
                        if words_a[ai + match_count].lower() == words_b[bi + match_count].lower():
                            match_count += 1
                        else:
                            break

                    if match_count > best_word_match:
                        best_word_match = match_count
                        best_a_word_idx = ai
                        best_b_word_idx = bi + match_count

                # Check for fuzzy word match (handles minor OCR differences)
                elif len(word_a) >= 3 and len(word_b) >= 3:
                    ratio = SequenceMatcher(None, word_a.lower(), word_b.lower()).ratio()
                    if ratio >= 0.75:
                        ai = len(words_a) - len(search_words_a) + i
                        if best_word_match == 0:
                            best_word_match = 1
                            best_a_word_idx = ai
                            best_b_word_idx = j + 1

        if best_word_match > 0:
            # Merge at word boundary
            merged_words = words_a[:best_a_word_idx + best_word_match] + words_b[best_b_word_idx:]
            merged = " ".join(merged_words)

            if self.verbose:
                self.logger.debug(
                    f"  Word boundary merge: matched {best_word_match} word(s) "
                    f"at A[{best_a_word_idx}], B[{best_b_word_idx - best_word_match}]"
                )

            return merged

        return None

    def _score_long_text(self, text: str, confidence: float) -> float:
        """
        Score a long text result for comparison between methods.
        
        Considers:
        - Confidence
        - Text length (longer is generally better for long text cells)
        - Text quality (spaces, alpha chars, etc.)
        """
        if not text:
            return 0.0

        score = confidence

        # Reward reasonable length
        text_len = len(text.strip())
        if text_len > 10:
            score += min(text_len * 0.3, 20)

        # Reward word separations (spaces)
        word_count = len(text.split())
        if word_count > 1:
            score += min(word_count * 2, 15)

        # Penalize garbled text indicators
        # - Many consecutive same characters
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2] and text[i].isalpha():
                score -= 5

        # - Very few spaces relative to length
        if text_len > 20 and word_count < text_len // 20:
            score -= 15

        # - Too many uppercase without spaces (garbled)
        upper_run = 0
        max_upper_run = 0
        for c in text:
            if c.isupper():
                upper_run += 1
                max_upper_run = max(max_upper_run, upper_run)
            else:
                upper_run = 0
        if max_upper_run > 15:
            score -= 10

        return score

    def _run_ocr_with_det(self, prepared: np.ndarray) -> Tuple[str, float]:
        """Run OCR with detection enabled (standard mode)."""
        try:
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
            self.logger.error(f"OCR with det error: {e}")
            return "", 0.0

    def _run_ocr_rec_only(self, prepared: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR in recognition-only mode (no detection).
        Treats the entire image as a single text region.
        """
        try:
            try:
                result = self.ocr.ocr(prepared, det=False, rec=True, cls=False)
            except TypeError:
                try:
                    result = self.ocr.ocr(prepared, det=False, rec=True)
                except:
                    return "", 0.0

            if result is None:
                return "", 0.0

            text = ""
            conf = 0.0

            if isinstance(result, list):
                for item in result:
                    if item is None:
                        continue

                    if isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, (list, tuple)) and len(sub_item) >= 2:
                                t, c = str(sub_item[0]).strip(), float(sub_item[1])
                                if t and 0 <= c <= 1:
                                    if len(t) > len(text):
                                        text = t
                                        conf = c
                            elif isinstance(sub_item, dict):
                                t = str(sub_item.get('text', sub_item.get('rec_text', ''))).strip()
                                c = float(sub_item.get('score', sub_item.get('rec_score', 0)))
                                if t and len(t) > len(text):
                                    text = t
                                    conf = c

                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        t, c = str(item[0]).strip(), float(item[1])
                        if t and 0 <= c <= 1:
                            if len(t) > len(text):
                                text = t
                                conf = c

                    elif isinstance(item, dict):
                        t = str(item.get('text', item.get('rec_text', ''))).strip()
                        c = float(item.get('score', item.get('rec_score', 0)))
                        if t and len(t) > len(text):
                            text = t
                            conf = c

            return text, conf * 100

        except Exception as e:
            self.logger.error(f"OCR rec-only error: {e}")
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
                            self.logger.debug(f"  '{orig}' -> '{corr}'")
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
                           f"({cs['exact_corrections']} exact, {cs['fuzzy_corrections']} fuzzy), "
                           f"Phrases: {cs['phrases_corrected']}")
        
        return results

    # =========================================================================
    # Dictionary and Correction API
    # =========================================================================
    
    def reload_dictionary(self):
        """Reload the dictionary from file."""
        if self.dictionary:
            self.dictionary.reload()
            self.logger.info("Dictionary reloaded")

    def add_word_correction(self, incorrect: str, correct: str) -> bool:
        """Add a word-level correction."""
        if self.dictionary:
            return self.dictionary.add_word_correction(incorrect, correct)
        return False

    def add_word_corrections_batch(self, correct_word: str, variations: List[str]) -> int:
        """Add multiple incorrect variations for a word."""
        if self.dictionary:
            return self.dictionary.add_word_corrections_batch(correct_word, variations)
        return 0

    def add_phrase_correction(self, incorrect: str, correct: str) -> bool:
        """Add a phrase-level correction."""
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
        """Get words that weren't corrected, sorted by frequency."""
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
        if stats['cells']  > 0:
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
# Tests
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing ScheduleDictionary loading...")
    
    # Create a test dictionary
    test_dict = {
        "terms": {
            "days": ["Monday", "Tuesday"],
            "subjects": ["Physics", "Mathematics"]
        },
        "word_corrections": {
            "Monday": ["Mond4y", "M0nday"],
            "Physics": ["Physlcs", "Phys1cs"],
            "Lecture": ["Lcture", "Lectur3"]
        },
        "phrase_corrections": {
            "Conf. dr.": ["Caat dc.", "Conf dc."],
            "T Petrisor": ["TRetris", "T Petriso"]
        },
        "typed_corrections": {
            "professor": {
                "Conf. dr.": ["Caat dc."],
                "T Petrisor": ["TRetris"]
            },
            "day": {
                "Monday": ["Mond4y"]
            }
        }
    }
    
    # Save test dictionary
    test_path = "test_ocr_dictionary.json"
    with open(test_path, 'w') as f:
        json.dump(test_dict, f, indent=2)
    
    # Test loading
    dictionary = ScheduleDictionary(test_path, verbose=True)
    print(f"\nStats: {dictionary.get_stats()}")
    
    # Test corrections
    corrector = WordLevelCorrector(dictionary, verbose=True)
    
    test_cases = [
        ("Mond4y Lcture", 65.0),
        ("Physlcs Lab", 70.0),
        ("Caat dc. TRetris", 60.0),
        ("Hello World", 80.0),
    ]
    
    print("\n--- Testing corrections ---")
    for text, conf in test_cases:
        result, new_conf, corrected, corrections = corrector.correct_text(text, conf)
        print(f"'{text}' -> '{result}' (corrected: {corrected})")
        if corrections:
            for orig, corr in corrections:
                print(f"    '{orig}' -> '{corr}'")
    
    # Cleanup
    import os
    os.remove(test_path)
    print("\n✓ Tests completed")