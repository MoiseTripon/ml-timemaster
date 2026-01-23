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
        languages: str = "en+ron",
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

        # Initialize dictionary, correctors
        if self.enable_validation:
            dict_path = dictionary_path or DEFAULT_DICTIONARY_PATH
            self.dictionary = ScheduleDictionary(dict_path, verbose=verbose_logging)
            self.corrector = WordLevelCorrector(self.dictionary, verbose=verbose_logging)
            self.char_corrector = CharacterLevelCorrector(self.dictionary, verbose=verbose_logging)
            self.logger.info(f"Character + Word-level correction enabled with dictionary: "
                            f"{self.dictionary.dictionary_path}")
        else:
            self.dictionary = None
            self.corrector = None
            self.char_corrector = None

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
            if w > h * 6 and w > 400:
                sw_text, sw_conf = self._run_ocr_sliding_window(prepared)
                full_text, full_conf = self._run_ocr_rec_only(prepared)
                det_text, det_conf = self._run_ocr_with_det(prepared)

                # Ensure no None values
                sw_text = sw_text or ""
                full_text = full_text or ""
                det_text = det_text or ""

                candidates = [
                    (sw_text, sw_conf, "sliding_window"),
                    (full_text, full_conf, "rec_only"),
                    (det_text, det_conf, "det"),
                ]

                # Filter to non-empty candidates
                valid_candidates = [(t, c, m) for t, c, m in candidates if t]

                if valid_candidates:
                    best_text, best_conf, best_method = max(
                        valid_candidates,
                        key=lambda x: self._score_long_text(x[0], x[1]),
                    )

                    if self.verbose:
                        self.logger.debug(f"Long text candidates:")
                        for t, c, m in candidates:
                            if t:
                                self.logger.debug(f"  [{m}] conf={c:.1f} '{t[:80]}'")
                        self.logger.debug(f"  Selected: [{best_method}]")

                    return best_text, best_conf

                # All candidates empty
                return "", 0.0

            # Standard path for normal-width cells
            text, conf = self._run_ocr_with_det(prepared)
            text = text or ""

            # Fallback for medium-length text with poor results
            if text and conf < 50 and len(text) > 20:
                rec_text, rec_conf = self._run_ocr_rec_only(prepared)
                rec_text = rec_text or ""
                if rec_text and rec_conf > conf:
                    return rec_text, rec_conf

            return text, conf

        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return "", 0.0

    def _run_ocr_sliding_window(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR using a sliding window approach for long text.
        """
        h, w = img.shape[:2]

        estimated_char_width = max(int(h * 0.55), 10)
        target_chars_per_window = 30
        window_width = min(estimated_char_width * target_chars_per_window, w)

        overlap = int(window_width * 0.4)

        if window_width < 100:
            window_width = min(400, w)
            overlap = int(window_width * 0.4)

        # If image isn't much wider than one window, just do full rec
        if w <= window_width * 1.3:
            return self._run_ocr_rec_only(img)

        # Generate window positions
        stride = window_width - overlap
        if stride <= 0:
            stride = max(window_width // 2, 1)

        windows = []
        x = 0
        while x < w:
            x_end = min(x + window_width, w)
            # Don't create tiny trailing windows
            if x > 0 and (x_end - x) < window_width * 0.4:
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

            if not window_img.flags['C_CONTIGUOUS']:
                window_img = np.ascontiguousarray(window_img)

            text, conf = self._run_ocr_rec_only(window_img)
            text = text or ""

            if not text:
                text, conf = self._run_ocr_with_det(window_img)
                text = text or ""

            window_results.append({
                'text': text,
                'conf': conf,
                'x_start': x_start,
                'x_end': x_end,
                'index': i,
            })

            if self.verbose:
                self.logger.debug(f"  Window {i} [{x_start}:{x_end}]: conf={conf:.1f} '{text}'")

        # Stitch windows together using overlap matching
        combined_text, avg_conf = self._stitch_window_results(window_results, overlap, window_width)
        combined_text = combined_text or ""

        return combined_text, avg_conf

    def _stitch_window_results(
        self,
        window_results: List[Dict[str, Any]],
        overlap_pixels: int,
        window_width: int
    ) -> Tuple[str, float]:
        """
        Stitch sliding window OCR results by matching overlapping text.
        """
        if not window_results:
            return "", 0.0

        valid_results = [r for r in window_results if r.get('text', '').strip()]

        if not valid_results:
            return "", 0.0

        if len(valid_results) == 1:
            return valid_results[0]['text'] or "", valid_results[0]['conf']

        overlap_fraction = overlap_pixels / window_width if window_width > 0 else 0.4

        combined_text = valid_results[0].get('text', '') or ""
        total_conf = valid_results[0].get('conf', 0.0)
        conf_count = 1

        for i in range(1, len(valid_results)):
            curr_text = valid_results[i].get('text', '') or ""
            curr_conf = valid_results[i].get('conf', 0.0)

            if not curr_text.strip():
                continue

            estimated_overlap_chars = max(
                int(len(curr_text) * overlap_fraction),
                int(len(combined_text) * overlap_fraction),
                3
            )

            merged = self._find_overlap_and_merge(combined_text, curr_text, estimated_overlap_chars)
            combined_text = merged or combined_text
            total_conf += curr_conf
            conf_count += 1

        avg_conf = total_conf / conf_count if conf_count > 0 else 0.0

        result = combined_text.strip() if combined_text else ""
        return result, avg_conf

    def _find_overlap_and_merge(
        self,
        text_a: str,
        text_b: str,
        estimated_overlap_chars: int
    ) -> str:
        """
        Find overlapping text between the suffix of text_a and prefix of text_b,
        then merge them.
        """
        if not text_a:
            return text_b or ""
        if not text_b:
            return text_a or ""

        search_len_a = min(len(text_a), int(estimated_overlap_chars * 1.8) + 10)
        search_len_b = min(len(text_b), int(estimated_overlap_chars * 1.8) + 10)

        suffix_a = text_a[-search_len_a:]
        prefix_b = text_b[:search_len_b]

        # Strategy 1: Find longest exact substring match
        best_match_len = 0
        best_b_start = 0

        min_match = max(3, estimated_overlap_chars // 3)

        for match_len in range(min(len(suffix_a), len(prefix_b)), min_match - 1, -1):
            suffix_end = suffix_a[-match_len:]
            prefix_start = prefix_b[:match_len]

            if suffix_end == prefix_start:
                best_match_len = match_len
                best_b_start = match_len
                break

        if best_match_len >= min_match:
            merged = text_a + text_b[best_b_start:]
            if self.verbose:
                self.logger.debug(
                    f"  Exact overlap match ({best_match_len} chars): "
                    f"'{suffix_a[-best_match_len:]}'"
                )
            return merged

        # Strategy 2: Fuzzy matching - find best alignment
        best_ratio = 0.0
        best_split_b = 0

        min_overlap = max(3, estimated_overlap_chars // 4)
        max_overlap = min(len(suffix_a), len(prefix_b), estimated_overlap_chars * 2)

        for overlap_len in range(min_overlap, max_overlap + 1):
            a_segment = text_a[-overlap_len:]
            b_segment = text_b[:overlap_len]

            if abs(len(a_segment) - len(b_segment)) > max(3, overlap_len * 0.3):
                continue

            ratio = SequenceMatcher(None, a_segment.lower(), b_segment.lower()).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_split_b = overlap_len

        if best_ratio >= 0.55:
            # If the overlap from A and B are very different, pick the better one
            if best_ratio < 0.75:
                overlap_a = text_a[-best_split_b:] if best_split_b <= len(text_a) else text_a
                overlap_b = text_b[:best_split_b]
                better_overlap = self._pick_better_overlap(overlap_a, overlap_b)
                split_a = len(text_a) - best_split_b if best_split_b <= len(text_a) else 0
                merged = text_a[:split_a] + better_overlap + text_b[best_split_b:]
            else:
                merged = text_a + text_b[best_split_b:]

            if self.verbose:
                self.logger.debug(
                    f"  Fuzzy overlap match (ratio={best_ratio:.2f}), skip {best_split_b} chars from B"
                )

            return merged

        # Strategy 3: Word boundary merge
        word_merge = self._try_word_boundary_merge(text_a, text_b, estimated_overlap_chars)
        if word_merge:
            return word_merge

        # Last resort: concatenate with trim
        trim_chars = max(2, estimated_overlap_chars // 3)
        if len(text_a) > trim_chars and len(text_b) > trim_chars:
            trimmed = text_a[:-trim_chars] + " " + text_b[trim_chars:]
        else:
            trimmed = text_a + " " + text_b

        if self.verbose:
            self.logger.debug(
                f"  No overlap found, concatenating with trim ({trim_chars} chars)"
            )

        return trimmed

    def _pick_better_overlap(self, overlap_a: str, overlap_b: str) -> str:
        """Pick the better version of overlapping text based on quality heuristics."""
        if not overlap_a:
            return overlap_b or ""
        if not overlap_b:
            return overlap_a or ""

        def score_text(t: str) -> float:
            if not t:
                return 0.0
            s = 0.0
            alpha_ratio = sum(1 for c in t if c.isalpha()) / len(t)
            s += alpha_ratio * 10

            space_ratio = t.count(' ') / max(len(t), 1)
            s += space_ratio * 5

            for i in range(len(t) - 2):
                if t[i] == t[i + 1] == t[i + 2]:
                    s -= 3

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
        """Try to merge texts by finding matching words at the boundary."""
        if not text_a or not text_b:
            return None

        words_a = text_a.split()
        words_b = text_b.split()

        if not words_a or not words_b:
            return None

        max_words_to_check = max(2, estimated_overlap_chars // 4)
        search_words_a = words_a[-max_words_to_check:]
        search_words_b = words_b[:max_words_to_check]

        best_word_match = 0
        best_a_word_idx = len(words_a)
        best_b_word_idx = 0

        for i, word_a in enumerate(search_words_a):
            if not word_a:
                continue
            for j, word_b in enumerate(search_words_b):
                if not word_b:
                    continue

                if word_a.lower() == word_b.lower() and len(word_a) >= 3:
                    match_count = 1
                    ai = len(words_a) - len(search_words_a) + i
                    bi = j

                    while (ai + match_count < len(words_a) and
                        bi + match_count < len(words_b)):
                        wa = words_a[ai + match_count]
                        wb = words_b[bi + match_count]
                        if wa and wb and wa.lower() == wb.lower():
                            match_count += 1
                        else:
                            break

                    if match_count > best_word_match:
                        best_word_match = match_count
                        best_a_word_idx = ai
                        best_b_word_idx = bi + match_count

                elif len(word_a) >= 3 and len(word_b) >= 3:
                    ratio = SequenceMatcher(None, word_a.lower(), word_b.lower()).ratio()
                    if ratio >= 0.75:
                        ai = len(words_a) - len(search_words_a) + i
                        if best_word_match == 0:
                            best_word_match = 1
                            best_a_word_idx = ai
                            best_b_word_idx = j + 1

        if best_word_match > 0:
            merged_words = words_a[:best_a_word_idx + best_word_match] + words_b[best_b_word_idx:]
            merged = " ".join(w for w in merged_words if w)

            if self.verbose:
                self.logger.debug(
                    f"  Word boundary merge: matched {best_word_match} word(s)"
                )

            return merged

        return None

    def _score_long_text(self, text: str, confidence: float) -> float:
        """Score a long text result for comparison between methods."""
        if not text:
            return 0.0

        score = confidence

        text_len = len(text.strip())
        if text_len > 10:
            score += min(text_len * 0.3, 20)

        word_count = len(text.split())
        if word_count > 1:
            score += min(word_count * 2, 15)

        # Penalize garbled text indicators
        for i in range(len(text) - 2):
            if text[i] == text[i + 1] == text[i + 2] and text[i].isalpha():
                score -= 5

        if text_len > 20 and word_count < text_len // 20:
            score -= 15

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

            texts = [t for t, c in parsed if t]
            confs = [c for t, c in parsed if t]

            if not texts:
                return "", 0.0

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
                            if sub_item is None:
                                continue
                            if isinstance(sub_item, (list, tuple)) and len(sub_item) >= 2:
                                try:
                                    t = str(sub_item[0]).strip() if sub_item[0] is not None else ""
                                    c = float(sub_item[1]) if sub_item[1] is not None else 0.0
                                except (ValueError, TypeError):
                                    continue
                                if t and 0 <= c <= 1:
                                    if len(t) > len(text):
                                        text = t
                                        conf = c
                            elif isinstance(sub_item, dict):
                                t = str(sub_item.get('text', sub_item.get('rec_text', '')) or "").strip()
                                try:
                                    c = float(sub_item.get('score', sub_item.get('rec_score', 0)) or 0)
                                except (ValueError, TypeError):
                                    c = 0.0
                                if t and len(t) > len(text):
                                    text = t
                                    conf = c

                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            t = str(item[0]).strip() if item[0] is not None else ""
                            c = float(item[1]) if item[1] is not None else 0.0
                        except (ValueError, TypeError):
                            continue
                        if t and 0 <= c <= 1:
                            if len(t) > len(text):
                                text = t
                                conf = c

                    elif isinstance(item, dict):
                        t = str(item.get('text', item.get('rec_text', '')) or "").strip()
                        try:
                            c = float(item.get('score', item.get('rec_score', 0)) or 0)
                        except (ValueError, TypeError):
                            c = 0.0
                        if t and len(t) > len(text):
                            text = t
                            conf = c

            return text or "", conf * 100

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

            # Guard against None from OCR
            if text is None:
                text = ""
            
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

                # Guard against None from OCR
                if text is None:
                    text = ""
                
                if text and self._is_valid_text(text, conf):
                    score = self._calculate_score(text, conf, rotated=True)
                    if score > best_score:
                        best_text, best_score, best_conf = text, score, conf
                    if score >= self.high_conf:
                        break

        self.stats['time'] += time.time() - start

        if best_score >= self.min_conf or (best_text and best_score >= 20):
            self.stats['success'] += 1

            if self.enable_validation:
                corrected_text = best_text or ""
                final_conf = best_conf

                # Step 1: Character-level corrections
                if self.char_corrector:
                    try:
                        result = self.char_corrector.correct_text(corrected_text, final_conf)
                        if result is not None and len(result) == 3:
                            char_text, char_conf, char_corrections = result
                            if char_text is not None:
                                corrected_text = char_text
                                final_conf = char_conf
                                if char_corrections and self.verbose:
                                    self.logger.debug(f"Char corrections on: '{best_text}'")
                                    for orig, fixed in char_corrections:
                                        self.logger.debug(f"  char: '{orig}' -> '{fixed}'")
                    except Exception as e:
                        self.logger.error(f"Char correction error: {e}")

                # Step 2: Word/phrase-level corrections
                if self.corrector:
                    try:
                        result = self.corrector.correct_text(corrected_text, final_conf)
                        if result is not None and len(result) == 4:
                            word_text, word_conf, was_corrected, word_corrections = result
                            if word_text is not None:
                                corrected_text = word_text
                                final_conf = word_conf
                                if was_corrected:
                                    self.stats['corrected'] += 1
                                    if self.verbose:
                                        for orig, corr in word_corrections:
                                            self.logger.debug(f"  word: '{orig}' -> '{corr}'")
                    except Exception as e:
                        self.logger.error(f"Word correction error: {e}")

                return self._clean_text(corrected_text)

            return self._clean_text(best_text)

        return ""

    def _clean_text(self, text: str) -> str:
        """Clean text."""
        if text is None:
            return ""
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


class CharacterLevelCorrector:
    """
    Corrects character-level OCR errors using:
    1. Character confusion matrix (which chars get misread as which)
    2. N-gram frequency analysis (which char sequences are plausible)
    3. Word structure analysis (detecting garbled regions)
    4. Space insertion heuristics
    5. Missing character recovery
    6. SymSpell + wordfreq for language-aware correction (en + ro)
    """

    def __init__(self, dictionary: Optional['ScheduleDictionary'] = None, verbose: bool = False):
        self.dictionary = dictionary
        self.verbose = verbose
        self.logger = logging.getLogger(__name__ + ".CharCorrector")

        # Character confusion matrix: OCR often confuses these
        # Format: misread_char -> [(correct_char, likelihood), ...]
        self.confusion_matrix: Dict[str, List[Tuple[str, float]]] = {
            # Shape-similar uppercase
            'A': [('H', 0.4), ('4', 0.3), ('R', 0.2)],
            'B': [('8', 0.5), ('D', 0.3), ('6', 0.2)],
            'C': [('G', 0.3), ('(', 0.2), ('O', 0.2)],
            'D': [('O', 0.4), ('0', 0.4), ('B', 0.2)],
            'E': [('F', 0.3), ('3', 0.3), ('L', 0.2)],
            'F': [('E', 0.3), ('P', 0.3), ('T', 0.2)],
            'G': [('C', 0.4), ('6', 0.3), ('O', 0.2)],
            'H': [('A', 0.4), ('N', 0.3), ('M', 0.2), ('II', 0.3)],
            'I': [('L', 0.5), ('1', 0.5), ('l', 0.4), ('T', 0.2)],
            'J': [('I', 0.3), ('1', 0.3), ('T', 0.2)],
            'K': [('X', 0.3), ('R', 0.2)],
            'L': [('I', 0.4), ('1', 0.4), ('E', 0.2), ('i', 0.3)],
            'M': [('N', 0.3), ('H', 0.3), ('IM', 0.2), ('IVI', 0.2)],
            'N': [('M', 0.3), ('H', 0.3), ('IV', 0.2)],
            'O': [('0', 0.6), ('D', 0.4), ('Q', 0.2), ('C', 0.2)],
            'P': [('F', 0.3), ('R', 0.3), ('D', 0.2)],
            'Q': [('O', 0.4), ('0', 0.3)],
            'R': [('A', 0.3), ('P', 0.3), ('K', 0.2)],
            'S': [('5', 0.5), ('$', 0.2), ('8', 0.2)],
            'T': [('I', 0.3), ('7', 0.3), ('F', 0.2), ('1', 0.2)],
            'U': [('V', 0.3), ('O', 0.2), ('LI', 0.2)],
            'V': [('U', 0.3), ('W', 0.2), ('Y', 0.2)],
            'W': [('VV', 0.4), ('M', 0.2)],
            'X': [('K', 0.3), ('Y', 0.2)],
            'Y': [('V', 0.3), ('X', 0.2), ('T', 0.2)],
            'Z': [('2', 0.4), ('7', 0.2)],
            # Digits
            '0': [('O', 0.6), ('D', 0.3), ('Q', 0.2)],
            '1': [('I', 0.5), ('L', 0.5), ('l', 0.4), ('7', 0.2)],
            '2': [('Z', 0.3), ('7', 0.2)],
            '3': [('E', 0.3), ('8', 0.2)],
            '4': [('A', 0.3), ('H', 0.2)],
            '5': [('S', 0.5), ('6', 0.2)],
            '6': [('G', 0.3), ('b', 0.3)],
            '7': [('T', 0.3), ('1', 0.2), ('2', 0.2)],
            '8': [('B', 0.5), ('S', 0.2)],
            '9': [('g', 0.3), ('q', 0.2)],
            # Lowercase
            'l': [('I', 0.5), ('1', 0.5), ('i', 0.3)],
            'i': [('l', 0.4), ('1', 0.3), ('I', 0.3)],
            'o': [('0', 0.5), ('a', 0.2)],
            'n': [('m', 0.2), ('ri', 0.3), ('r', 0.2)],
            'm': [('rn', 0.4), ('nn', 0.2), ('n', 0.2)],
            'r': [('n', 0.2), ('t', 0.2)],
            'c': [('e', 0.3), ('o', 0.2)],
            'e': [('c', 0.3), ('a', 0.2)],
            'a': [('o', 0.2), ('e', 0.2), ('d', 0.2)],
            'd': [('cl', 0.3), ('a', 0.2)],
            'u': [('v', 0.2), ('n', 0.2)],
            'h': [('b', 0.2), ('n', 0.2)],
            'b': [('h', 0.2), ('6', 0.3)],
        }

        # Build reverse confusion: correct_char -> [(misread_char, likelihood), ...]
        self.reverse_confusion: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for misread, corrections in self.confusion_matrix.items():
            for correct, likelihood in corrections:
                self.reverse_confusion[correct].append((misread, likelihood))

        # Romanian/Latin character patterns (common bigrams)
        self.common_bigrams = set([
            'TI', 'IA', 'IN', 'RE', 'AR', 'AT', 'CA', 'TE', 'DE', 'LA',
            'AN', 'RA', 'OR', 'SI', 'AL', 'RI', 'ST', 'LE', 'TA', 'NI',
            'LI', 'CE', 'IT', 'IC', 'TR', 'PR', 'UN', 'MA', 'CU', 'SE',
            'CO', 'PE', 'EN', 'RO', 'IE', 'UL', 'EL', 'LO', 'GE', 'FI',
            'DI', 'ME', 'PL', 'GR', 'BR', 'CR', 'DR', 'FR', 'SP', 'SC',
            'ED', 'UC', 'OL', 'OG', 'PS', 'IH', 'HO', 'FO', 'RM', 'AP',
            'AC', 'AD', 'AG', 'AJ', 'AM', 'AP', 'AS', 'AV', 'AZ',
            'BA', 'BE', 'BI', 'BL', 'BO', 'BU',
            'CI', 'CL', 'CN', 'CR', 'CT', 'CU',
            'DA', 'DO', 'DU',
            'EA', 'EC', 'EF', 'EG', 'EM', 'EP', 'ER', 'ES', 'ET', 'EU', 'EV', 'EX',
            'FA', 'FE', 'FL', 'FO', 'FU',
            'GA', 'GI', 'GL', 'GO', 'GU',
            'HA', 'HI',
            'ID', 'IF', 'IG', 'IL', 'IM', 'IO', 'IP', 'IR', 'IS', 'IU', 'IV', 'IZ',
            'LU', 'LT',
            'MI', 'MO', 'MU',
            'NA', 'NE', 'NO', 'NU',
            'OA', 'OB', 'OC', 'OD', 'OF', 'OI', 'OM', 'ON', 'OP', 'OS', 'OT', 'OU', 'OV',
            'PA', 'PI', 'PO', 'PU',
            'RU', 'RN', 'RM',
            'SA', 'SO', 'SU', 'SL', 'SM', 'SN',
            'TO', 'TU', 'TH',
            'UA', 'UB', 'UC', 'UD', 'UG', 'UI', 'UM', 'UP', 'UR', 'US', 'UT',
            'VA', 'VE', 'VI', 'VO', 'VR',
            'ZA', 'ZI',
        ])

        self.impossible_trigrams = set([
            'SSS', 'DDD', 'LLL', 'MMM', 'NNN', 'RRR', 'TTT', 'PPP',
            'BBB', 'CCC', 'FFF', 'GGG', 'HHH', 'JJJ', 'KKK', 'VVV',
            'WWW', 'XXX', 'YYY', 'ZZZ', 'QQQ',
            'DLG', 'MLG', 'SMD', 'DGR', 'LDG', 'MLD', 'SMA',
            'BCF', 'BKG', 'CFG', 'CKQ', 'DKP', 'FGK', 'FKP',
            'GKP', 'GKQ', 'JKQ', 'JPQ', 'KPQ', 'KQX', 'QPX',
        ])

        # Initialize SymSpell for en and ro
        self._symspell_en = None
        self._symspell_ro = None
        self._wordfreq_available = False
        self._symspell_available = False
        self._init_spell_checkers()

        self.stats = {
            'texts_processed': 0,
            'chars_corrected': 0,
            'spaces_inserted': 0,
            'chars_recovered': 0,
            'words_fixed': 0,
            'symspell_corrections': 0,
        }

    def _init_spell_checkers(self):
        """Initialize SymSpell dictionaries and wordfreq."""
        # Check wordfreq availability
        try:
            from wordfreq import word_frequency, top_n_list
            self._wordfreq_available = True
            self.logger.info("wordfreq available for en + ro")
        except ImportError:
            self._wordfreq_available = False
            self.logger.warning("wordfreq not installed. Install with: pip install wordfreq")

        # Initialize SymSpell
        try:
            from symspellpy import SymSpell, Verbosity

            # English SymSpell
            self._symspell_en = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            en_dict_loaded = False

            # Try to load the default English dictionary from symspellpy package
            import pkg_resources
            try:
                dict_path = pkg_resources.resource_filename(
                    "symspellpy", "frequency_dictionary_en_82_765.txt"
                )
                if os.path.exists(dict_path):
                    en_dict_loaded = self._symspell_en.load_dictionary(
                        dict_path, term_index=0, count_index=1
                    )
            except Exception:
                pass

            # Fallback: build from wordfreq
            if not en_dict_loaded and self._wordfreq_available:
                self.logger.info("Building English SymSpell dictionary from wordfreq...")
                en_dict_loaded = self._build_symspell_from_wordfreq(self._symspell_en, "en", 50000)

            if en_dict_loaded:
                self.logger.info("English SymSpell dictionary loaded")
            else:
                self.logger.warning("Failed to load English SymSpell dictionary")
                self._symspell_en = None

            # Romanian SymSpell - build from wordfreq
            if self._wordfreq_available:
                self._symspell_ro = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                ro_loaded = self._build_symspell_from_wordfreq(self._symspell_ro, "ro", 50000)
                if ro_loaded:
                    self.logger.info("Romanian SymSpell dictionary loaded from wordfreq")
                    # Add common Romanian academic terms
                    self._add_romanian_academic_terms(self._symspell_ro)
                else:
                    self.logger.warning("Failed to build Romanian SymSpell dictionary")
                    self._symspell_ro = None
            else:
                self._symspell_ro = None

            self._symspell_available = (self._symspell_en is not None or
                                         self._symspell_ro is not None)

            if self._symspell_available:
                self.logger.info("SymSpell correction enabled (en={}, ro={})".format(
                    self._symspell_en is not None, self._symspell_ro is not None
                ))

        except ImportError:
            self._symspell_available = False
            self.logger.warning("symspellpy not installed. Install with: pip install symspellpy")

    def _build_symspell_from_wordfreq(self, sym: 'SymSpell', lang: str, n_words: int) -> bool:
        """Build a SymSpell dictionary from wordfreq top words."""
        try:
            from wordfreq import top_n_list, word_frequency

            words = top_n_list(lang, n_words)
            if not words:
                return False

            count = 0
            for word in words:
                if len(word) < 2:
                    continue
                # Convert frequency to a count-like value
                freq = word_frequency(word, lang)
                # Scale to integer count (higher = more common)
                int_count =  max(1, int(freq * 1_000_000_000))
                sym.create_dictionary_entry(word, int_count)
                count += 1

            self.logger.info(f"Built {lang} SymSpell dict with {count} words")
            return count > 0

        except Exception as e:
            self.logger.error(f"Error building SymSpell dict for {lang}: {e}")
            return False

    def _add_romanian_academic_terms(self, sym: 'SymSpell'):
        """Add common Romanian academic/schedule terms to SymSpell."""
        academic_terms = [
            # Subjects
            "algebra", "liniara", "geometrie", "analitica", "diferentiala",
            "informatica", "aplicata", "programare", "calculatoare",
            "matematica", "analiza", "numerica", "statistica", "probabilitati",
            "fizica", "chimie", "mecanica", "termodinamica", "electrotehnica",
            "electronica", "automatica", "telecomunicatii", "retele",
            "psihologia", "educatiei", "pedagogie", "didactica", "metodica",
            "economia", "management", "marketing", "contabilitate", "finante",
            "drept", "sociologie", "filosofie", "istorie", "geografie",
            "biologie", "ecologie", "genetica", "anatomie", "fiziologie",
            "arhitectura", "constructii", "materiale", "rezistenta",
            "energetica", "hidraulica", "pneumatica", "tribologie",
            "comunicare", "literatura", "lingvistica", "traducere",
            # Course types
            "curs", "seminar", "laborator", "proiect", "practica", "examen",
            "colocviu", "verificare", "restanta", "consultatii",
            "prelegere", "dezbatere", "atelier", "stagiu",
            # Titles
            "profesor", "conferentiar", "lector", "asistent", "doctorand",
            "docent", "inginer", "doctor", "magistru", "academician",
            # Common words in schedules
            "sala", "amfiteatru", "laborator", "cabinet", "biblioteca",
            "etaj", "corp", "cladire", "campus", "facultatea",
            "universitatea", "departamentul", "catedra", "sectia",
            "seria", "grupa", "subgrupa", "anul", "semestrul",
            # Days
            "luni", "marti", "miercuri", "joi", "vineri", "sambata", "duminica",
            # Time
            "ora", "minute", "pauza", "interval",
            # Other common
            "obligatoriu", "optional", "facultativ", "intensiv",
            "saptamanal", "bisaptamanal", "zilnic",
            "introducere", "fundamentele", "bazele", "elemente",
            "tehnici", "metode", "principii", "concepte", "teorii",
            "aplicatii", "exercitii", "probleme", "studiu", "cercetare",
            "lucrari", "referat", "prezentare", "evaluare", "notare",
        ]

        count = 0
        for term in academic_terms:
            sym.create_dictionary_entry(term, 100000)
            # Also add uppercase version
            sym.create_dictionary_entry(term.upper(), 100000)
            # Capitalized
            sym.create_dictionary_entry(term.capitalize(), 100000)
            count += 1

        # Add terms from the OCR dictionary if available
        if self.dictionary:
            for term in self.dictionary.all_terms:
                if term and len(term) >= 2:
                    sym.create_dictionary_entry(term.lower(), 200000)
                    sym.create_dictionary_entry(term.upper(), 200000)
                    sym.create_dictionary_entry(term, 200000)
                    count += 1

        self.logger.info(f"Added {count} Romanian academic terms to SymSpell")

    def _get_word_frequency(self, word: str, lang: str = None) -> float:
        """
        Get word frequency using wordfreq library.
        Checks both en and ro, returns the higher frequency.
        """
        if not self._wordfreq_available or not word:
            return 0.0

        try:
            from wordfreq import word_frequency

            word_lower = word.lower()

            if lang:
                return word_frequency(word_lower, lang)

            # Check both languages, return higher
            freq_en = word_frequency(word_lower, 'en')
            freq_ro = word_frequency(word_lower, 'ro')
            return max(freq_en, freq_ro)

        except Exception:
            return 0.0

    def _is_known_word(self, word: str) -> bool:
        """
        Check if a word is known in any language or dictionary.
        """
        if not word or len(word) < 2:
            return False

        # Check OCR dictionary first
        if self.dictionary:
            if self.dictionary.is_valid_word(word):
                return True
            if self.dictionary.get_word_correction(word) is not None:
                return True

        # Check wordfreq
        freq = self._get_word_frequency(word)
        if freq > 1e-7:  # Reasonably common word
            return True

        return False

    def _symspell_lookup(self, word: str, max_edit_distance: int = 2) -> Optional[str]:
        """
        Look up a word using SymSpell in both en and ro.
        Returns the best single-word suggestion or None.
        NEVER returns multi-word suggestions.
        """
        if not self._symspell_available or not word or len(word) < 3:
            return None

        try:
            from symspellpy import Verbosity

            best_suggestion = None
            best_score = 0.0

            word_lower = word.lower()

            # Try Romanian first
            if self._symspell_ro:
                suggestions = self._symspell_ro.lookup(
                    word_lower,
                    Verbosity.CLOSEST,
                    max_edit_distance=max_edit_distance
                )
                for suggestion in suggestions:
                    # NEVER accept multi-word suggestions
                    if ' ' in suggestion.term:
                        continue
                    if suggestion.term == word_lower:
                        return None  # Word is already correct
                    # Must be similar length
                    if abs(len(suggestion.term) - len(word_lower)) > max(2, len(word_lower) * 0.3):
                        continue
                    score = suggestion.count / (suggestion.distance + 1)
                    freq = self._get_word_frequency(suggestion.term, 'ro')
                    if freq > 0:
                        score *= (1 + freq * 10000)
                    if score > best_score:
                        best_score = score
                        best_suggestion = suggestion.term

            # Try English
            if self._symspell_en:
                suggestions = self._symspell_en.lookup(
                    word_lower,
                    Verbosity.CLOSEST,
                    max_edit_distance=max_edit_distance
                )
                for suggestion in suggestions:
                    if ' ' in suggestion.term:
                        continue
                    if suggestion.term == word_lower:
                        return None
                    if abs(len(suggestion.term) - len(word_lower)) > max(2, len(word_lower) * 0.3):
                        continue
                    score = suggestion.count / (suggestion.distance + 1)
                    freq = self._get_word_frequency(suggestion.term, 'en')
                    if freq > 0:
                        score *= (1 + freq * 10000)
                    if score > best_score:
                        best_score = score
                        best_suggestion = suggestion.term

            if best_suggestion and best_suggestion != word_lower and ' ' not in best_suggestion:
                # Final validation
                if self._get_word_frequency(best_suggestion) > 1e-8:
                    return best_suggestion
                if self.dictionary and self.dictionary.is_valid_word(best_suggestion):
                    return best_suggestion

            return None

        except Exception as e:
            if self.verbose:
                self.logger.debug(f"SymSpell lookup error for '{word}': {e}")
            return None

    def _symspell_segment(self, word: str) -> Optional[str]:
        """
        Use SymSpell word segmentation with STRICT validation.
        Only returns a result if ALL segments are valid words of length >= 3.
        """
        if not self._symspell_available or not word or len(word) < 8:
            return None

        try:
            from symspellpy import Verbosity

            word_lower = word.lower()
            best_segmentation = None
            best_score = 0.0

            if self._symspell_ro:
                result = self._symspell_ro.word_segmentation(word_lower)
                if result and result.segmented_string:
                    score = self._validate_split(result.segmented_string, word)
                    if score > best_score:
                        best_score = score
                        best_segmentation = result.segmented_string

            if self._symspell_en:
                result = self._symspell_en.word_segmentation(word_lower)
                if result and result.segmented_string:
                    score = self._validate_split(result.segmented_string, word)
                    if score > best_score:
                        best_score = score
                        best_segmentation = result.segmented_string

            return best_segmentation

        except Exception as e:
            if self.verbose:
                self.logger.debug(f"SymSpell segment error: {e}")
            return None
        
    def correct_text(self, text: str, confidence: float) -> Tuple[str, float, List[Tuple[str, str]]]:
        """
        Apply character-level corrections to OCR text.

        Args:
            text: Raw OCR text
            confidence: OCR confidence (0-100)

        Returns:
            Tuple of (corrected_text, adjusted_confidence, list of corrections)
        """
        self.stats['texts_processed'] += 1

        if not text or not text.strip():
            return text or "", confidence, []

        original_text = text
        corrections = []

        # Step 1: Fix obvious character substitutions first (digit/letter context)
        text, char_corrections = self._fix_character_substitutions(text, confidence)
        if text is None:
            text = original_text
        corrections.extend(char_corrections)

        # Step 2: Fix missing spaces
        text, space_corrections = self._fix_missing_spaces(text, confidence)
        if text is None:
            text = original_text
        corrections.extend(space_corrections)

        # Step 3: Fix garbled words using dictionary + confusion matrix
        text, word_corrections = self._fix_garbled_words(text, confidence)
        if text is None:
            text = original_text
        corrections.extend(word_corrections)

        # Step 4: SymSpell-based word correction for remaining unknown words
        text, symspell_corrections = self._fix_with_symspell(text, confidence)
        if text is None:
            text = original_text
        corrections.extend(symspell_corrections)

        # Step 5: Recover missing characters
        text, recovery_corrections = self._recover_missing_characters(text, confidence)
        if text is None:
            text = original_text
        corrections.extend(recovery_corrections)

        # Adjust confidence
        if corrections:
            correction_count = len(corrections)
            if confidence < 50:
                conf_boost = min(correction_count * 8, 30)
            elif confidence < 70:
                conf_boost = min(correction_count * 5, 20)
            else:
                conf_boost = min(correction_count * 3, 10)
            confidence = min(confidence + conf_boost, 92)

        if self.verbose and corrections:
            self.logger.debug(f"CharCorrector: '{original_text}' -> '{text}'")
            for orig, fixed in corrections:
                self.logger.debug(f"  '{orig}' -> '{fixed}'")

        return text or "", confidence, corrections

    # def _fix_missing_spaces(self, text: str, confidence: float) -> Tuple[str, List[Tuple[str, str]]]:
    #     """
    #     Detect and fix missing spaces in text.
        
    #     CONSERVATIVE approach: Only insert spaces when there's strong evidence
    #     that two words were concatenated. Both sides must be valid words.
    #     """
    #     if not text:
    #         return text or "", []

    #     corrections = []
    #     result = text

    #     # Step 1: Fix obvious boundaries (letter-digit, digit-letter)
    #     # Only for clear cases like "APLICATA41" or "C41Profesor"
    #     result, digit_corrections = self._fix_digit_letter_boundaries(result)
    #     corrections.extend(digit_corrections)

    #     # Step 2: Try to split very long words (> 15 chars) that are likely concatenated
    #     words = result.split()
    #     fixed_words = []
    #     changed = False

    #     for word in words:
    #         if not word:
    #             fixed_words.append("")
    #             continue

    #         # Only try to split words that are suspiciously long
    #         # Normal words rarely exceed 15 characters
    #         if len(word) > 15 and word.isalpha():
    #             split_result = self._try_smart_word_split(word)
    #             if split_result and split_result != word:
    #                 corrections.append((word, split_result))
    #                 fixed_words.append(split_result)
    #                 changed = True
    #                 continue

    #         # Also try splitting if the word contains obvious concatenation patterns
    #         # like lowercase followed by uppercase: "aplicataConferentiar"
    #         if len(word) > 8 and self._has_case_boundary(word):
    #             split_result = self._split_at_case_boundary(word)
    #             if split_result and split_result != word:
    #                 corrections.append((word, split_result))
    #                 fixed_words.append(split_result)
    #                 changed = True
    #                 continue

    #         fixed_words.append(word)

    #     if changed:
    #         result = ' '.join(fixed_words)

    #     return result, corrections

    # def _fix_digit_letter_boundaries(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
    #     """
    #     Fix boundaries between letters and digits.
    #     Only for clear cases where a word and number are concatenated.
    #     """
    #     if not text:
    #         return text or "", []

    #     corrections = []
    #     result = []
    #     i = 0
    #     original = text

    #     while i < len(text):
    #         result.append(text[i])

    #         if i < len(text) - 1:
    #             curr = text[i]
    #             next_c = text[i + 1]

    #             # Letter to digit: "APLICATA41" -> "APLICATA 41"
    #             # But NOT for codes like "P03" or "C41"
    #             if curr.isalpha() and next_c.isdigit():
    #                 # Look back to see how many letters precede
    #                 letter_count = 0
    #                 j = i
    #                 while j >= 0 and text[j].isalpha():
    #                     letter_count += 1
    #                     j -= 1

    #                 # Look forward to see how many digits follow
    #                 digit_count = 0
    #                 k = i + 1
    #                 while k < len(text) and text[k].isdigit():
    #                     digit_count += 1
    #                     k += 1

    #                 # Only split if we have a substantial word (>3 letters) followed by digits
    #                 # This avoids splitting codes like "P03", "C41"
    #                 if letter_count > 3 and digit_count >= 1:
    #                     result.append(' ')

    #             # Digit to letter: "41Profesor" -> "41 Profesor"
    #             # But NOT for things like "3D" or "P03abc" (unlikely pattern)
    #             elif curr.isdigit() and next_c.isalpha():
    #                 # Look back to see digits
    #                 digit_count = 0
    #                 j = i
    #                 while j >= 0 and text[j].isdigit():
    #                     digit_count += 1
    #                     j -= 1

    #                 # Look forward to see letters
    #                 letter_count = 0
    #                 k = i + 1
    #                 while k < len(text) and text[k].isalpha():
    #                     letter_count += 1
    #                     k += 1

    #                 # Split if we have digits followed by a word (>3 letters)
    #                 if digit_count >= 1 and letter_count > 3:
    #                     result.append(' ')

    #         i += 1

    #     new_text = ''.join(result)
    #     if new_text != original:
    #         corrections.append((original, new_text))

    #     return new_text, corrections

    # def _has_case_boundary(self, word: str) -> bool:
    #     """Check if word has a lowercase-to-uppercase transition (likely concatenation)."""
    #     for i in range(len(word) - 1):
    #         if word[i].islower() and word[i + 1].isupper():
    #             return True
    #     return False

    # def _split_at_case_boundary(self, word: str) -> Optional[str]:
    #     """
    #     Split a word at case boundaries (lowercase followed by uppercase).
    #     e.g., "aplicataConferentiar" -> "aplicata Conferentiar"
        
    #     Only splits if both resulting parts look like valid words.
    #     """
    #     if not word or len(word) < 6:
    #         return None

    #     parts = []
    #     current_start = 0

    #     for i in range(len(word) - 1):
    #         if word[i].islower() and word[i + 1].isupper():
    #             part = word[current_start:i + 1]
    #             if len(part) >= 2:  # Minimum part length
    #                 parts.append(part)
    #                 current_start = i + 1

    #     # Add the last part
    #     if current_start < len(word):
    #         parts.append(word[current_start:])

    #     if len(parts) <= 1:
    #         return None

    #     # Validate: each part should be a plausible word (>= 3 chars or known)
    #     valid_parts = []
    #     for part in parts:
    #         if len(part) < 2:
    #             return None  # Too short, probably wrong split

    #         # Check if it's a known word or looks plausible
    #         if len(part) >= 3:
    #             if self._is_known_word(part):
    #                 valid_parts.append(part)
    #             elif self._looks_like_word(part):
    #                 valid_parts.append(part)
    #             else:
    #                 return None  # Unknown and doesn't look like a word
    #         else:
    #             # Very short parts (2 chars) - only accept if they're common words
    #             if part.upper() in {'SI', 'DE', 'LA', 'IN', 'CU', 'PE', 'UN', 'NU', 'CE', 'SE', 'NE', 'VA', 'FI', 'AI', 'AM', 'AU', 'EI', 'EU', 'II', 'OI', 'AR', 'AS', 'AT', 'OR'}:
    #                 valid_parts.append(part)
    #             else:
    #                 return None  # Short unknown word

    #     if len(valid_parts) == len(parts):
    #         return ' '.join(valid_parts)

    #     return None

    # def _try_smart_word_split(self, word: str) -> Optional[str]:
    #     """
    #     Try to split a very long word into multiple valid words.
    #     Uses multiple strategies but validates that ALL resulting parts are valid.
    #     """
    #     if not word or len(word) < 10:
    #         return None

    #     word_to_split = word

    #     # Strategy 1: SymSpell segmentation (if available)
    #     if self._symspell_available:
    #         segmented = self._symspell_segment_validated(word_to_split)
    #         if segmented:
    #             return segmented

    #     # Strategy 2: Dictionary-based dynamic programming split
    #     if self.dictionary:
    #         dp_split = self._dp_word_split(word_to_split)
    #         if dp_split:
    #             return dp_split

    #     # Strategy 3: Greedy longest-match from start
    #     greedy_split = self._greedy_word_split(word_to_split)
    #     if greedy_split:
    #         return greedy_split

    #     return None

    # def _symspell_segment_validated(self, word: str) -> Optional[str]:
    #     """
    #     Use SymSpell segmentation but strictly validate the result.
    #     """
    #     if not self._symspell_available or not word:
    #         return None

    #     try:
    #         from symspellpy import Verbosity

    #         word_lower = word.lower()
    #         best_result = None
    #         best_score = 0.0

    #         # Try Romanian first
    #         if self._symspell_ro:
    #             result = self._symspell_ro.word_segmentation(word_lower)
    #             if result and result.segmented_string and ' ' in result.segmented_string:
    #                 score = self._score_segmentation(result.segmented_string)
    #                 if score > best_score:
    #                     best_score = score
    #                     best_result = result.segmented_string

    #         # Try English
    #         if self._symspell_en:
    #             result = self._symspell_en.word_segmentation(word_lower)
    #             if result and result.segmented_string and ' ' in result.segmented_string:
    #                 score = self._score_segmentation(result.segmented_string)
    #                 if score > best_score:
    #                     best_score = score
    #                     best_result = result.segmented_string

    #         # Validate the result
    #         if best_result and best_score > 0.5:
    #             # Apply original case
    #             if word.isupper():
    #                 best_result = best_result.upper()
    #             elif word[0].isupper():
    #                 # Capitalize first letter of each word
    #                 best_result = ' '.join(w.capitalize() for w in best_result.split())

    #             return best_result

    #         return None

    #     except Exception as e:
    #         if self.verbose:
    #             self.logger.debug(f"SymSpell segment error: {e}")
    #         return None

    # def _score_segmentation(self, segmented: str) -> float:
    #     """
    #     Score a segmentation result.
    #     Higher score = better (more valid words, longer words preferred).
    #     """
    #     if not segmented:
    #         return 0.0

    #     words = segmented.split()
    #     if not words:
    #         return 0.0

    #     total_score = 0.0
    #     valid_count = 0

    #     for word in words:
    #         if not word:
    #             continue

    #         word_score = 0.0

    #         # Length bonus (longer words are better, avoid single chars)
    #         if len(word) == 1:
    #             word_score -= 1.0  # Penalize single characters heavily
    #         elif len(word) == 2:
    #             # Only accept common 2-letter words
    #             if word.upper() in {'SI', 'DE', 'LA', 'IN', 'CU', 'PE', 'UN', 'NU', 'CE', 'SE', 'NE'}:
    #                 word_score += 0.5
    #             else:
    #                 word_score -= 0.5
    #         elif len(word) >= 3:
    #             word_score += len(word) * 0.2

    #         # Check if it's a known word
    #         if self._is_known_word(word):
    #             word_score += 2.0
    #             valid_count += 1
    #         elif self._looks_like_word(word):
    #             word_score += 0.5
    #         else:
    #             word_score -= 1.0  # Unknown word penalty

    #         total_score += word_score

    #     # Bonus for having most words be valid
    #     if len(words) > 0:
    #         valid_ratio = valid_count / len(words)
    #         total_score += valid_ratio * 2.0

    #     # Normalize by number of words
    #     return total_score / len(words) if words else 0.0

    # def _dp_word_split(self, word: str) -> Optional[str]:
    #     """
    #     Use dynamic programming to find the best way to split a word.
    #     Only returns a result if ALL parts are valid words.
    #     """
    #     if not self.dictionary or not word:
    #         return None

    #     n = len(word)
    #     word_lower = word.lower()
    #     word_upper = word.upper()

    #     # dp[i] = (best_score, best_split_text) for word[:i]
    #     dp: List[Optional[Tuple[float, str, int]]] = [None] * (n + 1)
    #     dp[0] = (0.0, "", 0)  # (score, text, word_count)

    #     for i in range(1, n + 1):
    #         for j in range(max(0, i - 20), i):  # Max word length 20
    #             if dp[j] is None:
    #                 continue

    #             segment = word[j:i]
    #             segment_lower = segment.lower()
    #             segment_upper = segment.upper()

    #             # Check if this segment is a valid word
    #             is_valid = False
    #             segment_score = 0.0

    #             # Check dictionary
    #             if self.dictionary.is_valid_word(segment) or self.dictionary.is_valid_word(segment_lower) or self.dictionary.is_valid_word(segment_upper):
    #                 is_valid = True
    #                 segment_score = len(segment) * 2.0
    #             # Check wordfreq
    #             elif self._wordfreq_available:
    #                 freq = self._get_word_frequency(segment)
    #                 if freq > 1e-7:
    #                     is_valid = True
    #                     segment_score = len(segment) * 1.5

    #             # Accept common short words
    #             if not is_valid and len(segment) == 2:
    #                 if segment_upper in {'SI', 'DE', 'LA', 'IN', 'CU', 'PE', 'UN', 'NU', 'CE', 'SE', 'NE', 'VA', 'FI'}:
    #                     is_valid = True
    #                     segment_score = 1.0

    #             # Single letters only as last resort and only for 'A', 'I', etc.
    #             if not is_valid and len(segment) == 1:
    #                 if segment_upper in {'A', 'I', 'O', 'E'}:
    #                     # Only allow single letter if it's connecting two valid words
    #                     # This is handled by the overall validation
    #                     is_valid = True
    #                     segment_score = 0.1  # Very low score

    #             if is_valid:
    #                 prev_score, prev_text, prev_count = dp[j]
    #                 new_score = prev_score + segment_score
    #                 new_text = (prev_text + " " + segment).strip() if prev_text else segment
    #                 new_count = prev_count + 1

    #                 # Penalize too many small words
    #                 if len(segment) <= 2 and prev_count > 0:
    #                     new_score -= 0.5

    #                 if dp[i] is None or new_score > dp[i][0]:
    #                     dp[i] = (new_score, new_text, new_count)

    #     if dp[n] is None:
    #         return None

    #     final_score, final_text, word_count = dp[n]

    #     # Validate: must have at least 2 words, and average word length should be reasonable
    #     if word_count < 2:
    #         return None

    #     words = final_text.split()
    #     avg_len = sum(len(w) for w in words) / len(words) if words else 0

    #     # Reject if too many tiny words
    #     tiny_count = sum(1 for w in words if len(w) <= 2)
    #     if tiny_count > len(words) * 0.5:
    #         return None

    #     # Reject if average word length is too short (suggests over-splitting)
    #     if avg_len < 3:
    #         return None

    #     # Apply original case
    #     if word.isupper():
    #         final_text = final_text.upper()

    #     return final_text

    # def _greedy_word_split(self, word: str) -> Optional[str]:
    #     """
    #     Greedy approach: find the longest valid word from the start, repeat.
    #     """
    #     if not word or len(word) < 8:
    #         return None

    #     remaining = word
    #     parts = []

    #     while remaining:
    #         found = False

    #         # Try longest possible first
    #         for length in range(min(len(remaining), 15), 2, -1):
    #             candidate = remaining[:length]

    #             if self._is_known_word(candidate):
    #                 parts.append(candidate)
    #                 remaining = remaining[length:]
    #                 found = True
    #                 break

    #         if not found:
    #             # No valid word found at this position
    #             if len(remaining) <= 3 and len(parts) > 0:
    #                 # Append remainder to last word if short
    #                 parts[-1] = parts[-1] + remaining
    #                 remaining = ""
    #             else:
    #                 # Can't split properly
    #                 return None

    #     if len(parts) < 2:
    #         return None

    #     # Validate all parts
    #     for part in parts:
    #         if len(part) < 2:
    #             return None
    #         if len(part) == 2 and not self._is_known_word(part):
    #             # Reject unknown 2-letter parts
    #             return None

    #     # Apply original case
    #     result = ' '.join(parts)
    #     if word.isupper():
    #         result = result.upper()

    #     return result

    # def _looks_like_word(self, word: str) -> bool:
    #     """
    #     Check if a string looks like it could be a word (has vowels, reasonable structure).
    #     """
    #     if not word or len(word) < 2:
    #         return False

    #     word_upper = word.upper()

    #     # Must have at least one vowel (unless very short)
    #     vowels = sum(1 for c in word_upper if c in 'AEIOU')
    #     if len(word) > 3 and vowels == 0:
    #         return False

    #     # Check for impossible patterns
    #     for i in range(len(word_upper) - 2):
    #         if word_upper[i] == word_upper[i + 1] == word_upper[i + 2]:
    #             return False  # Three same chars in a row

    #     # Check vowel ratio
    #     alpha_chars = [c for c in word_upper if c.isalpha()]
    #     if alpha_chars:
    #         vowel_ratio = vowels / len(alpha_chars)
    #         if vowel_ratio < 0.1 or vowel_ratio > 0.8:
    #             return False

    #     return True

    # def _is_likely_word_boundary(self, text: str, pos: int) -> bool:
        """
        DEPRECATED - This method was too aggressive.
        Now we use _try_smart_word_split which requires full word validation.
        
        Keeping this as a stub in case other code references it.
        """
        # Always return False - we handle splitting differently now
        return False

    def _fix_missing_spaces(self, text: str, confidence: float) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Detect and fix missing spaces in text.
        
        CONSERVATIVE approach: only split when we're confident the merged
        word is NOT a valid word and the split produces meaningful results.
        """
        if not text:
            return text or "", []

        corrections = []
        result = text

        # Pattern 1: Only insert spaces at VERY clear boundaries
        # (lowercase->uppercase transitions, NOT based on bigram matching)
        new_result = []
        chars = list(result)
        i = 0

        while i < len(chars):
            new_result.append(chars[i])

            if i < len(chars) - 1:
                # Only split at clear case transitions within a token
                # (not at word boundaries that already have spaces)
                if (chars[i].isalpha() and chars[i + 1].isalpha() and
                        chars[i].islower() and chars[i + 1].isupper()):
                    # lowercase followed by uppercase is almost always a word boundary
                    # e.g., "aplicataC" -> "aplicata C"
                    new_result.append(' ')
                    self.stats['spaces_inserted'] += 1

                # Letter-to-digit transitions (only for long alpha sequences)
                elif (chars[i].isalpha() and chars[i + 1].isdigit() and
                    self._is_digit_boundary(result, i)):
                    new_result.append(' ')
                    self.stats['spaces_inserted'] += 1

                # Digit-to-letter transitions
                elif (chars[i].isdigit() and chars[i + 1].isalpha() and
                    self._is_digit_boundary(result, i)):
                    new_result.append(' ')
                    self.stats['spaces_inserted'] += 1

            i += 1

        new_text = ''.join(new_result)
        if new_text != result:
            corrections.append((result, new_text))
            result = new_text

        # Pattern 2: Try splitting ONLY long unknown words (not known words!)
        words = result.split()
        fixed_words = []
        changed = False

        for word in words:
            # Only attempt splitting if:
            # 1. Word is long enough (>15 chars - two real words merged)
            # 2. Word is NOT a known valid word
            # 3. Word is NOT a known word in wordfreq
            if (len(word) > 15 and word.isalpha() and
                    not self._is_known_word(word) and
                    self._detect_garble_level(word) < 0.4):

                split = self._try_smart_split(word)
                if split is not None and split != word:
                    corrections.append((word, split))
                    fixed_words.append(split)
                    changed = True
                    continue

            fixed_words.append(word)

        if changed:
            result = ' '.join(fixed_words)

        return result, corrections

    def _try_smart_split(self, word: str) -> Optional[str]:
        """
        Try to split a long concatenated word into real words.
        
        STRICT requirements:
        - Each resulting word must be at least 3 characters
        - Each resulting word must be a known word (dictionary or wordfreq)
        - The split must cover the entire input
        - Prefer fewer, longer words over many short ones
        """
        if not word or len(word) < 8:
            return None

        # Try SymSpell segmentation first, but validate strictly
        symspell_result = self._try_symspell_split_strict(word)
        if symspell_result:
            return symspell_result

        # Try dictionary-based split with strict validation
        if self.dictionary:
            dict_result = self._try_dictionary_split_strict(word)
            if dict_result:
                return dict_result

        return None

    def _try_symspell_split_strict(self, word: str) -> Optional[str]:
        """
        Use SymSpell segmentation with strict validation.
        Only accepts splits where ALL words are real, known words of length >= 3.
        """
        if not self._symspell_available or not word:
            return None

        try:
            from symspellpy import Verbosity

            word_lower = word.lower()
            best_segmentation = None
            best_score = 0.0

            # Try Romanian segmentation
            if self._symspell_ro:
                result = self._symspell_ro.word_segmentation(word_lower)
                if result and result.segmented_string:
                    score = self._validate_split(result.segmented_string, word)
                    if score > best_score:
                        best_score = score
                        best_segmentation = result.segmented_string

            # Try English segmentation
            if self._symspell_en:
                result = self._symspell_en.word_segmentation(word_lower)
                if result and result.segmented_string:
                    score = self._validate_split(result.segmented_string, word)
                    if score > best_score:
                        best_score = score
                        best_segmentation = result.segmented_string

            if best_segmentation and best_score > 0:
                # Apply original case
                if word.isupper():
                    return best_segmentation.upper()
                elif word.islower():
                    return best_segmentation
                else:
                    return best_segmentation.upper() if word[0].isupper() else best_segmentation

            return None

        except Exception as e:
            if self.verbose:
                self.logger.debug(f"SymSpell split error for '{word}': {e}")
            return None

    def _try_dictionary_split_strict(self, word: str) -> Optional[str]:
        """
        Try to split a word using dictionary with strict validation.
        Each segment must be >= 3 chars and a known word.
        """
        if not self.dictionary or not word:
            return None

        n = len(word)
        word_lower = word.lower()

        # dp[i] = (score, split_text, word_count) for best split of word[:i]
        dp: List[Optional[Tuple[float, str, int]]] = [None] * (n + 1)
        dp[0] = (0.0, "", 0)

        for i in range(1, n + 1):
            # Minimum segment length is 3 (no single/double char splits)
            for j in range(max(0, i - 15), max(0, i - 2)):
                if dp[j] is None:
                    continue

                segment = word_lower[j:i]

                # STRICT: segment must be at least 3 characters
                if len(segment) < 3:
                    continue

                # STRICT: segment must be a known word
                segment_known = (
                    self.dictionary.is_valid_word(segment) or
                    self._get_word_frequency(segment) > 1e-7
                )

                if not segment_known:
                    continue

                # Score: prefer longer words (penalize many short words)
                segment_score = len(segment) ** 1.5  # Superlinear reward for length
                total_score = dp[j][0] + segment_score
                prev_text = dp[j][1]
                word_count = dp[j][2] + 1
                new_text = (prev_text + " " + segment).strip() if prev_text else segment

                if dp[i] is None or total_score > dp[i][0]:
                    dp[i] = (total_score, new_text, word_count)

        if dp[n] is not None:
            split_text = dp[n][1]
            word_count = dp[n][2]

            # Must split into at least 2 words, and each must be >= 3 chars
            if word_count >= 2:
                split_words = split_text.split()
                all_valid = all(len(w) >= 3 for w in split_words)

                if all_valid:
                    # Apply original case
                    if word.isupper():
                        return split_text.upper()
                    return split_text

        return None

    def _validate_split(self, segmented: str, original: str) -> float:
        """
        Validate a word segmentation result.
        
        Returns a score > 0 if the split is valid, 0 if invalid.
        
        Requirements:
        - All words must be at least 3 characters
        - All words must be known (dictionary or wordfreq)
        - Must have at least 2 words
        - Must not have single-character words
        """
        if not segmented or ' ' not in segmented:
            return 0.0

        words = segmented.split()

        # Must produce at least 2 words
        if len(words) < 2:
            return 0.0

        # Check that rejoining matches original (no chars lost)
        rejoined = ''.join(words)
        if rejoined.lower() != original.lower().replace(' ', ''):
            return 0.0

        # ALL words must be at least 3 characters
        for w in words:
            if len(w) < 3:
                return 0.0

        # ALL words must be known
        known_count = 0
        total_score = 0.0

        for w in words:
            is_known = False

            # Check dictionary
            if self.dictionary and self.dictionary.is_valid_word(w):
                is_known = True
                total_score += len(w) ** 1.5

            # Check wordfreq
            if not is_known:
                freq = self._get_word_frequency(w)
                if freq > 1e-7:
                    is_known = True
                    total_score += len(w) ** 1.2
                elif freq > 1e-9:
                    # Marginal word - only count if long enough
                    if len(w) >= 5:
                        is_known = True
                        total_score += len(w) * 0.5

            if is_known:
                known_count += 1

        # ALL words must be known for the split to be accepted
        if known_count < len(words):
            return 0.0

        # Penalize splits with many small words (prefer fewer, longer words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 4:
            total_score *= 0.5

        return total_score

    def _is_likely_word_boundary(self, text: str, pos: int) -> bool:
        """
        Determine if position pos in text is likely a word boundary.
        
        VERY CONSERVATIVE: only return True for clear case transitions.
        We do NOT use bigram matching here to avoid over-splitting.
        """
        if pos < 1 or pos >= len(text) - 1:
            return False

        # ONLY split at lowercase->uppercase transitions
        # This is the most reliable indicator of a word boundary
        if text[pos].islower() and text[pos + 1].isupper():
            return True

        # Do NOT use bigram-based splitting or dictionary-based splitting here
        # Those are too aggressive and cause "LINIARA" -> "L IN IA R A"
        return False

    # def _fix_missing_spaces(self, text: str, confidence: float) -> Tuple[str, List[Tuple[str, str]]]:
    #     """
    #     Detect and fix missing spaces in text.
    #     """
    #     if not text:
    #         return text or "", []

    #     corrections = []
    #     result = text

    #     # Pattern 1: Insert spaces at likely word boundaries
    #     new_result = []
    #     chars = list(result)
    #     i = 0

    #     while i < len(chars):
    #         new_result.append(chars[i])

    #         if i < len(chars) - 1:
    #             if (chars[i].isalpha() and chars[i + 1].isalpha() and
    #                     self._is_likely_word_boundary(result, i)):
    #                 new_result.append(' ')
    #                 self.stats['spaces_inserted'] += 1

    #             elif (chars[i].isalpha() and chars[i + 1].isdigit() and
    #                   self._is_digit_boundary(result, i)):
    #                 new_result.append(' ')
    #                 self.stats['spaces_inserted'] += 1

    #             elif (chars[i].isdigit() and chars[i + 1].isalpha() and
    #                   self._is_digit_boundary(result, i)):
    #                 new_result.append(' ')
    #                 self.stats['spaces_inserted'] += 1

    #         i += 1

    #     new_text = ''.join(new_result)
    #     if new_text != result:
    #         corrections.append((result, new_text))
    #         result = new_text

    #     # Pattern 2: Try splitting long words using SymSpell segmentation
    #     words = result.split()
    #     fixed_words = []
    #     changed = False

    #     for word in words:
    #         if len(word) > 12 and word.isalpha():
    #             # Try SymSpell segmentation first
    #             segmented = self._symspell_segment(word)
    #             if segmented and segmented != word.lower():
    #                 # Apply original case
    #                 if word.isupper():
    #                     segmented = segmented.upper()
    #                 corrections.append((word, segmented))
    #                 fixed_words.append(segmented)
    #                 changed = True
    #                 continue

    #             # Fallback: try dictionary split
    #             if self.dictionary:
    #                 split = self._try_dictionary_split(word)
    #                 if split is not None and split != word:
    #                     corrections.append((word, split))
    #                     fixed_words.append(split)
    #                     changed = True
    #                     continue

    #         fixed_words.append(word)

    #     if changed:
    #         result = ' '.join(fixed_words)

    #     return result, corrections

    def _is_likely_word_boundary(self, text: str, pos: int) -> bool:
        """
        Determine if position pos in text is likely a word boundary.
        """
        if pos < 1 or pos >= len(text) - 1:
            return False

        before = text[max(0, pos - 5):pos + 1]
        after = text[pos + 1:min(len(text), pos + 7)]

        # Dictionary check
        if self.dictionary and len(after) >= 3:
            for length in range(min(len(after), 12), 2, -1):
                candidate = after[:length]
                if self.dictionary.is_valid_word(candidate):
                    for bl in range(min(len(before), 12), 2, -1):
                        before_candidate = before[-bl:]
                        if self.dictionary.is_valid_word(before_candidate):
                            return True

        # Heuristic: lowercase to uppercase transition
        if text[pos].islower() and text[pos + 1].isupper():
            return True

        # Common word starts
        common_starts = ['SI', 'DE', 'LA', 'IN', 'CU', 'PE', 'DI', 'PR', 'CO', 'AN', 'AP']
        if len(after) >= 2 and after[:2].upper() in common_starts:
            if before[-1].isalpha() and len(before) >= 3:
                return True

        # wordfreq check: see if splitting creates known words
        if self._wordfreq_available and len(before) >= 3 and len(after) >= 3:
            before_word = before[-min(len(before), 8):]
            after_word = after[:min(len(after), 8)]
            freq_before = self._get_word_frequency(before_word)
            freq_after = self._get_word_frequency(after_word)
            if freq_before > 1e-6 and freq_after > 1e-6:
                return True

        return False

    def _is_digit_boundary(self, text: str, pos: int) -> bool:
        """Determine if position between letter and digit is a word boundary."""
        start = pos
        while start > 0 and text[start - 1] != ' ':
            start -= 1
        end = pos + 1
        while end < len(text) and text[end] != ' ':
            end += 1

        token = text[start:end]

        letters = sum(1 for c in token if c.isalpha())
        digits = sum(1 for c in token if c.isdigit())

        # Keep codes like "P03", "C41" together
        if letters <= 2 and digits <= 3 and len(token) <= 5:
            return False

        # Split things like "APLICATA41" or "41Sl"
        if letters > 3 and digits > 0:
            return True

        return False

    def _try_dictionary_split(self, word: str) -> Optional[str]:
        """Try to split a long word into dictionary terms."""
        if not self.dictionary or not word:
            return None

        n = len(word)
        dp: List[Optional[Tuple[float, str]]] = [None] * (n + 1)
        dp[0] = (0.0, "")

        for i in range(1, n + 1):
            for j in range(max(0, i - 15), i):
                if dp[j] is None:
                    continue

                segment = word[j:i]
                segment_score = self._score_segment(segment)

                if segment_score > 0:
                    total_score = dp[j][0] + segment_score
                    prev_text = dp[j][1]
                    new_text = (prev_text + " " + segment).strip() if prev_text else segment

                    if dp[i] is None or total_score > dp[i][0]:
                        dp[i] = (total_score, new_text)

        if dp[n] is not None and dp[n][0] > len(word) * 0.3:
            split_text = dp[n][1]
            if ' ' in split_text:
                return split_text

        return None

    def _score_segment(self, segment: str) -> float:
        """Score a segment based on whether it's a valid word/code."""
        if not segment:
            return 0.0

        # Check dictionary
        if self.dictionary and self.dictionary.is_valid_word(segment):
            return len(segment) * 2.0

        # Check wordfreq
        freq = self._get_word_frequency(segment)
        if freq > 1e-6:
            return len(segment) * 1.8
        if freq > 1e-8:
            return len(segment) * 1.2

        # Short code (1-2 letters + digits)
        if len(segment) <= 4:
            letters = sum(1 for c in segment if c.isalpha())
            digits = sum(1 for c in segment if c.isdigit())
            if letters <= 2 and digits >= 1:
                return len(segment) * 1.5

        # Plausible word check
        vowels = sum(1 for c in segment.upper() if c in 'AEIOU')
        if len(segment) > 2 and vowels == 0:
            return 0.0

        if len(segment) >= 3 and vowels >= 1:
            return len(segment) * 0.5

        if len(segment) <= 2:
            common_short = {'SI', 'DE', 'LA', 'IN', 'CU', 'PE', 'A', 'I', 'C', 'S'}
            if segment.upper() in common_short:
                return len(segment) * 1.5
            return 0.1

        return 0.0


    def _fix_garbled_words(self, text: str, confidence: float) -> Tuple[str, List[Tuple[str, str]]]:
        """Fix garbled words by comparing against dictionary using confusion matrix."""
        if not self.dictionary or not text:
            return text or "", []

        corrections = []
        words = text.split()
        fixed_words = []

        for word in words:
            if not word:
                fixed_words.append("")
                continue

            if (len(word) < 3 or
                    word.isdigit() or
                    self._is_known_word(word)):
                fixed_words.append(word)
                continue

            garble_score = self._detect_garble_level(word)

            if garble_score < 0.3 and confidence > 60:
                fixed_words.append(word)
                continue

            best_correction = self._find_confusion_correction(word)

            if best_correction is not None and best_correction != word:
                # VALIDATE: correction should not be shorter than original
                # and should not introduce spaces (that's the job of _fix_missing_spaces)
                if ' ' not in best_correction and len(best_correction) >= len(word) - 1:
                    corrections.append((word, best_correction))
                    fixed_words.append(best_correction)
                    self.stats['words_fixed'] += 1
                    if self.verbose:
                        self.logger.debug(f"  Garble fix: '{word}' -> '{best_correction}'")
                else:
                    fixed_words.append(word)
            else:
                fixed_words.append(word)

        return ' '.join(fixed_words), corrections
    
    def _fix_with_symspell(self, text: str, confidence: float) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Fix remaining unknown words using SymSpell lookup.
        Does NOT split words - only substitutes characters.
        """
        if not self._symspell_available or not text:
            return text or "", []

        corrections = []
        words = text.split()
        fixed_words = []

        for word in words:
            if not word:
                fixed_words.append("")
                continue

            # Skip if already known, too short, or is a number/code
            if (len(word) < 3 or
                    word.isdigit() or
                    self._is_known_word(word) or
                    (len(word) <= 4 and any(c.isdigit() for c in word))):
                fixed_words.append(word)
                continue

            # Skip words that don't look garbled
            garble_score = self._detect_garble_level(word)
            if garble_score < 0.2 and confidence > 70:
                fixed_words.append(word)
                continue

            # Determine max edit distance based on word length
            if len(word) <= 4:
                max_dist = 1
            elif len(word) <= 7:
                max_dist = 2 if confidence < 60 else 1
            else:
                max_dist = 2

            suggestion = self._symspell_lookup(word, max_edit_distance=max_dist)

            if suggestion is not None and suggestion.lower() != word.lower():
                # CRITICAL: Do not accept suggestions that contain spaces
                if ' ' in suggestion:
                    fixed_words.append(word)
                    continue

                # Do not accept suggestions drastically different in length
                if abs(len(suggestion) - len(word)) > max(2, len(word) * 0.3):
                    fixed_words.append(word)
                    continue

                # Apply original case
                corrected = self._apply_case_pattern(word, suggestion)
                if corrected is None:
                    corrected = suggestion

                corrections.append((word, corrected))
                fixed_words.append(corrected)
                self.stats['symspell_corrections'] += 1

                if self.verbose:
                    self.logger.debug(f"  SymSpell fix: '{word}' -> '{corrected}'")
            else:
                fixed_words.append(word)

        return ' '.join(fixed_words), corrections
    
    def _detect_garble_level(self, word: str) -> float:
        """Detect how garbled a word is (0.0 = looks fine, 1.0 = very garbled)."""
        if not word or len(word) < 3:
            return 0.0

        score = 0.0
        upper = word.upper()
        n = len(upper)

        # Impossible trigrams
        for i in range(n - 2):
            trigram = upper[i:i + 3]
            if trigram in self.impossible_trigrams:
                score += 0.3

        # Long consonant clusters
        vowels_set = set('AEIOU')
        consonant_run = 0
        max_consonant_run = 0
        for c in upper:
            if c.isalpha() and c not in vowels_set:
                consonant_run += 1
                max_consonant_run = max(max_consonant_run, consonant_run)
            else:
                consonant_run = 0

        if max_consonant_run > 4:
            score += 0.2 * (max_consonant_run - 4)

        # Repeated characters
        for i in range(n - 2):
            if upper[i] == upper[i + 1] == upper[i + 2]:
                score += 0.4

        # Implausible bigrams
        implausible_bigrams = 0
        total_alpha_bigrams = 0
        for i in range(n - 1):
            bigram = upper[i:i + 2]
            if bigram[0].isalpha() and bigram[1].isalpha():
                total_alpha_bigrams += 1
                if bigram not in self.common_bigrams:
                    implausible_bigrams += 1

        if total_alpha_bigrams > 0:
            bigram_ratio = implausible_bigrams / total_alpha_bigrams
            if bigram_ratio > 0.6:
                score += 0.3

        # Vowel ratio
        alpha_chars = [c for c in upper if c.isalpha()]
        if alpha_chars:
            vowel_count = sum(1 for c in alpha_chars if c in vowels_set)
            vowel_ratio = vowel_count / len(alpha_chars)
            if vowel_ratio < 0.15 or vowel_ratio > 0.75:
                score += 0.2

        # wordfreq check: if word has zero frequency, it's more likely garbled
        if self._wordfreq_available and len(word) >= 4:
            freq = self._get_word_frequency(word)
            if freq == 0:
                score += 0.15

        return min(score, 1.0)

    def _find_confusion_correction(self, word: str) -> Optional[str]:
        """Find the best correction for a garbled word using the confusion matrix."""
        if not self.dictionary or not word:
            return None

        word_upper = word.upper()
        n = len(word_upper)

        best_match = None
        best_score = 0.0

        # Strategy 1: Confusion-aware similarity against dictionary
        for term_lower, term in self.dictionary.terms_lower_map.items():
            term_upper = term.upper()

            if abs(len(term_upper) - n) > max(2, n * 0.25):
                continue

            sim_score = self._confusion_similarity(word_upper, term_upper)

            if sim_score > best_score and sim_score > 0.65:
                best_score = sim_score
                best_match = term

        # Check word corrections
        for correct_word in self.dictionary.word_corrections_by_correct.keys():
            correct_upper = correct_word.upper()

            if abs(len(correct_upper) - n) > max(2, n * 0.25):
                continue

            sim_score = self._confusion_similarity(word_upper, correct_upper)

            if sim_score > best_score and sim_score > 0.65:
                best_score = sim_score
                best_match = correct_word

        if best_match is not None:
            result = self._apply_case_pattern(word, best_match)
            return result if result is not None else best_match

        # Strategy 2: Generate candidates by substitution
        candidates = self._generate_confusion_candidates(word_upper, max_substitutions=2)

        for candidate in candidates:
            if not candidate:
                continue
            if self.dictionary.is_valid_word(candidate):
                result = self._apply_case_pattern(word, candidate)
                return result if result is not None else candidate

            correction = self.dictionary.get_word_correction(candidate)
            if correction is not None:
                result = self._apply_case_pattern(word, correction)
                return result if result is not None else correction

            # Also check against wordfreq
            freq = self._get_word_frequency(candidate)
            if freq > 1e-6:
                result = self._apply_case_pattern(word, candidate)
                return result if result is not None else candidate

        return None

    def _confusion_similarity(self, word_a: str, word_b: str) -> float:
        """Calculate similarity considering character confusion."""
        if not word_a or not word_b:
            return 0.0

        n, m = len(word_a), len(word_b)

        if n == m:
            matches = 0
            confusion_matches = 0

            for i in range(n):
                if word_a[i] == word_b[i]:
                    matches += 1
                elif self._is_confusion_pair(word_a[i], word_b[i]):
                    confusion_matches += 1

            score = (matches + confusion_matches * 0.7) / n
            return score

        # Different lengths: alignment-based scoring
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if word_a[i - 1] == word_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1.0
                elif self._is_confusion_pair(word_a[i - 1], word_b[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1] + 0.7
                else:
                    dp[i][j] = max(
                        dp[i - 1][j] - 0.1,
                        dp[i][j - 1] - 0.1,
                        dp[i - 1][j - 1] - 0.3,
                    )

        max_len = max(n, m)
        return dp[n][m] / max_len if max_len > 0 else 0.0

    def _is_confusion_pair(self, char_a: str, char_b: str) -> bool:
        """Check if two characters are a known confusion pair."""
        if char_a in self.confusion_matrix:
            for confused, _ in self.confusion_matrix[char_a]:
                if confused == char_b:
                    return True

        if char_b in self.confusion_matrix:
            for confused, _ in self.confusion_matrix[char_b]:
                if confused == char_a:
                    return True

        return False

    def _generate_confusion_candidates(
        self,
        word: str,
        max_substitutions: int = 2
    ) -> List[str]:
        """Generate candidate corrections by applying confusion matrix substitutions."""
        candidates = set()
        n = len(word)

        for i in range(n):
            char = word[i]
            if char in self.confusion_matrix:
                for replacement, _ in self.confusion_matrix[char]:
                    if len(replacement) == 1:
                        candidate = word[:i] + replacement + word[i + 1:]
                        candidates.add(candidate)
                    elif len(replacement) == 2:
                        candidate = word[:i] + replacement + word[i + 1:]
                        candidates.add(candidate)

        if max_substitutions >= 2 and n <= 12:
            single_candidates = list(candidates)[:50]
            for base in single_candidates:
                for i in range(len(base)):
                    if i >= len(base):
                        break
                    char = base[i]
                    if char in self.confusion_matrix:
                        for replacement, likelihood in self.confusion_matrix[char]:
                            if likelihood >= 0.3 and len(replacement) == 1:
                                candidate = base[:i] + replacement + base[i + 1:]
                                candidates.add(candidate)

        return list(candidates)[:200]

    def _fix_character_substitutions(
        self,
        text: str,
        confidence: float
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Fix remaining character substitutions that don't require dictionary lookup."""
        if not text:
            return text or "", []

        corrections = []
        words = text.split()
        fixed_words = []

        for word in words:
            if not word:
                fixed_words.append("")
                continue
            fixed = self._fix_obvious_substitutions(word)
            if fixed != word:
                char_diff = sum(
                    1 for a, b in zip(word, fixed) if a != b
                ) if len(word) == len(fixed) else 1
                self.stats['chars_corrected'] += char_diff
                corrections.append((word, fixed))
            fixed_words.append(fixed)

        return ' '.join(fixed_words), corrections

    def _fix_obvious_substitutions(self, word: str) -> str:
        """Fix obvious character substitutions based on context."""
        if not word or len(word) < 2:
            return word or ""

        chars = list(word)
        alpha_count = sum(1 for c in chars if c.isalpha())
        digit_count = sum(1 for c in chars if c.isdigit())

        if alpha_count > digit_count * 2 and alpha_count >= 3:
            for i, c in enumerate(chars):
                if c == '0' and (i > 0 or len(chars) > 2):
                    chars[i] = 'O'
                elif c == '1':
                    if ((i > 0 and chars[i - 1].isupper()) or
                            (i < len(chars) - 1 and chars[i + 1].isupper())):
                        chars[i] = 'I'
                    elif i > 0 and chars[i - 1].islower():
                        chars[i] = 'l'
                elif c == '5' and alpha_count > 3:
                    chars[i] = 'S'
                elif c == '3' and alpha_count > 3:
                    chars[i] = 'E'
                elif c == '4' and alpha_count > 3:
                    chars[i] = 'A'
                elif c == '8' and alpha_count > 3:
                    chars[i] = 'B'

        elif digit_count > alpha_count * 2 and digit_count >= 2:
            for i, c in enumerate(chars):
                if c in ('O', 'o'):
                    chars[i] = '0'
                elif c in ('I', 'l'):
                    chars[i] = '1'
                elif c in ('S', 's'):
                    chars[i] = '5'
                elif c == 'B':
                    chars[i] = '8'
                elif c in ('Z', 'z'):
                    chars[i] = '2'

        return ''.join(chars)

    def _recover_missing_characters(
        self,
        text: str,
        confidence: float
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Try to recover missing characters (dropped by OCR)."""
        if not text:
            return text or "", []

        corrections = []
        words = text.split()
        fixed_words = []

        for word in words:
            if not word:
                fixed_words.append("")
                continue

            if (len(word) < 3 or
                    word.isdigit() or
                    self._is_known_word(word)):
                fixed_words.append(word)
                continue

            # Try inserting one character at each position (dictionary match)
            fixed = self._try_insert_missing_char(word)
            if fixed is not None and fixed != word:
                corrections.append((word, fixed))
                fixed_words.append(fixed)
                self.stats['chars_recovered'] += 1
                if self.verbose:
                    self.logger.debug(f"  Recovered char: '{word}' -> '{fixed}'")
                continue

            # Try SymSpell with edit distance that accounts for deletion
            if self._symspell_available and len(word) >= 4:
                suggestion = self._symspell_lookup(word, max_edit_distance=2)
                if (suggestion is not None and
                        suggestion.lower() != word.lower() and
                        len(suggestion) > len(word)):
                    corrected = self._apply_case_pattern(word, suggestion)
                    if corrected is None:
                        corrected = suggestion
                    # Validate the suggestion is actually a recovery (longer)
                    if len(corrected) >= len(word):
                        corrections.append((word, corrected))
                        fixed_words.append(corrected)
                        self.stats['chars_recovered'] += 1
                        if self.verbose:
                            self.logger.debug(f"  SymSpell recovered: '{word}' -> '{corrected}'")
                        continue

            fixed_words.append(word)

        return ' '.join(fixed_words), corrections

    def _try_insert_missing_char(self, word: str) -> Optional[str]:
        """Try inserting a missing character to match a dictionary term."""
        if not word:
            return None

        word_upper = word.upper()
        best_match = None
        best_score = 0.0

        # Check dictionary terms that are exactly 1 char longer
        if self.dictionary:
            for term_lower, term in self.dictionary.terms_lower_map.items():
                term_upper = term.upper()

                if len(term_upper) != len(word_upper) + 1:
                    continue

                if self._is_one_deletion_away(word_upper, term_upper):
                    score = len(term) / 10.0
                    if score > best_score:
                        best_score = score
                        best_match = term

            # Check word corrections
            if best_match is None:
                for correct_word in self.dictionary.word_corrections_by_correct.keys():
                    correct_upper = correct_word.upper()

                    if len(correct_upper) != len(word_upper) + 1:
                        continue

                    if self._is_one_deletion_away(word_upper, correct_upper):
                        best_match = correct_word
                        break

        # Also check wordfreq: try inserting common characters
        if best_match is None and self._wordfreq_available and len(word) >= 4:
            common_chars = 'AEIOURSTNLCDMPHGBFVWYKJXQZ'
            best_freq = 0.0

            for pos in range(len(word) + 1):
                for char in common_chars:
                    candidate = word[:pos] + char + word[pos:]
                    freq = self._get_word_frequency(candidate)
                    if freq > best_freq and freq > 1e-6:
                        best_freq = freq
                        best_match = candidate

        if best_match is not None:
            result = self._apply_case_pattern(word, best_match)
            return result if result is not None else best_match

        return None

    def _is_one_deletion_away(self, shorter: str, longer: str) -> bool:
        """Check if shorter is the same as longer with exactly one char deleted."""
        if len(longer) - len(shorter) != 1:
            return False

        i = j = 0
        diff_count = 0

        while i < len(shorter) and j < len(longer):
            if shorter[i] == longer[j]:
                i += 1
                j += 1
            else:
                j += 1
                diff_count += 1
                if diff_count > 1:
                    return False

        return True

    def _apply_case_pattern(self, original: str, corrected: str) -> str:
        """Apply the case pattern from original to corrected."""
        if not original or not corrected:
            return corrected or ""

        if original.isupper():
            return corrected.upper()
        if original.islower():
            return corrected.lower()
        if len(original) > 1 and original[0].isupper() and original[1:].islower():
            return corrected.capitalize()

        result = []
        for i, c in enumerate(corrected):
            if i < len(original):
                if original[i].isupper():
                    result.append(c.upper())
                else:
                    result.append(c.lower())
            else:
                result.append(c)

        return ''.join(result)

    def get_stats(self) -> Dict[str, int]:
        """Get correction statistics."""
        return self.stats.copy()
    
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