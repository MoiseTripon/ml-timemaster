"""
OCR Validation and Correction Module for ML Timemaster.
Provides dictionary-based validation and fuzzy matching correction for OCR results.
"""

import json
import logging
import re
import os
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class CorrectionResult:
    """Holds the result of a correction attempt."""
    original: str
    corrected: str
    was_corrected: bool
    confidence: float
    corrections_made: List[Tuple[str, str]]  # List of (original_word, corrected_word)


class OCRValidator:
    """
    Validates and corrects OCR results using dictionary-based matching.
    Optimized for schedule-related text with low computational overhead.
    """
    
    # Default dictionary filename
    DEFAULT_DICT_FILE = "ocr_dictionary.json"
    
    # Similarity thresholds
    HIGH_SIMILARITY_THRESHOLD = 0.85  # Auto-correct above this
    LOW_SIMILARITY_THRESHOLD = 0.60   # Consider correction above this
    
    # Character substitution patterns for common OCR errors
    CHAR_SUBSTITUTIONS = {
        '0': 'O', 'O': '0',
        '1': 'I', 'I': '1', 'l': '1', '|': 'I',
        '5': 'S', 'S': '5',
        '8': 'B', 'B': '8',
        '6': 'G', 'G': '6',
        '2': 'Z', 'Z': '2',
    }
    
    # Multi-character substitutions
    MULTI_CHAR_SUBSTITUTIONS = {
        'rn': 'm', 'm': 'rn',
        'cl': 'd', 'd': 'cl',
        'vv': 'w', 'w': 'vv',
        'nn': 'm',
        'ii': 'u',
    }
    
    def __init__(
        self,
        dictionary_path: Optional[str] = None,
        similarity_threshold: float = 0.80,
        enable_fuzzy_matching: bool = True,
        enable_char_substitution: bool = True,
        case_sensitive: bool = False,
        preserve_numbers: bool = True,
        max_word_length_diff: int = 2,
        verbose: bool = False
    ):
        """
        Initialize OCR Validator.
        
        Args:
            dictionary_path: Path to dictionary JSON file. If None, looks for default.
            similarity_threshold: Minimum similarity ratio for corrections (0.0-1.0).
            enable_fuzzy_matching: Enable fuzzy string matching.
            enable_char_substitution: Enable common OCR character substitution fixes.
            case_sensitive: Whether matching should be case-sensitive.
            preserve_numbers: Don't try to correct pure numbers.
            max_word_length_diff: Maximum length difference for fuzzy matching.
            verbose: Enable verbose logging.
        """
        self.similarity_threshold = similarity_threshold
        self.enable_fuzzy = enable_fuzzy_matching
        self.enable_char_sub = enable_char_substitution
        self.case_sensitive = case_sensitive
        self.preserve_numbers = preserve_numbers
        self.max_len_diff = max_word_length_diff
        self.verbose = verbose
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Initialize dictionary
        self.dictionary: Set[str] = set()
        self.dictionary_lower: Set[str] = set()  # Lowercase version for case-insensitive matching
        self.abbreviations: Dict[str, str] = {}
        self.ocr_error_map: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            'words_processed': 0,
            'words_corrected': 0,
            'fuzzy_matches': 0,
            'char_substitutions': 0,
            'abbreviation_expansions': 0
        }
        
        # Load dictionary
        self._load_dictionary(dictionary_path)
        
    def _find_dictionary_path(self, provided_path: Optional[str]) -> Optional[str]:
        """Find the dictionary file path."""
        if provided_path and os.path.exists(provided_path):
            return provided_path
        
        # Search in common locations
        search_paths = [
            self.DEFAULT_DICT_FILE,
            os.path.join(os.path.dirname(__file__), self.DEFAULT_DICT_FILE),
            os.path.join(os.path.dirname(__file__), 'config', self.DEFAULT_DICT_FILE),
            os.path.join(os.getcwd(), self.DEFAULT_DICT_FILE),
            os.path.join(os.getcwd(), 'config', self.DEFAULT_DICT_FILE),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_dictionary(self, dictionary_path: Optional[str]) -> None:
        """Load dictionary from JSON file."""
        path = self._find_dictionary_path(dictionary_path)
        
        if path is None:
            self.logger.warning(
                f"Dictionary file not found. Creating empty dictionary. "
                f"Create '{self.DEFAULT_DICT_FILE}' to enable corrections."
            )
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load word categories
            word_categories = [
                'days', 'months', 'time_words', 'schedule_terms',
                'common_names', 'status_words', 'numbers_as_words', 'custom_words'
            ]
            
            for category in word_categories:
                if category in data:
                    words = data[category]
                    self.dictionary.update(words)
                    self.dictionary_lower.update(w.lower() for w in words)
            
            # Load abbreviations
            if 'abbreviations' in data:
                self.abbreviations = data['abbreviations']
                # Add abbreviations to dictionary too
                self.dictionary.update(self.abbreviations.keys())
                self.dictionary_lower.update(k.lower() for k in self.abbreviations.keys())
            
            # Load OCR error patterns
            if 'common_ocr_errors' in data:
                self.ocr_error_map = data['common_ocr_errors']
            
            self.logger.info(
                f"Loaded dictionary from {path}: "
                f"{len(self.dictionary)} words, {len(self.abbreviations)} abbreviations"
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in dictionary file: {e}")
        except Exception as e:
            self.logger.error(f"Error loading dictionary: {e}")
    
    def add_word(self, word: str) -> None:
        """Add a word to the dictionary at runtime."""
        self.dictionary.add(word)
        self.dictionary_lower.add(word.lower())
    
    def add_words(self, words: List[str]) -> None:
        """Add multiple words to the dictionary at runtime."""
        for word in words:
            self.add_word(word)
    
    def add_abbreviation(self, abbrev: str, full_form: str) -> None:
        """Add an abbreviation mapping at runtime."""
        self.abbreviations[abbrev] = full_form
        self.dictionary.add(abbrev)
        self.dictionary_lower.add(abbrev.lower())
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text is purely numeric (including time/date formats)."""
        # Remove common separators
        cleaned = re.sub(r'[:\-/.,\s]', '', text)
        return cleaned.isdigit()
    
    def _is_time_format(self, text: str) -> bool:
        """Check if text appears to be a time format."""
        time_patterns = [
            r'^\d{1,2}:\d{2}$',           # 12:30
            r'^\d{1,2}:\d{2}:\d{2}$',     # 12:30:45
            r'^\d{1,2}:\d{2}\s*[AaPp][Mm]$',  # 12:30 PM
            r'^\d{1,2}\s*[AaPp][Mm]$',    # 12PM
        ]
        return any(re.match(p, text.strip()) for p in time_patterns)
    
    def _is_date_format(self, text: str) -> bool:
        """Check if text appears to be a date format."""
        date_patterns = [
            r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$',  # 01/15/2024
            r'^\d{2,4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}$',  # 2024-01-15
            r'^\d{1,2}[/\-\.]\d{1,2}$',                 # 01/15
        ]
        return any(re.match(p, text.strip()) for p in date_patterns)
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not self.case_sensitive:
            s1, s2 = s1.lower(), s2.lower()
        return SequenceMatcher(None, s1, s2).ratio()
    
    def _find_best_match(self, word: str) -> Optional[Tuple[str, float]]:
        """Find the best matching word from dictionary."""
        if not self.dictionary:
            return None
        
        word_lower = word.lower()
        word_len = len(word)
        
        best_match = None
        best_score = 0.0
        
        # Use appropriate dictionary based on case sensitivity
        search_dict = self.dictionary if self.case_sensitive else self.dictionary_lower
        original_dict = list(self.dictionary)  # Keep original for returning
        
        for i, dict_word in enumerate(search_dict):
            # Skip if length difference is too large
            if abs(len(dict_word) - word_len) > self.max_len_diff:
                continue
            
            # Calculate similarity
            compare_word = word if self.case_sensitive else word_lower
            compare_dict = dict_word if self.case_sensitive else dict_word.lower()
            
            similarity = self._calculate_similarity(compare_word, compare_dict)
            
            if similarity > best_score and similarity >= self.LOW_SIMILARITY_THRESHOLD:
                best_score = similarity
                # Return the original cased version
                if self.case_sensitive:
                    best_match = dict_word
                else:
                    # Find the original cased version
                    for orig in original_dict:
                        if orig.lower() == dict_word:
                            best_match = orig
                            break
        
        if best_match and best_score >= self.similarity_threshold:
            return (best_match, best_score)
        
        return None
    
    def _apply_char_substitutions(self, word: str) -> List[str]:
        """Generate possible variations using character substitutions."""
        variations = [word]
        
        # Single character substitutions
        for i, char in enumerate(word):
            if char in self.CHAR_SUBSTITUTIONS:
                new_word = word[:i] + self.CHAR_SUBSTITUTIONS[char] + word[i+1:]
                variations.append(new_word)
        
        # Multi-character substitutions
        for pattern, replacement in self.MULTI_CHAR_SUBSTITUTIONS.items():
            if pattern in word:
                new_word = word.replace(pattern, replacement, 1)
                variations.append(new_word)
        
        return variations
    
    def _try_char_substitution_match(self, word: str) -> Optional[Tuple[str, float]]:
        """Try to find a match using character substitutions."""
        if not self.enable_char_sub:
            return None
        
        variations = self._apply_char_substitutions(word)
        
        for variation in variations[1:]:  # Skip original
            # Check exact match first
            if variation in self.dictionary:
                return (variation, 0.95)  # High confidence for exact match after substitution
            if not self.case_sensitive and variation.lower() in self.dictionary_lower:
                # Find original cased version
                for orig in self.dictionary:
                    if orig.lower() == variation.lower():
                        return (orig, 0.95)
                        
            # Try fuzzy match on variation
            if self.enable_fuzzy:
                match = self._find_best_match(variation)
                if match and match[1] >= self.HIGH_SIMILARITY_THRESHOLD:
                    return match
        
        return None
    
    def _correct_word(self, word: str) -> Tuple[str, bool, float]:
        """
        Attempt to correct a single word.
        
        Returns:
            Tuple of (corrected_word, was_corrected, confidence)
        """
        self.stats['words_processed'] += 1
        
        original = word
        word_stripped = word.strip()
        
        # Skip empty words
        if not word_stripped:
            return (word, False, 1.0)
        
        # Preserve numbers if configured
        if self.preserve_numbers and self._is_numeric(word_stripped):
            return (word, False, 1.0)
        
        # Preserve time and date formats
        if self._is_time_format(word_stripped) or self._is_date_format(word_stripped):
            return (word, False, 1.0)
        
        # Check if already in dictionary (exact match)
        if word_stripped in self.dictionary:
            return (word, False, 1.0)
        if not self.case_sensitive and word_stripped.lower() in self.dictionary_lower:
            return (word, False, 1.0)
        
        # Check abbreviations
        if word_stripped in self.abbreviations:
            self.stats['abbreviation_expansions'] += 1
            # Preserve the abbreviation, don't expand (user might want short form)
            return (word, False, 1.0)
        
        # Try character substitution matching first (faster for common OCR errors)
        char_sub_match = self._try_char_substitution_match(word_stripped)
        if char_sub_match:
            corrected, confidence = char_sub_match
            self.stats['char_substitutions'] += 1
            self.stats['words_corrected'] += 1
            if self.verbose:
                self.logger.debug(f"Char substitution: '{word_stripped}' -> '{corrected}' ({confidence:.2f})")
            
            # Preserve original casing pattern if possible
            corrected = self._preserve_case_pattern(word_stripped, corrected)
            return (corrected, True, confidence)
        
        # Try fuzzy matching
        if self.enable_fuzzy:
            fuzzy_match = self._find_best_match(word_stripped)
            if fuzzy_match:
                corrected, confidence = fuzzy_match
                self.stats['fuzzy_matches'] += 1
                self.stats['words_corrected'] += 1
                if self.verbose:
                    self.logger.debug(f"Fuzzy match: '{word_stripped}' -> '{corrected}' ({confidence:.2f})")
                
                # Preserve original casing pattern if possible
                corrected = self._preserve_case_pattern(word_stripped, corrected)
                return (corrected, True, confidence)
        
        # No correction found
        return (word, False, 0.5)  # Lower confidence for unknown words
    
    def _preserve_case_pattern(self, original: str, corrected: str) -> str:
        """Try to preserve the case pattern of the original word."""
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
        
        # Default: return as-is from dictionary
        return corrected
    
    def _tokenize(self, text: str) -> List[Tuple[str, bool]]:
        """
        Tokenize text into words and separators.
        
        Returns:
            List of (token, is_word) tuples
        """
        tokens = []
        pattern = r'(\s+|[^\w\s]+)'  # Split on whitespace and punctuation
        
        parts = re.split(pattern, text)
        
        for part in parts:
            if not part:
                continue
            is_word = bool(re.match(r'^\w+$', part))
            tokens.append((part, is_word))
        
        return tokens
    
    def validate_and_correct(self, text: str, ocr_confidence: float = 100.0) -> CorrectionResult:
        """
        Validate and correct OCR text.
        
        Args:
            text: The OCR result text to validate/correct.
            ocr_confidence: The OCR confidence score (0-100). Lower confidence
                           texts are more aggressively corrected.
        
        Returns:
            CorrectionResult with original, corrected text, and metadata.
        """
        if not text or not text.strip():
            return CorrectionResult(
                original=text,
                corrected=text,
                was_corrected=False,
                confidence=1.0,
                corrections_made=[]
            )
        
        # Adjust similarity threshold based on OCR confidence
        # Lower OCR confidence = more aggressive correction
        if ocr_confidence < 60:
            effective_threshold = self.similarity_threshold - 0.10
        elif ocr_confidence < 80:
            effective_threshold = self.similarity_threshold - 0.05
        else:
            effective_threshold = self.similarity_threshold
        
        original_threshold = self.similarity_threshold
        self.similarity_threshold = max(0.5, effective_threshold)
        
        try:
            # Tokenize text
            tokens = self._tokenize(text)
            
            corrected_tokens = []
            corrections_made = []
            total_confidence = 0.0
            word_count = 0
            
            for token, is_word in tokens:
                if is_word:
                    corrected, was_corrected, confidence = self._correct_word(token)
                    corrected_tokens.append(corrected)
                    total_confidence += confidence
                    word_count += 1
                    
                    if was_corrected:
                        corrections_made.append((token, corrected))
                else:
                    # Keep separators and punctuation as-is
                    corrected_tokens.append(token)
            
            corrected_text = ''.join(corrected_tokens)
            avg_confidence = total_confidence / word_count if word_count > 0 else 1.0
            
            return CorrectionResult(
                original=text,
                corrected=corrected_text,
                was_corrected=len(corrections_made) > 0,
                confidence=avg_confidence,
                corrections_made=corrections_made
            )
            
        finally:
            # Restore original threshold
            self.similarity_threshold = original_threshold
    
    def correct_text(self, text: str, ocr_confidence: float = 100.0) -> str:
        """
        Convenience method that returns only the corrected text.
        
        Args:
            text: The OCR result text to validate/correct.
            ocr_confidence: The OCR confidence score (0-100).
        
        Returns:
            Corrected text string.
        """
        result = self.validate_and_correct(text, ocr_confidence)
        return result.corrected
    
    def get_stats(self) -> Dict:
        """Get validation statistics."""
        stats = self.stats.copy()
        if stats['words_processed'] > 0:
            stats['correction_rate'] = stats['words_corrected'] / stats['words_processed'] * 100
        else:
            stats['correction_rate'] = 0.0
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            'words_processed': 0,
            'words_corrected': 0,
            'fuzzy_matches': 0,
            'char_substitutions': 0,
            'abbreviation_expansions': 0
        }
    
    def save_dictionary(self, path: str) -> None:
        """Save current dictionary to a JSON file."""
        data = {
            'metadata': {
                'version': '1.0',
                'description': 'OCR Dictionary (exported)',
            },
            'custom_words': list(self.dictionary),
            'abbreviations': self.abbreviations,
            'common_ocr_errors': self.ocr_error_map
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Dictionary saved to {path}")


# Standalone test function
def test_validator():
    """Test the OCR validator."""
    print("=" * 60)
    print("Testing OCR Validator")
    print("=" * 60)
    
    # Create validator with verbose logging
    validator = OCRValidator(
        similarity_threshold=0.80,
        verbose=True
    )
    
    # Add some test words if dictionary wasn't loaded
    if len(validator.dictionary) == 0:
        validator.add_words([
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday", "Schedule", "Meeting", "Break",
            "Morning", "Afternoon", "Evening", "Employee", "Shift"
        ])
    
    # Test cases
    test_cases = [
        # (input_text, ocr_confidence, expected_contains)
        ("Mcnday", 70.0, "Monday"),           # Common OCR error: o->c
        ("Schedu1e", 75.0, "Schedule"),        # Common OCR error: l->1
        ("TUESDAY", 90.0, "TUESDAY"),          # Should preserve case
        ("M0nday", 65.0, "Monday"),            # Common OCR error: o->0
        ("12:30 PM", 95.0, "12:30 PM"),        # Time format - preserve
        ("Ernployee", 60.0, "Employee"),       # Common OCR error: m->rn
        ("Wednesdav", 70.0, "Wednesday"),      # Common OCR error: y->v
        ("01/15/2024", 90.0, "01/15/2024"),    # Date format - preserve
        ("Morninq", 75.0, "Morning"),          # Common OCR error: g->q
        ("8reak", 70.0, "Break"),              # Common OCR error: B->8
    ]
    
    print("\n--- Test Cases ---")
    passed = 0
    for text, confidence, expected in test_cases:
        result = validator.validate_and_correct(text, confidence)
        
        success = expected.lower() in result.corrected.lower()
        status = "✓" if success else "✗"
        
        if success:
            passed += 1
        
        print(f"{status} Input: '{text}' (conf={confidence})")
        print(f"   Output: '{result.corrected}'")
        print(f"   Expected contains: '{expected}'")
        if result.corrections_made:
            print(f"   Corrections: {result.corrections_made}")
        print()
    
    print(f"\n--- Results: {passed}/{len(test_cases)} passed ---")
    print("\n--- Statistics ---")
    print(validator.get_stats())


if __name__ == "__main__":
    test_validator()