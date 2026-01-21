"""
Word-based dictionary optimizer with proper phrase handling.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Shared default path
DEFAULT_DICTIONARY_PATH = "ocr_dictionary.json"

class DictionaryOptimizer:
    """
    Dictionary optimizer with word-based corrections and phrase support.
    """
    
    def __init__(self, dictionary_path: str = DEFAULT_DICTIONARY_PATH):
        """Initialize the optimizer with dictionary path."""
        self.dictionary_path = self._find_dictionary_path(dictionary_path)
        self.logger = logging.getLogger(__name__ + ".DictionaryOptimizer")
    
    def _find_dictionary_path(self, path: str) -> Path:
        """Find the dictionary file in various locations."""
        paths_to_check = [
            Path(path),
            Path("src") / path,
            Path("src") / "ocr_dictionary.json",
            Path("ocr_dictionary.json")
        ]
        
        for p in paths_to_check:
            if p.exists():
                return p
        
        return Path(paths_to_check[0])
    
    def load_dictionary(self) -> Dict:
        """Load the dictionary from file."""
        if not self.dictionary_path.exists():
            return self._create_empty_dictionary()
        
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'typed_corrections' not in data:
                data['typed_corrections'] = self._create_typed_corrections_structure()
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load dictionary: {e}")
            return self._create_empty_dictionary()
    
    def _create_empty_dictionary(self) -> Dict:
        """Create an empty dictionary structure."""
        return {
            "terms": {
                "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "subjects": [],
                "schedule_terms": ["Lecture", "Tutorial", "Lab", "Seminar", "Exam"],
                "custom": []
            },
            "word_corrections": {},
            "phrase_corrections": {},
            "typed_corrections": self._create_typed_corrections_structure()
        }
    
    def _create_typed_corrections_structure(self) -> Dict:
        """Create the typed corrections structure."""
        return {
            "subject": {},
            "class_type": {},
            "professor": {},
            "room": {},
            "day": {},
            "time": {},
            "course_code": {},
            "general": {}
        }
    
    def save_dictionary(self, data: Dict, backup: bool = True) -> None:
        """Save the dictionary to file with optional backup."""
        if backup and self.dictionary_path.exists():
            backup_path = self.dictionary_path.with_suffix('.backup.json')
            try:
                with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Created backup at {backup_path}")
            except Exception as e:
                self.logger.warning(f"Failed to create backup: {e}")
        
        with open(self.dictionary_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved dictionary to {self.dictionary_path}")
    
    def add_word_based_corrections(self, corrections: List[Dict[str, Any]]) -> Dict:
        """Add word-based corrections with proper phrase handling."""
        data = self.load_dictionary()
        
        if 'typed_corrections' not in data:
            data['typed_corrections'] = self._create_typed_corrections_structure()
        
        if 'word_corrections' not in data:
            data['word_corrections'] = {}
        
        if 'phrase_corrections' not in data:
            data['phrase_corrections'] = {}
        
        # Ensure all type categories exist
        for type_name in ['subject', 'class_type', 'professor', 'room', 'day', 'time', 'course_code', 'general']:
            if type_name not in data['typed_corrections']:
                data['typed_corrections'][type_name] = {}
        
        stats = {
            "total_processed": len(corrections),
            "corrections_added": 0,
            "word_corrections_added": 0,
            "phrase_corrections_added": 0,
            "by_type": defaultdict(int),
            "details": []
        }
        
        for correction in corrections:
            original = correction.get('original', '').strip()
            corrected = correction.get('corrected', '').strip()
            info_type = correction.get('type', 'general')
            is_phrase = correction.get('is_phrase', False)
            
            if not original or not corrected or original == corrected:
                continue
            
            self.logger.info(f"Processing: [{info_type}] '{original}' -> '{corrected}' (phrase={is_phrase})")
            
            # Add to typed_corrections
            if corrected not in data['typed_corrections'][info_type]:
                data['typed_corrections'][info_type][corrected] = []
            
            if original not in data['typed_corrections'][info_type][corrected]:
                data['typed_corrections'][info_type][corrected].append(original)
                stats['corrections_added'] += 1
                stats['by_type'][info_type] += 1
                
                stats['details'].append({
                    'original': original,
                    'corrected': corrected,
                    'type': info_type,
                    'is_phrase': is_phrase
                })
            
            # Also add to word_corrections or phrase_corrections for OCR lookup
            if is_phrase:
                # Add to phrase_corrections
                if corrected not in data['phrase_corrections']:
                    data['phrase_corrections'][corrected] = []
                if original not in data['phrase_corrections'][corrected]:
                    data['phrase_corrections'][corrected].append(original)
                    stats['phrase_corrections_added'] += 1
            else:
                # Add to word_corrections
                if corrected not in data['word_corrections']:
                    data['word_corrections'][corrected] = []
                if original not in data['word_corrections'][corrected]:
                    data['word_corrections'][corrected].append(original)
                    stats['word_corrections_added'] += 1
        
        self._sort_dictionary(data)
        self.save_dictionary(data, backup=False)
        
        stats['by_type'] = dict(stats['by_type'])
        
        return stats
    
    def add_corrections_with_tags(self, corrections: List[Dict[str, Any]]) -> Dict:
        """Wrapper for backwards compatibility."""
        return self.add_word_based_corrections(corrections)
    
    def add_corrections_and_optimize(self, corrections: List[Dict[str, str]]) -> Dict:
        """Wrapper for backwards compatibility."""
        # Convert to word-based format
        word_corrections = []
        for corr in corrections:
            word_corrections.append({
                "original": corr.get("original", ""),
                "corrected": corr.get("corrected", ""),
                "type": "general",
                "is_phrase": " " in corr.get("corrected", "")
            })
        return self.add_word_based_corrections(word_corrections)
    
    def _sort_dictionary(self, data: Dict) -> None:
        """Sort all lists in the dictionary for readability."""
        for type_name in data.get('typed_corrections', {}):
            for correct_val in data['typed_corrections'][type_name]:
                data['typed_corrections'][type_name][correct_val] = sorted(
                    list(set(data['typed_corrections'][type_name][correct_val]))
                )
        
        for correct_val in data.get('word_corrections', {}):
            data['word_corrections'][correct_val] = sorted(
                list(set(data['word_corrections'][correct_val]))
            )
        
        for correct_val in data.get('phrase_corrections', {}):
            data['phrase_corrections'][correct_val] = sorted(
                list(set(data['phrase_corrections'][correct_val]))
            )