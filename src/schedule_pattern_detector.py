"""
Schedule Pattern Detection Module for ML Timemaster.
Detects and classifies schedule-related patterns in table data.
Uses pattern-first approach with intelligent gap filling.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from enum import Enum


@dataclass
class CategoryMatch:
    """Represents a detected category within cell text."""
    category: str
    start: int
    end: int
    confidence: float = 1.0
    value: Optional[str] = None


@dataclass 
class CellInfo:
    """Information about a cell for pattern analysis."""
    row: int
    col: int
    text: str
    categories: List[CategoryMatch] = field(default_factory=list)
    course_type: Optional[str] = None
    is_header: bool = False
    is_group_header: bool = False
    expected_pattern: Optional[List[str]] = None  # Pattern this cell should follow


class SchedulePatternDetector:
    """
    Detects schedule-related patterns in table data.
    """
    
    # Base category order
    BASE_CATEGORY_ORDER = [
        'course_type',
        'course_name', 
        'professor',
        'group',
        'room',
        'course_code',
    ]
    
    # Categories that are often optional
    OPTIONAL_CATEGORIES = {'professor', 'group', 'course_code'}
    
    # Priority for overlap resolution
    CATEGORY_PRIORITY = {
        'day': 100,
        'time': 95,
        'course_type': 90,
        'professor': 80,
        'room': 70,
        'group': 65,
        'course_code': 60,
        'course_name': 50,
        'general': 10,
    }
    
    # Max rows/columns to check for headers
    MAX_HEADER_ROWS = 3
    MAX_HEADER_COLS = 3
    
    def __init__(self, verbose: bool = False):
        """Initialize the pattern detector."""
        self.logger = logging.getLogger(__name__ + ".PatternDetector")
        self.verbose = verbose
        
        # Pattern learning results
        self.detected_category_order: List[str] = []
        self.patterns_by_course_type: Dict[str, List[str]] = {}
        self.group_rows: Set[int] = set()
        self.group_columns: Set[int] = set()
        self.day_rows: Set[int] = set()
        self.day_columns: Set[int] = set()
        self.time_rows: Set[int] = set()
        self.time_columns: Set[int] = set()
        self.content_start_row: int = 0
        self.content_start_col: int = 0
        
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize all detection patterns."""
        
        # Day patterns (multiple languages)
        self.day_patterns = {
            # English full
            "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
            "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
            "sunday": "Sunday",
            # English short
            "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
            "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
            # Romanian full
            "luni": "Luni", "marti": "Marti", "marți": "Marți",
            "miercuri": "Miercuri", "joi": "Joi", "vineri": "Vineri",
            "sambata": "Sambata", "sâmbătă": "Sâmbătă", 
            "duminica": "Duminica", "duminică": "Duminică",
            # Romanian short
            "lu": "Luni", "ma": "Marti", "mi": "Miercuri",
            "jo": "Joi", "vi": "Vineri", "sa": "Sambata", "du": "Duminica",
        }
        
        # Time patterns
        self.time_patterns = [
            # HH:MM - HH:MM (range)
            re.compile(r'(\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2})'),
            # HH-HH (hour range)
            re.compile(r'(?<!\d)(\d{1,2}\s*[-–]\s*\d{1,2})(?!\d|:|\.)'),
            # Single time HH:MM
            re.compile(r'(?<!\d)(\d{1,2}:\d{2})(?!\s*[-–])'),
            # Hour with superscript: 8^00
            re.compile(r'(\d{1,2}\^\d{2})'),
        ]
        
        # Course type indicators
        self.course_type_patterns = [
            # Full words (higher priority)
            (re.compile(r'\b(Laborator)\b', re.IGNORECASE), 'lab'),
            (re.compile(r'\b(Laboratory)\b', re.IGNORECASE), 'lab'),
            (re.compile(r'\b(Seminar(?:y)?)\b', re.IGNORECASE), 'seminar'),
            (re.compile(r'\b(Lecture)\b', re.IGNORECASE), 'course'),
            (re.compile(r'\b(Course)\b', re.IGNORECASE), 'course'),
            (re.compile(r'\b(Curs)\b', re.IGNORECASE), 'course'),
            (re.compile(r'\b(Lab)\b', re.IGNORECASE), 'lab'),
            (re.compile(r'\b(Sem)\b', re.IGNORECASE), 'seminar'),
            # Single letters at start or after space
            (re.compile(r'^([CcKk])(?=\s|$)'), 'course'),
            (re.compile(r'(?<=\s)([CcKk])(?=\s|$)'), 'course'),
            (re.compile(r'^([Ll])(?=\s|$)'), 'lab'),
            (re.compile(r'(?<=\s)([Ll])(?=\s|$)'), 'lab'),
            (re.compile(r'^([Ss])(?=\s|$)'), 'seminar'),
            (re.compile(r'(?<=\s)([Ss])(?=\s|$)'), 'seminar'),
            (re.compile(r'^([Pp])(?=\s|$)'), 'practical'),
            (re.compile(r'(?<=\s)([Pp])(?=\s|$)'), 'practical'),
        ]
        
        # Professor patterns - academic titles followed by names
        self.professor_patterns = [
            # Romanian titles with various abbreviations
            re.compile(r'((?:Prof|Conf|Lect|Asist|Dr|Ing)\.?\s*(?:dr|ing|habil)?\.?\s+[A-ZĂÂÎȘȚ][a-zăâîșț]+(?:\s+[A-ZĂÂÎȘȚ][a-zăâîșț]+)*)', re.UNICODE),
            # English titles
            re.compile(r'((?:Prof|Dr|Mr|Mrs|Ms)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.UNICODE),
            # Abbreviated: I. Popescu, A.B. Smith
            re.compile(r'((?:[A-ZĂÂÎȘȚ]\.)+\s*[A-ZĂÂÎȘȚ][a-zăâîșț]+)', re.UNICODE),
            # Just two capitalized names (likely First Last or Last First)
            re.compile(r'((?<!\w)[A-ZĂÂÎȘȚ][a-zăâîșț]+\s+[A-ZĂÂÎȘȚ][a-zăâîșț]+(?!\w))', re.UNICODE),
        ]
        
        # Room patterns - flexible with/without dashes, dots, spaces
        self.room_patterns = [
            # Sala/Room prefix
            re.compile(r'((?:Sala|Room|Sală)\s*[A-Za-z0-9][-A-Za-z0-9\.\s]*)', re.IGNORECASE),
            # Amphitheater
            re.compile(r'((?:Amf|Amphitheater)\.?\s*[A-Za-z0-9]*)', re.IGNORECASE),
            # Building-Room formats: A-201, A201, A.201, A 201
            re.compile(r'(?:^|\s)([A-Z]\d?[-\.\s]?\d{2,4}[A-Za-z]?)(?:\s|$|,)'),
            # Just letter + numbers: A201, B105
            re.compile(r'(?:^|\s)([A-Z]\d{3,4})(?:\s|$|,)'),
            # Room numbers only (3-4 digits)
            re.compile(r'(?:^|\s)(\d{3,4})(?:\s|$)'),
            # Lab/room number: L1, L2, Lab1
            re.compile(r'(?:^|\s)(L\d{1,2})(?:\s|$)', re.IGNORECASE),
        ]
        
        # Group patterns for explicit group mentions
        self.explicit_group_patterns = [
            re.compile(r'((?:Gr|Group|Grupa)\.?\s*\d+[A-Za-z]?)', re.IGNORECASE),
            re.compile(r'((?:Ser|Seria|Series)\.?\s*[A-Z])', re.IGNORECASE),
            # Year-section format: 3A, 2B
            re.compile(r'(?:^|\s)(\d[A-Z])(?:\s|$)'),
        ]
        
        # Course code patterns
        self.course_code_patterns = [
            re.compile(r'\b([A-Z]{2,4}[-\s]?\d{2,4})\b'),
            re.compile(r'\(([A-Z]{2,4}\d{2,4})\)'),
        ]
        
        # Patterns for detecting group headers (incrementing values)
        self.incrementing_patterns = [
            # Numbers: 1, 2, 3 or 01, 02, 03
            re.compile(r'^(\d{1,2})$'),
            # Group format: G1, G2, Gr1, Gr.1
            re.compile(r'^(?:G|Gr)\.?\s*(\d+)$', re.IGNORECASE),
            # Year-section: 1A, 2B, 3C
            re.compile(r'^(\d)([A-Z])$'),
            # Roman numerals
            re.compile(r'^(I{1,3}|IV|VI{0,3}|IX|X{1,3})$', re.IGNORECASE),
            # Letters: A, B, C
            re.compile(r'^([A-Z])$'),
            # Combined: 3A1, 2B2
            re.compile(r'^(\d[A-Z]\d?)$'),
        ]
    
    def analyze_table(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Analyze table and add categories to grid cells.
        """
        if not table_data:
            return table_data
        
        data = table_data.get('data', table_data)
        table = data.get('table', {})
        grid = table.get('grid', [])
        
        if not grid:
            self.logger.warning("No grid found in table data")
            return table_data
        
        # Reset state
        self._reset_state()
        
        num_rows = len(grid)
        num_cols = max(len(row) for row in grid if isinstance(row, list)) if grid else 0
        
        # Phase 1: Build cell info matrix
        cell_matrix = self._build_cell_matrix(grid, num_rows, num_cols)
        
        # Phase 2: Detect structural patterns (headers in first few rows/cols)
        self._detect_structural_patterns(cell_matrix, num_rows, num_cols)
        
        # Phase 3: First pass - detect categories in content cells
        self._first_pass_detect_categories(cell_matrix)
        
        # Phase 4: Learn patterns from detected categories
        self._learn_patterns(cell_matrix)
        
        # Phase 5: Second pass - apply patterns and fill gaps
        self._second_pass_apply_patterns(cell_matrix)
        
        # Phase 6: Pattern-based correction - fix "general" tags that should match patterns
        self._apply_pattern_corrections(cell_matrix)
        
        # Phase 7: Write results back to grid
        self._write_results_to_grid(grid, cell_matrix)
        
        # Add schedule info
        schedule_info = self._build_schedule_info(num_rows, num_cols)
        table['schedule_info'] = schedule_info
        
        return table_data
    
    def _reset_state(self):
        """Reset all learned state for new table."""
        self.detected_category_order = []
        self.patterns_by_course_type = {}
        self.group_rows = set()
        self.group_columns = set()
        self.day_rows = set()
        self.day_columns = set()
        self.time_rows = set()
        self.time_columns = set()
        self.content_start_row = 0
        self.content_start_col = 0
    
    def _build_cell_matrix(self, grid: List[List[Dict]], num_rows: int, num_cols: int) -> List[List[Optional[CellInfo]]]:
        """Build a matrix of CellInfo objects from the grid."""
        matrix = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        
        for row_idx, row in enumerate(grid):
            if not isinstance(row, list):
                continue
            for col_idx, cell in enumerate(row):
                if not isinstance(cell, dict):
                    continue
                text = cell.get('text', '').strip()
                matrix[row_idx][col_idx] = CellInfo(
                    row=row_idx,
                    col=col_idx,
                    text=text
                )
        
        return matrix
    
    def _detect_structural_patterns(self, cell_matrix: List[List[Optional[CellInfo]]], 
                                     num_rows: int, num_cols: int):
        """
        Detect structural patterns focusing on first few rows/columns.
        """
        # Check first MAX_HEADER_ROWS rows for headers
        header_row_limit = min(self.MAX_HEADER_ROWS, num_rows)
        header_col_limit = min(self.MAX_HEADER_COLS, num_cols)
        
        # Detect day headers in first few rows
        for row_idx in range(header_row_limit):
            day_count = 0
            non_empty_count = 0
            for col_idx in range(num_cols):
                cell = cell_matrix[row_idx][col_idx]
                if cell and cell.text:
                    non_empty_count += 1
                    if self._is_day(cell.text):
                        day_count += 1
            # If most non-empty cells are days, it's a day header row
            if day_count >= 2 or (non_empty_count > 0 and day_count / non_empty_count > 0.5):
                self.day_rows.add(row_idx)
        
        # Detect time headers in first few columns
        for col_idx in range(header_col_limit):
            time_count = 0
            non_empty_count = 0
            for row_idx in range(num_rows):
                cell = cell_matrix[row_idx][col_idx]
                if cell and cell.text:
                    non_empty_count += 1
                    if self._is_time_only(cell.text):
                        time_count += 1
            # If most non-empty cells are times, it's a time column
            if time_count >= 2 or (non_empty_count > 0 and time_count / non_empty_count > 0.5):
                self.time_columns.add(col_idx)
        
        # Detect group headers (incrementing patterns)
        self._detect_group_headers(cell_matrix, num_rows, num_cols, header_row_limit, header_col_limit)
        
        # Determine where content starts
        self.content_start_row = 0
        self.content_start_col = 0
        
        # Find first row after headers
        header_rows = self.day_rows | self.group_rows
        if header_rows:
            self.content_start_row = max(header_rows) + 1
        
        # Find first column after headers  
        header_cols = self.time_columns | self.group_columns
        if header_cols:
            # Only consider columns that are actually in header area
            left_header_cols = {c for c in header_cols if c < header_col_limit}
            if left_header_cols:
                self.content_start_col = max(left_header_cols) + 1
        
        if self.verbose:
            self.logger.info(f"Structural patterns detected:")
            self.logger.info(f"  Content starts at: row {self.content_start_row}, col {self.content_start_col}")
            self.logger.info(f"  Day rows: {self.day_rows}")
            self.logger.info(f"  Time columns: {self.time_columns}")
            self.logger.info(f"  Group rows: {self.group_rows}")
            self.logger.info(f"  Group columns: {self.group_columns}")
    
    def _detect_group_headers(self, cell_matrix: List[List[Optional[CellInfo]]], 
                               num_rows: int, num_cols: int,
                               header_row_limit: int, header_col_limit: int):
        """
        Detect rows/columns that contain group headers in the header area.
        """
        # Check header rows for incrementing patterns
        for row_idx in range(header_row_limit):
            row_values = []
            for col_idx in range(num_cols):
                cell = cell_matrix[row_idx][col_idx]
                if cell and cell.text:
                    row_values.append((col_idx, cell.text.strip()))
            
            if len(row_values) >= 2 and self._is_incrementing_sequence(row_values):
                # Check if this row comes after day headers (typical pattern)
                if self.day_rows and row_idx > max(self.day_rows):
                    self.group_rows.add(row_idx)
                    for col_idx, _ in row_values:
                        if cell_matrix[row_idx][col_idx]:
                            cell_matrix[row_idx][col_idx].is_group_header = True
                # Or if no day headers but looks like groups
                elif not self.day_rows:
                    self.group_rows.add(row_idx)
                    for col_idx, _ in row_values:
                        if cell_matrix[row_idx][col_idx]:
                            cell_matrix[row_idx][col_idx].is_group_header = True
        
        # Check header columns for incrementing patterns
        for col_idx in range(header_col_limit):
            col_values = []
            for row_idx in range(num_rows):
                cell = cell_matrix[row_idx][col_idx]
                if cell and cell.text:
                    col_values.append((row_idx, cell.text.strip()))
            
            if len(col_values) >= 2 and self._is_incrementing_sequence(col_values):
                # Only mark as group column if it's after time columns
                if self.time_columns and col_idx > max(c for c in self.time_columns if c < header_col_limit):
                    self.group_columns.add(col_idx)
                    for row_idx, _ in col_values:
                        if cell_matrix[row_idx][col_idx]:
                            cell_matrix[row_idx][col_idx].is_group_header = True
    
    def _is_incrementing_sequence(self, values: List[Tuple[int, str]]) -> bool:
        """Check if values form an incrementing sequence."""
        if len(values) < 2:
            return False
        
        texts = [v[1] for v in values]
        
        # Try to extract numeric/sequential values
        extracted = []
        pattern_type = None
        
        for text in texts:
            text = text.strip()
            if not text:
                continue
            
            # Try each incrementing pattern
            for pattern in self.incrementing_patterns:
                match = pattern.match(text)
                if match:
                    if pattern.pattern == r'^(\d{1,2})$':
                        extracted.append(('num', int(match.group(1))))
                        pattern_type = 'num'
                    elif pattern.pattern == r'^(?:G|Gr)\.?\s*(\d+)$':
                        extracted.append(('group', int(match.group(1))))
                        pattern_type = 'group'
                    elif pattern.pattern == r'^(\d)([A-Z])$':
                        extracted.append(('year_section', (int(match.group(1)), match.group(2))))
                        pattern_type = 'year_section'
                    elif pattern.pattern == r'^(I{1,3}|IV|VI{0,3}|IX|X{1,3})$':
                        roman = match.group(1).upper()
                        roman_val = self._roman_to_int(roman)
                        extracted.append(('roman', roman_val))
                        pattern_type = 'roman'
                    elif pattern.pattern == r'^([A-Z])$':
                        extracted.append(('letter', ord(match.group(1))))
                        pattern_type = 'letter'
                    break
        
        if len(extracted) < 2:
            return False
        
        types = set(e[0] for e in extracted)
        if len(types) != 1:
            return False
        
        values_only = [e[1] for e in extracted]
        
        if pattern_type in ('num', 'group', 'roman', 'letter'):
            increasing = 0
            for i in range(1, len(values_only)):
                if values_only[i] > values_only[i-1]:
                    increasing += 1
            return increasing >= len(values_only) * 0.5
        
        return False
    
    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral to integer."""
        roman_values = {'I': 1, 'V': 5, 'X': 10}
        result = 0
        prev = 0
        for char in reversed(roman.upper()):
            val = roman_values.get(char, 0)
            if val < prev:
                result -= val
            else:
                result += val
            prev = val
        return result
    
    def _is_day(self, text: str) -> bool:
        """Check if text is a day of the week."""
        text_lower = text.lower().strip()
        return text_lower in self.day_patterns
    
    def _is_time_only(self, text: str) -> bool:
        """Check if text contains only a time pattern."""
        text = text.strip()
        for pattern in self.time_patterns:
            match = pattern.match(text)
            if match and match.group(0).strip() == text:
                return True
        return False
    
    def _first_pass_detect_categories(self, cell_matrix: List[List[Optional[CellInfo]]]):
        """First pass: Detect categories in content cells."""
        for row_idx, row in enumerate(cell_matrix):
            for col_idx, cell in enumerate(row):
                if cell is None or not cell.text:
                    continue
                
                # Mark header cells
                if row_idx < self.content_start_row or col_idx < self.content_start_col:
                    cell.is_header = True
                
                if cell.is_group_header:
                    continue
                
                if row_idx in self.day_rows or col_idx in self.day_columns:
                    cell.is_header = True
                if row_idx in self.time_rows or col_idx in self.time_columns:
                    cell.is_header = True
                
                # Detect categories
                categories = self._detect_categories(cell.text, cell)
                cell.categories = categories
                
                # Record course type
                for cat in categories:
                    if cat.category == 'course_type':
                        cell.course_type = cat.value
                        break
    
    def _detect_categories(self, text: str, cell: CellInfo) -> List[CategoryMatch]:
        """Detect categories in text."""
        matches = []
        used_ranges: List[Tuple[int, int]] = []
        
        # For header cells
        if cell.is_header or cell.is_group_header:
            if self._is_day(text):
                return [CategoryMatch(category='day', start=0, end=len(text), 
                                    value=self.day_patterns.get(text.lower().strip(), text))]
            
            time_match = self._find_time(text)
            if time_match:
                return [time_match]
            
            if cell.is_group_header:
                return [CategoryMatch(category='group', start=0, end=len(text), value=text)]
        
        # For content cells
        type_match = self._find_course_type(text)
        if type_match:
            matches.append(type_match)
            used_ranges.append((type_match.start, type_match.end))
        
        prof_match = self._find_professor(text, used_ranges)
        if prof_match:
            matches.append(prof_match)
            used_ranges.append((prof_match.start, prof_match.end))
        
        room_match = self._find_room(text, used_ranges)
        if room_match:
            matches.append(room_match)
            used_ranges.append((room_match.start, room_match.end))
        
        group_match = self._find_explicit_group(text, used_ranges)
        if group_match:
            matches.append(group_match)
            used_ranges.append((group_match.start, group_match.end))
        
        code_match = self._find_course_code(text, used_ranges)
        if code_match:
            matches.append(code_match)
            used_ranges.append((code_match.start, code_match.end))
        
        matches.sort(key=lambda m: m.start)
        return matches
    
    def _learn_patterns(self, cell_matrix: List[List[Optional[CellInfo]]]):
        """Learn patterns from detected categories."""
        # Group cells by course type
        cells_by_type: Dict[str, List[CellInfo]] = defaultdict(list)
        
        for row in cell_matrix:
            for cell in row:
                if cell and cell.course_type and not cell.is_header:
                    cells_by_type[cell.course_type].append(cell)
        
        # Learn pattern for each course type
        for course_type, cells in cells_by_type.items():
            sequences = []
            for cell in cells:
                if cell.categories:
                    seq = [cat.category for cat in sorted(cell.categories, key=lambda c: c.start)]
                    sequences.append(seq)
            
            if sequences:
                # Find most common pattern
                pattern_counter = Counter()
                for seq in sequences:
                    pattern_counter[tuple(seq)] += 1
                
                if pattern_counter:
                    most_common = pattern_counter.most_common(1)[0][0]
                    self.patterns_by_course_type[course_type] = list(most_common)
        
        # Determine global category order
        all_categories = set()
        position_sums = defaultdict(float)
        position_counts = defaultdict(int)
        
        for pattern in self.patterns_by_course_type.values():
            for pos, cat in enumerate(pattern):
                all_categories.add(cat)
                position_sums[cat] += pos
                position_counts[cat] += 1
        
        avg_positions = {}
        for cat in all_categories:
            if position_counts[cat] > 0:
                avg_positions[cat] = position_sums[cat] / position_counts[cat]
        
        self.detected_category_order = sorted(avg_positions.keys(), key=lambda c: avg_positions[c])
        
        # Ensure base categories are included
        for base_cat in self.BASE_CATEGORY_ORDER:
            if base_cat not in self.detected_category_order:
                self.detected_category_order.append(base_cat)
        
        if self.verbose:
            self.logger.info(f"Learned patterns by course type: {self.patterns_by_course_type}")
            self.logger.info(f"Global category order: {self.detected_category_order}")
    
    def _find_time(self, text: str) -> Optional[CategoryMatch]:
        """Find time pattern."""
        for pattern in self.time_patterns:
            match = pattern.search(text)
            if match:
                return CategoryMatch(
                    category='time',
                    start=match.start(),
                    end=match.end(),
                    value=match.group(1)
                )
        return None
    
    def _find_course_type(self, text: str) -> Optional[CategoryMatch]:
        """Find course type indicator."""
        for pattern, course_type in self.course_type_patterns:
            match = pattern.search(text)
            if match:
                start = match.start(1) if match.lastindex else match.start()
                end = match.end(1) if match.lastindex else match.end()
                return CategoryMatch(
                    category='course_type',
                    start=start,
                    end=end,
                    value=course_type
                )
        return None
    
    def _find_professor(self, text: str, used_ranges: List[Tuple[int, int]]) -> Optional[CategoryMatch]:
        """Find professor name."""
        for pattern in self.professor_patterns:
            for match in pattern.finditer(text):
                start = match.start(1) if match.lastindex else match.start()
                end = match.end(1) if match.lastindex else match.end()
                
                if not self._overlaps(start, end, used_ranges):
                    return CategoryMatch(
                        category='professor',
                        start=start,
                        end=end,
                        value=match.group(1) if match.lastindex else match.group(0)
                    )
        return None
    
    def _find_room(self, text: str, used_ranges: List[Tuple[int, int]]) -> Optional[CategoryMatch]:
        """Find room identifier."""
        for pattern in self.room_patterns:
            for match in pattern.finditer(text):
                start = match.start(1) if match.lastindex else match.start()
                end = match.end(1) if match.lastindex else match.end()
                
                if not self._overlaps(start, end, used_ranges):
                    return CategoryMatch(
                        category='room',
                        start=start,
                        end=end,
                        value=match.group(1).strip() if match.lastindex else match.group(0).strip()
                    )
        return None
    
    def _find_explicit_group(self, text: str, used_ranges: List[Tuple[int, int]]) -> Optional[CategoryMatch]:
        """Find explicit group mention."""
        for pattern in self.explicit_group_patterns:
            for match in pattern.finditer(text):
                start = match.start(1) if match.lastindex else match.start()
                end = match.end(1) if match.lastindex else match.end()
                
                if not self._overlaps(start, end, used_ranges):
                    return CategoryMatch(
                        category='group',
                        start=start,
                        end=end,
                        value=match.group(1) if match.lastindex else match.group(0)
                    )
        return None
    
    def _find_course_code(self, text: str, used_ranges: List[Tuple[int, int]]) -> Optional[CategoryMatch]:
        """Find course code."""
        for pattern in self.course_code_patterns:
            for match in pattern.finditer(text):
                start = match.start(1) if match.lastindex else match.start()
                end = match.end(1) if match.lastindex else match.end()
                
                if not self._overlaps(start, end, used_ranges):
                    return CategoryMatch(
                        category='course_code',
                        start=start,
                        end=end,
                        value=match.group(1) if match.lastindex else match.group(0)
                    )
        return None
    
    def _overlaps(self, start: int, end: int, used_ranges: List[Tuple[int, int]]) -> bool:
        """Check if range overlaps with used ranges."""
        for used_start, used_end in used_ranges:
            if not (end <= used_start or start >= used_end):
                return True
        return False
    
    def _second_pass_apply_patterns(self, cell_matrix: List[List[Optional[CellInfo]]]):
        """Second pass: Apply patterns and fill gaps."""
        for row in cell_matrix:
            for cell in row:
                if cell is None or not cell.text:
                    continue
                
                if cell.is_header or cell.is_group_header:
                    if cell.categories and len(cell.categories) == 1:
                        cat = cell.categories[0]
                        if cat.start != 0 or cat.end != len(cell.text):
                            cell.categories = [CategoryMatch(
                                category=cat.category,
                                start=0,
                                end=len(cell.text),
                                value=cat.value
                            )]
                    continue
                
                # Get expected pattern
                if cell.course_type and cell.course_type in self.patterns_by_course_type:
                    cell.expected_pattern = self.patterns_by_course_type[cell.course_type]
                
                cell.categories = self._fill_all_gaps(cell.text, cell.categories, cell.expected_pattern)
    
    def _fill_all_gaps(self, text: str, categories: List[CategoryMatch], 
                        expected_pattern: Optional[List[str]] = None) -> List[CategoryMatch]:
        """Fill all gaps in text."""
        if not text:
            return []
        
        text_len = len(text)
        categories = sorted(categories, key=lambda c: c.start)
        
        # Find gaps
        covered = [False] * text_len
        for cat in categories:
            for i in range(max(0, cat.start), min(text_len, cat.end)):
                covered[i] = True
        
        gaps = []
        gap_start = None
        for i, is_covered in enumerate(covered):
            if not is_covered:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gaps.append((gap_start, i))
                    gap_start = None
        if gap_start is not None:
            gaps.append((gap_start, text_len))
        
        found_categories = {cat.category for cat in categories}
        new_categories = list(categories)
        
        for gap_start, gap_end in gaps:
            # Skip pure whitespace
            if all(text[i].isspace() for i in range(gap_start, gap_end)):
                continue
            
            actual_start = gap_start
            while actual_start < gap_end and text[actual_start].isspace():
                actual_start += 1
            actual_end = gap_end
            while actual_end > actual_start and text[actual_end - 1].isspace():
                actual_end -= 1
            
            if actual_start >= actual_end:
                continue
            
            gap_text = text[actual_start:actual_end]
            
            # Skip if just punctuation
            if not any(c.isalnum() for c in gap_text):
                continue
            
            # Determine category
            category = self._determine_gap_category(
                gap_text, actual_start, text, found_categories, expected_pattern
            )
            
            new_categories.append(CategoryMatch(
                category=category,
                start=actual_start,
                end=actual_end,
                value=gap_text,  # Always include value, even for 'general'
                confidence=0.7 if category != 'general' else 0.3
            ))
            
            if category != 'general':
                found_categories.add(category)
        
        new_categories.sort(key=lambda c: c.start)
        return new_categories
    
    def _determine_gap_category(self, gap_text: str, gap_start: int, full_text: str,
                                 found_categories: Set[str], 
                                 expected_pattern: Optional[List[str]]) -> str:
        """Determine category for a gap."""
        # Skip very short text
        if len(gap_text.strip()) < 2:
            return 'general'
        
        # Check what's missing from expected pattern
        if expected_pattern:
            missing = [c for c in expected_pattern if c not in found_categories]
            
            # Try to match to missing categories in order
            for cat in missing:
                if cat == 'course_name' and 'course_name' not in found_categories:
                    # Course name usually comes after course type
                    if 'course_type' in found_categories:
                        # Check position is reasonable
                        if self._looks_like_course_name(gap_text):
                            return 'course_name'
                
                elif cat == 'professor' and self._could_be_professor(gap_text):
                    return 'professor'
                
                elif cat == 'room' and self._could_be_room(gap_text):
                    return 'room'
                
                elif cat == 'group' and self._could_be_group(gap_text):
                    return 'group'
        
        # If no course_name yet and text is substantial, likely course name
        if 'course_name' not in found_categories and self._looks_like_course_name(gap_text):
            return 'course_name'
        
        # Check heuristics
        if self._could_be_professor(gap_text):
            return 'professor'
        if self._could_be_room(gap_text):
            return 'room'
        if self._could_be_group(gap_text):
            return 'group'
        
        return 'general'
    
    def _looks_like_course_name(self, text: str) -> bool:
        """Check if text looks like course name."""
        text = text.strip()
        if len(text) < 3:
            return False
        
        words = text.split()
        
        # Multi-word or capitalized = likely course name
        if len(words) >= 2:
            return True
        
        if len(text) >= 4 and text[0].isupper():
            return True
        
        # Common course name patterns
        course_keywords = ['programming', 'mathematics', 'physics', 'chemistry', 
                          'algoritm', 'matematica', 'fizica', 'chimie', 'informatica']
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in course_keywords):
            return True
        
        return False
    
    def _could_be_professor(self, text: str) -> bool:
        """Check if text could be professor."""
        # Already checked by patterns?
        for pattern in self.professor_patterns:
            if pattern.search(text):
                return True
        
        # Two capitalized words
        words = text.split()
        if len(words) == 2:
            if all(w[0].isupper() for w in words if w):
                return True
        
        return False
    
    def _could_be_room(self, text: str) -> bool:
        """Check if text could be room."""
        text = text.strip()
        
        # Already checked by patterns?
        for pattern in self.room_patterns:
            if pattern.search(text):
                return True
        
        # Short alphanumeric
        if len(text) <= 6 and re.match(r'^[A-Z0-9][-\.\s]?[0-9]+$', text):
            return True
        
        return False
    
    def _could_be_group(self, text: str) -> bool:
        """Check if text could be group."""
        text = text.strip()
        
        for pattern in self.explicit_group_patterns:
            if pattern.search(text):
                return True
        
        # Very short and matches group-like pattern
        if len(text) <= 3 and re.match(r'^\d[A-Z]?$', text):
            return True
        
        return False
    
    def _apply_pattern_corrections(self, cell_matrix: List[List[Optional[CellInfo]]]):
        """
        Apply pattern-based corrections.
        If a cell has 'general' in a position where pattern expects specific category,
        and the general text could match that category, correct it.
        """
        for row in cell_matrix:
            for cell in row:
                if not cell or not cell.text or not cell.categories:
                    continue
                
                if cell.is_header or not cell.expected_pattern:
                    continue
                
                # Get current category sequence
                current_seq = [cat.category for cat in sorted(cell.categories, key=lambda c: c.start)]
                
                # Find 'general' categories
                general_indices = [i for i, cat in enumerate(cell.categories) if cat.category == 'general']
                
                if not general_indices:
                    continue
                
                # For each general category, check if it should be something else
                for idx in general_indices:
                    general_cat = cell.categories[idx]
                    
                    # Determine expected category at this position
                    # Remove 'general' from sequence to compare
                    non_general_seq = [cat for cat in current_seq if cat != 'general']
                    
                    # Find what category is missing from expected pattern
                    missing = []
                    for expected_cat in cell.expected_pattern:
                        if expected_cat not in non_general_seq:
                            missing.append(expected_cat)
                    
                    # Try to match general text to missing categories
                    if missing and general_cat.value:
                        for miss_cat in missing:
                            matched = False
                            
                            if miss_cat == 'course_type':
                                # Check if it's a single letter that could be course type
                                if len(general_cat.value.strip()) == 1:
                                    letter = general_cat.value.strip().upper()
                                    if letter in ['C', 'L', 'S', 'P']:
                                        cell.categories[idx] = CategoryMatch(
                                            category='course_type',
                                            start=general_cat.start,
                                            end=general_cat.end,
                                            value='course' if letter == 'C' else 
                                                  'lab' if letter == 'L' else 
                                                  'seminar' if letter == 'S' else 'practical',
                                            confidence=0.8
                                        )
                                        matched = True
                                        break
                            
                            elif miss_cat == 'course_name' and self._looks_like_course_name(general_cat.value):
                                cell.categories[idx] = CategoryMatch(
                                    category='course_name',
                                    start=general_cat.start,
                                    end=general_cat.end,
                                    value=general_cat.value,
                                    confidence=0.8
                                )
                                matched = True
                                break
                            
                            elif miss_cat == 'professor' and self._could_be_professor(general_cat.value):
                                cell.categories[idx] = CategoryMatch(
                                    category='professor',
                                    start=general_cat.start,
                                    end=general_cat.end,
                                    value=general_cat.value,
                                    confidence=0.8
                                )
                                matched = True
                                break
                            
                            elif miss_cat == 'room' and self._could_be_room(general_cat.value):
                                cell.categories[idx] = CategoryMatch(
                                    category='room',
                                    start=general_cat.start,
                                    end=general_cat.end,
                                    value=general_cat.value,
                                    confidence=0.8
                                )
                                matched = True
                                break
                            
                            elif miss_cat == 'group' and self._could_be_group(general_cat.value):
                                cell.categories[idx] = CategoryMatch(
                                    category='group',
                                    start=general_cat.start,
                                    end=general_cat.end,
                                    value=general_cat.value,
                                    confidence=0.8
                                )
                                matched = True
                                break
                            
                            if matched:
                                break
                
                # Re-sort after corrections
                cell.categories.sort(key=lambda c: c.start)
    
    def _write_results_to_grid(self, grid: List[List[Dict]], 
                                cell_matrix: List[List[Optional[CellInfo]]]):
        """Write category results back to grid."""
        for row_idx, row in enumerate(grid):
            if not isinstance(row, list):
                continue
            for col_idx, cell_dict in enumerate(row):
                if not isinstance(cell_dict, dict):
                    continue
                
                cell_info = cell_matrix[row_idx][col_idx] if (
                    row_idx < len(cell_matrix) and 
                    col_idx < len(cell_matrix[row_idx])
                ) else None
                
                if cell_info and cell_info.categories:
                    cell_dict['categories'] = [
                        {
                            'category': cat.category,
                            'start': cat.start,
                            'end': cat.end,
                            'value': cat.value  # Include value for all categories
                        }
                        for cat in cell_info.categories
                    ]
                else:
                    cell_dict['categories'] = []
    
    def _build_schedule_info(self, num_rows: int, num_cols: int) -> Dict[str, Any]:
        """Build schedule structure information."""
        info = {
            'dimensions': {'rows': num_rows, 'columns': num_cols},
            'content_area': {
                'start_row': self.content_start_row,
                'start_col': self.content_start_col
            },
            'detected_patterns': self.patterns_by_course_type,
            'detected_category_order': self.detected_category_order,
            'headers': {
                'group_rows': sorted(self.group_rows),
                'group_columns': sorted(self.group_columns),
                'day_rows': sorted(self.day_rows),
                'day_columns': sorted(self.day_columns),
                'time_rows': sorted(self.time_rows),
                'time_columns': sorted(self.time_columns),
            }
        }
        
        # Determine orientation
        if self.day_rows and min(self.day_rows) < self.MAX_HEADER_ROWS:
            info['orientation'] = 'days_as_columns'
        elif self.day_columns and min(self.day_columns) < self.MAX_HEADER_COLS:
            info['orientation'] = 'days_as_rows'
        elif self.time_columns and 0 in self.time_columns:
            info['orientation'] = 'times_as_rows'
        else:
            info['orientation'] = 'unknown'
        
        return info


def analyze_schedule_patterns(table_data: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Main entry point for pattern detection."""
    detector = SchedulePatternDetector(verbose=verbose)
    return detector.analyze_table(table_data)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    test_data = {
        "success": True,
        "data": {
            "metadata": {
                "file_name": "test.pdf",
                "processing_timestamp": "2026-01-19 17:20:32"
            },
            "table": {
                "bounds": {"x1": 112, "y1": 362, "x2": 5241, "y2": 3667},
                "dimensions": {"rows": 8, "columns": 7},
                "grid": [
                    # Row 0: Empty + Days
                    [
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "Monday", "rowspan": 1, "colspan": 1},
                        {"text": "Tuesday", "rowspan": 1, "colspan": 1},
                        {"text": "Wednesday", "rowspan": 1, "colspan": 1},
                        {"text": "Thursday", "rowspan": 1, "colspan": 1},
                        {"text": "Friday", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                    ],
                    # Row 1: Group headers
                    [
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "1", "rowspan": 1, "colspan": 1},
                        {"text": "2", "rowspan": 1, "colspan": 1},
                        {"text": "3", "rowspan": 1, "colspan": 1},
                        {"text": "4", "rowspan": 1, "colspan": 1},
                        {"text": "5", "rowspan": 1, "colspan": 1},
                        {"text": "6", "rowspan": 1, "colspan": 1},
                    ],
                    # Row 2: Time + Courses
                    [
                        {"text": "8:00-10:00", "rowspan": 1, "colspan": 1},
                        {"text": "C Advanced Programming Prof. Dr. Smith A-201", "rowspan": 1, "colspan": 1},
                        {"text": "L Physics Lab Prof. Brown B105", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "S Chemistry Seminar Dr. Jones C.302", "rowspan": 1, "colspan": 1},
                        {"text": "C Math", "rowspan": 1, "colspan": 1},  # Shortened name
                        {"text": "", "rowspan": 1, "colspan": 1},
                    ],
                    # Row 3: Pattern correction test
                    [
                        {"text": "10:00-12:00", "rowspan": 1, "colspan": 1},
                        {"text": "K Computer Science Conf. dr. Popescu C301", "rowspan": 1, "colspan": 1},  # K instead of C
                        {"text": "L Biology Lab B102", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "C English Literature Prof. Wilson Room 105", "rowspan": 1, "colspan": 1},
                        {"text": "L Network Lab D201", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                    ],
                    # Row 4: Break
                    [
                        {"text": "12:00-14:00", "rowspan": 1, "colspan": 1},
                        {"text": "Lunch Break", "rowspan": 1, "colspan": 6},
                    ],
                    # Row 5: More courses
                    [
                        {"text": "14:00-16:00", "rowspan": 1, "colspan": 1},
                        {"text": "C Algorithms Prof. Lee A101", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "S Database Systems Dr. Popa Sala A2", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "L OS Lab C401", "rowspan": 1, "colspan": 1},  # Shortened
                        {"text": "S SE Prof. Dan A.102", "rowspan": 1, "colspan": 1},  # Very shortened
                    ],
                    # Row 6: Column groups test
                    [
                        {"text": "16:00-18:00", "rowspan": 1, "colspan": 1},
                        {"text": "C Data Structures", "rowspan": 1, "colspan": 1},
                        {"text": "L ML Lab Prof. Kim E201", "rowspan": 1, "colspan": 1},  # Shortened
                        {"text": "S Statistics Dr. Miller E101", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "C AI", "rowspan": 1, "colspan": 1},  # Very short name
                        {"text": "", "rowspan": 1, "colspan": 1},
                    ],
                    # Row 7: Test spaces
                    [
                        {"text": " ", "rowspan": 1, "colspan": 1},  # Just space
                        {"text": "   ", "rowspan": 1, "colspan": 1},  # Multiple spaces
                        {"text": "C Networks   Dr. White   L3", "rowspan": 1, "colspan": 1},  # Extra spaces
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                        {"text": "", "rowspan": 1, "colspan": 1},
                    ],
                ]
            }
        }
    }
    
    print("Testing Schedule Pattern Detector with Improvements")
    print("=" * 80)
    
    result = analyze_schedule_patterns(test_data, verbose=True)
    
    print("\n📊 Schedule Structure:")
    schedule_info = result['data']['table'].get('schedule_info', {})
    print(f"   Content starts at: row {schedule_info['content_area']['start_row']}, "
          f"col {schedule_info['content_area']['start_col']}")
    print(f"   Orientation: {schedule_info.get('orientation')}")
    print(f"   Day rows: {schedule_info['headers']['day_rows']}")
    print(f"   Group rows: {schedule_info['headers']['group_rows']}")
    print(f"   Time columns: {schedule_info['headers']['time_columns']}")
    
    print(f"\n📋 Detected patterns by course type:")
    for ctype, pattern in schedule_info['detected_patterns'].items():
        print(f"   {ctype}: {' → '.join(pattern)}")
    
    # Print specific test cases
    grid = result['data']['table']['grid']
    
    print("\n🔍 Test Cases:")
    
    # Test 1: Shortened name
    print("\n1. Shortened course name (Row 2, Col 5 - 'C Math'):")
    cell = grid[2][5]
    print(f"   Text: '{cell['text']}'")
    for cat in cell['categories']:
        print(f"   - {cat['category']}: '{cat['value']}' [{cat['start']}:{cat['end']}]")
    
    # Test 2: Pattern correction (K -> course)
    print("\n2. Pattern correction (Row 3, Col 1 - 'K Computer Science...'):")
    cell = grid[3][1]
    print(f"   Text: '{cell['text']}'")
    for cat in cell['categories']:
        print(f"   - {cat['category']}: '{cat['value']}' [{cat['start']}:{cat['end']}]")
    
    # Test 3: Very short names
    print("\n3. Very short name (Row 6, Col 5 - 'C AI'):")
    cell = grid[6][5]
    print(f"   Text: '{cell['text']}'")
    for cat in cell['categories']:
        print(f"   - {cat['category']}: '{cat['value']}' [{cat['start']}:{cat['end']}]")
    
    # Test 4: Spaces handling
    print("\n4. Spaces handling (Row 7 cells):")
    for col in range(3):
        cell = grid[7][col]
        print(f"   Cell [{7},{col}]: '{cell['text']}' → categories: {len(cell['categories'])}")
        if cell['categories']:
            for cat in cell['categories']:
                print(f"     - {cat['category']}: '{cat['value']}' [{cat['start']}:{cat['end']}]")
    
    print("\n✅ Tests completed")