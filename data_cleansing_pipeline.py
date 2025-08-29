"""
Enterprise Data Cleansing Pipeline for PHERS
Natural Language Response Integration

This module provides Kaggle-level data cleaning and normalization capabilities
for any CSV/Excel dataset, enabling natural language queries on messy real-world data.

Features:
- Universal column name normalization (any format → standard snake_case)
- Intelligent data type detection and standardization  
- Value normalization (names, addresses, categories, dates, numbers)
- Fuzzy search preparation with phonetic indexing
- Missing data standardization
- Original data preservation with queryable clean versions
"""

import re
import pandas as pd
import numpy as np
import unicodedata
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import string
from collections import defaultdict, Counter
from pathlib import Path
import hashlib

# For fuzzy matching and phonetic algorithms
try:
    import jellyfish  # For Soundex, Metaphone, Levenshtein
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False

# For advanced text processing
try:
    import unidecode  # For unicode normalization
    UNICODE_PROCESSING_AVAILABLE = True
except ImportError:
    UNICODE_PROCESSING_AVAILABLE = False

@dataclass
class ColumnMapping:
    """Maps original column to cleaned version"""
    original_name: str
    cleaned_name: str
    data_type: str
    normalization_applied: List[str]
    sample_values: List[str]
    confidence_score: float

@dataclass
class DataCleaningReport:
    """Report of data cleaning operations"""
    original_shape: Tuple[int, int]
    cleaned_shape: Tuple[int, int]
    column_mappings: List[ColumnMapping]
    data_quality_issues: Dict[str, int]
    normalization_stats: Dict[str, Any]
    processing_time_seconds: float
    fuzzy_search_indexes: Dict[str, List[str]]

@dataclass
class FuzzySearchIndex:
    """Index for fuzzy matching and typo correction"""
    original_values: Set[str]
    normalized_values: Set[str]
    phonetic_codes: Dict[str, List[str]]  # Soundex/Metaphone mappings
    similarity_matrix: Dict[str, List[Tuple[str, float]]]  # Value → [(similar_value, score)]

class EnterpriseDataCleaner:
    """
    Enterprise-grade data cleaning pipeline that transforms any messy dataset
    into clean, queryable format while preserving original data integrity.
    """
    
    def __init__(self, activity_log_path: str = "data/activity.log"):
        self.activity_log_path = activity_log_path
        self.logger = logging.getLogger(__name__)
        
        # Column name normalization patterns
        self.column_patterns = {
            # Remove special characters but preserve meaning
            'special_chars': r'[^\w\s]',
            'multiple_spaces': r'\s+',
            'leading_trailing_spaces': r'^\s+|\s+$',
            # Common abbreviations to expand
            'abbreviations': {
                'emp': 'employee',
                'mgr': 'manager', 
                'dept': 'department',
                'addr': 'address',
                'tel': 'telephone',
                'dob': 'date_of_birth',
                'ssn': 'social_security_number'
            }
        }
        
        # Data type detection patterns
        self.data_type_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[\d\s\-\(\)\.]{10,}$',
            'date': [
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
                r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
            ],
            'currency': r'^[\$\€\£\¥]?[\d,]+\.?\d*$',
            'percentage': r'^\d+\.?\d*%$',
            'social_security': r'^\d{3}-\d{2}-\d{4}$',
            'zip_code': r'^\d{5}(-\d{4})?$',
            'url': r'^https?://[^\s]+$'
        }
        
        # Value normalization patterns
        self.value_normalizers = {
            'name_titles': {
                'mr.': 'Mr.',
                'mrs.': 'Mrs.',
                'ms.': 'Ms.',
                'dr.': 'Dr.',
                'prof.': 'Prof.'
            },
            'boolean_values': {
                'yes': True, 'y': True, '1': True, 'true': True, 'on': True,
                'no': False, 'n': False, '0': False, 'false': False, 'off': False
            },
            'null_indicators': [
                '', ' ', 'null', 'NULL', 'none', 'None', 'NONE', 
                'n/a', 'N/A', 'na', 'NA', 'nil', 'NIL',
                '-', '--', '—', 'tbd', 'TBD', 'unknown', 'Unknown'
            ],
            'country_codes': {
                'usa': 'United States', 'us': 'United States',
                'uk': 'United Kingdom', 'ca': 'Canada',
                'au': 'Australia', 'de': 'Germany', 'fr': 'France'
            }
        }
        
        self._log_activity("EnterpriseDataCleaner initialized", {
            "fuzzy_matching_available": FUZZY_MATCHING_AVAILABLE,
            "unicode_processing_available": UNICODE_PROCESSING_AVAILABLE
        })
    
    def _log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log activity to the activity log file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "EnterpriseDataCleaner",
                "activity": activity,
                "details": details or {}
            }
            
            Path(self.activity_log_path).parent.mkdir(exist_ok=True)
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log activity: {e}")
    
    def clean_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Tuple[pd.DataFrame, DataCleaningReport]:
        """
        Perform complete enterprise-level data cleaning on any dataset.
        
        Args:
            df: Input DataFrame with messy data
            dataset_name: Name for logging purposes
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        start_time = datetime.now()
        original_shape = df.shape
        
        self._log_activity(f"Starting enterprise data cleaning", {
            "dataset_name": dataset_name,
            "original_shape": original_shape,
            "original_columns": list(df.columns)
        })
        
        # Initialize tracking
        column_mappings = []
        data_quality_issues = defaultdict(int)
        fuzzy_search_indexes = {}
        
        # Step 1: Clean column names
        cleaned_df, col_mappings = self._normalize_column_names(df)
        column_mappings.extend(col_mappings)
        
        # Step 2: Detect and standardize data types
        cleaned_df = self._standardize_data_types(cleaned_df, data_quality_issues)
        
        # Step 3: Normalize values within columns
        cleaned_df = self._normalize_column_values(cleaned_df, data_quality_issues)
        
        # Step 4: Handle missing data consistently
        cleaned_df = self._standardize_missing_data(cleaned_df, data_quality_issues)
        
        # Step 5: Build fuzzy search indexes
        fuzzy_search_indexes = self._build_fuzzy_search_indexes(cleaned_df)
        
        # Step 6: Final validation and optimization
        cleaned_df = self._optimize_data_types(cleaned_df)
        
        # Generate cleaning report
        processing_time = (datetime.now() - start_time).total_seconds()
        cleaning_report = DataCleaningReport(
            original_shape=original_shape,
            cleaned_shape=cleaned_df.shape,
            column_mappings=column_mappings,
            data_quality_issues=dict(data_quality_issues),
            normalization_stats=self._calculate_normalization_stats(df, cleaned_df),
            processing_time_seconds=processing_time,
            fuzzy_search_indexes=fuzzy_search_indexes
        )
        
        self._log_activity("Enterprise data cleaning completed", {
            "dataset_name": dataset_name,
            "processing_time_seconds": processing_time,
            "quality_improvements": dict(data_quality_issues),
            "columns_normalized": len(column_mappings)
        })
        
        return cleaned_df, cleaning_report
    
    def _normalize_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[ColumnMapping]]:
        """Normalize column names to standard snake_case format"""
        column_mappings = []
        new_columns = {}
        
        for original_col in df.columns:
            # Step 1: Unicode normalization
            normalized = original_col
            if UNICODE_PROCESSING_AVAILABLE:
                normalized = unidecode.unidecode(str(normalized))
            else:
                # Fallback unicode handling
                normalized = unicodedata.normalize('NFKD', str(normalized))
                normalized = ''.join(c for c in normalized if not unicodedata.combining(c))
            
            # Step 2: Remove HTML entities and special formatting
            normalized = re.sub(r'&[a-zA-Z]+;', '', normalized)  # HTML entities
            normalized = re.sub(r'<[^>]+>', '', normalized)      # HTML tags
            
            # Step 3: Handle special characters and spaces
            normalized = re.sub(r'[^\w\s]', '_', normalized)     # Special chars → underscore
            normalized = re.sub(r'\s+', '_', normalized)         # Spaces → underscore
            normalized = re.sub(r'_+', '_', normalized)          # Multiple underscores → single
            normalized = normalized.strip('_')                   # Remove leading/trailing underscores
            
            # Step 4: Convert to lowercase
            normalized = normalized.lower()
            
            # Step 5: Expand common abbreviations
            for abbrev, expansion in self.column_patterns['abbreviations'].items():
                normalized = re.sub(rf'\b{abbrev}\b', expansion, normalized)
            
            # Step 6: Ensure valid Python identifier
            if not normalized or normalized[0].isdigit():
                normalized = f"col_{normalized}" if normalized else f"col_{len(new_columns)}"
            
            # Step 7: Handle duplicates
            original_normalized = normalized
            counter = 1
            while normalized in new_columns.values():
                normalized = f"{original_normalized}_{counter}"
                counter += 1
            
            new_columns[original_col] = normalized
            
            # Track normalization applied
            normalization_applied = []
            if original_col != normalized:
                normalization_applied.append("column_name_standardization")
            if re.search(r'[^\w]', original_col):
                normalization_applied.append("special_character_removal")
            if original_col != original_col.lower():
                normalization_applied.append("case_normalization")
            
            column_mappings.append(ColumnMapping(
                original_name=original_col,
                cleaned_name=normalized,
                data_type="string",  # Will be updated in data type detection
                normalization_applied=normalization_applied,
                sample_values=df[original_col].dropna().astype(str).head(3).tolist(),
                confidence_score=1.0 if original_col == normalized else 0.9
            ))
        
        # Create new DataFrame with cleaned column names
        cleaned_df = df.rename(columns=new_columns)
        
        return cleaned_df, column_mappings
    
    def _standardize_data_types(self, df: pd.DataFrame, issues: Dict[str, int]) -> pd.DataFrame:
        """Detect and standardize data types across columns"""
        cleaned_df = df.copy()
        
        for column in cleaned_df.columns:
            column_data = cleaned_df[column].dropna().astype(str)
            
            if len(column_data) == 0:
                continue
            
            # Detect dominant data type
            type_scores = defaultdict(int)
            
            for value in column_data.head(100):  # Sample first 100 values
                value_str = str(value).strip()
                
                # Email detection
                if re.match(self.data_type_patterns['email'], value_str):
                    type_scores['email'] += 1
                
                # Phone detection
                elif re.match(self.data_type_patterns['phone'], value_str):
                    type_scores['phone'] += 1
                
                # Date detection
                elif any(re.match(pattern, value_str) for pattern in self.data_type_patterns['date']):
                    type_scores['date'] += 1
                
                # Currency detection
                elif re.match(self.data_type_patterns['currency'], value_str):
                    type_scores['currency'] += 1
                
                # Numeric detection
                elif self._is_numeric(value_str):
                    type_scores['numeric'] += 1
                
                # Boolean detection
                elif value_str.lower() in self.value_normalizers['boolean_values']:
                    type_scores['boolean'] += 1
                
                else:
                    type_scores['text'] += 1
            
            # Determine dominant type
            if type_scores:
                dominant_type = max(type_scores.items(), key=lambda x: x[1])[0]
                
                # Apply type-specific normalization
                if dominant_type == 'date':
                    cleaned_df[column] = self._normalize_dates(cleaned_df[column])
                    issues['dates_normalized'] += 1
                    
                elif dominant_type == 'numeric':
                    cleaned_df[column] = self._normalize_numbers(cleaned_df[column])
                    issues['numbers_normalized'] += 1
                    
                elif dominant_type == 'currency':
                    cleaned_df[column] = self._normalize_currency(cleaned_df[column])
                    issues['currency_normalized'] += 1
                    
                elif dominant_type == 'boolean':
                    cleaned_df[column] = self._normalize_boolean(cleaned_df[column])
                    issues['booleans_normalized'] += 1
                    
                elif dominant_type == 'phone':
                    cleaned_df[column] = self._normalize_phone_numbers(cleaned_df[column])
                    issues['phones_normalized'] += 1
        
        return cleaned_df
    
    def _normalize_column_values(self, df: pd.DataFrame, issues: Dict[str, int]) -> pd.DataFrame:
        """Normalize values within columns for consistency"""
        cleaned_df = df.copy()
        
        for column in cleaned_df.columns:
            # Detect if column contains names (heuristic)
            if self._is_name_column(column, cleaned_df[column]):
                cleaned_df[column] = self._normalize_names(cleaned_df[column])
                issues['names_normalized'] += 1
            
            # Detect if column contains addresses
            elif self._is_address_column(column, cleaned_df[column]):
                cleaned_df[column] = self._normalize_addresses(cleaned_df[column])
                issues['addresses_normalized'] += 1
            
            # Detect if column contains categories
            elif self._is_category_column(cleaned_df[column]):
                cleaned_df[column] = self._normalize_categories(cleaned_df[column])
                issues['categories_normalized'] += 1
        
        return cleaned_df
    
    def _standardize_missing_data(self, df: pd.DataFrame, issues: Dict[str, int]) -> pd.DataFrame:
        """Standardize representation of missing/null data"""
        cleaned_df = df.copy()
        
        for column in cleaned_df.columns:
            # Convert various null representations to pandas NaN
            null_mask = cleaned_df[column].astype(str).str.strip().isin(self.value_normalizers['null_indicators'])
            null_count = null_mask.sum()
            
            if null_count > 0:
                cleaned_df.loc[null_mask, column] = np.nan
                issues['null_values_standardized'] += null_count
        
        return cleaned_df
    
    def _build_fuzzy_search_indexes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build indexes for fuzzy matching and typo correction"""
        if not FUZZY_MATCHING_AVAILABLE:
            return {}
        
        fuzzy_indexes = {}
        
        for column in df.columns:
            unique_values = df[column].dropna().astype(str).unique()
            
            if len(unique_values) < 2:
                continue
            
            # Build fuzzy search index for this column
            index = FuzzySearchIndex(
                original_values=set(unique_values),
                normalized_values=set(),
                phonetic_codes={},
                similarity_matrix={}
            )
            
            # Generate normalized versions for fuzzy matching
            for value in unique_values:
                normalized = self._normalize_for_fuzzy_search(value)
                index.normalized_values.add(normalized)
                
                # Generate phonetic codes
                if FUZZY_MATCHING_AVAILABLE:
                    soundex = jellyfish.soundex(value)
                    metaphone = jellyfish.metaphone(value)
                    
                    if soundex not in index.phonetic_codes:
                        index.phonetic_codes[soundex] = []
                    index.phonetic_codes[soundex].append(value)
                    
                    if metaphone and metaphone not in index.phonetic_codes:
                        index.phonetic_codes[metaphone] = []
                    index.phonetic_codes[metaphone].append(value)
            
            # Build similarity matrix for top values
            top_values = list(unique_values)[:50]  # Limit for performance
            for value in top_values:
                similarities = []
                for other_value in top_values:
                    if value != other_value:
                        if FUZZY_MATCHING_AVAILABLE:
                            distance = jellyfish.levenshtein_distance(value.lower(), other_value.lower())
                            max_len = max(len(value), len(other_value))
                            similarity = 1 - (distance / max_len) if max_len > 0 else 0
                            
                            if similarity > 0.6:  # Only store high-similarity pairs
                                similarities.append((other_value, similarity))
                
                if similarities:
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    index.similarity_matrix[value] = similarities[:5]  # Top 5 similar
            
            fuzzy_indexes[column] = asdict(index)
        
        return fuzzy_indexes
    
    def _is_numeric(self, value: str) -> bool:
        """Check if value represents a number"""
        try:
            float(value.replace(',', '').replace('$', '').replace('%', ''))
            return True
        except (ValueError, AttributeError):
            return False
    
    def _normalize_dates(self, series: pd.Series) -> pd.Series:
        """Normalize date formats to ISO standard"""
        return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
    
    def _normalize_numbers(self, series: pd.Series) -> pd.Series:
        """Normalize numeric values"""
        return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')
    
    def _normalize_currency(self, series: pd.Series) -> pd.Series:
        """Normalize currency values"""
        # Remove currency symbols and convert to float
        cleaned = series.astype(str).str.replace(r'[\$\€\£\¥,]', '', regex=True)
        return pd.to_numeric(cleaned, errors='coerce')
    
    def _normalize_boolean(self, series: pd.Series) -> pd.Series:
        """Normalize boolean values"""
        return series.astype(str).str.lower().map(self.value_normalizers['boolean_values'])
    
    def _normalize_phone_numbers(self, series: pd.Series) -> pd.Series:
        """Normalize phone number formats"""
        # Keep only digits and format consistently
        cleaned = series.astype(str).str.replace(r'[^\d]', '', regex=True)
        # Apply standard formatting for US numbers (customize as needed)
        return cleaned.apply(lambda x: f"({x[:3]}) {x[3:6]}-{x[6:]}" if len(x) == 10 else x)
    
    def _is_name_column(self, column_name: str, series: pd.Series) -> bool:
        """Detect if column contains person names"""
        name_indicators = ['name', 'first', 'last', 'full', 'employee', 'person', 'user']
        return any(indicator in column_name.lower() for indicator in name_indicators)
    
    def _is_address_column(self, column_name: str, series: pd.Series) -> bool:
        """Detect if column contains addresses"""
        address_indicators = ['address', 'addr', 'location', 'street', 'city', 'state', 'zip']
        return any(indicator in column_name.lower() for indicator in address_indicators)
    
    def _is_category_column(self, series: pd.Series) -> bool:
        """Detect if column contains categorical data"""
        unique_ratio = len(series.unique()) / len(series.dropna())
        return unique_ratio < 0.5  # Less than 50% unique values suggests categories
    
    def _normalize_names(self, series: pd.Series) -> pd.Series:
        """Normalize person names to standard format"""
        def normalize_name(name):
            if pd.isna(name):
                return name
            
            name = str(name).strip()
            
            # Handle titles
            for abbrev, full in self.value_normalizers['name_titles'].items():
                name = re.sub(rf'\b{re.escape(abbrev)}\b', full, name, flags=re.IGNORECASE)
            
            # Standardize casing: "JOHN SMITH" → "John Smith"
            name = ' '.join(word.capitalize() for word in name.split())
            
            # Clean up multiple spaces
            name = re.sub(r'\s+', ' ', name).strip()
            
            return name
        
        return series.apply(normalize_name)
    
    def _normalize_addresses(self, series: pd.Series) -> pd.Series:
        """Normalize address formats"""
        def normalize_address(addr):
            if pd.isna(addr):
                return addr
            
            addr = str(addr).strip()
            
            # Common address abbreviations
            abbreviations = {
                r'\bSt\.?\b': 'Street',
                r'\bAve\.?\b': 'Avenue', 
                r'\bBlvd\.?\b': 'Boulevard',
                r'\bRd\.?\b': 'Road',
                r'\bDr\.?\b': 'Drive',
                r'\bLn\.?\b': 'Lane'
            }
            
            for pattern, replacement in abbreviations.items():
                addr = re.sub(pattern, replacement, addr, flags=re.IGNORECASE)
            
            return addr
        
        return series.apply(normalize_address)
    
    def _normalize_categories(self, series: pd.Series) -> pd.Series:
        """Normalize categorical values"""
        def normalize_category(cat):
            if pd.isna(cat):
                return cat
            
            cat = str(cat).strip()
            
            # Standardize casing
            cat = cat.title()
            
            # Remove extra spaces
            cat = re.sub(r'\s+', ' ', cat)
            
            return cat
        
        return series.apply(normalize_category)
    
    def _normalize_for_fuzzy_search(self, value: str) -> str:
        """Normalize value for fuzzy matching"""
        if pd.isna(value):
            return ""
        
        normalized = str(value).lower().strip()
        
        # Remove punctuation
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        # Normalize spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        optimized_df = df.copy()
        
        for column in optimized_df.columns:
            # Optimize integer columns
            if optimized_df[column].dtype == 'float64':
                # Check if all values are integers
                non_null = optimized_df[column].dropna()
                if len(non_null) > 0 and (non_null % 1 == 0).all():
                    optimized_df[column] = optimized_df[column].astype('Int64')  # Nullable integer
            
            # Optimize string columns
            elif optimized_df[column].dtype == 'object':
                # Convert to category if low cardinality
                unique_ratio = len(optimized_df[column].unique()) / len(optimized_df[column].dropna())
                if unique_ratio < 0.5 and len(optimized_df[column].unique()) < 1000:
                    optimized_df[column] = optimized_df[column].astype('category')
        
        return optimized_df
    
    def _calculate_normalization_stats(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about the normalization process"""
        return {
            'columns_renamed': sum(1 for orig, clean in zip(original_df.columns, cleaned_df.columns) if orig != clean),
            'total_columns': len(original_df.columns),
            'memory_usage_reduction': original_df.memory_usage(deep=True).sum() - cleaned_df.memory_usage(deep=True).sum(),
            'null_values_standardized': sum(cleaned_df.isnull().sum()) - sum(original_df.isnull().sum()),
            'data_types_optimized': sum(1 for orig, clean in zip(original_df.dtypes, cleaned_df.dtypes) if orig != clean)
        }
    
    def find_similar_values(self, query: str, column_values: List[str], max_results: int = 5) -> List[Tuple[str, float]]:
        """Find similar values for fuzzy matching"""
        if not FUZZY_MATCHING_AVAILABLE:
            return []
        
        similarities = []
        query_normalized = self._normalize_for_fuzzy_search(query)
        
        for value in column_values:
            value_normalized = self._normalize_for_fuzzy_search(value)
            
            # Calculate similarity using Levenshtein distance
            distance = jellyfish.levenshtein_distance(query_normalized, value_normalized)
            max_len = max(len(query_normalized), len(value_normalized))
            similarity = 1 - (distance / max_len) if max_len > 0 else 0
            
            if similarity > 0.6:  # Threshold for considering similar
                similarities.append((value, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def export_cleaning_report(self, report: DataCleaningReport, filepath: str = None) -> Dict[str, Any]:
        """Export detailed cleaning report"""
        if not filepath:
            filepath = f"data/cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            Path(filepath).parent.mkdir(exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            self._log_activity("Data cleaning report exported", {"filepath": filepath})
            return {"success": True, "filepath": filepath, "report": report}
        
        except Exception as e:
            self.logger.error(f"Failed to export cleaning report: {e}")
            return {"success": False, "error": str(e), "report": report}

# Global instance for easy integration
enterprise_cleaner = EnterpriseDataCleaner()

def clean_dataset(df: pd.DataFrame, dataset_name: str = "dataset") -> Tuple[pd.DataFrame, DataCleaningReport]:
    """Convenience function for dataset cleaning"""
    return enterprise_cleaner.clean_dataset(df, dataset_name)

def find_similar_values(query: str, column_values: List[str], max_results: int = 5) -> List[Tuple[str, float]]:
    """Convenience function for fuzzy matching"""
    return enterprise_cleaner.find_similar_values(query, column_values, max_results)

if __name__ == "__main__":
    # Example usage and testing
    import pandas as pd
    
    # Create sample messy data
    messy_data = pd.DataFrame({
        'Employee Name@#$': ['JOHN SMITH', 'jane  doe', 'Bob Johnson Jr.', 'mary WILLIAMS'],
        '  Salary (USD)  ': ['$50,000', '$75000', '60000', 'N/A'],
        'Hire Date/Start': ['2020-01-15', '01/20/2021', '2022-03-10', 'unknown'],
        'Department!!!': ['sales', 'MARKETING', 'Sales', 'marketing'],
        'Phone #': ['(555) 123-4567', '555.987.6543', '5551234567', 'N/A']
    })
    
    print("Original Data:")
    print(messy_data)
    print("\nOriginal Columns:", list(messy_data.columns))
    
    # Clean the dataset
    cleaned_data, report = clean_dataset(messy_data, "employee_test_data")
    
    print("\nCleaned Data:")
    print(cleaned_data)
    print("\nCleaned Columns:", list(cleaned_data.columns))
    
    print(f"\nProcessing time: {report.processing_time_seconds:.3f} seconds")
    print(f"Quality issues fixed: {len(report.data_quality_issues)}")
    print(f"Columns normalized: {len(report.column_mappings)}")
    
    # Test fuzzy matching
    if FUZZY_MATCHING_AVAILABLE:
        similar = find_similar_values("jon smith", cleaned_data['employee_name'].tolist())
        print(f"\nFuzzy matching 'jon smith': {similar}")