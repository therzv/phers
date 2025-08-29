"""
Data Sanitization Module

This module handles all data cleaning, validation, and security operations
to prevent SQL injection and ensure data quality.

Functions:
- clean_csv_data(): Clean uploaded CSV data
- sanitize_column_names(): Clean column names
- sanitize_data_values(): Clean data values
- validate_user_input(): Clean user search inputs
- prevent_sql_injection(): Secure SQL parameter handling
- validate_data_quality(): Check data quality issues
"""

import re
import pandas as pd
import unicodedata
from typing import Dict, List, Tuple, Any, Optional
import logging

# Configure logging for data sanitization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSanitizer:
    """
    Comprehensive data sanitization and security class.
    Handles all aspects of data cleaning and SQL injection prevention.
    """
    
    def __init__(self):
        # Define problematic characters that need cleaning
        self.problematic_chars = {
            '@': 'at',
            '#': 'num',
            '$': 'dollar', 
            '%': 'percent',
            '&': 'and',
            '*': 'star',
            '+': 'plus',
            '=': 'equals',
            '<': 'lt',
            '>': 'gt',
            '|': 'pipe',
            '\\': 'backslash',
            '/': 'slash',
            '?': 'question',
            '!': 'exclamation'
        }
        
        # SQL injection patterns to detect
        self.sql_injection_patterns = [
            r"('|(\\))",  # Single quotes and backslashes
            r"(;|\s*;\s*)",  # Semicolons
            r"(-{2}|/\*|\*/)",  # SQL comments
            r"\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b",  # SQL keywords
            r"\b(script|javascript|vbscript)\b",  # Script injection
        ]
        
    def clean_csv_data(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Main function to clean uploaded CSV data.
        
        Args:
            df: Raw DataFrame from uploaded CSV
            filename: Original filename for reporting
            
        Returns:
            Dict containing cleaned DataFrame and cleaning report
        """
        logger.info(f"Starting data cleaning for file: {filename}")
        
        cleaning_report = {
            "filename": filename,
            "original_shape": df.shape,
            "issues_found": [],
            "issues_fixed": [],
            "columns_cleaned": 0,
            "values_cleaned": 0,
            "needs_cleaning": False
        }
        
        try:
            # Step 1: Clean column names
            df_cleaned, column_report = self.sanitize_column_names(df)
            cleaning_report["columns_cleaned"] = column_report["columns_changed"]
            cleaning_report["column_mapping"] = column_report["mapping"]
            
            if column_report["columns_changed"] > 0:
                cleaning_report["needs_cleaning"] = True
                cleaning_report["issues_found"].extend(column_report["issues"])
                cleaning_report["issues_fixed"].append(f"{column_report['columns_changed']} column names sanitized")
            
            # Step 2: Clean data values
            df_cleaned, value_report = self.sanitize_data_values(df_cleaned)
            cleaning_report["values_cleaned"] = value_report["values_changed"]
            
            if value_report["values_changed"] > 0:
                cleaning_report["needs_cleaning"] = True
                cleaning_report["issues_found"].extend(value_report["issues"])
                cleaning_report["issues_fixed"].append(f"{value_report['values_changed']} data values cleaned")
                
            # Step 3: Validate data quality
            quality_report = self.validate_data_quality(df_cleaned)
            if quality_report["quality_issues"]:
                cleaning_report["needs_cleaning"] = True
                cleaning_report["issues_found"].extend(quality_report["quality_issues"])
                cleaning_report["quality_warnings"] = quality_report["warnings"]
            
            cleaning_report["final_shape"] = df_cleaned.shape
            logger.info(f"Data cleaning completed. Issues found: {len(cleaning_report['issues_found'])}")
            
            return {
                "cleaned_df": df_cleaned,
                "original_df": df,
                "report": cleaning_report,
                "needs_cleaning": cleaning_report["needs_cleaning"]
            }
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            cleaning_report["error"] = str(e)
            return {
                "cleaned_df": df,  # Return original if cleaning fails
                "original_df": df,
                "report": cleaning_report,
                "needs_cleaning": False
            }
    
    def sanitize_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean and sanitize column names for database safety.
        
        Args:
            df: DataFrame to clean column names
            
        Returns:
            Tuple of (cleaned_df, report_dict)
        """
        report = {
            "columns_changed": 0,
            "mapping": {},
            "issues": []
        }
        
        new_columns = []
        
        for original_col in df.columns:
            cleaned_col = self._clean_column_name(original_col)
            
            if cleaned_col != original_col:
                report["columns_changed"] += 1
                report["mapping"][original_col] = cleaned_col
                
                # Track specific issues found
                issues = []
                if ' ' in original_col:
                    issues.append("' ' → '_'")
                if any(char in original_col for char in self.problematic_chars.keys()):
                    issues.append("special characters removed")
                if original_col != original_col.strip():
                    issues.append("whitespace trimmed")
                    
                report["issues"].append({
                    "type": "column_name",
                    "original": original_col,
                    "cleaned": cleaned_col,
                    "issues": issues
                })
            
            new_columns.append(cleaned_col)
        
        # Create new DataFrame with cleaned column names
        df_cleaned = df.copy()
        df_cleaned.columns = new_columns
        
        return df_cleaned, report
    
    def _clean_column_name(self, column_name: str) -> str:
        """
        Clean individual column name.
        
        Args:
            column_name: Original column name
            
        Returns:
            Cleaned column name safe for database use
        """
        if not isinstance(column_name, str):
            column_name = str(column_name)
        
        # Step 1: Normalize unicode
        cleaned = unicodedata.normalize('NFKD', column_name)
        
        # Step 2: Remove/replace problematic characters
        for char, replacement in self.problematic_chars.items():
            cleaned = cleaned.replace(char, '_')
        
        # Step 3: Replace spaces with underscores
        cleaned = re.sub(r'\s+', '_', cleaned)
        
        # Step 4: Remove other special characters
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        
        # Step 5: Clean up multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Step 6: Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        # Step 7: Ensure it starts with letter or underscore
        if cleaned and cleaned[0].isdigit():
            cleaned = 'col_' + cleaned
        
        # Step 8: Fallback if empty
        if not cleaned:
            cleaned = 'unnamed_column'
            
        return cleaned
    
    def sanitize_data_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean and sanitize data values in DataFrame.
        
        Args:
            df: DataFrame to clean values
            
        Returns:
            Tuple of (cleaned_df, report_dict)
        """
        report = {
            "values_changed": 0,
            "issues": [],
            "columns_affected": []
        }
        
        df_cleaned = df.copy()
        
        for column in df_cleaned.columns:
            column_changes = 0
            
            # Only process string columns
            if df_cleaned[column].dtype == 'object':
                original_values = df_cleaned[column].astype(str)
                cleaned_values = original_values.apply(self._clean_data_value)
                
                # Count changes
                changes_mask = original_values != cleaned_values
                column_changes = changes_mask.sum()
                
                if column_changes > 0:
                    df_cleaned[column] = cleaned_values
                    report["values_changed"] += column_changes
                    report["columns_affected"].append(column)
                    
                    report["issues"].append({
                        "column": column,
                        "issues_fixed": column_changes,
                        "description": "Cleaned quotes, semicolons, and whitespace"
                    })
        
        return df_cleaned, report
    
    def _clean_data_value(self, value: Any) -> str:
        """
        Clean individual data value.
        
        Args:
            value: Original data value
            
        Returns:
            Cleaned data value
        """
        if pd.isna(value) or value is None:
            return ""
        
        # Convert to string
        cleaned = str(value)
        
        # Step 1: Normalize unicode
        cleaned = unicodedata.normalize('NFKD', cleaned)
        
        # Step 2: Trim whitespace
        cleaned = cleaned.strip()
        
        # Step 3: Normalize internal whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Step 4: Escape single quotes for SQL safety
        cleaned = cleaned.replace("'", "''")
        
        # Step 5: Remove or escape other problematic characters
        # Remove null bytes and control characters
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned)
        
        # Step 6: Limit length to prevent extremely long values
        if len(cleaned) > 1000:
            cleaned = cleaned[:997] + "..."
            
        return cleaned
    
    def sanitize_user_input(self, user_input: str, input_type: str = "general") -> str:
        """
        Sanitize user input for search queries.
        
        Args:
            user_input: Raw user input
            input_type: Type of input ("asset_tag", "username", "manufacturer", "general")
            
        Returns:
            Sanitized user input safe for SQL queries
        """
        if not user_input or not isinstance(user_input, str):
            return ""
        
        # Step 1: Basic cleaning
        cleaned = user_input.strip()
        
        # Step 2: Remove potential SQL injection patterns
        for pattern in self.sql_injection_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Step 3: Type-specific cleaning
        if input_type == "asset_tag":
            # Keep alphanumeric, hyphens, underscores
            cleaned = re.sub(r'[^A-Za-z0-9\-_]', '', cleaned)
        elif input_type == "username":
            # Keep letters, spaces, some punctuation
            cleaned = re.sub(r'[^A-Za-z0-9\s\.\-_]', '', cleaned)
        elif input_type == "manufacturer":
            # Keep letters, numbers, spaces, common punctuation
            cleaned = re.sub(r'[^A-Za-z0-9\s\.\-_&,]', '', cleaned)
        else:
            # General cleaning - remove most special characters
            cleaned = re.sub(r'[<>"|\\;]', '', cleaned)
        
        # Step 4: Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Step 5: Limit length
        if len(cleaned) > 200:
            cleaned = cleaned[:200]
        
        return cleaned
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and identify potential issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing quality assessment
        """
        quality_report = {
            "quality_issues": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check for empty DataFrame
        if df.empty:
            quality_report["quality_issues"].append("DataFrame is empty")
            return quality_report
        
        # Analyze each column
        for column in df.columns:
            col_stats = {
                "total_rows": len(df),
                "null_count": df[column].isnull().sum(),
                "unique_count": df[column].nunique(),
                "empty_strings": 0
            }
            
            # Check for empty strings in object columns
            if df[column].dtype == 'object':
                col_stats["empty_strings"] = (df[column].astype(str).str.strip() == "").sum()
            
            # Calculate null percentage
            null_percentage = (col_stats["null_count"] / col_stats["total_rows"]) * 100
            
            # Quality warnings
            if null_percentage > 50:
                quality_report["warnings"].append(f"Column '{column}' has {null_percentage:.1f}% null values")
            
            if col_stats["empty_strings"] > col_stats["total_rows"] * 0.3:
                quality_report["warnings"].append(f"Column '{column}' has many empty values")
            
            if col_stats["unique_count"] == 1:
                quality_report["warnings"].append(f"Column '{column}' has only one unique value")
            
            quality_report["stats"][column] = col_stats
        
        return quality_report
    
    def create_safe_sql_params(self, sql_template: str, params: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Create safe SQL with parameterized queries to prevent SQL injection.
        
        Args:
            sql_template: SQL template with named parameters
            params: Dictionary of parameter values
            
        Returns:
            Tuple of (safe_sql, parameter_list)
        """
        # Convert named parameters to positional parameters for safety
        safe_params = []
        safe_sql = sql_template
        
        for param_name, param_value in params.items():
            # Sanitize the parameter value
            if isinstance(param_value, str):
                safe_value = self.sanitize_user_input(param_value)
            else:
                safe_value = param_value
            
            safe_params.append(safe_value)
            # Replace named parameter with ? placeholder
            safe_sql = safe_sql.replace(f":{param_name}", "?")
        
        return safe_sql, safe_params

# Global instance for easy importing
data_sanitizer = DataSanitizer()