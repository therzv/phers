"""
Column Intelligence Module

This module automatically detects and maps column purposes from any CSV structure.
No hardcoded column names - works with any data schema.

Functions:
- detect_column_roles(): Analyze columns and detect their semantic roles
- create_column_mapping(): Create dynamic mapping for query generation
- analyze_data_patterns(): Understand data types and patterns
- suggest_column_matches(): Find best matching columns for queries
"""

import pandas as pd
import re
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import Counter
import difflib

logger = logging.getLogger(__name__)

class ColumnIntelligence:
    """
    Intelligent column detection and mapping system.
    Automatically understands CSV structure without hardcoded assumptions.
    """
    
    def __init__(self):
        # Semantic patterns for different column types
        self.role_patterns = {
            'identifier': {
                'name_patterns': [
                    r'id', r'identifier', r'tag', r'code', r'number', r'ref',
                    r'asset', r'serial', r'sku', r'barcode', r'uuid'
                ],
                'data_patterns': [
                    r'^[A-Z0-9\-_]+$',  # Uppercase codes
                    r'^\d+$',           # Pure numbers
                    r'^[A-Z]+-\d+',     # PREFIX-123 format
                    r'^\w{3,}-\w+'      # Multi-part identifiers
                ]
            },
            'person_name': {
                'name_patterns': [
                    r'name', r'user', r'person', r'employee', r'staff',
                    r'owner', r'contact', r'individual', r'member'
                ],
                'data_patterns': [
                    r'^[A-Z][a-z]+ [A-Z][a-z]+',  # FirstName LastName
                    r'^[A-Z]+$',                   # UPPERCASE names
                    r'^[A-Za-z\s\.]+$'             # Names with spaces/dots
                ]
            },
            'manufacturer': {
                'name_patterns': [
                    r'manufacturer', r'brand', r'make', r'company', r'vendor',
                    r'supplier', r'producer', r'maker', r'corp'
                ],
                'data_patterns': [
                    r'Inc\.?$', r'Corp\.?$', r'Ltd\.?$', r'Co\.?$',
                    r'^[A-Z][a-z]+ [A-Z]',  # Company Name Format
                    r'&', r'\bLLC\b', r'\bInc\b'
                ]
            },
            'product': {
                'name_patterns': [
                    r'item', r'product', r'model', r'type', r'category',
                    r'description', r'title', r'device', r'equipment'
                ],
                'data_patterns': [
                    r'[A-Z0-9]+ \d+',    # MODEL 123 format
                    r'\d+[A-Z]+',        # 24GB format
                    r'".*"',             # Quoted descriptions
                ]
            },
            'location': {
                'name_patterns': [
                    r'location', r'place', r'site', r'building', r'office',
                    r'floor', r'room', r'address', r'where'
                ],
                'data_patterns': [
                    r'Floor|floor|Lt\.|Lantai',
                    r'Building|building|Gedung',
                    r'Room|room|Ruang'
                ]
            },
            'money': {
                'name_patterns': [
                    r'price', r'cost', r'amount', r'value', r'dollar',
                    r'currency', r'money', r'budget', r'expense'
                ],
                'data_patterns': [
                    r'^\$\d+', r'^Rp\s*\d+', r'^\d+\.\d{2}$',
                    r'^\d{1,3}(,\d{3})*(\.\d{2})?$'
                ]
            },
            'date': {
                'name_patterns': [
                    r'date', r'time', r'when', r'created', r'modified',
                    r'purchased', r'updated', r'timestamp'
                ],
                'data_patterns': [
                    r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}',
                    r'\d{2}-\d{2}-\d{4}', r'\d{4}/\d{2}/\d{2}'
                ]
            },
            'status': {
                'name_patterns': [
                    r'status', r'state', r'condition', r'stage', r'phase'
                ],
                'data_patterns': [
                    r'^(Active|Inactive|New|Used|Broken|Repair)$',
                    r'^(Available|Unavailable|In Use|Retired)$'
                ]
            }
        }
        
    def analyze_table_structure(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """
        Analyze complete table structure and detect column roles.
        
        Args:
            df: DataFrame to analyze
            table_name: Name of the table
            
        Returns:
            Complete structure analysis with column mappings
        """
        logger.info(f"Analyzing table structure for: {table_name}")
        
        analysis = {
            "table_name": table_name,
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "column_roles": {},
            "role_mappings": {},
            "confidence_scores": {},
            "data_quality": {}
        }
        
        # Analyze each column
        for column in df.columns:
            column_analysis = self._analyze_single_column(df, column)
            analysis["column_roles"][column] = column_analysis
            
            # Build role mappings (role -> column_name)
            detected_role = column_analysis["detected_role"]
            confidence = column_analysis["confidence"]
            
            if detected_role != "unknown" and confidence > 0.5:
                if detected_role not in analysis["role_mappings"]:
                    analysis["role_mappings"][detected_role] = []
                
                analysis["role_mappings"][detected_role].append({
                    "column": column,
                    "confidence": confidence
                })
                
        # Sort role mappings by confidence
        for role in analysis["role_mappings"]:
            analysis["role_mappings"][role].sort(key=lambda x: x["confidence"], reverse=True)
            
        # Generate summary statistics
        analysis["detected_roles_summary"] = self._generate_role_summary(analysis["role_mappings"])
        
        logger.info(f"Analysis complete. Detected roles: {list(analysis['role_mappings'].keys())}")
        return analysis
    
    def _analyze_single_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Analyze a single column to detect its semantic role.
        
        Args:
            df: DataFrame containing the column
            column: Column name to analyze
            
        Returns:
            Column analysis results
        """
        column_data = df[column].dropna()
        if len(column_data) == 0:
            return {
                "detected_role": "unknown",
                "confidence": 0.0,
                "reasons": ["Column is entirely empty"],
                "data_type": "empty"
            }
        
        analysis = {
            "detected_role": "unknown",
            "confidence": 0.0,
            "reasons": [],
            "data_type": str(column_data.dtype),
            "sample_values": column_data.head(5).tolist(),
            "unique_count": column_data.nunique(),
            "null_percentage": (df[column].isna().sum() / len(df)) * 100
        }
        
        # Test each role pattern
        role_scores = {}
        
        for role_name, patterns in self.role_patterns.items():
            score = self._calculate_role_score(column, column_data, patterns)
            role_scores[role_name] = score
            
        # Find best matching role
        if role_scores:
            best_role = max(role_scores, key=role_scores.get)
            best_score = role_scores[best_role]
            
            if best_score > 0.3:  # Minimum confidence threshold
                analysis["detected_role"] = best_role
                analysis["confidence"] = best_score
                analysis["reasons"].append(f"Best match: {best_role} (score: {best_score:.2f})")
        
        # Additional analysis
        analysis["all_role_scores"] = role_scores
        analysis = self._add_data_insights(analysis, column_data)
        
        return analysis
    
    def _calculate_role_score(self, column_name: str, column_data: pd.Series, patterns: Dict[str, List[str]]) -> float:
        """
        Calculate how well a column matches a specific role.
        
        Args:
            column_name: Name of the column
            column_data: Column data
            patterns: Role patterns to match against
            
        Returns:
            Score between 0.0 and 1.0
        """
        name_score = 0.0
        data_score = 0.0
        
        # Test column name patterns
        column_lower = column_name.lower()
        name_patterns = patterns.get('name_patterns', [])
        
        for pattern in name_patterns:
            if re.search(pattern, column_lower):
                name_score = max(name_score, 0.8)  # High score for name match
                break
        else:
            # Fuzzy matching for partial matches
            for pattern in name_patterns:
                # Remove regex characters for fuzzy matching
                clean_pattern = re.sub(r'[^\w]', '', pattern)
                if clean_pattern in column_lower:
                    name_score = max(name_score, 0.4)  # Lower score for partial match
        
        # Test data patterns
        data_patterns = patterns.get('data_patterns', [])
        if data_patterns:
            sample_size = min(20, len(column_data))
            sample_data = column_data.head(sample_size).astype(str)
            
            matches = 0
            for value in sample_data:
                for pattern in data_patterns:
                    if re.search(pattern, value):
                        matches += 1
                        break
            
            data_score = matches / sample_size if sample_size > 0 else 0.0
        
        # Combine scores (name patterns are weighted higher)
        final_score = (name_score * 0.7) + (data_score * 0.3)
        return min(final_score, 1.0)
    
    def _add_data_insights(self, analysis: Dict[str, Any], column_data: pd.Series) -> Dict[str, Any]:
        """
        Add additional insights about the column data.
        
        Args:
            analysis: Current analysis dict
            column_data: Column data to analyze
            
        Returns:
            Enhanced analysis with additional insights
        """
        # Basic statistics
        analysis["statistics"] = {
            "total_values": len(column_data),
            "unique_values": column_data.nunique(),
            "most_common": column_data.value_counts().head(3).to_dict() if len(column_data) > 0 else {}
        }
        
        # Data type detection
        sample_values = column_data.astype(str).head(10)
        
        # Check if numeric
        try:
            pd.to_numeric(column_data.head(10))
            analysis["likely_numeric"] = True
        except (ValueError, TypeError):
            analysis["likely_numeric"] = False
        
        # Check if date
        date_like = 0
        for value in sample_values:
            if re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', str(value)):
                date_like += 1
        analysis["likely_date"] = date_like / len(sample_values) > 0.5 if len(sample_values) > 0 else False
        
        return analysis
    
    def _generate_role_summary(self, role_mappings: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate summary of detected roles.
        
        Args:
            role_mappings: Detected role mappings
            
        Returns:
            Role summary statistics
        """
        summary = {
            "total_roles_detected": len(role_mappings),
            "high_confidence_roles": {},
            "role_coverage": {}
        }
        
        for role, columns in role_mappings.items():
            high_conf_columns = [col for col in columns if col["confidence"] > 0.7]
            summary["high_confidence_roles"][role] = len(high_conf_columns)
            summary["role_coverage"][role] = {
                "total_columns": len(columns),
                "best_match": columns[0] if columns else None
            }
        
        return summary
    
    def get_column_for_role(self, analysis: Dict[str, Any], role: str, min_confidence: float = 0.5) -> Optional[str]:
        """
        Get the best column for a specific semantic role.
        
        Args:
            analysis: Table analysis results
            role: Semantic role to find
            min_confidence: Minimum confidence threshold
            
        Returns:
            Column name or None if no suitable column found
        """
        role_mappings = analysis.get("role_mappings", {})
        
        if role not in role_mappings:
            return None
        
        candidates = role_mappings[role]
        for candidate in candidates:
            if candidate["confidence"] >= min_confidence:
                return candidate["column"]
        
        return None
    
    def suggest_query_columns(self, analysis: Dict[str, Any], query_intent: str) -> List[Dict[str, Any]]:
        """
        Suggest relevant columns based on query intent.
        
        Args:
            analysis: Table analysis results
            query_intent: User's query intent
            
        Returns:
            List of suggested columns with relevance scores
        """
        suggestions = []
        query_lower = query_intent.lower()
        
        # Map query terms to semantic roles
        intent_role_mapping = {
            'asset': 'identifier',
            'tag': 'identifier', 
            'id': 'identifier',
            'user': 'person_name',
            'name': 'person_name',
            'employee': 'person_name',
            'manufacturer': 'manufacturer',
            'brand': 'manufacturer',
            'company': 'manufacturer',
            'product': 'product',
            'item': 'product',
            'model': 'product',
            'location': 'location',
            'price': 'money',
            'cost': 'money',
            'date': 'date',
            'status': 'status'
        }
        
        # Find relevant roles based on query
        relevant_roles = []
        for term, role in intent_role_mapping.items():
            if term in query_lower:
                relevant_roles.append(role)
        
        # Get columns for relevant roles
        for role in relevant_roles:
            column = self.get_column_for_role(analysis, role)
            if column:
                confidence = 0.0
                role_mappings = analysis.get("role_mappings", {})
                if role in role_mappings:
                    for candidate in role_mappings[role]:
                        if candidate["column"] == column:
                            confidence = candidate["confidence"]
                            break
                
                suggestions.append({
                    "column": column,
                    "role": role,
                    "relevance": confidence,
                    "reason": f"Matches query term for {role}"
                })
        
        # Sort by relevance
        suggestions.sort(key=lambda x: x["relevance"], reverse=True)
        return suggestions

# Global instance
column_intelligence = ColumnIntelligence()