"""
Dynamic Column Mapping Module

This module creates and manages dynamic column mappings for any CSV structure.
Replaces hardcoded column references with intelligent, adaptive mappings.

Functions:
- create_table_mapping(): Generate column mapping for a table
- get_mapped_column(): Get actual column name for semantic role
- build_dynamic_query(): Build queries using semantic roles
- update_mapping_confidence(): Learn from successful queries
"""

import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from column_intelligence import column_intelligence

logger = logging.getLogger(__name__)

class DynamicColumnMapper:
    """
    Manages dynamic column mappings across all tables.
    Provides semantic role to actual column name translation.
    """
    
    def __init__(self):
        # Cache of table analyses
        self.table_analyses: Dict[str, Dict[str, Any]] = {}
        
        # Global role mappings across all tables
        self.global_mappings: Dict[str, Dict[str, List[str]]] = {}
        
        # Success tracking for learning
        self.mapping_success_rates: Dict[str, Dict[str, float]] = {}
        
    def analyze_and_map_table(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """
        Analyze table structure and create dynamic mappings.
        
        Args:
            df: DataFrame to analyze
            table_name: Name of the table
            
        Returns:
            Analysis results with mappings
        """
        logger.info(f"Creating dynamic mappings for table: {table_name}")
        
        # Analyze table structure
        analysis = column_intelligence.analyze_table_structure(df, table_name)
        
        # Cache the analysis
        self.table_analyses[table_name] = analysis
        
        # Update global mappings
        self._update_global_mappings(table_name, analysis)
        
        # Create quick-access mapping
        role_mapping = self._create_role_mapping(analysis)
        analysis["quick_mapping"] = role_mapping
        
        logger.info(f"Mapped {len(role_mapping)} semantic roles for table {table_name}")
        return analysis
    
    def _update_global_mappings(self, table_name: str, analysis: Dict[str, Any]) -> None:
        """
        Update global role mappings with new table analysis.
        
        Args:
            table_name: Name of the table
            analysis: Analysis results
        """
        role_mappings = analysis.get("role_mappings", {})
        
        for role, columns in role_mappings.items():
            if role not in self.global_mappings:
                self.global_mappings[role] = {}
            
            if table_name not in self.global_mappings[role]:
                self.global_mappings[role][table_name] = []
            
            # Store column names with confidence scores
            for col_info in columns:
                self.global_mappings[role][table_name].append(col_info["column"])
    
    def _create_role_mapping(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Create simplified role -> column mapping for quick access.
        
        Args:
            analysis: Table analysis results
            
        Returns:
            Role to column mapping
        """
        role_mapping = {}
        role_mappings = analysis.get("role_mappings", {})
        
        for role, columns in role_mappings.items():
            if columns:
                # Use highest confidence column
                best_column = columns[0]["column"]
                role_mapping[role] = best_column
        
        return role_mapping
    
    def get_column_for_role(self, table_name: str, role: str, min_confidence: float = 0.5) -> Optional[str]:
        """
        Get the best column for a semantic role in a specific table.
        
        Args:
            table_name: Name of the table
            role: Semantic role to find
            min_confidence: Minimum confidence threshold
            
        Returns:
            Column name or None if no suitable column found
        """
        if table_name not in self.table_analyses:
            logger.warning(f"Table {table_name} not analyzed yet")
            return None
        
        analysis = self.table_analyses[table_name]
        return column_intelligence.get_column_for_role(analysis, role, min_confidence)
    
    def get_columns_for_role_any_table(self, role: str, min_confidence: float = 0.5) -> Dict[str, str]:
        """
        Get columns for a role across all analyzed tables.
        
        Args:
            role: Semantic role to find
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary mapping table_name -> column_name
        """
        results = {}
        
        for table_name in self.table_analyses:
            column = self.get_column_for_role(table_name, role, min_confidence)
            if column:
                results[table_name] = column
        
        return results
    
    def build_dynamic_where_conditions(self, table_name: str, semantic_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build WHERE conditions using semantic roles instead of hardcoded columns.
        
        Args:
            table_name: Name of the table
            semantic_conditions: Conditions using semantic roles
            
        Returns:
            Actual WHERE conditions with real column names
        """
        actual_conditions = {}
        
        for semantic_role, value in semantic_conditions.items():
            actual_column = self.get_column_for_role(table_name, semantic_role)
            if actual_column:
                actual_conditions[actual_column] = value
            else:
                logger.warning(f"No column found for role '{semantic_role}' in table '{table_name}'")
        
        return actual_conditions
    
    def suggest_tables_for_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Suggest tables that have columns matching a semantic role.
        
        Args:
            role: Semantic role to search for
            
        Returns:
            List of table suggestions with confidence scores
        """
        suggestions = []
        
        for table_name in self.table_analyses:
            column = self.get_column_for_role(table_name, role)
            if column:
                analysis = self.table_analyses[table_name]
                confidence = 0.0
                
                # Find confidence score
                role_mappings = analysis.get("role_mappings", {})
                if role in role_mappings:
                    for col_info in role_mappings[role]:
                        if col_info["column"] == column:
                            confidence = col_info["confidence"]
                            break
                
                suggestions.append({
                    "table_name": table_name,
                    "column_name": column,
                    "confidence": confidence,
                    "total_columns": analysis["total_columns"],
                    "total_rows": analysis["total_rows"]
                })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions
    
    def generate_semantic_query(self, semantic_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actual SQL query from semantic query description.
        
        Args:
            semantic_query: Query described in semantic terms
            
        Returns:
            Query components with actual column/table names
        """
        result = {
            "tables": [],
            "columns": {},
            "where_conditions": {},
            "errors": []
        }
        
        # Extract semantic roles from query
        required_roles = semantic_query.get("roles", [])
        conditions = semantic_query.get("conditions", {})
        
        # Find tables that have required roles
        table_scores = {}
        
        for role in required_roles:
            table_suggestions = self.suggest_tables_for_role(role)
            for suggestion in table_suggestions:
                table_name = suggestion["table_name"]
                confidence = suggestion["confidence"]
                
                if table_name not in table_scores:
                    table_scores[table_name] = {"score": 0.0, "roles_found": 0, "roles": {}}
                
                table_scores[table_name]["score"] += confidence
                table_scores[table_name]["roles_found"] += 1
                table_scores[table_name]["roles"][role] = {
                    "column": suggestion["column_name"],
                    "confidence": confidence
                }
        
        # Select best table(s)
        if table_scores:
            # Sort by total score and roles found
            sorted_tables = sorted(table_scores.items(), 
                                 key=lambda x: (x[1]["roles_found"], x[1]["score"]), 
                                 reverse=True)
            
            best_table, best_info = sorted_tables[0]
            result["tables"].append(best_table)
            result["columns"] = best_info["roles"]
            
            # Build WHERE conditions
            for role, value in conditions.items():
                if role in best_info["roles"]:
                    actual_column = best_info["roles"][role]["column"]
                    result["where_conditions"][actual_column] = value
                else:
                    result["errors"].append(f"Role '{role}' not found in selected table")
        else:
            result["errors"].append("No suitable tables found for required roles")
        
        return result
    
    def record_query_success(self, table_name: str, role: str, column: str, success: bool) -> None:
        """
        Record success/failure of role->column mapping for learning.
        
        Args:
            table_name: Name of the table
            role: Semantic role used
            column: Actual column used
            success: Whether the query was successful
        """
        if table_name not in self.mapping_success_rates:
            self.mapping_success_rates[table_name] = {}
        
        mapping_key = f"{role}->{column}"
        
        if mapping_key not in self.mapping_success_rates[table_name]:
            self.mapping_success_rates[table_name][mapping_key] = {"successes": 0, "attempts": 0}
        
        self.mapping_success_rates[table_name][mapping_key]["attempts"] += 1
        if success:
            self.mapping_success_rates[table_name][mapping_key]["successes"] += 1
        
        # Calculate success rate
        stats = self.mapping_success_rates[table_name][mapping_key]
        success_rate = stats["successes"] / stats["attempts"]
        
        logger.info(f"Mapping {mapping_key} in {table_name}: {success_rate:.2f} success rate ({stats['successes']}/{stats['attempts']})")
    
    def get_mapping_confidence(self, table_name: str, role: str, column: str) -> float:
        """
        Get historical confidence for a role->column mapping.
        
        Args:
            table_name: Name of the table
            role: Semantic role
            column: Column name
            
        Returns:
            Confidence score based on historical success
        """
        if table_name not in self.mapping_success_rates:
            return 0.5  # Default confidence
        
        mapping_key = f"{role}->{column}"
        
        if mapping_key not in self.mapping_success_rates[table_name]:
            return 0.5  # Default confidence
        
        stats = self.mapping_success_rates[table_name][mapping_key]
        return stats["successes"] / stats["attempts"] if stats["attempts"] > 0 else 0.5
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of all analyzed tables and their mappings.
        
        Returns:
            Summary of analysis results
        """
        summary = {
            "total_tables": len(self.table_analyses),
            "tables": {},
            "global_role_coverage": {},
            "mapping_statistics": {}
        }
        
        # Summarize each table
        for table_name, analysis in self.table_analyses.items():
            summary["tables"][table_name] = {
                "columns": analysis["total_columns"],
                "rows": analysis["total_rows"],
                "roles_detected": len(analysis.get("role_mappings", {})),
                "high_confidence_roles": len([r for r, cols in analysis.get("role_mappings", {}).items() if cols and cols[0]["confidence"] > 0.7])
            }
        
        # Global role coverage
        for role in self.global_mappings:
            tables_with_role = len(self.global_mappings[role])
            summary["global_role_coverage"][role] = tables_with_role
        
        return summary

# Global instance
dynamic_mapper = DynamicColumnMapper()