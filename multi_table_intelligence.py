"""
Multi-Table Intelligence Module

Handles complex queries that span multiple tables and require intelligent joins.
Optimizes query performance and handles relationship detection between tables.

Features:
- Automatic join detection and optimization
- Cross-table relationship analysis
- Query performance optimization
- Data consistency validation
- Smart aggregation across tables
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)

class MultiTableIntelligence:
    """
    Intelligent system for handling multi-table queries and joins.
    Analyzes table relationships and optimizes cross-table operations.
    """
    
    def __init__(self):
        # Join strategy preferences (performance-ordered)
        self.join_strategies = [
            'inner_join',      # Fastest for exact matches
            'left_join',       # Good for optional relationships  
            'outer_join',      # Complete data, slower
            'cross_join'       # Last resort, very slow
        ]
        
        # Relationship strength indicators
        self.relationship_indicators = {
            'strong': {
                'same_column_name': 0.9,
                'similar_data_patterns': 0.8,
                'foreign_key_pattern': 0.85,
                'shared_unique_values': 0.9
            },
            'medium': {
                'similar_column_names': 0.6,
                'overlapping_values': 0.5,
                'semantic_role_match': 0.7
            },
            'weak': {
                'name_similarity': 0.3,
                'data_type_match': 0.2,
                'table_name_similarity': 0.1
            }
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'small_table': 1000,      # rows
            'medium_table': 10000,    # rows  
            'large_table': 100000,    # rows
            'max_join_tables': 5,     # maximum tables in one join
            'max_cross_product': 1000000  # maximum result size
        }
    
    def analyze_table_relationships(self, tables_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze relationships between all available tables.
        
        Args:
            tables_info: Dictionary of table information from dynamic mapper
            
        Returns:
            Relationship analysis results
        """
        logger.info(f"Analyzing relationships between {len(tables_info)} tables")
        
        relationships = {
            'direct_relationships': [],
            'potential_joins': [],
            'relationship_strength': {},
            'join_recommendations': {},
            'performance_warnings': []
        }
        
        table_names = list(tables_info.keys())
        
        # Analyze all table pairs
        for i, table1 in enumerate(table_names):
            for j, table2 in enumerate(table_names[i+1:], i+1):
                relationship = self._analyze_table_pair(
                    table1, tables_info[table1],
                    table2, tables_info[table2]
                )
                
                if relationship['strength'] > 0.3:  # Minimum threshold
                    relationships['direct_relationships'].append(relationship)
                    
                    # Generate join recommendation
                    join_rec = self._create_join_recommendation(relationship)
                    relationships['join_recommendations'][f"{table1}__{table2}"] = join_rec
        
        # Sort relationships by strength
        relationships['direct_relationships'].sort(
            key=lambda x: x['strength'], reverse=True
        )
        
        # Generate performance warnings
        relationships['performance_warnings'] = self._generate_performance_warnings(tables_info)
        
        logger.info(f"Found {len(relationships['direct_relationships'])} potential relationships")
        return relationships
    
    def _analyze_table_pair(self, table1_name: str, table1_info: Dict[str, Any],
                           table2_name: str, table2_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationship between two specific tables."""
        
        relationship = {
            'table1': table1_name,
            'table2': table2_name,
            'strength': 0.0,
            'join_columns': [],
            'relationship_type': 'unknown',
            'evidence': [],
            'confidence': 0.0
        }
        
        table1_roles = table1_info.get('role_mappings', {})
        table2_roles = table2_info.get('role_mappings', {})
        
        # Find matching semantic roles
        common_roles = set(table1_roles.keys()) & set(table2_roles.keys())
        
        if common_roles:
            strength_score = 0.0
            join_candidates = []
            
            for role in common_roles:
                # Get best column for this role in each table
                col1 = table1_roles[role][0]['column'] if table1_roles[role] else None
                col2 = table2_roles[role][0]['column'] if table2_roles[role] else None
                
                if col1 and col2:
                    # Calculate confidence for this join
                    conf1 = table1_roles[role][0]['confidence']
                    conf2 = table2_roles[role][0]['confidence']
                    join_confidence = (conf1 + conf2) / 2
                    
                    join_candidates.append({
                        'role': role,
                        'column1': col1,
                        'column2': col2,
                        'confidence': join_confidence
                    })
                    
                    # Add to strength based on role importance
                    role_weights = {
                        'identifier': 0.9,    # Primary keys - strongest
                        'person_name': 0.7,   # User relationships
                        'manufacturer': 0.6,  # Business relationships
                        'location': 0.5,      # Spatial relationships
                        'product': 0.4,       # Product categories
                        'date': 0.3,          # Temporal relationships
                        'money': 0.2,         # Value relationships
                        'status': 0.1         # State relationships
                    }
                    
                    role_weight = role_weights.get(role, 0.1)
                    strength_score += role_weight * join_confidence
            
            relationship['strength'] = min(strength_score, 1.0)
            relationship['join_columns'] = sorted(join_candidates, 
                                                key=lambda x: x['confidence'], reverse=True)
            
            # Determine relationship type
            if 'identifier' in common_roles:
                relationship['relationship_type'] = 'primary_key'
                relationship['evidence'].append('Shared identifier role (likely primary key relationship)')
            elif 'person_name' in common_roles:
                relationship['relationship_type'] = 'user_based'
                relationship['evidence'].append('Shared person name (user-based relationship)')
            elif len(common_roles) >= 3:
                relationship['relationship_type'] = 'multi_attribute'
                relationship['evidence'].append(f'Multiple shared roles: {list(common_roles)}')
            else:
                relationship['relationship_type'] = 'attribute_based'
                relationship['evidence'].append(f'Shared attribute role: {list(common_roles)[0]}')
        
        return relationship
    
    def _create_join_recommendation(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Create a specific join recommendation based on relationship analysis."""
        
        recommendation = {
            'join_type': 'INNER JOIN',  # default
            'primary_table': relationship['table1'],
            'secondary_table': relationship['table2'], 
            'join_condition': '',
            'expected_performance': 'medium',
            'risk_factors': [],
            'optimization_suggestions': []
        }
        
        if not relationship['join_columns']:
            return recommendation
        
        best_join = relationship['join_columns'][0]
        
        # Build join condition
        recommendation['join_condition'] = (
            f"{relationship['table1']}.{best_join['column1']} = "
            f"{relationship['table2']}.{best_join['column2']}"
        )
        
        # Determine optimal join type based on relationship strength and type
        if relationship['relationship_type'] == 'primary_key' and relationship['strength'] > 0.8:
            recommendation['join_type'] = 'INNER JOIN'
            recommendation['expected_performance'] = 'high'
        elif relationship['strength'] > 0.6:
            recommendation['join_type'] = 'LEFT JOIN'  # Safer for medium confidence
            recommendation['expected_performance'] = 'medium'
        else:
            recommendation['join_type'] = 'LEFT JOIN'
            recommendation['expected_performance'] = 'low'
            recommendation['risk_factors'].append('Low relationship confidence')
        
        # Add optimization suggestions
        if best_join['confidence'] < 0.7:
            recommendation['optimization_suggestions'].append(
                'Consider verifying join column data consistency before execution'
            )
        
        if relationship['relationship_type'] == 'multi_attribute':
            recommendation['optimization_suggestions'].append(
                'Multiple join conditions possible - consider performance testing'
            )
        
        return recommendation
    
    def _generate_performance_warnings(self, tables_info: Dict[str, Any]) -> List[str]:
        """Generate performance warnings for multi-table operations."""
        warnings = []
        
        large_tables = []
        for table_name, info in tables_info.items():
            row_count = info.get('total_rows', 0)
            if row_count > self.performance_thresholds['large_table']:
                large_tables.append((table_name, row_count))
        
        if len(large_tables) > 1:
            warnings.append(
                f"Multiple large tables detected: {[t[0] for t in large_tables]}. "
                f"Joins may be slow."
            )
        
        if len(tables_info) > self.performance_thresholds['max_join_tables']:
            warnings.append(
                f"Many tables available ({len(tables_info)}). "
                f"Limit joins to essential tables only."
            )
        
        return warnings
    
    def plan_multi_table_query(self, required_roles: List[str], tables_info: Dict[str, Any],
                              relationships: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan an optimal multi-table query execution strategy.
        
        Args:
            required_roles: Semantic roles needed for the query
            tables_info: Available table information
            relationships: Pre-analyzed table relationships
            
        Returns:
            Multi-table query execution plan
        """
        logger.info(f"Planning multi-table query for roles: {required_roles}")
        
        plan = {
            'strategy': 'multi_table',
            'primary_table': None,
            'join_sequence': [],
            'estimated_cost': 0,
            'risk_level': 'low',
            'alternative_strategies': [],
            'performance_optimizations': []
        }
        
        # Find tables that cover required roles
        table_role_coverage = {}
        for table_name, table_info in tables_info.items():
            role_mappings = table_info.get('role_mappings', {})
            coverage = []
            for role in required_roles:
                if role in role_mappings and role_mappings[role]:
                    confidence = role_mappings[role][0]['confidence']
                    coverage.append((role, confidence))
            table_role_coverage[table_name] = coverage
        
        # Score tables by role coverage
        table_scores = {}
        for table_name, coverage in table_role_coverage.items():
            if coverage:  # Only tables that cover at least one role
                score = sum(conf for _, conf in coverage) / len(required_roles)
                table_scores[table_name] = {
                    'score': score,
                    'roles_covered': len(coverage),
                    'coverage_details': coverage
                }
        
        if not table_scores:
            plan['strategy'] = 'no_suitable_tables'
            return plan
        
        # Select primary table (highest coverage)
        primary_table = max(table_scores, key=lambda t: table_scores[t]['score'])
        plan['primary_table'] = primary_table
        
        # Check if single table is sufficient
        primary_coverage = table_scores[primary_table]['roles_covered']
        if primary_coverage >= len(required_roles):
            plan['strategy'] = 'single_table_sufficient'
            return plan
        
        # Plan joins to cover missing roles
        covered_roles = set(role for role, _ in table_role_coverage[primary_table])
        missing_roles = set(required_roles) - covered_roles
        
        join_sequence = []
        current_tables = {primary_table}
        
        for role in missing_roles:
            # Find best table for this role that can join with current tables
            best_candidate = self._find_best_join_candidate(
                role, current_tables, table_role_coverage, relationships
            )
            
            if best_candidate:
                join_info = best_candidate['join_info']
                join_sequence.append({
                    'table': best_candidate['table'],
                    'role_needed': role,
                    'join_type': join_info['join_type'],
                    'join_condition': join_info['join_condition'],
                    'confidence': best_candidate['confidence']
                })
                current_tables.add(best_candidate['table'])
        
        plan['join_sequence'] = join_sequence
        plan['estimated_cost'] = self._estimate_query_cost(plan, tables_info)
        plan['risk_level'] = self._assess_risk_level(plan, tables_info)
        
        # Generate alternative strategies
        plan['alternative_strategies'] = self._generate_alternatives(
            required_roles, table_scores, relationships
        )
        
        # Add performance optimizations
        plan['performance_optimizations'] = self._suggest_optimizations(plan, tables_info)
        
        logger.info(f"Multi-table plan created with {len(join_sequence)} joins")
        return plan
    
    def _find_best_join_candidate(self, role: str, current_tables: Set[str],
                                table_role_coverage: Dict[str, List[Tuple[str, float]]],
                                relationships: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the best table to join for a specific role."""
        
        candidates = []
        
        # Find tables that have this role
        for table_name, coverage in table_role_coverage.items():
            if table_name in current_tables:
                continue  # Already included
            
            role_confidence = None
            for covered_role, confidence in coverage:
                if covered_role == role:
                    role_confidence = confidence
                    break
            
            if role_confidence is None:
                continue  # This table doesn't have the role
            
            # Check if this table can join with any current table
            best_join = None
            best_join_strength = 0.0
            
            for current_table in current_tables:
                join_key = f"{current_table}__{table_name}"
                alt_join_key = f"{table_name}__{current_table}"
                
                join_rec = relationships.get('join_recommendations', {}).get(
                    join_key, relationships.get('join_recommendations', {}).get(alt_join_key)
                )
                
                if join_rec:
                    # Find relationship strength
                    for rel in relationships.get('direct_relationships', []):
                        if ((rel['table1'] == current_table and rel['table2'] == table_name) or
                            (rel['table1'] == table_name and rel['table2'] == current_table)):
                            if rel['strength'] > best_join_strength:
                                best_join_strength = rel['strength']
                                best_join = join_rec
                                break
            
            if best_join and best_join_strength > 0.3:  # Minimum join strength
                candidates.append({
                    'table': table_name,
                    'role_confidence': role_confidence,
                    'join_strength': best_join_strength,
                    'join_info': best_join,
                    'confidence': (role_confidence + best_join_strength) / 2
                })
        
        # Return best candidate
        if candidates:
            return max(candidates, key=lambda c: c['confidence'])
        return None
    
    def _estimate_query_cost(self, plan: Dict[str, Any], tables_info: Dict[str, Any]) -> int:
        """Estimate the computational cost of executing the query plan."""
        
        if plan['strategy'] == 'single_table_sufficient':
            primary_table_rows = tables_info.get(plan['primary_table'], {}).get('total_rows', 0)
            return max(1, primary_table_rows // 1000)  # Base cost
        
        cost = 0
        
        # Base cost from primary table
        primary_rows = tables_info.get(plan['primary_table'], {}).get('total_rows', 0)
        cost += max(1, primary_rows // 1000)
        
        # Add join costs
        current_result_size = primary_rows
        for join in plan['join_sequence']:
            join_table_rows = tables_info.get(join['table'], {}).get('total_rows', 0)
            
            # Estimate join result size (simplified)
            if join['join_type'] == 'INNER JOIN':
                estimated_result = min(current_result_size, join_table_rows)
            else:  # LEFT JOIN, OUTER JOIN
                estimated_result = max(current_result_size, join_table_rows)
            
            # Join cost is proportional to the product of sizes
            join_cost = (current_result_size * join_table_rows) // 10000
            cost += max(1, join_cost)
            
            current_result_size = estimated_result
        
        return min(cost, 100)  # Cap at 100 for scale
    
    def _assess_risk_level(self, plan: Dict[str, Any], tables_info: Dict[str, Any]) -> str:
        """Assess the risk level of the query plan."""
        
        if plan['strategy'] == 'single_table_sufficient':
            return 'low'
        
        risk_factors = 0
        
        # Risk from number of joins
        join_count = len(plan['join_sequence'])
        if join_count > 3:
            risk_factors += 2
        elif join_count > 1:
            risk_factors += 1
        
        # Risk from join confidence
        for join in plan['join_sequence']:
            if join['confidence'] < 0.5:
                risk_factors += 2
            elif join['confidence'] < 0.7:
                risk_factors += 1
        
        # Risk from table sizes
        total_rows = sum(info.get('total_rows', 0) for info in tables_info.values())
        if total_rows > self.performance_thresholds['large_table'] * 2:
            risk_factors += 2
        elif total_rows > self.performance_thresholds['medium_table']:
            risk_factors += 1
        
        # Risk from estimated cost
        if plan['estimated_cost'] > 50:
            risk_factors += 2
        elif plan['estimated_cost'] > 20:
            risk_factors += 1
        
        if risk_factors >= 5:
            return 'high'
        elif risk_factors >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_alternatives(self, required_roles: List[str], table_scores: Dict[str, Any],
                             relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative query strategies."""
        alternatives = []
        
        # Alternative 1: Use multiple single-table queries instead of joins
        if len(table_scores) > 1:
            alternatives.append({
                'strategy': 'multiple_single_queries',
                'description': 'Execute separate queries on each table and combine results',
                'pros': ['Lower complexity', 'More predictable performance'],
                'cons': ['Requires application-level joining', 'May miss relationships'],
                'estimated_performance': 'medium'
            })
        
        # Alternative 2: Use the best single table only
        best_table = max(table_scores, key=lambda t: table_scores[t]['score'])
        if table_scores[best_table]['roles_covered'] >= len(required_roles) * 0.7:
            alternatives.append({
                'strategy': 'best_single_table',
                'table': best_table,
                'description': f'Use only {best_table} which covers most required roles',
                'pros': ['Fastest execution', 'No join complexity'],
                'cons': ['May miss some information', 'Incomplete results'],
                'estimated_performance': 'high'
            })
        
        return alternatives
    
    def _suggest_optimizations(self, plan: Dict[str, Any], tables_info: Dict[str, Any]) -> List[str]:
        """Suggest performance optimizations for the query plan."""
        optimizations = []
        
        if plan['estimated_cost'] > 20:
            optimizations.append("Consider adding LIMIT clause to reduce result set size")
        
        if plan['risk_level'] == 'high':
            optimizations.append("High-risk query - consider testing with smaller data subset first")
        
        if len(plan['join_sequence']) > 2:
            optimizations.append("Multiple joins detected - ensure join columns are indexed")
        
        # Check for large tables in joins
        for join in plan['join_sequence']:
            table_rows = tables_info.get(join['table'], {}).get('total_rows', 0)
            if table_rows > self.performance_thresholds['large_table']:
                optimizations.append(f"Large table {join['table']} in join - consider filtering first")
        
        if not optimizations:
            optimizations.append("Query plan looks optimal for current data size")
        
        return optimizations
    
    def validate_join_feasibility(self, plan: Dict[str, Any], sample_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate that planned joins will actually work with real data.
        
        Args:
            plan: Multi-table query plan
            sample_data: Sample DataFrames for validation
            
        Returns:
            Validation results with warnings and recommendations
        """
        logger.info("Validating join feasibility with sample data")
        
        validation = {
            'feasible': True,
            'warnings': [],
            'join_validations': [],
            'data_quality_issues': [],
            'recommended_actions': []
        }
        
        if plan['strategy'] == 'single_table_sufficient':
            validation['join_validations'].append({
                'status': 'not_applicable',
                'message': 'Single table query - no joins to validate'
            })
            return validation
        
        primary_table = plan['primary_table']
        if primary_table not in sample_data:
            validation['feasible'] = False
            validation['warnings'].append(f"Primary table {primary_table} not available for validation")
            return validation
        
        primary_df = sample_data[primary_table]
        
        # Validate each planned join
        for join in plan['join_sequence']:
            join_table = join['table']
            
            if join_table not in sample_data:
                validation['warnings'].append(f"Join table {join_table} not available for validation")
                continue
            
            join_df = sample_data[join_table]
            
            # Validate join condition
            join_validation = self._validate_single_join(
                primary_df, join_df, join, primary_table, join_table
            )
            
            validation['join_validations'].append(join_validation)
            
            if not join_validation['valid']:
                validation['feasible'] = False
        
        # Generate recommendations based on validation results
        if not validation['feasible']:
            validation['recommended_actions'].append("Review and fix join column issues before execution")
        
        failed_joins = sum(1 for jv in validation['join_validations'] if not jv.get('valid', True))
        if failed_joins > 0:
            validation['recommended_actions'].append(f"Fix {failed_joins} failed join validation(s)")
        
        logger.info(f"Join validation complete - feasible: {validation['feasible']}")
        return validation
    
    def _validate_single_join(self, primary_df: pd.DataFrame, join_df: pd.DataFrame,
                            join_info: Dict[str, Any], primary_table: str, join_table: str) -> Dict[str, Any]:
        """Validate a single join operation with sample data."""
        
        validation = {
            'valid': False,
            'primary_table': primary_table,
            'join_table': join_table,
            'join_type': join_info['join_type'],
            'issues': [],
            'statistics': {}
        }
        
        # Parse join condition to extract column names
        # Format: "table1.column1 = table2.column2"
        condition = join_info['join_condition']
        match = re.search(rf'{primary_table}\.(\w+)\s*=\s*{join_table}\.(\w+)', condition)
        
        if not match:
            validation['issues'].append(f"Cannot parse join condition: {condition}")
            return validation
        
        primary_col = match.group(1)
        join_col = match.group(2)
        
        # Check if columns exist
        if primary_col not in primary_df.columns:
            validation['issues'].append(f"Primary column {primary_col} not found in {primary_table}")
            return validation
        
        if join_col not in join_df.columns:
            validation['issues'].append(f"Join column {join_col} not found in {join_table}")
            return validation
        
        # Analyze join compatibility
        primary_values = set(primary_df[primary_col].dropna().astype(str))
        join_values = set(join_df[join_col].dropna().astype(str))
        
        overlap = primary_values & join_values
        overlap_ratio = len(overlap) / max(len(primary_values), 1)
        
        validation['statistics'] = {
            'primary_unique_values': len(primary_values),
            'join_unique_values': len(join_values),
            'overlapping_values': len(overlap),
            'overlap_ratio': overlap_ratio
        }
        
        # Validate based on join type and overlap
        if join_info['join_type'] == 'INNER JOIN':
            if overlap_ratio < 0.1:  # Less than 10% overlap
                validation['issues'].append(
                    f"Low overlap ({overlap_ratio:.1%}) for INNER JOIN - may return very few results"
                )
            elif overlap_ratio > 0.1:
                validation['valid'] = True
        else:  # LEFT JOIN, OUTER JOIN
            if overlap_ratio > 0:
                validation['valid'] = True
            else:
                validation['issues'].append("No overlapping values found - join will not match any records")
        
        # Additional data quality checks
        primary_nulls = primary_df[primary_col].isna().sum()
        join_nulls = join_df[join_col].isna().sum()
        
        if primary_nulls > len(primary_df) * 0.5:
            validation['issues'].append(f"Primary join column has {primary_nulls} null values ({primary_nulls/len(primary_df):.1%})")
        
        if join_nulls > len(join_df) * 0.5:
            validation['issues'].append(f"Join column has {join_nulls} null values ({join_nulls/len(join_df):.1%})")
        
        return validation

# Global instance
multi_table_intelligence = MultiTableIntelligence()