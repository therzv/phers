"""
Query Refinement and Optimization Module

Provides intelligent query refinement, error correction, and optimization.
Learns from failed queries and suggests improvements.

Features:
- Query error analysis and correction
- Performance optimization suggestions
- Query rewriting for better results
- Learning from query success/failure patterns
- Intelligent query alternative generation
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import re
import time
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class QueryRefinement:
    """
    Intelligent query refinement and optimization system.
    Analyzes query performance and suggests improvements.
    """
    
    def __init__(self):
        # Common query error patterns and their fixes
        self.error_patterns = {
            'column_not_found': {
                'pattern': r"no such column[:\s]+([`'\w\s]+)",
                'fix_strategy': 'suggest_column_alternatives',
                'confidence': 0.8
            },
            'table_not_found': {
                'pattern': r"no such table[:\s]+([`'\w\s]+)",
                'fix_strategy': 'suggest_table_alternatives', 
                'confidence': 0.9
            },
            'syntax_error': {
                'pattern': r"syntax error|near|unexpected",
                'fix_strategy': 'fix_syntax_issues',
                'confidence': 0.6
            },
            'ambiguous_column': {
                'pattern': r"ambiguous column name[:\s]+([`'\w\s]+)",
                'fix_strategy': 'add_table_qualifiers',
                'confidence': 0.8
            },
            'data_type_mismatch': {
                'pattern': r"type mismatch|cannot compare|invalid comparison",
                'fix_strategy': 'fix_data_types',
                'confidence': 0.7
            }
        }
        
        # Performance optimization rules
        self.optimization_rules = {
            'add_limit': {
                'condition': lambda sql: not re.search(r'\bLIMIT\b', sql, re.IGNORECASE),
                'suggestion': 'Add LIMIT clause to control result set size',
                'priority': 'high',
                'auto_fix': lambda sql: sql.rstrip(';') + ' LIMIT 100'
            },
            'avoid_select_star': {
                'condition': lambda sql: 'SELECT *' in sql.upper(),
                'suggestion': 'Replace SELECT * with specific column names for better performance',
                'priority': 'medium',
                'auto_fix': None  # Requires knowledge of needed columns
            },
            'add_where_clause': {
                'condition': lambda sql: not re.search(r'\bWHERE\b', sql, re.IGNORECASE),
                'suggestion': 'Consider adding WHERE clause to filter results',
                'priority': 'medium', 
                'auto_fix': None  # Requires domain knowledge
            },
            'optimize_joins': {
                'condition': lambda sql: sql.upper().count('JOIN') > 2,
                'suggestion': 'Complex joins detected - ensure proper indexing and consider query restructuring',
                'priority': 'high',
                'auto_fix': None
            },
            'use_exists_over_in': {
                'condition': lambda sql: re.search(r'\bIN\s*\(SELECT', sql, re.IGNORECASE),
                'suggestion': 'Consider using EXISTS instead of IN with subquery for better performance',
                'priority': 'medium',
                'auto_fix': None
            }
        }
        
        # Query success tracking
        self.query_history = []
        self.success_patterns = defaultdict(int)
        self.failure_patterns = defaultdict(int)
        
        # Alternative query generation templates
        self.alternative_templates = {
            'exact_to_fuzzy': {
                'description': 'Convert exact matches to fuzzy searches',
                'transform': lambda sql: re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql)
            },
            'add_case_insensitive': {
                'description': 'Make text comparisons case-insensitive', 
                'transform': lambda sql: re.sub(r"LIKE\s+'([^']+)'", r"LIKE UPPER('\1')", sql.replace('LIKE', 'UPPER(column) LIKE'))
            },
            'broaden_date_range': {
                'description': 'Broaden date range filters',
                'transform': lambda sql: re.sub(r"=\s*'(\d{4}-\d{2}-\d{2})'", r">= '\1' AND date < date('\1', '+1 day')", sql)
            },
            'simplify_joins': {
                'description': 'Simplify complex joins by using single table',
                'transform': self._simplify_to_single_table
            }
        }
    
    def analyze_query_failure(self, sql: str, error_message: str, 
                             execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a failed query and suggest fixes.
        
        Args:
            sql: The failed SQL query
            error_message: Database error message
            execution_context: Context from query execution
            
        Returns:
            Analysis results with suggested fixes
        """
        logger.info(f"Analyzing query failure: {error_message[:100]}...")
        
        analysis = {
            'error_type': 'unknown',
            'root_cause': 'Unable to determine',
            'suggested_fixes': [],
            'alternative_queries': [],
            'confidence': 0.0,
            'auto_fixable': False,
            'learning_insights': []
        }
        
        # Identify error pattern
        for error_type, pattern_info in self.error_patterns.items():
            if re.search(pattern_info['pattern'], error_message, re.IGNORECASE):
                analysis['error_type'] = error_type
                analysis['confidence'] = pattern_info['confidence']
                
                # Extract problematic element
                match = re.search(pattern_info['pattern'], error_message, re.IGNORECASE)
                problematic_element = match.group(1) if match and len(match.groups()) > 0 else None
                
                # Generate fixes based on strategy
                fixes = self._generate_fixes(
                    pattern_info['fix_strategy'], sql, problematic_element, execution_context
                )
                
                analysis['suggested_fixes'] = fixes
                analysis['auto_fixable'] = any(fix.get('auto_apply', False) for fix in fixes)
                break
        
        # Add general optimization suggestions
        optimization_suggestions = self._analyze_query_optimizations(sql)
        analysis['suggested_fixes'].extend(optimization_suggestions)
        
        # Generate alternative query approaches
        alternatives = self._generate_alternative_queries(sql, analysis['error_type'])
        analysis['alternative_queries'] = alternatives
        
        # Record failure pattern for learning
        self._record_failure_pattern(sql, error_message, analysis['error_type'])
        
        # Generate learning insights
        analysis['learning_insights'] = self._generate_learning_insights(sql, analysis['error_type'])
        
        logger.info(f"Analysis complete: {analysis['error_type']} with {len(analysis['suggested_fixes'])} fixes")
        return analysis
    
    def optimize_query(self, sql: str, performance_metrics: Optional[Dict[str, Any]] = None,
                      table_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze and optimize a working query for better performance.
        
        Args:
            sql: SQL query to optimize
            performance_metrics: Execution performance data
            table_info: Available table information
            
        Returns:
            Optimization analysis and suggestions
        """
        logger.info(f"Optimizing query: {sql[:50]}...")
        
        optimization = {
            'original_sql': sql,
            'optimized_sql': sql,
            'optimizations_applied': [],
            'suggestions': [],
            'performance_impact': 'unknown',
            'confidence': 0.8,
            'manual_review_needed': False
        }
        
        optimized_sql = sql
        
        # Apply automatic optimizations
        for rule_name, rule_info in self.optimization_rules.items():
            if rule_info['condition'](sql):
                suggestion = {
                    'rule': rule_name,
                    'description': rule_info['suggestion'],
                    'priority': rule_info['priority'],
                    'auto_applicable': rule_info['auto_fix'] is not None
                }
                
                if rule_info['auto_fix'] and rule_info['priority'] == 'high':
                    # Apply high-priority automatic fixes
                    try:
                        new_sql = rule_info['auto_fix'](optimized_sql)
                        if new_sql != optimized_sql:
                            optimized_sql = new_sql
                            optimization['optimizations_applied'].append(rule_name)
                            suggestion['applied'] = True
                    except Exception as e:
                        logger.warning(f"Failed to apply optimization {rule_name}: {e}")
                        suggestion['applied'] = False
                
                optimization['suggestions'].append(suggestion)
        
        # Analyze table-specific optimizations
        if table_info:
            table_optimizations = self._analyze_table_specific_optimizations(sql, table_info)
            optimization['suggestions'].extend(table_optimizations)
        
        # Analyze performance metrics if available
        if performance_metrics:
            performance_suggestions = self._analyze_performance_metrics(sql, performance_metrics)
            optimization['suggestions'].extend(performance_suggestions)
        
        # Determine if manual review is needed
        high_priority_suggestions = [s for s in optimization['suggestions'] if s.get('priority') == 'high']
        optimization['manual_review_needed'] = len(high_priority_suggestions) > 2
        
        optimization['optimized_sql'] = optimized_sql
        
        # Record successful optimization pattern
        if optimization['optimizations_applied']:
            self._record_success_pattern(sql, optimization['optimizations_applied'])
        
        logger.info(f"Optimization complete: {len(optimization['optimizations_applied'])} applied, {len(optimization['suggestions'])} suggestions")
        return optimization
    
    def suggest_query_improvements(self, sql: str, result_analysis: Dict[str, Any],
                                 user_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest improvements based on query results and user feedback.
        
        Args:
            sql: Original SQL query
            result_analysis: Analysis of query results  
            user_feedback: Optional user feedback about results
            
        Returns:
            Improvement suggestions
        """
        logger.info("Generating query improvement suggestions")
        
        improvements = {
            'result_quality_issues': [],
            'suggested_modifications': [],
            'alternative_approaches': [],
            'user_intent_clarifications': [],
            'confidence': 0.7
        }
        
        # Analyze result quality
        if result_analysis:
            row_count = result_analysis.get('row_count', 0)
            
            if row_count == 0:
                improvements['result_quality_issues'].append('No results returned')
                improvements['suggested_modifications'].extend([
                    {
                        'type': 'broaden_search',
                        'description': 'Try using LIKE instead of exact matches',
                        'example_sql': self.alternative_templates['exact_to_fuzzy']['transform'](sql)
                    },
                    {
                        'type': 'remove_filters', 
                        'description': 'Remove some WHERE conditions to get more results',
                        'example_sql': self._remove_restrictive_where_conditions(sql)
                    }
                ])
            
            elif row_count > 1000:
                improvements['result_quality_issues'].append('Very large result set')
                improvements['suggested_modifications'].append({
                    'type': 'add_filters',
                    'description': 'Add more specific WHERE conditions or LIMIT',
                    'example_sql': sql.rstrip(';') + ' LIMIT 100'
                })
        
        # Process user feedback
        if user_feedback:
            feedback_suggestions = self._analyze_user_feedback(sql, user_feedback)
            improvements['user_intent_clarifications'] = feedback_suggestions
        
        # Generate alternative approaches
        alternatives = []
        for alt_name, alt_info in self.alternative_templates.items():
            try:
                alt_sql = alt_info['transform'](sql)
                if alt_sql != sql:
                    alternatives.append({
                        'name': alt_name,
                        'description': alt_info['description'],
                        'sql': alt_sql
                    })
            except Exception as e:
                logger.warning(f"Failed to generate alternative {alt_name}: {e}")
        
        improvements['alternative_approaches'] = alternatives
        
        return improvements
    
    def _generate_fixes(self, strategy: str, sql: str, problematic_element: Optional[str],
                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific fixes based on error strategy."""
        
        fixes = []
        
        if strategy == 'suggest_column_alternatives':
            if problematic_element:
                # Try to find similar column names
                available_columns = context.get('available_columns', [])
                suggestions = self._find_similar_items(problematic_element, available_columns)
                
                for suggestion in suggestions[:3]:
                    fixes.append({
                        'description': f"Replace '{problematic_element}' with '{suggestion}'",
                        'sql': sql.replace(problematic_element, suggestion),
                        'auto_apply': False,
                        'confidence': 0.7
                    })
        
        elif strategy == 'suggest_table_alternatives':
            if problematic_element:
                available_tables = context.get('available_tables', [])
                suggestions = self._find_similar_items(problematic_element, available_tables)
                
                for suggestion in suggestions[:2]:
                    fixes.append({
                        'description': f"Replace table '{problematic_element}' with '{suggestion}'", 
                        'sql': re.sub(rf'\b{re.escape(problematic_element)}\b', suggestion, sql),
                        'auto_apply': True,
                        'confidence': 0.8
                    })
        
        elif strategy == 'fix_syntax_issues':
            # Common syntax fixes
            common_fixes = [
                {
                    'pattern': r"(\w+)\s*=\s*(\w+)",
                    'replacement': r"\1 = '\2'",
                    'description': "Add quotes around string values"
                },
                {
                    'pattern': r"WHERE\s+AND",
                    'replacement': r"WHERE",
                    'description': "Remove redundant AND after WHERE"
                },
                {
                    'pattern': r"ORDER BY\s*$",
                    'replacement': r"",
                    'description': "Remove incomplete ORDER BY"
                }
            ]
            
            for fix in common_fixes:
                if re.search(fix['pattern'], sql):
                    fixed_sql = re.sub(fix['pattern'], fix['replacement'], sql)
                    fixes.append({
                        'description': fix['description'],
                        'sql': fixed_sql,
                        'auto_apply': True,
                        'confidence': 0.6
                    })
        
        elif strategy == 'add_table_qualifiers':
            if problematic_element:
                # Add table qualifiers to ambiguous columns
                # This is a simplified approach - would need more context in practice
                fixes.append({
                    'description': f"Add table qualifier to column '{problematic_element}'",
                    'sql': sql,  # Would need actual table context to fix
                    'auto_apply': False,
                    'confidence': 0.5
                })
        
        return fixes
    
    def _analyze_query_optimizations(self, sql: str) -> List[Dict[str, Any]]:
        """Analyze query for general optimization opportunities."""
        
        optimizations = []
        
        # Check for common optimization opportunities
        if not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
            optimizations.append({
                'rule': 'add_limit_general',
                'description': 'Consider adding LIMIT to control result size',
                'priority': 'medium',
                'auto_applicable': True
            })
        
        if re.search(r'SELECT\s+\*', sql, re.IGNORECASE):
            optimizations.append({
                'rule': 'specific_columns',
                'description': 'Select specific columns instead of * for better performance',
                'priority': 'medium',
                'auto_applicable': False
            })
        
        # Check for potentially expensive operations
        if re.search(r'\bLIKE\s+[\'"]%.*%[\'"]', sql, re.IGNORECASE):
            optimizations.append({
                'rule': 'wildcard_optimization',
                'description': 'Wildcard at start of LIKE pattern prevents index usage',
                'priority': 'high',
                'auto_applicable': False
            })
        
        return optimizations
    
    def _generate_alternative_queries(self, sql: str, error_type: str) -> List[Dict[str, Any]]:
        """Generate alternative query approaches."""
        
        alternatives = []
        
        # Generate alternatives based on error type
        if error_type == 'column_not_found':
            # Try selecting all columns to see what's available
            table_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1)
                alternatives.append({
                    'approach': 'explore_columns',
                    'description': f'Explore available columns in {table_name}',
                    'sql': f'SELECT * FROM {table_name} LIMIT 5'
                })
        
        elif error_type == 'syntax_error':
            # Simplify the query
            simplified = self._simplify_query(sql)
            if simplified != sql:
                alternatives.append({
                    'approach': 'simplify',
                    'description': 'Simplified version of the query',
                    'sql': simplified
                })
        
        # Apply general alternative templates
        for template_name, template_info in self.alternative_templates.items():
            try:
                alt_sql = template_info['transform'](sql)
                if alt_sql != sql:
                    alternatives.append({
                        'approach': template_name,
                        'description': template_info['description'],
                        'sql': alt_sql
                    })
            except Exception as e:
                logger.warning(f"Failed to generate {template_name} alternative: {e}")
        
        return alternatives[:5]  # Limit to top 5 alternatives
    
    def _find_similar_items(self, target: str, candidates: List[str]) -> List[str]:
        """Find items similar to target using fuzzy matching."""
        import difflib
        
        # Clean target
        target_clean = target.strip('`\'"')
        
        # Find close matches
        matches = difflib.get_close_matches(
            target_clean.lower(), 
            [c.lower() for c in candidates],
            n=3, 
            cutoff=0.6
        )
        
        # Return original case versions
        results = []
        for match in matches:
            for candidate in candidates:
                if candidate.lower() == match:
                    results.append(candidate)
                    break
        
        return results
    
    def _simplify_to_single_table(self, sql: str) -> str:
        """Simplify multi-table query to single table."""
        # Extract first table mentioned
        table_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if table_match:
            table_name = table_match.group(1)
            # Create simple SELECT from first table
            return f"SELECT * FROM {table_name} LIMIT 10"
        return sql
    
    def _simplify_query(self, sql: str) -> str:
        """Create a simplified version of a complex query."""
        
        # Remove complex WHERE conditions
        simplified = re.sub(r'WHERE\s+.+?(?=ORDER|GROUP|LIMIT|$)', 'WHERE 1=1 ', sql, flags=re.IGNORECASE)
        
        # Remove ORDER BY, GROUP BY
        simplified = re.sub(r'\s+(ORDER|GROUP)\s+BY\s+[^;]+', '', simplified, flags=re.IGNORECASE)
        
        # Add LIMIT if not present
        if not re.search(r'\bLIMIT\b', simplified, re.IGNORECASE):
            simplified = simplified.rstrip(';') + ' LIMIT 5'
        
        return simplified
    
    def _remove_restrictive_where_conditions(self, sql: str) -> str:
        """Remove overly restrictive WHERE conditions."""
        
        # Convert exact matches to LIKE patterns
        modified = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql)
        
        # Remove some AND conditions (keep first one)
        parts = modified.split(' AND ')
        if len(parts) > 2:
            modified = ' AND '.join(parts[:2])
        
        return modified
    
    def _analyze_table_specific_optimizations(self, sql: str, table_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze optimizations specific to table characteristics."""
        
        optimizations = []
        
        # Check table sizes
        for table_name, info in table_info.items():
            if table_name.lower() in sql.lower():
                row_count = info.get('total_rows', 0)
                
                if row_count > 50000:
                    optimizations.append({
                        'rule': f'large_table_{table_name}',
                        'description': f'Table {table_name} has {row_count:,} rows - ensure WHERE clause filters effectively',
                        'priority': 'high',
                        'auto_applicable': False
                    })
                
                # Check column count
                col_count = info.get('total_columns', 0)
                if col_count > 10 and 'SELECT *' in sql.upper():
                    optimizations.append({
                        'rule': f'wide_table_{table_name}',
                        'description': f'Table {table_name} has {col_count} columns - select specific columns for better performance',
                        'priority': 'medium',
                        'auto_applicable': False
                    })
        
        return optimizations
    
    def _analyze_performance_metrics(self, sql: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance metrics and suggest improvements."""
        
        suggestions = []
        
        execution_time = metrics.get('execution_time_ms', 0)
        row_count = metrics.get('rows_returned', 0)
        
        if execution_time > 1000:  # > 1 second
            suggestions.append({
                'rule': 'slow_query',
                'description': f'Query took {execution_time}ms - consider adding indexes or optimizing joins',
                'priority': 'high',
                'auto_applicable': False
            })
        
        if row_count > 10000:
            suggestions.append({
                'rule': 'large_result_set',
                'description': f'Query returned {row_count:,} rows - consider adding LIMIT or more restrictive WHERE conditions',
                'priority': 'high',
                'auto_applicable': True
            })
        
        return suggestions
    
    def _analyze_user_feedback(self, sql: str, feedback: str) -> List[Dict[str, Any]]:
        """Analyze user feedback to understand query issues."""
        
        clarifications = []
        feedback_lower = feedback.lower()
        
        if any(word in feedback_lower for word in ['not what i wanted', 'wrong results', 'incorrect']):
            clarifications.append({
                'issue': 'incorrect_results',
                'suggestion': 'The query may need different WHERE conditions or JOIN criteria'
            })
        
        if any(word in feedback_lower for word in ['too many', 'too much', 'overwhelming']):
            clarifications.append({
                'issue': 'too_many_results',
                'suggestion': 'Add more specific filters or reduce the LIMIT'
            })
        
        if any(word in feedback_lower for word in ['missing', 'not enough', 'incomplete']):
            clarifications.append({
                'issue': 'missing_results',
                'suggestion': 'The query may be too restrictive - try broader search criteria'
            })
        
        return clarifications
    
    def _record_failure_pattern(self, sql: str, error: str, error_type: str) -> None:
        """Record failure pattern for learning."""
        
        pattern_key = f"{error_type}:{sql[:50]}"
        self.failure_patterns[pattern_key] += 1
        
        # Add to history
        self.query_history.append({
            'timestamp': time.time(),
            'sql': sql,
            'error': error,
            'error_type': error_type,
            'success': False
        })
    
    def _record_success_pattern(self, sql: str, optimizations: List[str]) -> None:
        """Record successful optimization pattern."""
        
        pattern_key = f"optimized:{','.join(optimizations)}"
        self.success_patterns[pattern_key] += 1
        
        # Add to history
        self.query_history.append({
            'timestamp': time.time(),
            'sql': sql,
            'optimizations': optimizations,
            'success': True
        })
    
    def _generate_learning_insights(self, sql: str, error_type: str) -> List[str]:
        """Generate insights based on historical patterns."""
        
        insights = []
        
        # Check if this type of error is common
        similar_failures = sum(1 for pattern in self.failure_patterns 
                             if pattern.startswith(error_type))
        
        if similar_failures > 5:
            insights.append(f"This type of error ({error_type}) has occurred {similar_failures} times - consider reviewing query patterns")
        
        # Check for successful patterns
        if self.success_patterns:
            most_common_success = Counter(self.success_patterns).most_common(1)[0]
            insights.append(f"Most successful optimization pattern: {most_common_success[0]} (used {most_common_success[1]} times)")
        
        return insights
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning patterns and insights."""
        
        return {
            'total_queries_analyzed': len(self.query_history),
            'success_rate': sum(1 for q in self.query_history if q.get('success', False)) / max(len(self.query_history), 1),
            'common_failure_types': dict(Counter(self.failure_patterns).most_common(5)),
            'successful_optimizations': dict(Counter(self.success_patterns).most_common(5)),
            'recent_trends': self._analyze_recent_trends()
        }
    
    def _analyze_recent_trends(self) -> Dict[str, Any]:
        """Analyze trends in recent query patterns."""
        
        recent_cutoff = time.time() - 3600  # Last hour
        recent_queries = [q for q in self.query_history if q.get('timestamp', 0) > recent_cutoff]
        
        if not recent_queries:
            return {'no_recent_activity': True}
        
        recent_failures = [q for q in recent_queries if not q.get('success', False)]
        
        return {
            'recent_query_count': len(recent_queries),
            'recent_success_rate': (len(recent_queries) - len(recent_failures)) / len(recent_queries),
            'trending_error_types': [q.get('error_type') for q in recent_failures],
        }

# Global instance
query_refinement = QueryRefinement()