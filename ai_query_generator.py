"""
AI-Powered Query Generation Module

Enhances LLM integration with intelligent context and analysis from Phase 0-2.
Creates optimized, accurate SQL queries using advanced prompt engineering.

Features:
- Context-aware prompt engineering
- Multi-step query generation with validation
- Error recovery and query correction
- Intelligent fallback strategies
- Performance-optimized LLM usage
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class AIQueryGenerator:
    """
    Intelligent SQL query generation system using LLM with advanced context.
    Integrates with Phase 0-2 components for optimal results.
    """
    
    def __init__(self):
        # Prompt templates for different query types
        self.prompt_templates = {
            'search': {
                'system': "You are an expert SQL query generator. Generate precise SQL queries based on natural language input and database schema.",
                'template': """
Context: User is searching for specific information.
Intent: {intent} (confidence: {confidence:.2f})
Entities detected: {entities}
Available tables: {tables}
Column mappings: {column_mappings}

User question: "{question}"

Database schema:
{schema}

Generate a SQL query that:
1. Uses the correct table names and column names from the schema
2. Addresses the user's specific search intent
3. Includes appropriate WHERE conditions based on detected entities
4. Returns relevant columns for the search

SQL Query:"""
            },
            'count': {
                'system': "You are an expert SQL analyst. Generate COUNT queries that accurately measure data.",
                'template': """
Context: User wants to count records or calculate quantities.
Intent: {intent} (confidence: {confidence:.2f})
Entities detected: {entities}
Available tables: {tables}
Column mappings: {column_mappings}

User question: "{question}"

Database schema:
{schema}

Generate a COUNT SQL query that:
1. Uses appropriate COUNT() functions
2. Includes proper GROUP BY if needed
3. Applies relevant WHERE conditions
4. Returns meaningful count results

SQL Query:"""
            },
            'compare': {
                'system': "You are an expert data analyst. Generate comparison queries with proper aggregation.",
                'template': """
Context: User wants to compare different data points or categories.
Intent: {intent} (confidence: {confidence:.2f})
Entities detected: {entities}
Available tables: {tables}
Column mappings: {column_mappings}

User question: "{question}"

Database schema:
{schema}

Generate a comparison SQL query that:
1. Uses appropriate aggregation functions (AVG, SUM, COUNT, etc.)
2. Includes proper GROUP BY for comparison categories
3. Orders results meaningfully
4. Handles potential NULL values appropriately

SQL Query:"""
            },
            'filter': {
                'system': "You are an expert database query specialist. Generate filtered result queries.",
                'template': """
Context: User wants to filter data based on specific criteria.
Intent: {intent} (confidence: {confidence:.2f})
Entities detected: {entities}
Available tables: {tables}
Column mappings: {column_mappings}

User question: "{question}"

Database schema:
{schema}

Generate a filtered SQL query that:
1. Uses appropriate WHERE conditions based on user criteria
2. Applies correct operators (=, LIKE, IN, BETWEEN, etc.)
3. Handles text matching case-insensitively when appropriate
4. Returns all relevant columns for the filtered results

SQL Query:"""
            },
            'multi_table': {
                'system': "You are an expert database architect. Generate complex multi-table queries with optimal joins.",
                'template': """
Context: User query requires data from multiple tables.
Intent: {intent} (confidence: {confidence:.2f})
Entities detected: {entities}
Required tables: {tables}
Suggested joins: {join_suggestions}
Column mappings: {column_mappings}

User question: "{question}"

Database schema:
{schema}

Join recommendations:
{join_details}

Generate a multi-table SQL query that:
1. Uses the recommended JOIN strategy
2. Includes all necessary tables with proper aliases
3. Applies WHERE conditions after joins
4. Optimizes performance with appropriate JOIN order
5. Returns relevant columns from all joined tables

SQL Query:"""
            }
        }
        
        # Validation patterns for generated SQL
        self.validation_patterns = {
            'basic_structure': r'^\s*SELECT\s+.+\s+FROM\s+.+',
            'table_references': r'FROM\s+(["`\w]+)',
            'column_references': r'SELECT\s+(.+?)\s+FROM',
            'where_clause': r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|\s+GROUP\s+BY|\s+LIMIT|$)',
            'sql_injection': [
                r';\s*(DROP|DELETE|INSERT|UPDATE)\s+',
                r'--\s*$',
                r'/\*.*\*/',
                r'UNION\s+SELECT',
                r'xp_cmdshell',
                r'sp_executesql'
            ]
        }
        
        # Query optimization suggestions
        self.optimization_patterns = {
            'missing_limit': {
                'pattern': r'SELECT\s+.*\s+FROM\s+.*(?!\s+LIMIT\s+)',
                'suggestion': 'Consider adding LIMIT clause to prevent large result sets'
            },
            'no_where_clause': {
                'pattern': r'SELECT\s+.*\s+FROM\s+[^W]*(?!WHERE)',
                'suggestion': 'Query returns all records - consider adding WHERE conditions'
            },
            'select_star': {
                'pattern': r'SELECT\s+\*\s+FROM',
                'suggestion': 'Consider selecting specific columns instead of * for better performance'
            },
            'complex_joins': {
                'pattern': r'JOIN.*JOIN.*JOIN',
                'suggestion': 'Complex multi-join query - verify performance and consider indexing'
            }
        }
    
    def generate_enhanced_query(self, question: str, query_analysis: Dict[str, Any], 
                              table_info: Dict[str, Any], schema: str,
                              context: Optional[Dict[str, Any]] = None,
                              multi_table_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an enhanced SQL query using AI with full context from Phase 0-2.
        
        Args:
            question: Original user question
            query_analysis: Analysis from query_intelligence
            table_info: Table information from dynamic_mapper
            schema: Database schema description
            context: Query context from context_engine
            multi_table_plan: Multi-table plan from multi_table_intelligence
            
        Returns:
            Enhanced query generation results
        """
        logger.info(f"Generating enhanced AI query for: {question[:50]}...")
        
        result = {
            'sql': '',
            'confidence': 0.0,
            'reasoning': [],
            'optimizations': [],
            'fallback_used': False,
            'validation_results': {},
            'prompt_used': '',
            'generation_strategy': 'ai_enhanced',
            'performance_hints': []
        }
        
        try:
            # Step 1: Select appropriate prompt template
            intent = query_analysis.get('intent', 'search')
            complexity = query_analysis.get('complexity_score', 0)
            
            # Choose template based on complexity and multi-table needs
            if multi_table_plan and multi_table_plan.get('strategy') == 'multi_table':
                template_key = 'multi_table'
            else:
                template_key = intent if intent in self.prompt_templates else 'search'
            
            result['generation_strategy'] = f"ai_enhanced_{template_key}"
            
            # Step 2: Prepare enhanced prompt context
            prompt_context = self._prepare_prompt_context(
                question, query_analysis, table_info, schema, context, multi_table_plan
            )
            
            # Step 3: Generate prompt
            prompt = self._build_enhanced_prompt(template_key, prompt_context)
            result['prompt_used'] = prompt[:200] + "..." if len(prompt) > 200 else prompt
            
            # Step 4: Get LLM response
            sql_response = self._query_llm(prompt, template_key)
            
            # Step 5: Extract and clean SQL
            extracted_sql = self._extract_sql_from_response(sql_response)
            
            if not extracted_sql:
                raise Exception("No valid SQL extracted from LLM response")
            
            # Step 6: Validate generated SQL
            validation_results = self._validate_generated_sql(extracted_sql, table_info)
            result['validation_results'] = validation_results
            
            if not validation_results['is_valid']:
                # Try to fix common issues
                fixed_sql = self._attempt_sql_fixes(extracted_sql, validation_results, table_info)
                if fixed_sql:
                    extracted_sql = fixed_sql
                    result['reasoning'].append("Applied automatic SQL fixes")
            
            # Step 7: Generate optimizations and performance hints
            optimizations = self._generate_optimizations(extracted_sql, query_analysis)
            result['optimizations'] = optimizations
            
            performance_hints = self._generate_performance_hints(extracted_sql, table_info, complexity)
            result['performance_hints'] = performance_hints
            
            # Step 8: Calculate confidence
            confidence = self._calculate_generation_confidence(
                query_analysis, validation_results, template_key, len(optimizations)
            )
            
            result['sql'] = extracted_sql
            result['confidence'] = confidence
            result['reasoning'].append(f"Used {template_key} template with {confidence:.2f} confidence")
            
            logger.info(f"Enhanced query generated with {confidence:.2f} confidence")
            
        except Exception as e:
            logger.error(f"Enhanced query generation failed: {e}")
            
            # Fallback to simpler generation
            fallback_result = self._generate_fallback_query(
                question, query_analysis, table_info, schema
            )
            
            result.update(fallback_result)
            result['fallback_used'] = True
            result['reasoning'].append(f"Used fallback due to error: {str(e)}")
        
        return result
    
    def _prepare_prompt_context(self, question: str, query_analysis: Dict[str, Any],
                               table_info: Dict[str, Any], schema: str,
                               context: Optional[Dict[str, Any]], 
                               multi_table_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare comprehensive context for prompt generation."""
        
        # Extract entities with confidence
        entities_info = []
        for entity_type, entity_list in query_analysis.get('entities', {}).items():
            for entity in entity_list:
                entities_info.append(f"{entity_type}: '{entity['text']}' (conf: {entity['confidence']:.2f})")
        
        # Prepare column mappings
        column_mappings = {}
        for table_name, table_data in table_info.items():
            role_mappings = table_data.get('role_mappings', {})
            for role, columns in role_mappings.items():
                if columns:
                    column_mappings[role] = f"{table_name}.{columns[0]['column']}"
        
        # Context from conversation history
        context_hints = []
        if context:
            recent_entities = context.get('recent_entities', {})
            if recent_entities:
                context_hints.append(f"Recent context: {list(recent_entities.keys())[:3]}")
            
            table_prefs = context.get('table_preferences', {})
            if table_prefs:
                preferred_table = max(table_prefs, key=table_prefs.get)
                context_hints.append(f"User often queries: {preferred_table}")
        
        # Multi-table join information
        join_details = "N/A"
        if multi_table_plan and multi_table_plan.get('join_sequence'):
            join_info = []
            for join in multi_table_plan['join_sequence']:
                join_info.append(f"{join['join_type']} {join['table']} ON {join.get('join_condition', 'auto-detected')}")
            join_details = "; ".join(join_info)
        
        return {
            'question': question,
            'intent': query_analysis.get('intent', 'search'),
            'confidence': query_analysis.get('confidence', 0.5),
            'entities': ", ".join(entities_info) if entities_info else "None detected",
            'tables': list(table_info.keys()),
            'column_mappings': column_mappings,
            'schema': schema,
            'context_hints': context_hints,
            'join_suggestions': multi_table_plan.get('join_sequence', []) if multi_table_plan else [],
            'join_details': join_details,
            'complexity_score': query_analysis.get('complexity_score', 0)
        }
    
    def _build_enhanced_prompt(self, template_key: str, context: Dict[str, Any]) -> str:
        """Build enhanced prompt using selected template and context."""
        
        template_info = self.prompt_templates[template_key]
        
        # Format the template with context
        formatted_prompt = template_info['template'].format(**context)
        
        # Add context hints if available
        if context.get('context_hints'):
            context_section = "\nConversation context:\n" + "\n".join(context['context_hints'])
            formatted_prompt = formatted_prompt.replace("User question:", context_section + "\n\nUser question:")
        
        # Add system message
        system_msg = template_info['system']
        full_prompt = f"{system_msg}\n\n{formatted_prompt}"
        
        return full_prompt
    
    def _query_llm(self, prompt: str, template_key: str) -> str:
        """Query the LLM with the prepared prompt."""
        try:
            # Import LLM functionality
            from core import get_llm
            
            llm = get_llm()
            
            # Add specific instructions for SQL extraction
            enhanced_prompt = prompt + "\n\nIMPORTANT: Return ONLY the SQL query, no explanations. Wrap the SQL in <SQL> tags."
            
            response = llm.predict(enhanced_prompt)
            
            if not response:
                raise Exception("Empty response from LLM")
            
            logger.info(f"LLM responded with {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        
        # Try to extract from SQL tags
        sql_match = re.search(r'<SQL>(.*?)</SQL>', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Try to find SQL-like patterns
        lines = response.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)', line, re.IGNORECASE):
                sql_lines.append(line)
            elif sql_lines and re.match(r'^\s*(FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|JOIN|UNION)', line, re.IGNORECASE):
                sql_lines.append(line)
            elif sql_lines and line and not line.startswith('#') and not line.startswith('--'):
                sql_lines.append(line)
            elif sql_lines and not line:
                break
        
        if sql_lines:
            return '\n'.join(sql_lines).strip()
        
        # Last resort - return the response if it looks like SQL
        if re.search(r'\bSELECT\b.*\bFROM\b', response, re.IGNORECASE):
            return response.strip()
        
        return ""
    
    def _validate_generated_sql(self, sql: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated SQL for correctness and security."""
        
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'security_issues': [],
            'performance_concerns': []
        }
        
        # Check basic SQL structure
        if not re.match(self.validation_patterns['basic_structure'], sql, re.IGNORECASE):
            validation['is_valid'] = False
            validation['errors'].append("Invalid SQL structure - missing SELECT/FROM")
        
        # Check for SQL injection patterns
        for pattern in self.validation_patterns['sql_injection']:
            if re.search(pattern, sql, re.IGNORECASE):
                validation['is_valid'] = False
                validation['security_issues'].append(f"Potential SQL injection pattern: {pattern}")
        
        # Check table references
        table_matches = re.findall(r'FROM\s+["`]?(\w+)["`]?', sql, re.IGNORECASE)
        for table in table_matches:
            if table not in table_info:
                validation['warnings'].append(f"Table '{table}' not found in available tables")
        
        # Check for performance concerns
        if 'SELECT *' in sql.upper():
            validation['performance_concerns'].append("Using SELECT * may impact performance")
        
        if not re.search(r'\bLIMIT\b', sql, re.IGNORECASE) and not re.search(r'\bWHERE\b', sql, re.IGNORECASE):
            validation['performance_concerns'].append("Query may return large result set - consider adding WHERE or LIMIT")
        
        return validation
    
    def _attempt_sql_fixes(self, sql: str, validation_results: Dict[str, Any], 
                          table_info: Dict[str, Any]) -> Optional[str]:
        """Attempt to automatically fix common SQL issues."""
        
        fixed_sql = sql
        
        # Fix table name issues
        for error in validation_results.get('warnings', []):
            if 'Table' in error and 'not found' in error:
                # Extract problematic table name
                match = re.search(r"Table '(\w+)' not found", error)
                if match:
                    wrong_table = match.group(1)
                    # Find best matching table
                    available_tables = list(table_info.keys())
                    best_match = self._find_best_table_match(wrong_table, available_tables)
                    if best_match:
                        fixed_sql = re.sub(rf'\b{wrong_table}\b', best_match, fixed_sql, flags=re.IGNORECASE)
                        logger.info(f"Fixed table reference: {wrong_table} -> {best_match}")
        
        # Add LIMIT if missing and no WHERE clause
        if ('large result set' in str(validation_results.get('performance_concerns', [])) and 
            not re.search(r'\bLIMIT\b', fixed_sql, re.IGNORECASE)):
            fixed_sql = fixed_sql.rstrip(';') + ' LIMIT 100'
            logger.info("Added LIMIT 100 to prevent large result sets")
        
        return fixed_sql if fixed_sql != sql else None
    
    def _find_best_table_match(self, target: str, available: List[str]) -> Optional[str]:
        """Find the best matching table name using similarity."""
        import difflib
        
        best_match = difflib.get_close_matches(target.lower(), 
                                               [t.lower() for t in available], 
                                               n=1, cutoff=0.5)
        
        if best_match:
            # Return the original case version
            for table in available:
                if table.lower() == best_match[0]:
                    return table
        
        return None
    
    def _generate_optimizations(self, sql: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions for the SQL query."""
        optimizations = []
        
        for opt_name, opt_info in self.optimization_patterns.items():
            if re.search(opt_info['pattern'], sql, re.IGNORECASE):
                optimizations.append(opt_info['suggestion'])
        
        # Add context-specific optimizations
        complexity = query_analysis.get('complexity_score', 0)
        if complexity > 7:
            optimizations.append("High complexity query - consider breaking into simpler parts")
        
        return optimizations
    
    def _generate_performance_hints(self, sql: str, table_info: Dict[str, Any], 
                                   complexity: int) -> List[str]:
        """Generate performance hints based on the query and table information."""
        hints = []
        
        # Check table sizes
        large_tables = []
        for table_name, table_data in table_info.items():
            if table_name.lower() in sql.lower():
                rows = table_data.get('total_rows', 0)
                if rows > 10000:
                    large_tables.append((table_name, rows))
        
        if large_tables:
            hints.append(f"Query involves large table(s): {', '.join([f'{t}({r:,} rows)' for t, r in large_tables])}")
        
        # Check for joins
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        if join_count > 2:
            hints.append(f"Complex query with {join_count} joins - ensure proper indexing")
        
        # Check for aggregation
        if re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\b', sql, re.IGNORECASE):
            hints.append("Aggregation query - performance depends on data distribution")
        
        return hints
    
    def _calculate_generation_confidence(self, query_analysis: Dict[str, Any],
                                       validation_results: Dict[str, Any],
                                       template_key: str, optimization_count: int) -> float:
        """Calculate confidence score for the generated query."""
        
        base_confidence = query_analysis.get('confidence', 0.5)
        
        # Adjust based on validation results
        if validation_results['is_valid']:
            validation_bonus = 0.3
        else:
            validation_bonus = -0.5
        
        # Adjust based on template match
        template_bonus = 0.2 if template_key != 'search' else 0.0
        
        # Adjust based on optimizations needed
        optimization_penalty = min(optimization_count * 0.05, 0.2)
        
        final_confidence = base_confidence + validation_bonus + template_bonus - optimization_penalty
        
        return max(0.0, min(1.0, final_confidence))
    
    def _generate_fallback_query(self, question: str, query_analysis: Dict[str, Any],
                                table_info: Dict[str, Any], schema: str) -> Dict[str, Any]:
        """Generate a simple fallback query when advanced generation fails."""
        
        logger.warning("Using fallback query generation")
        
        # Very basic query generation
        intent = query_analysis.get('intent', 'search')
        entities = query_analysis.get('entities', {})
        
        # Pick the first available table
        table_name = list(table_info.keys())[0] if table_info else 'unknown_table'
        
        # Basic SELECT query
        if intent == 'count':
            fallback_sql = f"SELECT COUNT(*) FROM {table_name}"
        else:
            fallback_sql = f"SELECT * FROM {table_name} LIMIT 10"
        
        # Add basic WHERE if entities detected
        if entities:
            entity_conditions = []
            for entity_type, entity_list in entities.items():
                if entity_list:
                    entity_value = entity_list[0]['text']
                    # Try to guess a column name
                    if entity_type == 'asset_id':
                        entity_conditions.append(f"asset_tag LIKE '%{entity_value}%'")
                    elif entity_type == 'person':
                        entity_conditions.append(f"name LIKE '%{entity_value}%'")
            
            if entity_conditions:
                fallback_sql += f" WHERE {' AND '.join(entity_conditions)}"
        
        return {
            'sql': fallback_sql,
            'confidence': 0.3,
            'reasoning': ["Used basic fallback generation"],
            'optimizations': ["This is a basic fallback query - consider manual refinement"],
            'validation_results': {'is_valid': True, 'errors': [], 'warnings': []},
            'performance_hints': ["Fallback query may not be optimal for your specific needs"]
        }

# Global instance
ai_query_generator = AIQueryGenerator()