"""
Advanced Query Intelligence Module

Enhances natural language understanding for complex queries.
Builds on Phase 1 column intelligence to understand user intent at deeper levels.

Features:
- Intent classification (search, filter, aggregate, compare, list)
- Entity extraction from natural language
- Query complexity analysis and optimization
- Context-aware query interpretation
- Relationship detection between entities
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import difflib

logger = logging.getLogger(__name__)

class QueryIntelligence:
    """
    Advanced natural language query understanding system.
    Analyzes user intent and extracts structured query requirements.
    """
    
    def __init__(self):
        # Intent classification patterns
        self.intent_patterns = {
            'search': [
                r'what is|what are|find|search|show me|tell me|get|display',
                r'who has|who owns|who is using',
                r'where is|where are',
                r'which.*has|which.*contains'
            ],
            'filter': [
                r'all.*where|everything.*where|items.*where',
                r'filter.*by|show.*only|only.*with',
                r'exclude|without|not|except'
            ],
            'count': [
                r'how many|count|total|number of',
                r'sum of|total of'
            ],
            'compare': [
                r'compare|difference|versus|vs|better|worse',
                r'higher|lower|more|less',
                r'between.*and'
            ],
            'list': [
                r'list all|show all|display all',
                r'everything|all items|all records'
            ],
            'aggregate': [
                r'average|mean|maximum|minimum|max|min',
                r'sum|total|group by'
            ]
        }
        
        # Entity type patterns (building on Phase 1 roles)
        self.entity_patterns = {
            'person': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith format
                r'\b[A-Z]+\b',  # UPPERCASE names
                r'user|employee|person|staff|owner'
            ],
            'asset_id': [
                r'\b[A-Z]+-[A-Z0-9\-]+\b',  # ASSET-123 format
                r'\b[A-Z]{2,}\d+\b',  # ABC123 format
                r'\basset|tag|id|identifier'
            ],
            'company': [
                r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*(?:\s(?:Inc|Corp|LLC|Ltd))\.?\b',
                r'company|manufacturer|brand|vendor'
            ],
            'location': [
                r'floor\s+\d+|level\s+\d+|building\s+[A-Z]',
                r'room\s+\d+|office\s+\d+',
                r'location|place|site|where'
            ],
            'product': [
                r'laptop|desktop|computer|device|equipment',
                r'model|product|item|type'
            ],
            'attribute': [
                r'price|cost|value|amount',
                r'status|condition|state',
                r'date|time|when'
            ]
        }
        
        # Relationship keywords
        self.relationship_keywords = {
            'ownership': ['has', 'owns', 'assigned to', 'belongs to', 'used by'],
            'location': ['in', 'at', 'located at', 'placed in', 'stored at'],
            'specification': ['is', 'type', 'model', 'brand', 'made by'],
            'temporal': ['before', 'after', 'during', 'since', 'until'],
            'comparison': ['more than', 'less than', 'equal to', 'different from']
        }
        
        # Question word analysis
        self.question_analysis = {
            'what': {'expects': ['attribute', 'specification'], 'type': 'search'},
            'who': {'expects': ['person'], 'type': 'search'},
            'where': {'expects': ['location'], 'type': 'search'},
            'when': {'expects': ['date', 'time'], 'type': 'search'},
            'which': {'expects': ['identifier', 'selection'], 'type': 'filter'},
            'how many': {'expects': ['count'], 'type': 'count'},
            'how much': {'expects': ['amount', 'price'], 'type': 'search'}
        }

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query to understand intent and extract structured information.
        
        Args:
            query: Natural language query string
            
        Returns:
            Structured analysis of query intent and requirements
        """
        logger.info(f"Analyzing query intent: {query}")
        
        analysis = {
            'original_query': query,
            'intent': 'search',  # default
            'confidence': 0.0,
            'entities': {},
            'relationships': [],
            'question_type': None,
            'complexity_score': 0,
            'required_operations': [],
            'expected_result_type': 'single_record',
            'semantic_roles_needed': [],
            'filters': [],
            'sort_requirements': []
        }
        
        query_lower = query.lower().strip()
        
        # 1. Classify intent
        analysis['intent'], analysis['confidence'] = self._classify_intent(query_lower)
        
        # 2. Extract entities
        analysis['entities'] = self._extract_entities(query)
        
        # 3. Detect relationships
        analysis['relationships'] = self._detect_relationships(query_lower)
        
        # 4. Analyze question structure
        analysis['question_type'] = self._analyze_question_type(query_lower)
        
        # 5. Determine complexity
        analysis['complexity_score'] = self._calculate_complexity(analysis)
        
        # 6. Identify required semantic roles
        analysis['semantic_roles_needed'] = self._identify_required_roles(analysis)
        
        # 7. Extract filters and conditions
        analysis['filters'] = self._extract_filters(query_lower, analysis['entities'])
        
        # 8. Determine expected result type
        analysis['expected_result_type'] = self._determine_result_type(analysis)
        
        logger.info(f"Intent analysis complete: {analysis['intent']} (confidence: {analysis['confidence']:.2f})")
        return analysis
    
    def _classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify the primary intent of the query."""
        intent_scores = defaultdict(float)
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                if matches > 0:
                    intent_scores[intent] += matches * 0.7
                    
                    # Bonus for exact matches
                    if re.search(r'\b' + pattern.replace('.*', r'\w*') + r'\b', query, re.IGNORECASE):
                        intent_scores[intent] += 0.3
        
        if not intent_scores:
            return 'search', 0.5  # Default fallback
            
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[best_intent], 1.0)
        
        return best_intent, confidence
    
    def _extract_entities(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from the query using pattern matching."""
        entities = defaultdict(list)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entity_info = {
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8 if match.group(0).isupper() else 0.6,
                        'pattern_matched': pattern
                    }
                    entities[entity_type].append(entity_info)
        
        # Remove duplicates and sort by confidence
        for entity_type in entities:
            entities[entity_type] = sorted(entities[entity_type], 
                                         key=lambda x: x['confidence'], reverse=True)
            
            # Remove overlapping matches (keep highest confidence)
            cleaned = []
            for entity in entities[entity_type]:
                overlap = any(
                    (entity['start'] < existing['end'] and entity['end'] > existing['start'])
                    for existing in cleaned
                )
                if not overlap:
                    cleaned.append(entity)
            entities[entity_type] = cleaned
        
        return dict(entities)
    
    def _detect_relationships(self, query: str) -> List[Dict[str, Any]]:
        """Detect relationships between entities in the query."""
        relationships = []
        
        for rel_type, keywords in self.relationship_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    # Find context around the relationship keyword
                    pattern = rf'(\w+(?:\s+\w+)*)\s+{re.escape(keyword)}\s+(\w+(?:\s+\w+)*)'
                    matches = re.finditer(pattern, query, re.IGNORECASE)
                    
                    for match in matches:
                        relationship = {
                            'type': rel_type,
                            'keyword': keyword,
                            'subject': match.group(1).strip(),
                            'object': match.group(2).strip(),
                            'confidence': 0.7,
                            'context': match.group(0)
                        }
                        relationships.append(relationship)
        
        return relationships
    
    def _analyze_question_type(self, query: str) -> Optional[str]:
        """Analyze the question structure to understand what's being asked."""
        for question_word, info in self.question_analysis.items():
            if query.startswith(question_word.lower()):
                return question_word
        
        # Check for question patterns in the middle
        question_patterns = ['what is', 'who has', 'where is', 'which', 'how many']
        for pattern in question_patterns:
            if pattern in query:
                return pattern.split()[0]
        
        return None
    
    def _calculate_complexity(self, analysis: Dict[str, Any]) -> int:
        """Calculate query complexity score based on various factors."""
        complexity = 0
        
        # Base complexity from entities
        total_entities = sum(len(entities) for entities in analysis['entities'].values())
        complexity += min(total_entities, 5)  # Cap at 5 points
        
        # Complexity from relationships
        complexity += len(analysis['relationships']) * 2
        
        # Intent-based complexity
        intent_complexity = {
            'search': 1,
            'filter': 2,
            'count': 2,
            'compare': 3,
            'aggregate': 4,
            'list': 1
        }
        complexity += intent_complexity.get(analysis['intent'], 1)
        
        # Question type complexity
        if analysis['question_type'] in ['which', 'how many']:
            complexity += 1
        
        return min(complexity, 10)  # Cap at 10
    
    def _identify_required_roles(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify which semantic roles are needed based on the query analysis."""
        required_roles = set()
        
        # Map entity types to semantic roles
        entity_to_role_mapping = {
            'person': 'person_name',
            'asset_id': 'identifier',
            'company': 'manufacturer',
            'location': 'location',
            'product': 'product',
            'attribute': ['money', 'status', 'date']  # Multiple possible roles
        }
        
        # Add roles based on detected entities
        for entity_type, entities in analysis['entities'].items():
            if entities:  # Only if we found entities of this type
                roles = entity_to_role_mapping.get(entity_type, [])
                if isinstance(roles, list):
                    required_roles.update(roles)
                else:
                    required_roles.add(roles)
        
        # Add roles based on intent
        intent_role_mapping = {
            'search': ['identifier', 'person_name'],
            'filter': ['identifier', 'manufacturer', 'location'],
            'count': ['identifier'],
            'compare': ['identifier', 'money', 'product'],
            'aggregate': ['money', 'identifier'],
            'list': ['identifier', 'person_name', 'manufacturer']
        }
        
        intent_roles = intent_role_mapping.get(analysis['intent'], [])
        required_roles.update(intent_roles)
        
        return sorted(list(required_roles))
    
    def _extract_filters(self, query: str, entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract filter conditions from the query."""
        filters = []
        
        # Extract explicit filters from entities
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity['confidence'] > 0.7:
                    filter_info = {
                        'type': entity_type,
                        'value': entity['text'],
                        'operation': '=',  # Default to equality
                        'confidence': entity['confidence']
                    }
                    filters.append(filter_info)
        
        # Extract comparison operators
        comparison_patterns = {
            'greater_than': [r'more than|greater than|above|over|>', r'\d+'],
            'less_than': [r'less than|below|under|<', r'\d+'],
            'equal_to': [r'equals?|is|=', r'\w+'],
            'not_equal': [r'not|isn\'t|!=', r'\w+'],
            'contains': [r'contains|includes|with|has', r'\w+']
        }
        
        for op_type, (op_pattern, value_pattern) in comparison_patterns.items():
            pattern = rf'{op_pattern}\s+({value_pattern})'
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                filter_info = {
                    'type': 'comparison',
                    'operation': op_type,
                    'value': match.group(1),
                    'confidence': 0.8
                }
                filters.append(filter_info)
        
        return filters
    
    def _determine_result_type(self, analysis: Dict[str, Any]) -> str:
        """Determine what type of result the user expects."""
        intent = analysis['intent']
        
        if intent == 'count':
            return 'scalar'
        elif intent == 'list':
            return 'multiple_records'
        elif intent in ['compare', 'aggregate']:
            return 'summary'
        elif len(analysis.get('filters', [])) == 0 and intent == 'search':
            return 'multiple_records'  # General search
        else:
            return 'single_record'  # Specific search with filters
    
    def generate_query_plan(self, analysis: Dict[str, Any], available_tables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an execution plan for the analyzed query.
        
        Args:
            analysis: Query analysis from analyze_query_intent
            available_tables: Available table information from dynamic mapper
            
        Returns:
            Query execution plan
        """
        logger.info("Generating query execution plan")
        
        plan = {
            'execution_strategy': 'single_table',  # or 'multi_table'
            'primary_table': None,
            'required_joins': [],
            'where_conditions': [],
            'select_columns': [],
            'order_by': [],
            'limit': None,
            'aggregation': None,
            'estimated_complexity': analysis['complexity_score'],
            'optimization_suggestions': []
        }
        
        # Find best table(s) for required roles
        table_scores = self._score_tables_for_roles(analysis['semantic_roles_needed'], available_tables)
        
        if table_scores:
            best_table = max(table_scores, key=table_scores.get)
            plan['primary_table'] = best_table
            
            # Check if multi-table query is needed
            coverage = self._check_role_coverage(analysis['semantic_roles_needed'], best_table, available_tables)
            if coverage < 0.8:  # Less than 80% of roles covered
                plan['execution_strategy'] = 'multi_table'
                plan['required_joins'] = self._plan_joins(analysis['semantic_roles_needed'], available_tables)
        
        # Build WHERE conditions from filters
        for filter_item in analysis['filters']:
            if filter_item['type'] in ['asset_id', 'person', 'company']:
                plan['where_conditions'].append({
                    'semantic_role': self._map_filter_to_role(filter_item['type']),
                    'operation': filter_item.get('operation', '='),
                    'value': filter_item['value'],
                    'confidence': filter_item['confidence']
                })
        
        # Determine SELECT columns based on intent
        if analysis['intent'] == 'count':
            plan['select_columns'] = ['COUNT(*)']
            plan['aggregation'] = 'count'
        elif analysis['intent'] == 'list':
            plan['select_columns'] = ['*']
            plan['limit'] = 100  # Reasonable default
        else:
            # Select relevant columns based on semantic roles
            plan['select_columns'] = self._select_relevant_columns(analysis, available_tables)
        
        # Add optimization suggestions
        plan['optimization_suggestions'] = self._generate_optimization_suggestions(plan, analysis)
        
        logger.info(f"Query plan generated: {plan['execution_strategy']} strategy")
        return plan
    
    def _score_tables_for_roles(self, required_roles: List[str], available_tables: Dict[str, Any]) -> Dict[str, float]:
        """Score tables based on how well they match required semantic roles."""
        scores = {}
        
        # This would interface with the dynamic mapper from Phase 1
        # For now, using a placeholder structure
        for table_name, table_info in available_tables.items():
            score = 0.0
            role_mappings = table_info.get('role_mappings', {})
            
            for role in required_roles:
                if role in role_mappings:
                    # Weight by confidence of the role mapping
                    role_confidence = role_mappings[role][0].get('confidence', 0.5) if role_mappings[role] else 0.5
                    score += role_confidence
            
            # Normalize by number of required roles
            if required_roles:
                scores[table_name] = score / len(required_roles)
        
        return scores
    
    def _check_role_coverage(self, required_roles: List[str], table: str, available_tables: Dict[str, Any]) -> float:
        """Check what percentage of required roles are covered by a single table."""
        if not required_roles:
            return 1.0
            
        table_info = available_tables.get(table, {})
        role_mappings = table_info.get('role_mappings', {})
        
        covered_roles = sum(1 for role in required_roles if role in role_mappings)
        return covered_roles / len(required_roles)
    
    def _plan_joins(self, required_roles: List[str], available_tables: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan JOIN operations for multi-table queries."""
        # Simplified join planning - would be more sophisticated in production
        joins = []
        
        # Find tables that share common identifier roles
        identifier_tables = []
        for table_name, table_info in available_tables.items():
            role_mappings = table_info.get('role_mappings', {})
            if 'identifier' in role_mappings:
                identifier_tables.append(table_name)
        
        # Create joins between tables with shared identifiers
        if len(identifier_tables) > 1:
            primary_table = identifier_tables[0]
            for other_table in identifier_tables[1:]:
                join_info = {
                    'type': 'INNER JOIN',
                    'table': other_table,
                    'condition': f'{primary_table}.identifier_column = {other_table}.identifier_column',
                    'confidence': 0.7
                }
                joins.append(join_info)
        
        return joins
    
    def _map_filter_to_role(self, filter_type: str) -> str:
        """Map filter types to semantic roles."""
        mapping = {
            'asset_id': 'identifier',
            'person': 'person_name',
            'company': 'manufacturer',
            'location': 'location',
            'product': 'product'
        }
        return mapping.get(filter_type, 'identifier')
    
    def _select_relevant_columns(self, analysis: Dict[str, Any], available_tables: Dict[str, Any]) -> List[str]:
        """Select relevant columns based on query intent and entities."""
        if analysis['intent'] == 'search' and analysis['expected_result_type'] == 'single_record':
            return ['*']  # Return all columns for specific searches
        
        # Select columns based on detected entities and relationships
        relevant_columns = set()
        
        for entity_type in analysis['entities'].keys():
            role = self._map_filter_to_role(entity_type)
            relevant_columns.add(role)
        
        # Add related columns based on relationships
        for rel in analysis['relationships']:
            if rel['type'] == 'ownership':
                relevant_columns.update(['identifier', 'person_name'])
            elif rel['type'] == 'specification':
                relevant_columns.update(['product', 'manufacturer'])
        
        return list(relevant_columns) if relevant_columns else ['*']
    
    def _generate_optimization_suggestions(self, plan: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions for the query plan."""
        suggestions = []
        
        if plan['execution_strategy'] == 'multi_table':
            suggestions.append("Consider indexing join columns for better performance")
        
        if len(plan['where_conditions']) > 3:
            suggestions.append("Complex WHERE clause - consider breaking into simpler queries")
        
        if analysis['complexity_score'] > 7:
            suggestions.append("High complexity query - consider caching results")
        
        if plan.get('limit') is None and analysis['intent'] == 'list':
            suggestions.append("Add LIMIT clause to prevent large result sets")
        
        return suggestions

# Global instance
query_intelligence = QueryIntelligence()