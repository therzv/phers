"""
Context Engine Module

Manages query context and provides intelligent context-aware query interpretation.
Maintains conversation history and learns from user interactions.

Features:
- Query context tracking and management
- Conversation history analysis
- Context-aware query disambiguation
- User intent learning and adaptation
- Query refinement suggestions
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, defaultdict, Counter
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class ContextEngine:
    """
    Intelligent context management system for query interpretation.
    Tracks conversation history and provides context-aware assistance.
    """
    
    def __init__(self, max_history_size: int = 50):
        # Conversation history storage
        self.query_history = deque(maxlen=max_history_size)
        self.context_stack = []  # Stack of active contexts
        self.user_preferences = {}
        self.session_stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'session_start': datetime.now()
        }
        
        # Context types and their importance weights
        self.context_types = {
            'table_context': 0.9,      # Recently used tables
            'column_context': 0.8,     # Recently used columns/roles
            'entity_context': 0.7,     # Recently mentioned entities
            'intent_context': 0.6,     # Recent query intents
            'temporal_context': 0.4,   # Time-based context
            'domain_context': 0.5      # Business domain context
        }
        
        # Context decay factors (how quickly context loses relevance)
        self.decay_factors = {
            'immediate': 1.0,    # Current query
            'recent': 0.8,       # Last 1-2 queries
            'session': 0.6,      # This session
            'historical': 0.3    # Previous sessions
        }
        
        # Entity relationship tracking
        self.entity_relationships = defaultdict(list)
        self.frequent_patterns = defaultdict(int)
    
    def add_query_to_history(self, query: str, query_analysis: Dict[str, Any], 
                           execution_result: Dict[str, Any]) -> None:
        """
        Add a completed query to the conversation history.
        
        Args:
            query: Original user query
            query_analysis: Analysis results from query intelligence
            execution_result: Results from query execution
        """
        timestamp = datetime.now()
        
        history_entry = {
            'timestamp': timestamp,
            'original_query': query,
            'analysis': query_analysis,
            'result': execution_result,
            'success': execution_result.get('success', False),
            'context_used': self._extract_context_used(query_analysis),
            'entities_mentioned': self._extract_entities(query_analysis),
            'tables_accessed': execution_result.get('tables_used', []),
            'performance_metrics': execution_result.get('performance', {})
        }
        
        self.query_history.append(history_entry)
        self.session_stats['queries_processed'] += 1
        
        if history_entry['success']:
            self.session_stats['successful_queries'] += 1
        else:
            self.session_stats['failed_queries'] += 1
        
        # Update context and learn from this query
        self._update_context_from_query(history_entry)
        self._learn_user_patterns(history_entry)
        
        logger.info(f"Added query to history. Total queries: {len(self.query_history)}")
    
    def get_context_for_query(self, current_query: str) -> Dict[str, Any]:
        """
        Get relevant context information for interpreting the current query.
        
        Args:
            current_query: The query being processed
            
        Returns:
            Context information to enhance query understanding
        """
        logger.info(f"Gathering context for query: {current_query[:50]}...")
        
        context = {
            'active_context': {},
            'recent_entities': {},
            'table_preferences': {},
            'column_preferences': {},
            'intent_patterns': {},
            'disambiguation_hints': [],
            'query_suggestions': [],
            'context_confidence': 0.0
        }
        
        if not self.query_history:
            logger.info("No query history available - returning empty context")
            return context
        
        # Gather context from recent queries
        recent_queries = list(self.query_history)[-5:]  # Last 5 queries
        
        # 1. Extract recent entities and their relationships
        context['recent_entities'] = self._gather_entity_context(recent_queries, current_query)
        
        # 2. Identify table usage patterns
        context['table_preferences'] = self._analyze_table_preferences(recent_queries)
        
        # 3. Analyze column/role usage patterns
        context['column_preferences'] = self._analyze_column_preferences(recent_queries)
        
        # 4. Detect intent patterns
        context['intent_patterns'] = self._analyze_intent_patterns(recent_queries)
        
        # 5. Generate disambiguation hints
        context['disambiguation_hints'] = self._generate_disambiguation_hints(current_query, recent_queries)
        
        # 6. Suggest query refinements
        context['query_suggestions'] = self._generate_query_suggestions(current_query, recent_queries)
        
        # 7. Calculate overall context confidence
        context['context_confidence'] = self._calculate_context_confidence(context)
        
        logger.info(f"Context gathered with confidence: {context['context_confidence']:.2f}")
        return context
    
    def _extract_context_used(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract what context was used in the query analysis."""
        return {
            'semantic_roles': query_analysis.get('semantic_roles_needed', []),
            'entities': list(query_analysis.get('entities', {}).keys()),
            'intent': query_analysis.get('intent', 'unknown'),
            'complexity': query_analysis.get('complexity_score', 0)
        }
    
    def _extract_entities(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Extract mentioned entities from query analysis."""
        entities = []
        entity_dict = query_analysis.get('entities', {})
        
        for entity_type, entity_list in entity_dict.items():
            for entity in entity_list:
                entities.append(entity.get('text', ''))
        
        return entities
    
    def _update_context_from_query(self, history_entry: Dict[str, Any]) -> None:
        """Update active context based on a completed query."""
        
        # Update entity relationships
        entities = history_entry['entities_mentioned']
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                relationship = {
                    'entity': entity2,
                    'co_occurrence_count': 1,
                    'last_seen': history_entry['timestamp'],
                    'context': history_entry['analysis']['intent']
                }
                
                # Update or add relationship
                existing = next((r for r in self.entity_relationships[entity1] 
                               if r['entity'] == entity2), None)
                if existing:
                    existing['co_occurrence_count'] += 1
                    existing['last_seen'] = history_entry['timestamp']
                else:
                    self.entity_relationships[entity1].append(relationship)
        
        # Update frequent patterns
        intent = history_entry['analysis']['intent']
        entities_key = tuple(sorted(entities))
        pattern_key = f"{intent}:{entities_key}"
        self.frequent_patterns[pattern_key] += 1
    
    def _learn_user_patterns(self, history_entry: Dict[str, Any]) -> None:
        """Learn user preferences and patterns from query history."""
        
        # Learn table preferences
        for table in history_entry['tables_accessed']:
            if 'table_preferences' not in self.user_preferences:
                self.user_preferences['table_preferences'] = defaultdict(int)
            self.user_preferences['table_preferences'][table] += 1
        
        # Learn intent preferences
        intent = history_entry['analysis']['intent']
        if 'intent_preferences' not in self.user_preferences:
            self.user_preferences['intent_preferences'] = defaultdict(int)
        self.user_preferences['intent_preferences'][intent] += 1
        
        # Learn entity patterns
        entities = history_entry['entities_mentioned']
        for entity in entities:
            if 'entity_usage' not in self.user_preferences:
                self.user_preferences['entity_usage'] = defaultdict(int)
            self.user_preferences['entity_usage'][entity] += 1
    
    def _gather_entity_context(self, recent_queries: List[Dict[str, Any]], current_query: str) -> Dict[str, Any]:
        """Gather context about recently mentioned entities."""
        entity_context = defaultdict(list)
        current_query_lower = current_query.lower()
        
        for entry in reversed(recent_queries):  # Most recent first
            age_weight = self._calculate_age_weight(entry['timestamp'])
            
            for entity in entry['entities_mentioned']:
                if entity.lower() in current_query_lower:
                    # This entity is mentioned in current query - high relevance
                    relevance = 0.9 * age_weight
                else:
                    # Check if entity is related to current query entities
                    relevance = 0.3 * age_weight
                
                entity_context[entity].append({
                    'query': entry['original_query'],
                    'intent': entry['analysis']['intent'],
                    'timestamp': entry['timestamp'],
                    'relevance': relevance,
                    'tables_used': entry['tables_accessed']
                })
        
        # Sort by relevance and keep top entries
        for entity in entity_context:
            entity_context[entity].sort(key=lambda x: x['relevance'], reverse=True)
            entity_context[entity] = entity_context[entity][:3]  # Keep top 3
        
        return dict(entity_context)
    
    def _analyze_table_preferences(self, recent_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze which tables the user has been working with recently."""
        table_scores = defaultdict(float)
        
        for entry in recent_queries:
            age_weight = self._calculate_age_weight(entry['timestamp'])
            success_weight = 1.0 if entry['success'] else 0.5
            
            for table in entry['tables_accessed']:
                table_scores[table] += age_weight * success_weight
        
        # Normalize scores
        max_score = max(table_scores.values()) if table_scores else 1.0
        return {table: score / max_score for table, score in table_scores.items()}
    
    def _analyze_column_preferences(self, recent_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze which semantic roles/columns the user queries frequently."""
        role_scores = defaultdict(float)
        
        for entry in recent_queries:
            age_weight = self._calculate_age_weight(entry['timestamp'])
            success_weight = 1.0 if entry['success'] else 0.5
            
            for role in entry['context_used']['semantic_roles']:
                role_scores[role] += age_weight * success_weight
        
        # Normalize scores
        max_score = max(role_scores.values()) if role_scores else 1.0
        return {role: score / max_score for role, score in role_scores.items()}
    
    def _analyze_intent_patterns(self, recent_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in user query intents."""
        intent_analysis = {
            'recent_intents': [],
            'intent_frequency': defaultdict(int),
            'intent_transitions': defaultdict(int),
            'most_common_intent': None
        }
        
        previous_intent = None
        
        for entry in recent_queries:
            intent = entry['analysis']['intent']
            timestamp = entry['timestamp']
            
            intent_analysis['recent_intents'].append({
                'intent': intent,
                'timestamp': timestamp,
                'success': entry['success']
            })
            
            intent_analysis['intent_frequency'][intent] += 1
            
            if previous_intent:
                transition = f"{previous_intent} -> {intent}"
                intent_analysis['intent_transitions'][transition] += 1
            
            previous_intent = intent
        
        # Find most common intent
        if intent_analysis['intent_frequency']:
            intent_analysis['most_common_intent'] = max(
                intent_analysis['intent_frequency'], 
                key=intent_analysis['intent_frequency'].get
            )
        
        return intent_analysis
    
    def _generate_disambiguation_hints(self, current_query: str, recent_queries: List[Dict[str, Any]]) -> List[str]:
        """Generate hints to help disambiguate the current query."""
        hints = []
        current_lower = current_query.lower()
        
        # Check for ambiguous entities
        ambiguous_entities = []
        for entry in recent_queries:
            for entity in entry['entities_mentioned']:
                if entity.lower() in current_lower:
                    # Check if this entity was used in different contexts
                    contexts = set()
                    for e in recent_queries:
                        if entity in e['entities_mentioned']:
                            contexts.add(e['analysis']['intent'])
                    
                    if len(contexts) > 1:
                        ambiguous_entities.append((entity, contexts))
        
        if ambiguous_entities:
            for entity, contexts in ambiguous_entities:
                hint = f"'{entity}' was recently used in {len(contexts)} different contexts: {', '.join(contexts)}"
                hints.append(hint)
        
        # Check for incomplete queries
        if len(current_query.strip().split()) < 3:
            hints.append("Query seems short - consider providing more specific details")
        
        # Check for pronoun references
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them']
        if any(pronoun in current_lower.split() for pronoun in pronouns):
            if recent_queries:
                last_entities = recent_queries[-1]['entities_mentioned']
                if last_entities:
                    hint = f"Pronoun detected - might refer to recent entities: {', '.join(last_entities[:3])}"
                    hints.append(hint)
        
        return hints
    
    def _generate_query_suggestions(self, current_query: str, recent_queries: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions to improve the current query."""
        suggestions = []
        current_lower = current_query.lower()
        
        # Suggest related queries based on recent patterns
        pattern_counter = Counter(self.frequent_patterns)
        for pattern_key, count in pattern_counter.most_common(3):
            if count > 1:  # Only suggest if pattern has occurred multiple times
                intent, entities = pattern_key.split(':', 1)
                if intent not in current_lower:
                    suggestions.append(f"Consider a {intent} query - you've used this pattern {count} times")
        
        # Suggest additional information based on successful recent queries
        successful_queries = [q for q in recent_queries if q['success']]
        if successful_queries:
            common_roles = defaultdict(int)
            for query in successful_queries[-3:]:  # Last 3 successful
                for role in query['context_used']['semantic_roles']:
                    common_roles[role] += 1
            
            if common_roles:
                most_common_role = max(common_roles, key=common_roles.get)
                if most_common_role not in current_lower:
                    suggestions.append(f"You often query {most_common_role} - consider including it")
        
        # Suggest table context
        table_prefs = self._analyze_table_preferences(recent_queries)
        if table_prefs:
            preferred_table = max(table_prefs, key=table_prefs.get)
            suggestions.append(f"You frequently work with {preferred_table} - check if relevant")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _calculate_age_weight(self, timestamp: datetime) -> float:
        """Calculate weight based on how recent the timestamp is."""
        now = datetime.now()
        age = now - timestamp
        
        if age < timedelta(minutes=5):
            return self.decay_factors['immediate']
        elif age < timedelta(minutes=30):
            return self.decay_factors['recent']
        elif age < timedelta(hours=2):
            return self.decay_factors['session']
        else:
            return self.decay_factors['historical']
    
    def _calculate_context_confidence(self, context: Dict[str, Any]) -> float:
        """Calculate overall confidence in the gathered context."""
        confidence_factors = []
        
        # Factor 1: Amount of recent entities
        entity_count = len(context['recent_entities'])
        entity_confidence = min(entity_count / 3, 1.0)  # Up to 3 entities
        confidence_factors.append(entity_confidence * 0.3)
        
        # Factor 2: Table preferences strength
        table_prefs = context['table_preferences']
        if table_prefs:
            max_table_pref = max(table_prefs.values())
            confidence_factors.append(max_table_pref * 0.3)
        else:
            confidence_factors.append(0.0)
        
        # Factor 3: Intent pattern consistency
        intent_patterns = context['intent_patterns']
        if intent_patterns['most_common_intent']:
            intent_frequency = intent_patterns['intent_frequency'][intent_patterns['most_common_intent']]
            intent_confidence = min(intent_frequency / 5, 1.0)  # Up to 5 occurrences
            confidence_factors.append(intent_confidence * 0.2)
        else:
            confidence_factors.append(0.0)
        
        # Factor 4: Query history depth
        history_depth = min(len(self.query_history) / 10, 1.0)  # Up to 10 queries
        confidence_factors.append(history_depth * 0.2)
        
        return sum(confidence_factors)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the current session."""
        session_duration = datetime.now() - self.session_stats['session_start']
        
        success_rate = 0.0
        if self.session_stats['queries_processed'] > 0:
            success_rate = self.session_stats['successful_queries'] / self.session_stats['queries_processed']
        
        # Analyze query types in this session
        intent_distribution = defaultdict(int)
        for entry in self.query_history:
            intent_distribution[entry['analysis']['intent']] += 1
        
        # Find most productive time periods
        query_times = [entry['timestamp'].hour for entry in self.query_history]
        most_active_hour = max(set(query_times), key=query_times.count) if query_times else None
        
        return {
            'session_duration': str(session_duration).split('.')[0],  # Remove microseconds
            'queries_processed': self.session_stats['queries_processed'],
            'successful_queries': self.session_stats['successful_queries'],
            'failed_queries': self.session_stats['failed_queries'],
            'success_rate': f"{success_rate:.1%}",
            'intent_distribution': dict(intent_distribution),
            'most_active_hour': most_active_hour,
            'top_entities': list(self.user_preferences.get('entity_usage', {}).keys())[:5],
            'preferred_tables': list(self.user_preferences.get('table_preferences', {}).keys())[:3]
        }

# Global instance
context_engine = ContextEngine()