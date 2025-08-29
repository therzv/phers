"""
Conversation Manager Module

Manages conversational query enhancement and multi-turn interactions.
Provides natural follow-up handling and context-aware query continuations.

Features:
- Multi-turn conversation handling
- Follow-up query interpretation
- Context-aware query refinement
- Natural language interaction flow
- Conversation state management
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import re

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages conversational interactions and multi-turn query sessions.
    Enhances queries based on conversation flow and context.
    """
    
    def __init__(self, max_conversation_length: int = 20):
        # Active conversation tracking
        self.current_conversation = deque(maxlen=max_conversation_length)
        self.conversation_context = {
            'active_topic': None,
            'referenced_entities': {},
            'current_table_focus': None,
            'query_chain_depth': 0,
            'last_successful_query': None,
            'pending_clarifications': []
        }
        
        # Conversation patterns for follow-up detection
        self.followup_patterns = {
            'pronoun_reference': {
                'patterns': [r'\b(it|this|that|these|those|they|them)\b'],
                'handler': 'resolve_pronoun_reference',
                'confidence': 0.8
            },
            'quantity_followup': {
                'patterns': [r'\bhow many\b', r'\bcount\b', r'\btotal\b'],
                'handler': 'convert_to_count_query',
                'confidence': 0.9
            },
            'detail_expansion': {
                'patterns': [r'\bmore details?\b', r'\bshow me more\b', r'\bexpand\b'],
                'handler': 'expand_previous_query',
                'confidence': 0.8
            },
            'comparison_followup': {
                'patterns': [r'\bcompare (with|to)\b', r'\bversus\b', r'\bvs\b'],
                'handler': 'create_comparison_query',
                'confidence': 0.9
            },
            'filter_refinement': {
                'patterns': [r'\bbut only\b', r'\bexcept\b', r'\bwithout\b', r'\bfilter\b'],
                'handler': 'refine_with_additional_filters',
                'confidence': 0.8
            },
            'related_question': {
                'patterns': [r'\bwhat about\b', r'\bhow about\b', r'\band\b.*\?'],
                'handler': 'handle_related_question',
                'confidence': 0.7
            },
            'clarification_response': {
                'patterns': [r'\byes\b', r'\bno\b', r'\bcorrect\b', r'\bwrong\b'],
                'handler': 'process_clarification_response',
                'confidence': 0.6
            }
        }
        
        # Context resolution strategies
        self.context_resolvers = {
            'pronoun_resolution': {
                'it': lambda ctx: ctx.get('last_mentioned_entity'),
                'this': lambda ctx: ctx.get('current_focus_entity'), 
                'that': lambda ctx: ctx.get('previous_focus_entity'),
                'they': lambda ctx: ctx.get('entity_group', []),
                'them': lambda ctx: ctx.get('entity_group', [])
            },
            'implicit_subject': {
                'handler': self._resolve_implicit_subject,
                'confidence_threshold': 0.7
            }
        }
        
        # Query enhancement templates
        self.enhancement_templates = {
            'add_context_filter': "SELECT * FROM {table} WHERE {original_conditions} AND {context_conditions}",
            'expand_columns': "SELECT {expanded_columns} FROM {table} WHERE {conditions}",
            'add_ordering': "{original_query} ORDER BY {context_ordering}",
            'add_grouping': "SELECT {group_columns}, COUNT(*) as count FROM {table} WHERE {conditions} GROUP BY {group_columns}",
            'create_comparison': "SELECT {entity1}, {metric} as {entity1}_value UNION SELECT {entity2}, {metric} as {entity2}_value FROM {table} WHERE {conditions}"
        }
    
    def process_conversational_query(self, query: str, query_analysis: Dict[str, Any],
                                   previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a query in conversational context with follow-up handling.
        
        Args:
            query: User's current query
            query_analysis: Analysis from query intelligence
            previous_results: Results from previous query if any
            
        Returns:
            Enhanced query processing results
        """
        logger.info(f"Processing conversational query: {query[:50]}...")
        
        result = {
            'original_query': query,
            'enhanced_query': query,
            'conversation_type': 'initial',
            'followup_detected': False,
            'context_applied': False,
            'enhancements': [],
            'clarification_needed': False,
            'suggested_followups': [],
            'conversation_state': 'active'
        }
        
        # Check if this is a follow-up query
        followup_analysis = self._detect_followup_pattern(query)
        
        if followup_analysis['is_followup']:
            result['followup_detected'] = True
            result['conversation_type'] = followup_analysis['type']
            
            # Process the follow-up
            enhanced_result = self._handle_followup_query(
                query, followup_analysis, query_analysis, previous_results
            )
            
            result.update(enhanced_result)
        
        # Apply general conversation context
        context_enhancements = self._apply_conversation_context(query, query_analysis)
        if context_enhancements:
            result['context_applied'] = True
            result['enhancements'].extend(context_enhancements)
        
        # Update conversation state
        self._update_conversation_state(query, query_analysis, result)
        
        # Generate suggested follow-ups
        result['suggested_followups'] = self._generate_suggested_followups(
            query, query_analysis, result
        )
        
        # Check if clarification is needed
        clarification = self._check_clarification_needed(query, query_analysis)
        if clarification:
            result['clarification_needed'] = True
            result['clarification_questions'] = clarification
        
        logger.info(f"Conversation processing complete: {result['conversation_type']}")
        return result
    
    def _detect_followup_pattern(self, query: str) -> Dict[str, Any]:
        """Detect if query is a follow-up and classify the type."""
        
        analysis = {
            'is_followup': False,
            'type': 'initial',
            'confidence': 0.0,
            'matched_patterns': [],
            'handler': None
        }
        
        query_lower = query.lower().strip()
        
        # Check against follow-up patterns
        for followup_type, pattern_info in self.followup_patterns.items():
            for pattern in pattern_info['patterns']:
                if re.search(pattern, query_lower):
                    analysis['is_followup'] = True
                    analysis['type'] = followup_type
                    analysis['confidence'] = pattern_info['confidence']
                    analysis['matched_patterns'].append(pattern)
                    analysis['handler'] = pattern_info['handler']
                    break
            
            if analysis['is_followup']:
                break
        
        # Additional heuristics for follow-up detection
        if not analysis['is_followup']:
            # Short queries are often follow-ups
            if len(query.split()) < 4 and self.current_conversation:
                analysis['is_followup'] = True
                analysis['type'] = 'implicit_followup'
                analysis['confidence'] = 0.6
            
            # Questions starting with "and" or "also"
            if re.match(r'^(and|also|what about|how about)', query_lower):
                analysis['is_followup'] = True
                analysis['type'] = 'related_question'
                analysis['confidence'] = 0.7
        
        return analysis
    
    def _handle_followup_query(self, query: str, followup_analysis: Dict[str, Any],
                              query_analysis: Dict[str, Any],
                              previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle specific types of follow-up queries."""
        
        result = {
            'enhanced_query': query,
            'enhancements': [],
            'processing_notes': []
        }
        
        handler_name = followup_analysis.get('handler')
        if not handler_name:
            return result
        
        # Call appropriate handler
        try:
            if hasattr(self, handler_name):
                handler_method = getattr(self, handler_name)
                enhancement_result = handler_method(query, query_analysis, previous_results)
                result.update(enhancement_result)
            else:
                logger.warning(f"Handler {handler_name} not found")
                result['processing_notes'].append(f"Handler {handler_name} not implemented")
        
        except Exception as e:
            logger.error(f"Error in followup handler {handler_name}: {e}")
            result['processing_notes'].append(f"Error processing followup: {str(e)}")
        
        return result
    
    def resolve_pronoun_reference(self, query: str, query_analysis: Dict[str, Any],
                                 previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve pronoun references in the query."""
        
        result = {
            'enhanced_query': query,
            'enhancements': ['pronoun_resolution'],
            'processing_notes': []
        }
        
        # Find pronouns in the query
        pronouns = re.findall(r'\b(it|this|that|these|those|they|them)\b', query.lower())
        
        if not pronouns:
            return result
        
        enhanced_query = query
        
        for pronoun in pronouns:
            resolution = self._resolve_pronoun(pronoun)
            if resolution:
                # Replace pronoun with resolved entity
                enhanced_query = re.sub(
                    rf'\b{pronoun}\b', 
                    resolution, 
                    enhanced_query, 
                    flags=re.IGNORECASE,
                    count=1
                )
                result['processing_notes'].append(f"Resolved '{pronoun}' to '{resolution}'")
        
        result['enhanced_query'] = enhanced_query
        return result
    
    def convert_to_count_query(self, query: str, query_analysis: Dict[str, Any],
                              previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert a previous query to a count-based query."""
        
        result = {
            'enhanced_query': query,
            'enhancements': ['count_conversion'],
            'processing_notes': []
        }
        
        # Get the last successful query from conversation
        last_query_info = self._get_last_successful_query()
        
        if last_query_info and last_query_info.get('sql'):
            original_sql = last_query_info['sql']
            
            # Convert SELECT ... to SELECT COUNT(*)
            count_sql = re.sub(
                r'^SELECT\s+.*?\s+FROM', 
                'SELECT COUNT(*) as total_count FROM',
                original_sql,
                flags=re.IGNORECASE
            )
            
            # Remove ORDER BY, LIMIT for count query
            count_sql = re.sub(r'\s+ORDER\s+BY\s+[^;]+', '', count_sql, flags=re.IGNORECASE)
            count_sql = re.sub(r'\s+LIMIT\s+\d+', '', count_sql, flags=re.IGNORECASE)
            
            result['enhanced_query'] = f"Count query: {query}"
            result['suggested_sql'] = count_sql
            result['processing_notes'].append("Converted previous query to count query")
        else:
            result['processing_notes'].append("No previous query found to convert")
        
        return result
    
    def expand_previous_query(self, query: str, query_analysis: Dict[str, Any],
                             previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Expand the previous query to show more details."""
        
        result = {
            'enhanced_query': query,
            'enhancements': ['query_expansion'],
            'processing_notes': []
        }
        
        last_query_info = self._get_last_successful_query()
        
        if last_query_info and last_query_info.get('sql'):
            original_sql = last_query_info['sql']
            
            # Remove LIMIT or increase it
            if 'LIMIT' in original_sql.upper():
                expanded_sql = re.sub(r'LIMIT\s+\d+', 'LIMIT 50', original_sql, flags=re.IGNORECASE)
            else:
                expanded_sql = original_sql.rstrip(';') + ' LIMIT 50'
            
            # If it was SELECT *, keep it; if specific columns, expand to *
            if 'SELECT *' not in original_sql.upper():
                # Try to expand to SELECT *
                table_match = re.search(r'FROM\s+(\w+)', original_sql, re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1)
                    expanded_sql = re.sub(
                        r'SELECT\s+.*?\s+FROM',
                        f'SELECT * FROM',
                        expanded_sql,
                        flags=re.IGNORECASE
                    )
            
            result['suggested_sql'] = expanded_sql
            result['processing_notes'].append("Expanded previous query to show more details")
        else:
            result['processing_notes'].append("No previous query found to expand")
        
        return result
    
    def create_comparison_query(self, query: str, query_analysis: Dict[str, Any],
                               previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a comparison query based on current context."""
        
        result = {
            'enhanced_query': query,
            'enhancements': ['comparison_creation'],
            'processing_notes': []
        }
        
        # Extract what to compare from the query
        compare_match = re.search(r'compare\s+(.*?)\s+(?:with|to)\s+(.*?)(?:\s|$)', query, re.IGNORECASE)
        
        if compare_match:
            entity1 = compare_match.group(1).strip()
            entity2 = compare_match.group(2).strip()
            
            # Use context to build comparison query
            context_table = self.conversation_context.get('current_table_focus')
            if context_table:
                # Create a basic comparison structure
                enhanced_query = f"Compare {entity1} and {entity2} from {context_table}"
                result['enhanced_query'] = enhanced_query
                result['processing_notes'].append(f"Created comparison between '{entity1}' and '{entity2}'")
            else:
                result['processing_notes'].append("No table context available for comparison")
        else:
            result['processing_notes'].append("Could not extract comparison entities from query")
        
        return result
    
    def refine_with_additional_filters(self, query: str, query_analysis: Dict[str, Any],
                                      previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Add additional filters to the previous query."""
        
        result = {
            'enhanced_query': query,
            'enhancements': ['filter_refinement'],
            'processing_notes': []
        }
        
        last_query_info = self._get_last_successful_query()
        
        if last_query_info and last_query_info.get('sql'):
            original_sql = last_query_info['sql']
            
            # Extract the additional filter from current query
            filter_patterns = [
                r'but only\s+(.*?)(?:\s|$)',
                r'except\s+(.*?)(?:\s|$)',
                r'without\s+(.*?)(?:\s|$)'
            ]
            
            additional_filter = None
            for pattern in filter_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    additional_filter = match.group(1)
                    break
            
            if additional_filter:
                # Add the filter to the WHERE clause
                if 'WHERE' in original_sql.upper():
                    refined_sql = original_sql.replace('WHERE', f"WHERE {additional_filter} AND")
                else:
                    refined_sql = original_sql.rstrip(';') + f" WHERE {additional_filter}"
                
                result['suggested_sql'] = refined_sql
                result['processing_notes'].append(f"Added filter: {additional_filter}")
            else:
                result['processing_notes'].append("Could not extract additional filter from query")
        else:
            result['processing_notes'].append("No previous query found to refine")
        
        return result
    
    def handle_related_question(self, query: str, query_analysis: Dict[str, Any],
                               previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle related questions that build on previous context."""
        
        result = {
            'enhanced_query': query,
            'enhancements': ['related_question_handling'],
            'processing_notes': []
        }
        
        # Extract the new subject from "what about X" or "how about X"
        related_patterns = [
            r'what about\s+(.*?)(?:\s|$|\?)',
            r'how about\s+(.*?)(?:\s|$|\?)',
            r'and\s+(.*?)(?:\s|$|\?)'
        ]
        
        new_subject = None
        for pattern in related_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                new_subject = match.group(1).strip()
                break
        
        if new_subject:
            # Use the same query structure but with new subject
            last_query_info = self._get_last_successful_query()
            
            if last_query_info:
                # Try to substitute the new subject
                enhanced_query = f"Show me information about {new_subject}"
                result['enhanced_query'] = enhanced_query
                result['processing_notes'].append(f"Created related query for: {new_subject}")
            else:
                result['processing_notes'].append("No previous query context for related question")
        else:
            result['processing_notes'].append("Could not extract subject from related question")
        
        return result
    
    def process_clarification_response(self, query: str, query_analysis: Dict[str, Any],
                                     previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process yes/no responses to clarification questions."""
        
        result = {
            'enhanced_query': query,
            'enhancements': ['clarification_processing'],
            'processing_notes': []
        }
        
        response_lower = query.lower().strip()
        
        if any(word in response_lower for word in ['yes', 'correct', 'right', 'exactly']):
            # User confirmed - proceed with pending action
            result['user_confirmation'] = True
            result['processing_notes'].append("User confirmed previous suggestion")
        
        elif any(word in response_lower for word in ['no', 'wrong', 'incorrect', 'not']):
            # User disagreed - need alternative approach
            result['user_confirmation'] = False
            result['processing_notes'].append("User rejected previous suggestion - need alternative")
        
        # Clear pending clarifications
        self.conversation_context['pending_clarifications'] = []
        
        return result
    
    def _resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve a pronoun to its referenced entity."""
        
        resolver = self.context_resolvers['pronoun_resolution'].get(pronoun.lower())
        if resolver:
            return resolver(self.conversation_context)
        
        return None
    
    def _resolve_implicit_subject(self, query: str) -> Optional[str]:
        """Resolve implicit subject based on conversation context."""
        
        # If there's an active topic, use it
        active_topic = self.conversation_context.get('active_topic')
        if active_topic:
            return active_topic
        
        # Look for the most recently mentioned entity
        recent_entities = self.conversation_context.get('referenced_entities', {})
        if recent_entities:
            # Return most recent entity
            sorted_entities = sorted(recent_entities.items(), key=lambda x: x[1], reverse=True)
            return sorted_entities[0][0]
        
        return None
    
    def _apply_conversation_context(self, query: str, query_analysis: Dict[str, Any]) -> List[str]:
        """Apply general conversation context to enhance the query."""
        
        enhancements = []
        
        # Add table focus if not specified
        if not self._has_table_reference(query) and self.conversation_context.get('current_table_focus'):
            enhancements.append('added_table_context')
        
        # Add entity context if relevant
        query_entities = query_analysis.get('entities', {})
        if not query_entities and self.conversation_context.get('referenced_entities'):
            enhancements.append('added_entity_context')
        
        return enhancements
    
    def _update_conversation_state(self, query: str, query_analysis: Dict[str, Any], 
                                  result: Dict[str, Any]) -> None:
        """Update the conversation state with new information."""
        
        # Add to conversation history
        conversation_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'analysis': query_analysis,
            'result': result
        }
        self.current_conversation.append(conversation_entry)
        
        # Update context
        self.conversation_context['query_chain_depth'] += 1
        
        # Update entity references
        entities = query_analysis.get('entities', {})
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_text = entity.get('text', '')
                if entity_text:
                    self.conversation_context['referenced_entities'][entity_text] = datetime.now()
        
        # Update active topic
        if entities:
            # Set active topic to most confident entity
            highest_confidence = 0
            best_entity = None
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.get('confidence', 0) > highest_confidence:
                        highest_confidence = entity.get('confidence', 0)
                        best_entity = entity.get('text')
            
            if best_entity:
                self.conversation_context['active_topic'] = best_entity
    
    def _generate_suggested_followups(self, query: str, query_analysis: Dict[str, Any],
                                    result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggested follow-up questions."""
        
        suggestions = []
        
        # Based on query intent
        intent = query_analysis.get('intent', 'search')
        
        if intent == 'search':
            suggestions.extend([
                {'text': 'How many results are there?', 'type': 'count'},
                {'text': 'Show me more details', 'type': 'expansion'},
                {'text': 'What about related items?', 'type': 'related'}
            ])
        
        elif intent == 'count':
            suggestions.extend([
                {'text': 'Show me the actual records', 'type': 'details'},
                {'text': 'Break down by category', 'type': 'grouping'}
            ])
        
        # Based on entities in query
        entities = query_analysis.get('entities', {})
        if 'company' in entities:
            suggestions.append({'text': 'Compare with other manufacturers', 'type': 'comparison'})
        
        if 'location' in entities:
            suggestions.append({'text': 'What about other locations?', 'type': 'related'})
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _check_clarification_needed(self, query: str, query_analysis: Dict[str, Any]) -> Optional[List[str]]:
        """Check if clarification is needed for the query."""
        
        clarifications = []
        
        # Low confidence queries need clarification
        if query_analysis.get('confidence', 1.0) < 0.5:
            clarifications.append(f"I'm not sure about your query '{query}'. Could you rephrase it?")
        
        # Ambiguous entities
        entities = query_analysis.get('entities', {})
        for entity_type, entity_list in entities.items():
            if len(entity_list) > 1:
                entity_texts = [e.get('text', '') for e in entity_list]
                clarifications.append(f"Which {entity_type} do you mean: {', '.join(entity_texts)}?")
        
        # Missing context for pronouns
        if re.search(r'\b(it|this|that)\b', query.lower()) and not self.conversation_context.get('active_topic'):
            clarifications.append("What does 'it/this/that' refer to?")
        
        return clarifications if clarifications else None
    
    def _get_last_successful_query(self) -> Optional[Dict[str, Any]]:
        """Get the last successful query from conversation history."""
        
        for entry in reversed(self.current_conversation):
            if entry.get('result', {}).get('success', False):
                return entry
        
        return self.conversation_context.get('last_successful_query')
    
    def _has_table_reference(self, query: str) -> bool:
        """Check if query has explicit table reference."""
        
        table_keywords = ['from', 'table', 'in the']
        query_lower = query.lower()
        
        return any(keyword in query_lower for keyword in table_keywords)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation state."""
        
        return {
            'conversation_length': len(self.current_conversation),
            'query_chain_depth': self.conversation_context['query_chain_depth'],
            'active_topic': self.conversation_context.get('active_topic'),
            'current_table_focus': self.conversation_context.get('current_table_focus'),
            'recent_entities': list(self.conversation_context.get('referenced_entities', {}).keys()),
            'pending_clarifications': self.conversation_context.get('pending_clarifications', []),
            'conversation_duration': (
                datetime.now() - self.current_conversation[0]['timestamp']
            ).total_seconds() if self.current_conversation else 0
        }
    
    def reset_conversation(self) -> None:
        """Reset conversation state for a new session."""
        
        self.current_conversation.clear()
        self.conversation_context = {
            'active_topic': None,
            'referenced_entities': {},
            'current_table_focus': None,
            'query_chain_depth': 0,
            'last_successful_query': None,
            'pending_clarifications': []
        }
        
        logger.info("Conversation state reset for new session")

# Global instance
conversation_manager = ConversationManager()