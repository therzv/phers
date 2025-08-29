"""
Natural Language Response Generator for PHERS
Phi4 LLM Integration for Conversational Data Interactions

This module transforms technical database responses into natural, conversational
responses using Phi4 LLM integration with intelligent context awareness.

Features:
- Phi4 LLM integration for natural response generation
- Context-aware response templates for different scenarios
- Data-driven response personalization
- Intent-based response tone adjustment
- Conversation memory and follow-up suggestions
- Performance-optimized response caching
"""

import os
import time
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import pandas as pd
from pathlib import Path

# Import core PHERS components
try:
    from core import get_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import Phase 4 performance optimization if available
try:
    from intelligent_cache import get_intelligent_cache
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

class ResponseType(Enum):
    """Types of responses the system can generate"""
    SUCCESS_SINGLE = "success_single"           # Found 1 record
    SUCCESS_MULTIPLE = "success_multiple"       # Found multiple records  
    NO_RESULTS = "no_results"                  # No matches found
    SUGGESTION_NEEDED = "suggestion_needed"    # Similar matches available
    DATA_EXPLORATION = "data_exploration"      # Show available data overview
    ERROR_RECOVERY = "error_recovery"          # Handle query errors gracefully
    CLARIFICATION = "clarification"            # Ask for more specific info
    FOLLOW_UP = "follow_up"                   # Continue conversation

@dataclass
class QueryContext:
    """Context information for generating appropriate responses"""
    original_query: str
    query_intent: str                          # search, count, list, explore, etc.
    data_domain: str                          # hr, sales, inventory, etc.
    user_expertise: str                       # beginner, intermediate, expert
    conversation_history: List[str]           # Previous queries in session
    available_columns: List[str]              # Columns in dataset
    dataset_description: str                  # Brief description of data
    similar_queries: List[str]                # Similar successful queries

@dataclass
class ResponseContext:
    """Context about the query results for response generation"""
    result_count: int
    result_data: List[Dict[str, Any]]         # Actual query results
    execution_time: float
    suggested_alternatives: List[Tuple[str, float]]  # Similar values found
    column_mappings: Dict[str, str]           # Original → cleaned column names
    data_insights: Dict[str, Any]             # Patterns found in results
    confidence_score: float                   # How confident we are in results

@dataclass
class GeneratedResponse:
    """Complete natural language response"""
    response_text: str
    response_type: ResponseType
    suggestions: List[str]                    # Follow-up suggestions
    data_summary: Optional[Dict[str, Any]]    # Structured data summary
    confidence: float
    generation_time: float
    cached: bool

class NaturalResponseGenerator:
    """
    Generates natural, conversational responses using Phi4 LLM integration.
    Transforms technical database results into helpful, human-like responses.
    """
    
    def __init__(self, activity_log_path: str = "data/activity.log"):
        self.activity_log_path = activity_log_path
        self.logger = logging.getLogger(__name__)
        
        # Response templates for different scenarios
        self.response_templates = {
            ResponseType.SUCCESS_SINGLE: [
                "I found {name} in your data! {details} Is this who you were looking for?",
                "Great! I located {name}. {details} Does this help?",
                "Here's what I found for {name}: {details}",
                "Perfect match! {name} - {details}"
            ],
            
            ResponseType.SUCCESS_MULTIPLE: [
                "I found {count} people matching your query. {summary}",
                "Great! Your search returned {count} results. {summary}",
                "Here are the {count} matches I found: {summary}",
                "I discovered {count} records that match. {summary}"
            ],
            
            ResponseType.NO_RESULTS: [
                "I couldn't find an exact match for '{query}' in your data. {suggestions}",
                "No direct matches for '{query}', but I have some suggestions: {suggestions}",
                "'{query}' didn't return any results. Here's what I found instead: {suggestions}",
                "I don't see '{query}' in the data, but these might be what you're looking for: {suggestions}"
            ],
            
            ResponseType.DATA_EXPLORATION: [
                "I'm working with your {domain} database containing {count} records. {overview}",
                "Your {domain} data has {count} entries. {overview}",
                "Here's what I can help you explore in your {domain} dataset: {overview}"
            ]
        }
        
        # Context-aware prompt templates for Phi4
        self.llm_prompts = {
            'natural_response': """
You are a helpful data assistant. Transform this technical query result into a natural, conversational response.

Query: "{query}"
Results: {results}
Context: {context}

Generate a natural response that:
1. Addresses the user's question directly
2. Uses conversational tone, not technical jargon
3. Provides specific, helpful information
4. Offers relevant follow-up suggestions if appropriate
5. Handles typos and similar names gracefully

Response:""",
            
            'suggestion_response': """
The user searched for "{query}" but no exact match was found. 
Similar values found: {similar_values}
Available data: {data_overview}

Generate a helpful response that:
1. Acknowledges the failed search politely
2. Suggests the most relevant alternatives
3. Asks clarifying questions if needed
4. Maintains a helpful, supportive tone

Response:""",
            
            'data_exploration': """
The user asked: "{query}"
Dataset overview: {dataset_info}
Available columns: {columns}
Sample data: {sample_data}

Generate an engaging response that:
1. Explains what data is available
2. Suggests interesting questions they could ask
3. Provides 3-4 specific example queries
4. Uses an encouraging, exploratory tone

Response:"""
        }
        
        # Initialize cache if available
        self.cache = None
        if CACHING_AVAILABLE:
            try:
                self.cache = get_intelligent_cache()
            except Exception as e:
                self.logger.warning(f"Cache initialization failed: {e}")
        
        self._log_activity("NaturalResponseGenerator initialized", {
            "llm_available": LLM_AVAILABLE,
            "caching_available": CACHING_AVAILABLE,
            "response_templates": len(self.response_templates)
        })
    
    def _log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log activity to the activity log file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "NaturalResponseGenerator",
                "activity": activity,
                "details": details or {}
            }
            
            Path(self.activity_log_path).parent.mkdir(exist_ok=True)
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log activity: {e}")
    
    def generate_response(self, query_context: QueryContext, 
                         response_context: ResponseContext) -> GeneratedResponse:
        """
        Generate a natural language response for query results.
        
        Args:
            query_context: Context about the user's query
            response_context: Context about the query results
            
        Returns:
            GeneratedResponse with natural language text and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(query_context, response_context)
        cached_response = None
        
        if self.cache:
            cached_response = self.cache.get(cache_key, category="natural_responses")
            if cached_response:
                self._log_activity("Response cache hit", {"query": query_context.original_query})
                cached_response['cached'] = True
                cached_response['generation_time'] = time.time() - start_time
                return GeneratedResponse(**cached_response)
        
        # Determine response type
        response_type = self._determine_response_type(response_context)
        
        # Generate response based on type and context
        if LLM_AVAILABLE and self._should_use_llm(query_context, response_context):
            response_text = self._generate_llm_response(query_context, response_context, response_type)
        else:
            response_text = self._generate_template_response(query_context, response_context, response_type)
        
        # Generate follow-up suggestions
        suggestions = self._generate_suggestions(query_context, response_context, response_type)
        
        # Create data summary
        data_summary = self._create_data_summary(response_context) if response_context.result_data else None
        
        # Calculate confidence score
        confidence = self._calculate_response_confidence(query_context, response_context, response_type)
        
        generation_time = time.time() - start_time
        
        # Create response object
        generated_response = GeneratedResponse(
            response_text=response_text,
            response_type=response_type,
            suggestions=suggestions,
            data_summary=data_summary,
            confidence=confidence,
            generation_time=generation_time,
            cached=False
        )
        
        # Cache the response
        if self.cache and confidence > 0.8:
            try:
                self.cache.set(cache_key, asdict(generated_response), category="natural_responses")
            except Exception as e:
                self.logger.warning(f"Failed to cache response: {e}")
        
        self._log_activity("Natural response generated", {
            "query": query_context.original_query,
            "response_type": response_type.value,
            "generation_time": generation_time,
            "confidence": confidence,
            "used_llm": LLM_AVAILABLE and self._should_use_llm(query_context, response_context)
        })
        
        return generated_response
    
    def _determine_response_type(self, response_context: ResponseContext) -> ResponseType:
        """Determine the appropriate response type based on results"""
        if response_context.result_count == 0:
            if response_context.suggested_alternatives:
                return ResponseType.SUGGESTION_NEEDED
            else:
                return ResponseType.NO_RESULTS
        elif response_context.result_count == 1:
            return ResponseType.SUCCESS_SINGLE
        else:
            return ResponseType.SUCCESS_MULTIPLE
    
    def _should_use_llm(self, query_context: QueryContext, response_context: ResponseContext) -> bool:
        """Decide whether to use LLM for response generation"""
        if not LLM_AVAILABLE:
            return False
        
        # Use LLM for complex scenarios
        complex_conditions = [
            len(response_context.suggested_alternatives) > 0,  # Need smart suggestions
            response_context.result_count > 10,               # Complex multi-result
            'explore' in query_context.query_intent.lower(),  # Data exploration
            len(query_context.conversation_history) > 2       # Ongoing conversation
        ]
        
        return any(complex_conditions)
    
    def _generate_llm_response(self, query_context: QueryContext, 
                              response_context: ResponseContext, 
                              response_type: ResponseType) -> str:
        """Generate response using Phi4 LLM"""
        try:
            llm = get_llm()
            
            # Choose appropriate prompt template
            if response_type == ResponseType.SUGGESTION_NEEDED:
                prompt_template = self.llm_prompts['suggestion_response']
                prompt = prompt_template.format(
                    query=query_context.original_query,
                    similar_values=[alt[0] for alt in response_context.suggested_alternatives[:5]],
                    data_overview=query_context.dataset_description
                )
            elif 'explore' in query_context.query_intent.lower():
                prompt_template = self.llm_prompts['data_exploration']
                prompt = prompt_template.format(
                    query=query_context.original_query,
                    dataset_info=query_context.dataset_description,
                    columns=query_context.available_columns[:10],
                    sample_data=str(response_context.result_data[:3]) if response_context.result_data else "No sample available"
                )
            else:
                prompt_template = self.llm_prompts['natural_response']
                prompt = prompt_template.format(
                    query=query_context.original_query,
                    results=self._format_results_for_llm(response_context),
                    context=self._format_context_for_llm(query_context)
                )
            
            # Generate response with Phi4
            llm_response = llm.predict(prompt)
            
            # Clean up LLM response
            cleaned_response = self._clean_llm_response(llm_response)
            
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {e}")
            # Fallback to template response
            return self._generate_template_response(query_context, response_context, response_type)
    
    def _generate_template_response(self, query_context: QueryContext,
                                   response_context: ResponseContext,
                                   response_type: ResponseType) -> str:
        """Generate response using templates (fallback when LLM unavailable)"""
        templates = self.response_templates.get(response_type, [])
        if not templates:
            return f"Found {response_context.result_count} results for your query."
        
        # Choose template based on context
        template = templates[0]  # Use first template as default
        
        # Fill template based on response type
        if response_type == ResponseType.SUCCESS_SINGLE:
            if response_context.result_data:
                record = response_context.result_data[0]
                name = self._extract_name_from_record(record)
                details = self._extract_details_from_record(record, query_context)
                
                return template.format(name=name, details=details)
        
        elif response_type == ResponseType.SUCCESS_MULTIPLE:
            summary = self._create_multi_result_summary(response_context, query_context)
            return template.format(count=response_context.result_count, summary=summary)
        
        elif response_type in [ResponseType.NO_RESULTS, ResponseType.SUGGESTION_NEEDED]:
            suggestions = self._format_suggestions_for_template(response_context.suggested_alternatives)
            return template.format(query=query_context.original_query, suggestions=suggestions)
        
        elif response_type == ResponseType.DATA_EXPLORATION:
            overview = self._create_data_overview(query_context)
            return template.format(
                domain=query_context.data_domain,
                count=len(response_context.result_data) if response_context.result_data else 0,
                overview=overview
            )
        
        return f"Found {response_context.result_count} results for your query."
    
    def _extract_name_from_record(self, record: Dict[str, Any]) -> str:
        """Extract a person's name from a database record"""
        # Look for name fields in order of preference
        name_fields = ['name', 'full_name', 'employee_name', 'person_name', 'first_name', 'last_name']
        
        for field in name_fields:
            for key in record.keys():
                if field in key.lower():
                    return str(record[key])
        
        # If no name field, use first non-ID field
        for key, value in record.items():
            if 'id' not in key.lower() and str(value).strip():
                return str(value)
        
        return "this person"
    
    def _extract_details_from_record(self, record: Dict[str, Any], query_context: QueryContext) -> str:
        """Extract relevant details from a record based on query context"""
        details = []
        
        # Extract age if available and query mentions age
        if 'age' in query_context.original_query.lower():
            for key, value in record.items():
                if 'age' in key.lower() and value:
                    details.append(f"age {value}")
        
        # Extract department/role information  
        dept_fields = ['department', 'dept', 'team', 'division']
        for field in dept_fields:
            for key, value in record.items():
                if field in key.lower() and value:
                    details.append(f"works in {value}")
        
        # Extract title/position
        title_fields = ['title', 'position', 'role', 'job']
        for field in title_fields:
            for key, value in record.items():
                if field in key.lower() and value:
                    details.append(f"position: {value}")
        
        # Extract salary if mentioned in query
        if any(term in query_context.original_query.lower() for term in ['salary', 'pay', 'wage']):
            for key, value in record.items():
                if any(term in key.lower() for term in ['salary', 'pay', 'wage']) and value:
                    details.append(f"salary: ${value:,.2f}" if isinstance(value, (int, float)) else f"salary: {value}")
        
        return ', '.join(details) if details else "information available in your database"
    
    def _create_multi_result_summary(self, response_context: ResponseContext, 
                                   query_context: QueryContext) -> str:
        """Create a summary for multiple results"""
        if not response_context.result_data:
            return "Multiple records found."
        
        # Group by common attributes
        summary_parts = []
        
        # If looking for people, group by department
        dept_groups = {}
        for record in response_context.result_data[:10]:  # Limit to first 10
            for key, value in record.items():
                if 'dept' in key.lower() or 'department' in key.lower():
                    if value not in dept_groups:
                        dept_groups[value] = 0
                    dept_groups[value] += 1
        
        if dept_groups:
            summary_parts.append(f"Departments: {', '.join(f'{k} ({v})' for k, v in list(dept_groups.items())[:3])}")
        
        # Add sample names
        names = []
        for record in response_context.result_data[:3]:
            name = self._extract_name_from_record(record)
            if name != "this person":
                names.append(name)
        
        if names:
            summary_parts.append(f"Including: {', '.join(names)}")
            if response_context.result_count > 3:
                summary_parts.append(f"and {response_context.result_count - 3} more")
        
        return '. '.join(summary_parts) if summary_parts else "Here are the details:"
    
    def _format_suggestions_for_template(self, suggested_alternatives: List[Tuple[str, float]]) -> str:
        """Format suggestions for template responses"""
        if not suggested_alternatives:
            return "Let me know if you'd like to explore your data differently."
        
        suggestions = []
        for value, confidence in suggested_alternatives[:3]:
            suggestions.append(f"• {value} (similarity: {confidence:.0%})")
        
        return '\n' + '\n'.join(suggestions) + '\n\nWhich of these matches what you were looking for?'
    
    def _create_data_overview(self, query_context: QueryContext) -> str:
        """Create an overview of available data for exploration"""
        overview_parts = []
        
        # Categorize columns
        people_cols = [col for col in query_context.available_columns if any(term in col.lower() for term in ['name', 'employee', 'person'])]
        work_cols = [col for col in query_context.available_columns if any(term in col.lower() for term in ['dept', 'title', 'position', 'salary'])]
        date_cols = [col for col in query_context.available_columns if any(term in col.lower() for term in ['date', 'hire', 'start'])]
        
        if people_cols:
            overview_parts.append(f"👥 People: {', '.join(people_cols[:3])}")
        if work_cols:
            overview_parts.append(f"💼 Work Info: {', '.join(work_cols[:3])}")  
        if date_cols:
            overview_parts.append(f"📅 Dates: {', '.join(date_cols[:3])}")
        
        overview_parts.append("Try asking: 'Who works in Sales?', 'Show me recent hires', or 'What's the average salary?'")
        
        return '\n' + '\n'.join(overview_parts)
    
    def _generate_suggestions(self, query_context: QueryContext,
                            response_context: ResponseContext,
                            response_type: ResponseType) -> List[str]:
        """Generate follow-up suggestions based on context"""
        suggestions = []
        
        # Suggestions based on response type
        if response_type == ResponseType.SUCCESS_SINGLE:
            suggestions.extend([
                "Show me others in the same department",
                "What's their manager's information?",
                "Find similar employees"
            ])
        
        elif response_type == ResponseType.SUCCESS_MULTIPLE:
            suggestions.extend([
                "Filter by department",
                "Show only recent hires", 
                "Sort by salary"
            ])
        
        elif response_type in [ResponseType.NO_RESULTS, ResponseType.SUGGESTION_NEEDED]:
            suggestions.extend([
                "Show me all available names",
                "Search by department instead",
                "Try a partial name search"
            ])
        
        # Context-aware suggestions
        if 'name' in query_context.original_query.lower():
            suggestions.append("Search by employee ID instead")
        
        if query_context.data_domain == 'hr':
            suggestions.extend([
                "Show department breakdown",
                "Find employees by hire date",
                "Compare salary ranges"
            ])
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _create_data_summary(self, response_context: ResponseContext) -> Optional[Dict[str, Any]]:
        """Create structured summary of the data"""
        if not response_context.result_data:
            return None
        
        summary = {
            'total_records': response_context.result_count,
            'sample_records': response_context.result_data[:3],
            'columns_shown': list(response_context.result_data[0].keys()) if response_context.result_data else [],
            'execution_time': response_context.execution_time
        }
        
        return summary
    
    def _calculate_response_confidence(self, query_context: QueryContext,
                                     response_context: ResponseContext,
                                     response_type: ResponseType) -> float:
        """Calculate confidence score for the generated response"""
        confidence = 0.8  # Base confidence
        
        # Adjust based on result count
        if response_context.result_count == 1:
            confidence += 0.1  # Single result is more confident
        elif response_context.result_count == 0:
            confidence -= 0.2  # No results is less confident
        
        # Adjust based on execution time (faster = more confident)
        if response_context.execution_time < 1.0:
            confidence += 0.05
        
        # Adjust based on suggestions available
        if response_context.suggested_alternatives:
            confidence += 0.05
        
        # Adjust based on LLM usage
        if LLM_AVAILABLE and self._should_use_llm(query_context, response_context):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _format_results_for_llm(self, response_context: ResponseContext) -> str:
        """Format results for LLM processing"""
        if not response_context.result_data:
            return "No results found"
        
        if response_context.result_count == 1:
            return f"Single record: {response_context.result_data[0]}"
        else:
            return f"{response_context.result_count} records, sample: {response_context.result_data[:3]}"
    
    def _format_context_for_llm(self, query_context: QueryContext) -> str:
        """Format query context for LLM processing"""
        context_parts = [
            f"Intent: {query_context.query_intent}",
            f"Domain: {query_context.data_domain}",
            f"Available columns: {', '.join(query_context.available_columns[:10])}"
        ]
        
        if query_context.conversation_history:
            context_parts.append(f"Previous queries: {query_context.conversation_history[-3:]}")
        
        return "; ".join(context_parts)
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean and format LLM response"""
        # Remove common LLM artifacts
        cleaned = response.strip()
        
        # Remove "Response:" prefix if present
        if cleaned.startswith("Response:"):
            cleaned = cleaned[9:].strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r'  +', ' ', cleaned)
        
        # Ensure proper ending punctuation
        if cleaned and not cleaned[-1] in '.!?':
            cleaned += '.'
        
        return cleaned
    
    def _generate_cache_key(self, query_context: QueryContext, 
                          response_context: ResponseContext) -> str:
        """Generate cache key for response caching"""
        # Create unique key based on query and result characteristics
        key_parts = [
            query_context.original_query.lower().strip(),
            str(response_context.result_count),
            str(len(response_context.suggested_alternatives)),
            query_context.data_domain,
            str(hash(tuple(query_context.available_columns)))
        ]
        
        key_string = "|".join(key_parts)
        return f"nlr_{abs(hash(key_string))}"

# Global instance for easy integration
natural_response_generator = NaturalResponseGenerator()

def generate_natural_response(query_context: QueryContext, 
                            response_context: ResponseContext) -> GeneratedResponse:
    """Convenience function for generating natural responses"""
    return natural_response_generator.generate_response(query_context, response_context)

if __name__ == "__main__":
    # Example usage and testing
    
    # Create sample query context
    query_context = QueryContext(
        original_query="what is the age of john smith",
        query_intent="search",
        data_domain="hr",
        user_expertise="beginner",
        conversation_history=[],
        available_columns=["employee_name", "age", "department", "salary", "hire_date"],
        dataset_description="Employee database with 245 records",
        similar_queries=[]
    )
    
    # Test successful single result
    response_context = ResponseContext(
        result_count=1,
        result_data=[{"employee_name": "John Smith", "age": 34, "department": "Marketing", "salary": 65000}],
        execution_time=0.15,
        suggested_alternatives=[],
        column_mappings={"employee_name": "Employee Name"},
        data_insights={},
        confidence_score=0.95
    )
    
    response = generate_natural_response(query_context, response_context)
    print("Single Result Response:")
    print(response.response_text)
    print(f"Confidence: {response.confidence}")
    print(f"Suggestions: {response.suggestions}")
    
    # Test no results with suggestions
    response_context_no_results = ResponseContext(
        result_count=0,
        result_data=[],
        execution_time=0.08,
        suggested_alternatives=[("John Johnson", 0.85), ("Jane Smith", 0.78), ("John Williams", 0.72)],
        column_mappings={},
        data_insights={},
        confidence_score=0.6
    )
    
    query_context.original_query = "benjamin howard"
    response = generate_natural_response(query_context, response_context_no_results)
    print("\nNo Results with Suggestions:")
    print(response.response_text)
    print(f"Suggestions: {response.suggestions}")