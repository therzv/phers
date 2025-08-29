"""
Smart Suggestion Engine for PHERS
Intelligent Query Enhancement with Fuzzy Matching

This module provides intelligent suggestions for failed or ambiguous queries,
including fuzzy matching, typo correction, and contextual recommendations.

Features:
- Fuzzy string matching for typo correction
- Phonetic matching for name variations
- Contextual suggestions based on data patterns
- Query enhancement recommendations
- Similar value discovery with confidence scoring
- Learning from user interactions
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd

# Fuzzy matching libraries
try:
    import jellyfish
    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False

try:
    from difflib import SequenceMatcher, get_close_matches
    DIFFLIB_AVAILABLE = True
except ImportError:
    DIFFLIB_AVAILABLE = False

@dataclass
class SuggestionMatch:
    """Represents a suggested match for user query"""
    original_value: str
    suggested_value: str
    match_type: str                    # fuzzy, phonetic, partial, contextual
    confidence: float                  # 0.0 to 1.0
    reason: str                       # Explanation for the suggestion
    column_name: Optional[str] = None # Which column the match came from
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class QueryEnhancement:
    """Suggestions for improving a query"""
    original_query: str
    enhanced_queries: List[str]       # Alternative query formulations
    missing_filters: List[str]        # Suggested additional filters
    query_type_suggestions: List[str] # Different types of queries to try
    data_exploration_tips: List[str]  # Ways to explore the data
    confidence: float

@dataclass
class SuggestionContext:
    """Context for generating appropriate suggestions"""
    available_columns: List[str]
    column_data_samples: Dict[str, List[Any]]  # Column -> sample values
    data_domain: str                          # hr, sales, inventory, etc.
    user_query_history: List[str]             # Previous successful queries
    common_search_patterns: Dict[str, int]    # Pattern -> frequency
    dataset_size: int
    column_types: Dict[str, str]              # Column -> inferred type

class SmartSuggestionEngine:
    """
    Intelligent suggestion system that provides helpful recommendations
    when queries fail or return unexpected results.
    """
    
    def __init__(self, activity_log_path: str = "data/activity.log"):
        self.activity_log_path = activity_log_path
        self.logger = logging.getLogger(__name__)
        
        # Fuzzy matching configuration
        self.fuzzy_config = {
            'min_similarity_threshold': 0.6,
            'phonetic_threshold': 0.8,
            'partial_match_threshold': 0.7,
            'max_suggestions': 5
        }
        
        # Common name variations and patterns
        self.name_variations = {
            'common_nicknames': {
                'william': ['bill', 'will', 'willy'],
                'robert': ['bob', 'rob', 'bobby'],
                'richard': ['rick', 'rich', 'dick'],
                'michael': ['mike', 'mick'],
                'elizabeth': ['liz', 'beth', 'betty'],
                'jennifer': ['jen', 'jenny'],
                'christopher': ['chris'],
                'matthew': ['matt'],
                'benjamin': ['ben'],
                'alexander': ['alex']
            },
            'title_variations': {
                'mr': ['mister', 'mr.'],
                'mrs': ['mrs.', 'missus'],
                'ms': ['ms.', 'miss'],
                'dr': ['dr.', 'doctor'],
                'prof': ['prof.', 'professor']
            }
        }
        
        # Query pattern templates
        self.query_patterns = {
            'person_search': [
                "Find {name}",
                "Show me {name}",
                "What is {field} of {name}",
                "Where is {name}",
                "{name}'s information"
            ],
            'data_exploration': [
                "Show me all {column} values",
                "List everyone in {department}",
                "Who works in {location}",
                "Find all {category} items"
            ],
            'statistical': [
                "Average {field}",
                "Count of {category}",
                "Total {field} by {group}",
                "Highest {field}",
                "Lowest {field}"
            ]
        }
        
        # Learning storage
        self.interaction_patterns = defaultdict(int)
        self.successful_corrections = defaultdict(list)
        
        self._log_activity("SmartSuggestionEngine initialized", {
            "fuzzy_matching_available": JELLYFISH_AVAILABLE,
            "difflib_available": DIFFLIB_AVAILABLE,
            "suggestion_types": list(self.query_patterns.keys())
        })
    
    def _log_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log activity to the activity log file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "SmartSuggestionEngine",
                "activity": activity,
                "details": details or {}
            }
            
            Path(self.activity_log_path).parent.mkdir(exist_ok=True)
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to log activity: {e}")
    
    def find_similar_values(self, query_term: str, context: SuggestionContext,
                           target_column: Optional[str] = None) -> List[SuggestionMatch]:
        """
        Find similar values for a query term using multiple matching strategies.
        
        Args:
            query_term: The term user searched for
            context: Context about available data
            target_column: Specific column to search in (optional)
            
        Returns:
            List of SuggestionMatch objects with potential matches
        """
        suggestions = []
        query_lower = query_term.lower().strip()
        
        # Determine which columns to search
        columns_to_search = [target_column] if target_column else context.available_columns
        
        for column in columns_to_search:
            if column not in context.column_data_samples:
                continue
            
            column_values = context.column_data_samples[column]
            
            # Strategy 1: Exact substring matching (highest confidence)
            exact_matches = self._find_exact_substring_matches(query_lower, column_values, column)
            suggestions.extend(exact_matches)
            
            # Strategy 2: Fuzzy string matching
            if JELLYFISH_AVAILABLE or DIFFLIB_AVAILABLE:
                fuzzy_matches = self._find_fuzzy_matches(query_lower, column_values, column)
                suggestions.extend(fuzzy_matches)
            
            # Strategy 3: Phonetic matching for names
            if self._is_likely_name_column(column):
                phonetic_matches = self._find_phonetic_matches(query_lower, column_values, column)
                suggestions.extend(phonetic_matches)
            
            # Strategy 4: Nickname and variation matching
            if self._is_likely_name_column(column):
                variation_matches = self._find_name_variation_matches(query_lower, column_values, column)
                suggestions.extend(variation_matches)
            
            # Strategy 5: Partial word matching
            partial_matches = self._find_partial_matches(query_lower, column_values, column)
            suggestions.extend(partial_matches)
        
        # Remove duplicates and sort by confidence
        suggestions = self._deduplicate_suggestions(suggestions)
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to top suggestions
        suggestions = suggestions[:self.fuzzy_config['max_suggestions']]
        
        self._log_activity("Similar values found", {
            "query_term": query_term,
            "target_column": target_column,
            "suggestions_count": len(suggestions),
            "top_confidence": suggestions[0].confidence if suggestions else 0
        })
        
        return suggestions
    
    def generate_query_enhancements(self, original_query: str, 
                                   context: SuggestionContext) -> QueryEnhancement:
        """
        Generate enhanced query suggestions when original query fails or needs improvement.
        
        Args:
            original_query: User's original query
            context: Context about available data
            
        Returns:
            QueryEnhancement with alternative query formulations
        """
        enhanced_queries = []
        missing_filters = []
        query_type_suggestions = []
        data_exploration_tips = []
        
        query_lower = original_query.lower()
        
        # Analyze query intent
        query_intent = self._analyze_query_intent(query_lower)
        
        # Generate enhanced queries based on intent
        if query_intent == 'person_search':
            enhanced_queries.extend(self._generate_person_search_enhancements(query_lower, context))
            data_exploration_tips.extend([
                "Try searching by last name only",
                "Search by department or role instead",
                "Use 'Show me all employees' to browse"
            ])
        
        elif query_intent == 'data_exploration':
            enhanced_queries.extend(self._generate_exploration_enhancements(query_lower, context))
            data_exploration_tips.extend([
                "Try 'Show me all [column] values' to see options",
                "Use 'Count by [column]' for summaries",
                "Ask 'What data do you have?' for overview"
            ])
        
        elif query_intent == 'statistical':
            enhanced_queries.extend(self._generate_statistical_enhancements(query_lower, context))
            query_type_suggestions.extend([
                "Try grouping by department or category",
                "Use 'average', 'total', or 'count' for calculations",
                "Add filters like 'for department Marketing'"
            ])
        
        # Add column-specific suggestions
        relevant_columns = self._identify_relevant_columns(query_lower, context)
        for column in relevant_columns:
            missing_filters.append(f"Filter by {column}")
            enhanced_queries.append(f"{original_query} in {column}")
        
        # Add context-aware suggestions
        if context.data_domain == 'hr':
            data_exploration_tips.extend([
                "Try: 'Who works in [department]?'",
                "Ask: 'Show me recent hires'",
                "Query: 'Average salary by department'"
            ])
        
        # Calculate confidence based on number and quality of suggestions
        confidence = min(0.9, len(enhanced_queries) * 0.15 + len(data_exploration_tips) * 0.05)
        
        enhancement = QueryEnhancement(
            original_query=original_query,
            enhanced_queries=enhanced_queries[:8],  # Limit to 8 suggestions
            missing_filters=missing_filters[:5],
            query_type_suggestions=query_type_suggestions[:5],
            data_exploration_tips=data_exploration_tips[:6],
            confidence=confidence
        )
        
        self._log_activity("Query enhancements generated", {
            "original_query": original_query,
            "enhanced_queries_count": len(enhanced_queries),
            "confidence": confidence,
            "query_intent": query_intent
        })
        
        return enhancement
    
    def learn_from_interaction(self, original_query: str, successful_alternative: str,
                             user_selected: bool = True):
        """
        Learn from user interactions to improve future suggestions.
        
        Args:
            original_query: What user originally asked
            successful_alternative: What actually worked
            user_selected: Whether user explicitly selected this option
        """
        if user_selected:
            # Store successful corrections
            pattern = self._generalize_query_pattern(original_query, successful_alternative)
            self.successful_corrections[pattern].append({
                'original': original_query,
                'successful': successful_alternative,
                'timestamp': datetime.now().isoformat()
            })
            
            # Track interaction patterns
            self.interaction_patterns[pattern] += 1
            
            self._log_activity("Learning from interaction", {
                "original_query": original_query,
                "successful_alternative": successful_alternative,
                "pattern": pattern
            })
    
    def get_contextual_suggestions(self, partial_query: str, 
                                 context: SuggestionContext) -> List[str]:
        """
        Generate contextual autocomplete suggestions as user types.
        
        Args:
            partial_query: Partial query user has typed
            context: Context about available data
            
        Returns:
            List of completion suggestions
        """
        suggestions = []
        partial_lower = partial_query.lower().strip()
        
        # Common query starters
        if len(partial_lower) < 4:
            suggestions.extend([
                "Show me all employees",
                "Who works in",
                "What is the average",
                "Count of",
                "Find"
            ])
            return suggestions
        
        # Column-based suggestions
        for column in context.available_columns:
            column_lower = column.lower()
            
            # If partial matches column name
            if partial_lower in column_lower or column_lower.startswith(partial_lower):
                suggestions.append(f"Show me all {column} values")
                suggestions.append(f"Count by {column}")
                
                # Add value-based suggestions
                if column in context.column_data_samples:
                    sample_values = context.column_data_samples[column][:5]
                    for value in sample_values:
                        if str(value).lower().startswith(partial_lower):
                            suggestions.append(f"Find {value}")
        
        # Pattern-based suggestions from learning
        for pattern, frequency in self.interaction_patterns.most_common(10):
            if partial_lower in pattern.lower():
                suggestions.append(pattern)
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            if suggestion.lower() not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion.lower())
        
        return unique_suggestions[:8]  # Limit to 8 suggestions
    
    def _find_exact_substring_matches(self, query_term: str, values: List[Any], 
                                    column: str) -> List[SuggestionMatch]:
        """Find exact substring matches"""
        matches = []
        
        for value in values:
            value_str = str(value).lower()
            if query_term in value_str:
                confidence = 0.95 if value_str == query_term else 0.85
                matches.append(SuggestionMatch(
                    original_value=query_term,
                    suggested_value=str(value),
                    match_type="exact_substring",
                    confidence=confidence,
                    reason=f"Contains '{query_term}'",
                    column_name=column
                ))
        
        return matches
    
    def _find_fuzzy_matches(self, query_term: str, values: List[Any], 
                          column: str) -> List[SuggestionMatch]:
        """Find fuzzy string matches using available libraries"""
        matches = []
        
        for value in values:
            value_str = str(value).lower()
            
            if JELLYFISH_AVAILABLE:
                # Use Jellyfish for more accurate matching
                distance = jellyfish.levenshtein_distance(query_term, value_str)
                max_len = max(len(query_term), len(value_str))
                similarity = 1 - (distance / max_len) if max_len > 0 else 0
            elif DIFFLIB_AVAILABLE:
                # Fallback to difflib
                similarity = SequenceMatcher(None, query_term, value_str).ratio()
            else:
                continue
            
            if similarity >= self.fuzzy_config['min_similarity_threshold']:
                matches.append(SuggestionMatch(
                    original_value=query_term,
                    suggested_value=str(value),
                    match_type="fuzzy",
                    confidence=similarity * 0.9,  # Slightly lower confidence for fuzzy
                    reason=f"Similar spelling ({similarity:.0%} match)",
                    column_name=column
                ))
        
        return matches
    
    def _find_phonetic_matches(self, query_term: str, values: List[Any], 
                             column: str) -> List[SuggestionMatch]:
        """Find phonetic matches for name-like values"""
        if not JELLYFISH_AVAILABLE:
            return []
        
        matches = []
        query_soundex = jellyfish.soundex(query_term)
        query_metaphone = jellyfish.metaphone(query_term)
        
        for value in values:
            value_str = str(value)
            value_soundex = jellyfish.soundex(value_str)
            value_metaphone = jellyfish.metaphone(value_str)
            
            # Check Soundex match
            if query_soundex == value_soundex and query_soundex != '0000':
                matches.append(SuggestionMatch(
                    original_value=query_term,
                    suggested_value=value_str,
                    match_type="phonetic_soundex",
                    confidence=0.8,
                    reason="Sounds similar (Soundex match)",
                    column_name=column
                ))
            
            # Check Metaphone match
            elif query_metaphone and query_metaphone == value_metaphone:
                matches.append(SuggestionMatch(
                    original_value=query_term,
                    suggested_value=value_str,
                    match_type="phonetic_metaphone",
                    confidence=0.75,
                    reason="Sounds similar (Metaphone match)",
                    column_name=column
                ))
        
        return matches
    
    def _find_name_variation_matches(self, query_term: str, values: List[Any], 
                                   column: str) -> List[SuggestionMatch]:
        """Find matches using common name variations and nicknames"""
        matches = []
        
        # Check if query could be a nickname
        for full_name, nicknames in self.name_variations['common_nicknames'].items():
            if query_term.lower() in nicknames:
                # Look for full name in values
                for value in values:
                    value_str = str(value).lower()
                    if full_name in value_str:
                        matches.append(SuggestionMatch(
                            original_value=query_term,
                            suggested_value=str(value),
                            match_type="nickname_expansion",
                            confidence=0.85,
                            reason=f"'{query_term}' is often short for '{full_name}'",
                            column_name=column
                        ))
        
        # Check reverse: if query is full name, suggest nicknames
        query_term_clean = query_term.lower()
        if query_term_clean in self.name_variations['common_nicknames']:
            nicknames = self.name_variations['common_nicknames'][query_term_clean]
            for value in values:
                value_str = str(value).lower()
                for nickname in nicknames:
                    if nickname in value_str:
                        matches.append(SuggestionMatch(
                            original_value=query_term,
                            suggested_value=str(value),
                            match_type="full_to_nickname",
                            confidence=0.8,
                            reason=f"'{nickname}' is a common nickname for '{query_term}'",
                            column_name=column
                        ))
        
        return matches
    
    def _find_partial_matches(self, query_term: str, values: List[Any], 
                            column: str) -> List[SuggestionMatch]:
        """Find partial word matches"""
        matches = []
        query_words = query_term.split()
        
        if len(query_words) > 1:  # Multi-word query
            for value in values:
                value_str = str(value).lower()
                value_words = value_str.split()
                
                # Check if any query words match value words
                matching_words = sum(1 for qw in query_words 
                                   for vw in value_words if qw in vw or vw in qw)
                
                if matching_words > 0:
                    confidence = (matching_words / len(query_words)) * 0.7
                    if confidence >= self.fuzzy_config['partial_match_threshold']:
                        matches.append(SuggestionMatch(
                            original_value=query_term,
                            suggested_value=str(value),
                            match_type="partial_words",
                            confidence=confidence,
                            reason=f"Matches {matching_words} of {len(query_words)} words",
                            column_name=column
                        ))
        
        return matches
    
    def _deduplicate_suggestions(self, suggestions: List[SuggestionMatch]) -> List[SuggestionMatch]:
        """Remove duplicate suggestions, keeping highest confidence"""
        seen = {}
        
        for suggestion in suggestions:
            key = (suggestion.suggested_value.lower(), suggestion.column_name)
            
            if key not in seen or suggestion.confidence > seen[key].confidence:
                seen[key] = suggestion
        
        return list(seen.values())
    
    def _is_likely_name_column(self, column: str) -> bool:
        """Heuristic to detect if column likely contains names"""
        name_indicators = ['name', 'first', 'last', 'full', 'employee', 'person', 'user', 'customer']
        return any(indicator in column.lower() for indicator in name_indicators)
    
    def _analyze_query_intent(self, query: str) -> str:
        """Analyze what type of query the user is trying to make"""
        query_lower = query.lower()
        
        # Person search patterns
        if any(word in query_lower for word in ['who', 'find', 'show', 'age', 'name']):
            return 'person_search'
        
        # Statistical patterns
        elif any(word in query_lower for word in ['average', 'count', 'total', 'sum', 'max', 'min']):
            return 'statistical'
        
        # Exploration patterns
        elif any(word in query_lower for word in ['all', 'list', 'what', 'how many', 'browse']):
            return 'data_exploration'
        
        return 'general'
    
    def _generate_person_search_enhancements(self, query: str, 
                                           context: SuggestionContext) -> List[str]:
        """Generate enhanced queries for person searches"""
        enhancements = []
        
        # Extract potential name from query
        name_match = re.search(r'\b([a-z]+(?:\s+[a-z]+)*)\b', query)
        if name_match:
            potential_name = name_match.group(1)
            
            enhancements.extend([
                f"Show me {potential_name}",
                f"Find employee {potential_name}",
                f"Who is {potential_name}",
                f"Search for {potential_name}",
                f"{potential_name}'s information"
            ])
            
            # Try variations with different name parts
            name_parts = potential_name.split()
            if len(name_parts) > 1:
                enhancements.extend([
                    f"Find {name_parts[0]}",  # First name only
                    f"Search {name_parts[-1]}",  # Last name only
                ])
        
        return enhancements
    
    def _generate_exploration_enhancements(self, query: str, 
                                         context: SuggestionContext) -> List[str]:
        """Generate enhanced queries for data exploration"""
        enhancements = []
        
        # Suggest column exploration
        for column in context.available_columns[:5]:  # Top 5 columns
            enhancements.extend([
                f"Show me all {column} values",
                f"List unique {column}",
                f"Count by {column}"
            ])
        
        # Domain-specific suggestions
        if context.data_domain == 'hr':
            enhancements.extend([
                "Show all departments",
                "List all employees",
                "Who was hired recently",
                "Show salary ranges"
            ])
        
        return enhancements
    
    def _generate_statistical_enhancements(self, query: str, 
                                         context: SuggestionContext) -> List[str]:
        """Generate enhanced queries for statistical analysis"""
        enhancements = []
        
        # Find numeric columns for statistical operations
        numeric_columns = [col for col, type_hint in context.column_types.items() 
                          if type_hint in ['numeric', 'currency', 'integer']]
        
        # Find categorical columns for grouping
        categorical_columns = [col for col in context.available_columns 
                             if any(term in col.lower() for term in ['dept', 'category', 'type', 'status'])]
        
        # Generate statistical query variations
        for num_col in numeric_columns[:3]:
            enhancements.extend([
                f"Average {num_col}",
                f"Total {num_col}",
                f"Maximum {num_col}",
                f"Minimum {num_col}"
            ])
            
            # Add grouping if categorical columns available
            for cat_col in categorical_columns[:2]:
                enhancements.extend([
                    f"Average {num_col} by {cat_col}",
                    f"Total {num_col} by {cat_col}"
                ])
        
        return enhancements
    
    def _identify_relevant_columns(self, query: str, context: SuggestionContext) -> List[str]:
        """Identify columns that might be relevant to the query"""
        relevant = []
        query_words = set(query.lower().split())
        
        for column in context.available_columns:
            column_words = set(column.lower().replace('_', ' ').split())
            
            # Check for word overlap
            if query_words.intersection(column_words):
                relevant.append(column)
        
        return relevant[:3]  # Limit to 3 most relevant
    
    def _generalize_query_pattern(self, original: str, successful: str) -> str:
        """Create a generalized pattern from successful query corrections"""
        # Simple pattern extraction - replace specific values with placeholders
        pattern = successful.lower()
        
        # Replace potential names with [NAME]
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        pattern = re.sub(name_pattern, '[NAME]', pattern)
        
        # Replace numbers with [NUMBER]
        number_pattern = r'\b\d+\b'
        pattern = re.sub(number_pattern, '[NUMBER]', pattern)
        
        return pattern

# Global instance for easy integration
smart_suggestion_engine = SmartSuggestionEngine()

def find_similar_values(query_term: str, context: SuggestionContext,
                       target_column: Optional[str] = None) -> List[SuggestionMatch]:
    """Convenience function for finding similar values"""
    return smart_suggestion_engine.find_similar_values(query_term, context, target_column)

def generate_query_enhancements(original_query: str, 
                               context: SuggestionContext) -> QueryEnhancement:
    """Convenience function for generating query enhancements"""
    return smart_suggestion_engine.generate_query_enhancements(original_query, context)

def learn_from_interaction(original_query: str, successful_alternative: str, 
                         user_selected: bool = True):
    """Convenience function for learning from interactions"""
    smart_suggestion_engine.learn_from_interaction(original_query, successful_alternative, user_selected)

if __name__ == "__main__":
    # Example usage and testing
    
    # Create sample context
    context = SuggestionContext(
        available_columns=['employee_name', 'age', 'department', 'salary'],
        column_data_samples={
            'employee_name': ['John Smith', 'Jane Doe', 'Robert Johnson', 'Mary Williams', 'Benjamin Harris'],
            'department': ['Marketing', 'Sales', 'Engineering', 'HR'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000]
        },
        data_domain='hr',
        user_query_history=[],
        common_search_patterns={},
        dataset_size=245,
        column_types={'age': 'numeric', 'salary': 'currency', 'employee_name': 'text', 'department': 'category'}
    )
    
    # Test similar value finding
    print("Testing similar values for 'jon smith':")
    similar = find_similar_values('jon smith', context, 'employee_name')
    for match in similar:
        print(f"  {match.suggested_value} ({match.confidence:.2f}) - {match.reason}")
    
    # Test query enhancements
    print("\nTesting query enhancements for 'benjamin howard':")
    enhancements = generate_query_enhancements('benjamin howard', context)
    print("Enhanced queries:")
    for query in enhancements.enhanced_queries:
        print(f"  - {query}")
    
    print("Data exploration tips:")
    for tip in enhancements.data_exploration_tips:
        print(f"  - {tip}")
    
    # Test contextual suggestions
    print("\nTesting contextual suggestions for 'show':")
    suggestions = smart_suggestion_engine.get_contextual_suggestions('show', context)
    for suggestion in suggestions:
        print(f"  - {suggestion}")