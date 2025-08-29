"""
SQL Security Module

This module handles all SQL-related security operations including:
- Parameterized query generation
- SQL injection prevention
- Safe query execution
- Input validation for SQL operations

Dependencies: sanitization.py
"""

import sqlite3
import pandas as pd
import re
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from sqlalchemy import text, Engine
from sanitization import data_sanitizer

logger = logging.getLogger(__name__)

class SQLSecurityManager:
    """
    Manages all SQL security operations to prevent injection attacks
    and ensure safe database interactions.
    """
    
    def __init__(self):
        # SQL keywords that should never appear in user input
        self.dangerous_keywords = {
            'drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update',
            'exec', 'execute', 'sp_', 'xp_', 'union', 'script', 'declare',
            'cursor', 'backup', 'restore', 'shutdown', 'grant', 'revoke'
        }
        
    def execute_safe_query(self, 
                          query_template: str, 
                          params: Dict[str, Any],
                          connection: Union[sqlite3.Connection, Engine]) -> pd.DataFrame:
        """
        Execute a SQL query safely using parameterized queries.
        
        Args:
            query_template: SQL query template with named parameters
            params: Dictionary of parameters to bind
            connection: Database connection (sqlite3 or SQLAlchemy)
            
        Returns:
            DataFrame with query results
        """
        try:
            # Step 1: Sanitize all parameters
            safe_params = self._sanitize_query_params(params)
            
            # Step 2: Create parameterized query
            safe_sql, param_list = data_sanitizer.create_safe_sql_params(query_template, safe_params)
            
            # Step 3: Validate the final SQL
            self._validate_sql_safety(safe_sql)
            
            # Step 4: Execute based on connection type
            if isinstance(connection, Engine):
                return self._execute_with_sqlalchemy(safe_sql, param_list, connection)
            else:
                return self._execute_with_sqlite(safe_sql, param_list, connection)
                
        except Exception as e:
            logger.error(f"Error executing safe query: {str(e)}")
            logger.error(f"Query template: {query_template}")
            logger.error(f"Parameters: {params}")
            raise Exception(f"Safe query execution failed: {str(e)}")
    
    def _sanitize_query_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize all query parameters before use.
        
        Args:
            params: Raw parameters dictionary
            
        Returns:
            Sanitized parameters dictionary
        """
        safe_params = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Determine input type for appropriate sanitization
                input_type = self._detect_input_type(key, value)
                safe_params[key] = data_sanitizer.sanitize_user_input(value, input_type)
            elif isinstance(value, (int, float)):
                safe_params[key] = value
            elif value is None:
                safe_params[key] = None
            else:
                # Convert other types to string and sanitize
                safe_params[key] = data_sanitizer.sanitize_user_input(str(value))
        
        return safe_params
    
    def _detect_input_type(self, param_name: str, param_value: str) -> str:
        """
        Detect the type of input parameter for appropriate sanitization.
        
        Args:
            param_name: Parameter name
            param_value: Parameter value
            
        Returns:
            Input type string
        """
        param_lower = param_name.lower()
        
        if any(term in param_lower for term in ['asset', 'tag', 'id', 'code']):
            return "asset_tag"
        elif any(term in param_lower for term in ['user', 'name', 'employee']):
            return "username"
        elif any(term in param_lower for term in ['manufacturer', 'brand', 'company']):
            return "manufacturer"
        else:
            return "general"
    
    def _validate_sql_safety(self, sql: str) -> None:
        """
        Validate that SQL query is safe from injection attacks.
        
        Args:
            sql: SQL query string to validate
            
        Raises:
            Exception if unsafe SQL detected
        """
        sql_lower = sql.lower()
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if keyword in sql_lower:
                # Allow SELECT statements with some keywords in specific contexts
                if keyword in ['union', 'insert', 'update', 'delete'] and sql_lower.strip().startswith('select'):
                    # This might be legitimate in subqueries, but log it
                    logger.warning(f"Potentially risky keyword '{keyword}' found in SELECT query")
                elif keyword in ['drop', 'truncate', 'alter', 'create', 'exec', 'execute']:
                    raise Exception(f"Dangerous SQL keyword '{keyword}' detected in query")
        
        # Check for comment patterns that might indicate injection
        if '--' in sql or '/*' in sql or '*/' in sql:
            raise Exception("SQL comments detected - possible injection attempt")
        
        # Check for multiple statements (semicolons outside of strings)
        # This is a simple check - more sophisticated parsing could be added
        semicolon_count = sql.count(';')
        if semicolon_count > 1:
            logger.warning(f"Multiple statements detected in SQL: {semicolon_count} semicolons")
    
    def _execute_with_sqlalchemy(self, sql: str, params: List[Any], engine: Engine) -> pd.DataFrame:
        """
        Execute query using SQLAlchemy engine with parameterized query.
        
        Args:
            sql: Parameterized SQL query
            params: List of parameter values
            engine: SQLAlchemy engine
            
        Returns:
            DataFrame with results
        """
        try:
            with engine.connect() as connection:
                # Use text() for parameterized queries
                query = text(sql)
                result = connection.execute(query, params)
                
                # Convert to DataFrame
                columns = result.keys()
                rows = result.fetchall()
                
                if rows:
                    return pd.DataFrame(rows, columns=columns)
                else:
                    return pd.DataFrame(columns=columns)
                    
        except Exception as e:
            logger.error(f"SQLAlchemy execution error: {str(e)}")
            raise
    
    def _execute_with_sqlite(self, sql: str, params: List[Any], connection: sqlite3.Connection) -> pd.DataFrame:
        """
        Execute query using sqlite3 connection with parameterized query.
        
        Args:
            sql: Parameterized SQL query
            params: List of parameter values  
            connection: sqlite3 connection
            
        Returns:
            DataFrame with results
        """
        try:
            cursor = connection.cursor()
            cursor.execute(sql, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Get all rows
            rows = cursor.fetchall()
            
            if rows and columns:
                return pd.DataFrame(rows, columns=columns)
            else:
                return pd.DataFrame(columns=columns)
                
        except Exception as e:
            logger.error(f"SQLite execution error: {str(e)}")
            raise
    
    def build_safe_select_query(self, 
                               table_name: str,
                               columns: List[str] = None,
                               where_conditions: Dict[str, Any] = None,
                               limit: int = None,
                               order_by: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Build a safe SELECT query with parameterized conditions.
        
        Args:
            table_name: Name of the table to query
            columns: List of column names (None for SELECT *)
            where_conditions: Dictionary of column: value conditions
            limit: Maximum number of rows to return
            order_by: Column name to order by
            
        Returns:
            Tuple of (query_template, parameters_dict)
        """
        # Sanitize table name
        safe_table = self._sanitize_identifier(table_name)
        
        # Build SELECT clause
        if columns:
            safe_columns = [self._sanitize_identifier(col) for col in columns]
            select_clause = f"SELECT {', '.join(safe_columns)}"
        else:
            select_clause = "SELECT *"
        
        # Build FROM clause
        from_clause = f"FROM \"{safe_table}\""
        
        # Build WHERE clause
        where_clause = ""
        where_params = {}
        if where_conditions:
            where_parts = []
            for column, value in where_conditions.items():
                safe_column = self._sanitize_identifier(column)
                param_name = f"param_{len(where_params)}"
                where_parts.append(f"\"{safe_column}\" = :{param_name}")
                where_params[param_name] = value
            
            if where_parts:
                where_clause = f"WHERE {' AND '.join(where_parts)}"
        
        # Build ORDER BY clause
        order_clause = ""
        if order_by:
            safe_order_col = self._sanitize_identifier(order_by)
            order_clause = f"ORDER BY \"{safe_order_col}\""
        
        # Build LIMIT clause
        limit_clause = ""
        if limit and isinstance(limit, int) and limit > 0:
            limit_clause = f"LIMIT {limit}"
        
        # Combine all clauses
        query_parts = [select_clause, from_clause]
        if where_clause:
            query_parts.append(where_clause)
        if order_clause:
            query_parts.append(order_clause)
        if limit_clause:
            query_parts.append(limit_clause)
        
        query_template = " ".join(query_parts)
        
        return query_template, where_params
    
    def _sanitize_identifier(self, identifier: str) -> str:
        """
        Sanitize database identifier (table name, column name) for safe use.
        
        Args:
            identifier: Raw identifier
            
        Returns:
            Sanitized identifier
        """
        if not identifier or not isinstance(identifier, str):
            raise Exception("Invalid identifier provided")
        
        # Remove dangerous characters
        sanitized = re.sub(r'[^\w]', '_', identifier)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'col_' + sanitized
        
        # Ensure it's not empty
        if not sanitized:
            raise Exception("Identifier became empty after sanitization")
        
        return sanitized

# Global instance for easy importing
sql_security = SQLSecurityManager()