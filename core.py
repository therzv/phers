"""
PHERS Core - Streamlined Data Chat System
6-Step Flow: Upload → Profile → AI Clean → Index → Chat → Results
"""

import os
import pandas as pd
import numpy as np
import redis
import mysql.connector
from typing import Dict, List, Any, Optional
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', 3306)),
    'user': os.getenv('MYSQL_USER', 'rizvi'),
    'password': os.getenv('MYSQL_PASSWORD', 'cooln3tt3r'),
    'database': os.getenv('MYSQL_DATABASE', 'phers')
}

REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'), 
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': int(os.getenv('REDIS_DB', 0))
}

UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', './data'))
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Global connections
redis_client = None
mysql_conn = None

def initialize_connections():
    """Initialize Redis and MySQL connections"""
    global redis_client, mysql_conn
    
    try:
        redis_client = redis.Redis(**REDIS_CONFIG, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connected")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        
    try:
        mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
        print("✅ MySQL connected")
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")

def get_dataset_id(filename: str) -> str:
    """Generate unique dataset ID from filename"""
    return hashlib.md5(filename.encode()).hexdigest()[:8]

class DataProfiler:
    """Step 2: Profile data and identify issues"""
    
    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data profile with issues identification"""
        profile = {
            'basic_info': {
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'memory_usage': int(df.memory_usage(deep=True).sum()),
                'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
            },
            'missing_data': {
                'missing_counts': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
                'missing_percentage': {str(k): float(v) for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()}
            },
            'column_issues': {},
            'suggested_fixes': []
        }
        
        # Identify column issues
        for col in df.columns:
            issues = []
            
            # Check for mixed types
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col].dropna())
                    issues.append("Mixed numeric/text data")
                except:
                    pass
            
            # Check for whitespace issues
            if df[col].dtype == 'object':
                has_whitespace = df[col].dropna().str.contains(r'^\s+|\s+$').any()
                if has_whitespace:
                    issues.append("Leading/trailing whitespace")
            
            # Check for inconsistent casing
            if df[col].dtype == 'object':
                unique_values = df[col].dropna().unique()
                if len(unique_values) > 1:
                    lower_unique = set(str(v).lower() for v in unique_values)
                    if len(lower_unique) < len(unique_values):
                        issues.append("Inconsistent casing")
            
            if issues:
                profile['column_issues'][col] = issues
        
        return profile

class AIDataCleaner:
    """Step 3: AI suggests cleaning operations with explanations"""
    
    @staticmethod
    def suggest_cleaning_operations(profile: Dict[str, Any]) -> List[Dict[str, str]]:
        """Use Phi-4 to suggest data cleaning operations"""
        suggestions = []
        
        # Based on profile, create suggestions
        for col, issues in profile['column_issues'].items():
            for issue in issues:
                if issue == "Leading/trailing whitespace":
                    suggestions.append({
                        'column': col,
                        'operation': 'strip_whitespace',
                        'explanation': f"Remove leading/trailing spaces from '{col}' to ensure data consistency",
                        'code': f"df['{col}'] = df['{col}'].str.strip()"
                    })
                elif issue == "Inconsistent casing":
                    suggestions.append({
                        'column': col,
                        'operation': 'normalize_case',
                        'explanation': f"Standardize casing in '{col}' to improve data quality",
                        'code': f"df['{col}'] = df['{col}'].str.title()"
                    })
                elif issue == "Mixed numeric/text data":
                    suggestions.append({
                        'column': col,
                        'operation': 'convert_numeric',
                        'explanation': f"Convert '{col}' to numeric format, handling text values appropriately",
                        'code': f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
                    })
        
        return suggestions

class DataCleaner:
    """Step 4: Clean data and index it"""
    
    @staticmethod
    def apply_cleaning_operations(df: pd.DataFrame, operations: List[Dict[str, str]]) -> pd.DataFrame:
        """Apply suggested cleaning operations"""
        df_clean = df.copy()
        
        for op in operations:
            try:
                if op['operation'] == 'strip_whitespace':
                    col = op['column']
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                elif op['operation'] == 'normalize_case':
                    col = op['column']
                    df_clean[col] = df_clean[col].astype(str).str.title()
                elif op['operation'] == 'convert_numeric':
                    col = op['column']
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                print(f"✅ Applied {op['operation']} to {op['column']}")
            except Exception as e:
                print(f"❌ Failed to apply {op['operation']} to {op['column']}: {e}")
        
        return df_clean
    
    @staticmethod
    def index_data(dataset_id: str, df: pd.DataFrame):
        """Index cleaned data for fast querying"""
        try:
            # Store in Redis for fast access
            if redis_client:
                # Store metadata
                metadata = {
                    'columns': df.columns.tolist(),
                    'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
                    'shape': df.shape,
                    'indexed_at': pd.Timestamp.now().isoformat()
                }
                redis_client.set(f"dataset:{dataset_id}:metadata", json.dumps(metadata))
                
                # Store sample data for quick preview
                sample = df.head(10).to_dict('records')
                redis_client.set(f"dataset:{dataset_id}:sample", json.dumps(sample, default=str))
                
                print(f"✅ Data indexed in Redis: {dataset_id}")
            
            # Store full data in MySQL for persistence
            if mysql_conn:
                cursor = mysql_conn.cursor()
                
                # Create table if not exists
                table_name = f"dataset_{dataset_id}"
                
                # Generate CREATE TABLE statement
                columns_sql = []
                for col in df.columns:
                    col_clean = col.replace(' ', '_').replace('-', '_')
                    if df[col].dtype in ['int64', 'int32']:
                        columns_sql.append(f"`{col_clean}` INT")
                    elif df[col].dtype in ['float64', 'float32']:
                        columns_sql.append(f"`{col_clean}` DECIMAL(15,4)")
                    else:
                        columns_sql.append(f"`{col_clean}` TEXT")
                
                create_sql = f"""
                CREATE TABLE IF NOT EXISTS `{table_name}` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    {', '.join(columns_sql)}
                )
                """
                cursor.execute(create_sql)
                
                # Insert data
                df_clean_cols = df.copy()
                df_clean_cols.columns = [col.replace(' ', '_').replace('-', '_') for col in df_clean_cols.columns]
                
                placeholders = ', '.join(['%s'] * len(df_clean_cols.columns))
                insert_sql = f"INSERT INTO `{table_name}` ({', '.join([f'`{col}`' for col in df_clean_cols.columns])}) VALUES ({placeholders})"
                
                data_to_insert = [tuple(row) for row in df_clean_cols.values]
                cursor.executemany(insert_sql, data_to_insert)
                mysql_conn.commit()
                
                print(f"✅ Data stored in MySQL: {table_name}")
                
        except Exception as e:
            print(f"❌ Indexing failed: {e}")

class ChatSession:
    """Step 5: Manage natural language chat sessions"""
    
    @staticmethod
    def create_session(dataset_id: str) -> str:
        """Create new chat session"""
        session_id = hashlib.md5(f"{dataset_id}_{pd.Timestamp.now()}".encode()).hexdigest()[:12]
        
        if redis_client:
            session_data = {
                'dataset_id': dataset_id,
                'created_at': pd.Timestamp.now().isoformat(),
                'message_count': 0
            }
            redis_client.set(f"session:{session_id}", json.dumps(session_data))
            redis_client.expire(f"session:{session_id}", int(os.getenv('SESSION_TIMEOUT', 3600)))
            
        return session_id
    
    @staticmethod
    def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if redis_client:
            session_data = redis_client.get(f"session:{session_id}")
            if session_data:
                return json.loads(session_data)
        return None
    
    @staticmethod
    def add_message(session_id: str, question: str, response: Dict[str, Any]):
        """Add message to session history"""
        if redis_client:
            message_key = f"session:{session_id}:messages"
            message = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'question': question,
                'response': response
            }
            redis_client.lpush(message_key, json.dumps(message, default=str))
            redis_client.expire(message_key, int(os.getenv('SESSION_TIMEOUT', 3600)))

class NLProcessor:
    """Step 6: Convert natural language to code and return conversational results"""
    
    @staticmethod
    def process_question(dataset_id: str, question: str) -> Dict[str, Any]:
        """Process natural language question using PandasAI + Phi-4"""
        
        try:
            # Get dataset metadata
            if redis_client:
                metadata_str = redis_client.get(f"dataset:{dataset_id}:metadata")
                if not metadata_str:
                    return {"error": "Dataset not found", "success": False}
                
                metadata = json.loads(metadata_str)
                
                # For now, return a structured response
                # TODO: Integrate PandasAI + Phi-4 here
                response = {
                    "success": True,
                    "question": question,
                    "answer": f"I understand you're asking: '{question}'. I have access to a dataset with {len(metadata['columns'])} columns: {', '.join(metadata['columns'][:5])}{'...' if len(metadata['columns']) > 5 else ''}. Let me analyze this for you.",
                    "data": {},
                    "explanation": "This is a conversational response from PHERS AI system.",
                    "dataset_info": {
                        "columns": metadata['columns'],
                        "shape": metadata['shape']
                    }
                }
                
                return response
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
        
        return {"error": "Redis not available", "success": False}

# Initialize connections on import
initialize_connections()