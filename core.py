"""Core helpers and global state for the HR Data Chat app.
This module holds DB connection, table state, indexing, LLM helper, and utility functions.
"""
from typing import Dict, Any, List, Optional
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd
import os
import io
import re
import json
import threading
import datetime
import math
import difflib
from fastapi import HTTPException  # lightweight dependency used for validation errors

# Optional pandas-ai integration (fast prototype mode) — lazy import in routes to avoid hard dependency at module import
PANDAS_AI_AVAILABLE = False
PANDASAI_LANGCHAIN_AVAILABLE = False

# sqlparse is used for SQL validation (separate from optional chroma)
SQLPARSE_AVAILABLE = False
try:
    import sqlparse
    SQLPARSE_AVAILABLE = True
except Exception:
    sqlparse = None
    SQLPARSE_AVAILABLE = False

# Optional RAG (Chroma) support
CHROMA_AVAILABLE = False
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # use local persistent chroma (filesystem) by default
    try:
        chroma_client = chromadb.Client(settings={"chroma_db_impl": "duckdb+parquet", "persist_directory": os.path.join(os.getcwd(), "data", "chroma")})
    except Exception:
        chroma_client = chromadb.Client()
    EMB_MODEL_NAME = os.environ.get("EMB_MODEL", "all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer(EMB_MODEL_NAME)
    chroma_collection = chroma_client.get_or_create_collection("hr_docs")
    CHROMA_IDS_BY_FILE: Dict[str, list] = {}
    CHROMA_AVAILABLE = True
    print("Chroma + embedding model available; RAG enabled.")
except Exception as e:
    chroma_client = None
    chroma_collection = None
    embedding_model = None
    CHROMA_IDS_BY_FILE = {}
    CHROMA_AVAILABLE = False
    print("Chroma or embedding model not available (optional). Install chromadb + sentence-transformers to enable RAG.", e)

# indexing status per filename
INDEXING_STATUS: Dict[str, Dict[str, Any]] = {}

# Database and table metadata
DATA_DIR = "data"
DB_PATH = os.environ.get("DB_PATH") or os.environ.get("DATABASE_URL")
ENGINE: Optional[Engine] = None

# Prefer a DATABASE_URL or explicit MySQL env vars if provided, otherwise default to a local sqlite file
def _init_engine():
    global ENGINE
    if ENGINE:
        return ENGINE
    # If DATABASE_URL provided (SQLAlchemy format), use it
    db_url = os.environ.get('DATABASE_URL') or os.environ.get('DB_PATH')
    if db_url:
        ENGINE = create_engine(db_url, future=True)
        return ENGINE
    # else try explicit MySQL components
    mysql_host = os.environ.get('MYSQL_HOST')
    mysql_port = os.environ.get('MYSQL_PORT', '3306')
    mysql_user = os.environ.get('MYSQL_USER')
    mysql_pass = os.environ.get('MYSQL_PASS')
    mysql_db = os.environ.get('MYSQL_DB')
    if mysql_host and mysql_user and mysql_db:
        user = mysql_user
        password = mysql_pass or ''
        host = mysql_host
        port = mysql_port
        db = mysql_db
        ENGINE = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}", future=True)
        return ENGINE
    # fallback to a local sqlite file in data/
    os.makedirs(DATA_DIR, exist_ok=True)
    sqlite_path = os.path.join(DATA_DIR, 'phers.db')
    ENGINE = create_engine(f"sqlite:///{sqlite_path}", future=True)
    return ENGINE

# For backwards compatibility with simple sqlite3 usage elsewhere, provide a thin sqlite3 connection for now
_init_engine()
try:
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'phers.db'), check_same_thread=False)
except Exception:
    conn = None
TABLE_COLUMNS: Dict[str, list] = {}
UPLOADED_FILES: Dict[str, str] = {}
COLUMN_NAME_MAP: Dict[str, Dict[str, str]] = {}
ACTIVE_FILES: Dict[str, bool] = {}  # Track which files are active for queries
QUESTION_REPLACEMENTS: List = []


def index_dataframe_to_chroma(df: pd.DataFrame, filename: str, table_name: str):
    """Index rows from a dataframe into chroma; store ids in CHROMA_IDS_BY_FILE."""
    if not CHROMA_AVAILABLE:
        return
    INDEXING_STATUS[filename] = {"status": "indexing", "started": datetime.datetime.now().isoformat(), "progress": 0.0, "message": ""}
    ids = []
    metadatas = []
    docs = []
    CHUNK_ROWS = int(os.environ.get("RAG_CHUNK_ROWS", "8"))
    rows = []
    total_rows = len(df)
    total_docs = max(1, math.ceil(total_rows / CHUNK_ROWS))
    doc_count = 0
    for i, row in df.iterrows():
        rows.append((i, row))
        if len(rows) >= CHUNK_ROWS:
            idxs = [str(r[0]) for r in rows]
            text = '\n'.join([' | '.join([f"{k}: {r[1][k]}" for k in df.columns]) for r in rows])
            doc_id = f"{filename}::{table_name}::{rows[0][0]}-{rows[-1][0]}"
            docs.append(text)
            metadatas.append({"file": filename, "table": table_name, "rows": idxs, "columns": list(df.columns)})
            ids.append(doc_id)
            rows = []
            doc_count += 1
            try:
                INDEXING_STATUS[filename]["progress"] = min(0.6, (doc_count / total_docs) * 0.6)
            except Exception:
                pass
    if rows:
        idxs = [str(r[0]) for r in rows]
        text = '\n'.join([' | '.join([f"{k}: {r[1][k]}" for k in df.columns]) for r in rows])
        doc_id = f"{filename}::{table_name}::{rows[0][0]}-{rows[-1][0]}"
        docs.append(text)
        metadatas.append({"file": filename, "table": table_name, "rows": idxs, "columns": list(df.columns)})
        ids.append(doc_id)
        doc_count += 1
        try:
            INDEXING_STATUS[filename]["progress"] = min(0.6, (doc_count / total_docs) * 0.6)
        except Exception:
            pass

    try:
        INDEXING_STATUS[filename]["progress"] = max(INDEXING_STATUS[filename].get("progress", 0.0), 0.0)
        embs = embedding_model.encode(docs).tolist()
        INDEXING_STATUS[filename]["progress"] = 0.9
        try:
            sanitized_metas = []
            for m in metadatas:
                sm = {}
                for k, v in m.items():
                    if v is None or isinstance(v, (str, int, float, bool)):
                        sm[k] = v
                    else:
                        try:
                            sm[k] = json.dumps(v)
                        except Exception:
                            sm[k] = str(v)
                sanitized_metas.append(sm)
        except Exception:
            sanitized_metas = metadatas

        chroma_collection.add(ids=ids, documents=docs, metadatas=sanitized_metas, embeddings=embs)
        CHROMA_IDS_BY_FILE[filename] = ids
        INDEXING_STATUS[filename]["progress"] = 1.0
        INDEXING_STATUS[filename]["status"] = "done"
        INDEXING_STATUS[filename]["finished"] = datetime.datetime.now().isoformat()
        INDEXING_STATUS[filename]["count"] = len(ids)
    except Exception as e:
        INDEXING_STATUS[filename]["status"] = "error"
        INDEXING_STATUS[filename]["message"] = str(e)
        INDEXING_STATUS[filename]["finished"] = datetime.datetime.now().isoformat()


def update_question_replacements():
    global QUESTION_REPLACEMENTS
    repls = []
    for table, mapping in COLUMN_NAME_MAP.items():
        try:
            cre = re.compile(r"\b" + re.escape(table) + r"\b", flags=re.I)
            repls.append((cre, table))
        except re.error:
            pass
        for sanitized, orig in mapping.items():
            if not orig:
                continue
            words = re.split(r"\s+", orig.strip())
            if len(words) > 1:
                sep_pattern = r"[\s_\-]+"
                last = re.escape(words[-1]) + r"s?"
                pat = r"\b" + sep_pattern.join([re.escape(w) for w in words[:-1]] + [last]) + r"\b"
            else:
                pat = r"\b" + re.escape(orig) + r"s?\b"
            try:
                cre = re.compile(pat, flags=re.I)
                repls.append((cre, sanitized))
            except re.error:
                try:
                    cre = re.compile(r"\b" + re.escape(orig) + r"\b", flags=re.I)
                    repls.append((cre, sanitized))
                except Exception:
                    pass
    QUESTION_REPLACEMENTS = repls


def normalize_question_text(q: str) -> str:
    """Enhanced question normalization with better HR terminology mapping."""
    out = q
    
    # Apply existing regex replacements
    for cre, repl in QUESTION_REPLACEMENTS:
        out = cre.sub(repl, out)
    
    # Add HR-specific term mappings
    hr_term_mappings = {
        'assets': 'Asset ID',
        'asset': 'Asset ID', 
        'repair': 'In Repair',
        'repairing': 'In Repair',
        'broken': 'In Repair',
        'active': 'Active',
        'working': 'Active',
        'disposed': 'Disposed',
        'discarded': 'Disposed',
        'deleted': 'Disposed',
        'machinery': 'Machinery',
        'equipment': 'IT Equipment',
        'vehicles': 'Vehicle',
        'vehicle': 'Vehicle',
        'cars': 'Vehicle',
        'real estate': 'Real Estate',
        'property': 'Real Estate',
        'properties': 'Real Estate',
        'buildings': 'Real Estate',
        'cost': 'Value ($)',
        'costs': 'Value ($)',
        'value': 'Value ($)',
        'values': 'Value ($)',
        'price': 'Value ($)',
        'prices': 'Value ($)',
        'worth': 'Value ($)',
        'location': 'Location',
        'locations': 'Location',
        'site': 'Location',
        'sites': 'Location',
        'city': 'Location',
        'cities': 'Location'
    }
    
    try:
        cand_map = {}
        
        # Add HR-specific mappings
        cand_map.update(hr_term_mappings)
        
        # Add table and column mappings
        for tbl, mapping in COLUMN_NAME_MAP.items():
            cand_map[tbl.lower()] = tbl
            for san, orig in mapping.items():
                if not orig:
                    continue
                cand_map[orig.lower()] = san
                cand_map[orig.lower() + 's'] = san
                # Split compound words and map them
                for w in re.split(r"[^\w]+", orig.lower()):
                    if w and len(w) > 2:
                        cand_map[w] = san
                        cand_map[w + 's'] = san
        
        # Enhanced tokenization and replacement
        tokens = re.split(r"(\W+)", out)
        for i, tok in enumerate(tokens):
            lower = tok.lower().strip()
            if lower and lower.isalpha() and len(lower) > 2:
                if lower in cand_map:
                    tokens[i] = cand_map[lower]
                else:
                    # Try fuzzy matching with higher cutoff for better precision
                    choices = list(cand_map.keys())
                    matches = difflib.get_close_matches(lower, choices, n=1, cutoff=0.75)
                    if matches:
                        tokens[i] = cand_map[matches[0]]
        
        out = ''.join(tokens)
    except Exception:
        pass
    
    return out


def validate_sql_against_schema(sql: str):
    """A more robust SQL schema validator.

    Strategy:
    - Remove quoted strings so literal values aren't treated as identifiers.
    - Find table names by searching known table safe names in the SQL text.
    - Extract candidate column identifiers from the SELECT clause, FROM/JOIN (tables), and left-hand side
      of comparison operators in WHERE (e.g. `status = 'In Repair'` -> validate `status` only).
    - Only validate candidates (not arbitrary tokens) against `TABLE_COLUMNS` case-insensitively.
    """
    try:
        # remove quoted strings (single & double) to avoid treating values as identifiers
        try:
            cleaned = re.sub(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")", "", sql)
        except Exception:
            cleaned = sql
        txt = cleaned.lower()

        # find mentioned tables (by sanitized table name) - check both quoted and unquoted forms
        used_tables = []
        for t in TABLE_COLUMNS.keys():
            # Check for unquoted table name
            if re.search(r"\b" + re.escape(t.lower()) + r"\b", txt):
                used_tables.append(t)
            # Check for quoted table name with double quotes
            elif re.search(r"\"" + re.escape(t.lower()) + r"\"", txt):
                used_tables.append(t)
            # Check for quoted table name with backticks
            elif re.search(r"`" + re.escape(t.lower()) + r"`", txt):
                used_tables.append(t)
                
        if not used_tables:
            # More detailed error for debugging
            print(f"SQL validation failed - no tables found in: {txt}")
            print(f"Available tables: {list(TABLE_COLUMNS.keys())}")
            raise Exception("SQL references unknown tables.")

        candidates = set()
        # 1) columns in SELECT clause: between SELECT and FROM
        SQL_FUNCTIONS = {"count", "sum", "avg", "min", "max"}
        m = re.search(r"select\s+(.*?)\s+from\b", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if m:
            sel = m.group(1)
            # split by commas and extract likely identifiers (skip SQL functions)
            for part in re.split(r",", sel):
                # capture the first identifier-like token
                idm = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", part)
                if idm:
                    tok = idm.group(1)
                    if tok.lower() in SQL_FUNCTIONS:
                        # try to find alias after 'as' or the next identifier
                        alias = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\b", part, flags=re.IGNORECASE)
                        if alias:
                            candidates.add(alias.group(1))
                        else:
                            # skip function names as column candidates
                            continue
                    else:
                        candidates.add(tok)

        # 2) left-hand side identifiers in WHERE (col = ... , col IN (...), col LIKE ...)
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(?:=|<>|!=|<|>|\bin\b|\blike\b)", cleaned, flags=re.IGNORECASE):
            candidates.add(match.group(1))

        # 3) ORDER BY / GROUP BY columns
        for kw in ("order by", "group by", "having"):
            mm = re.search(rf"{kw}\s+(.*?)(?:$|limit|offset|where|order by|group by)", cleaned, flags=re.IGNORECASE | re.DOTALL)
            if mm:
                for part in re.split(r",", mm.group(1)):
                    idm = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", part)
                    if idm:
                        candidates.add(idm.group(1))

        # validate each candidate against known columns (case-insensitive)
        all_cols_map = {}
        for tbl, cols in TABLE_COLUMNS.items():
            for c in cols:
                all_cols_map[c.lower()] = (tbl, c)

        # filter out common SQL keywords and very short tokens (e.g., 'in')
        SQL_KEYWORDS = {"in", "and", "or", "like", "not", "is", "on", "as", "by", "from", "select", "where", "group", "order", "having", "limit", "offset", "join", "true", "false", "null"}

        for ident in list(candidates):
            if not ident or len(ident) < 3:
                # ignore very short tokens (likely SQL keywords like 'in')
                continue
            if ident.lower() in SQL_KEYWORDS:
                continue
            if ident.lower() in all_cols_map:
                # found as column
                continue
            # also allow table names to appear as identifiers in some expressions
            if ident.lower() in [t.lower() for t in TABLE_COLUMNS.keys()]:
                continue
            # not found -> gather suggestions using our enhanced function
            suggestions = suggest_column_alternatives(ident)
            if suggestions:
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?"
            else:
                # Fallback to original logic
                suggestions = difflib.get_close_matches(ident, list(all_cols_map.keys()), n=3, cutoff=0.6)
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise Exception(f"Column '{ident}' not found.{suggestion_text}")

        return True
    except Exception:
        raise


# Ollama wrapper import order (try official langchain_ollama, then langchain_community, then legacy)
LLM_CLASS = None
LLM_SOURCE = None
try:
    from langchain_ollama import OllamaLLM
    LLM_CLASS = OllamaLLM
    LLM_SOURCE = 'langchain_ollama.OllamaLLM'
    print('Using OllamaLLM from langchain_ollama')
except Exception:
    try:
        from langchain_community.llms import Ollama
        LLM_CLASS = Ollama
        LLM_SOURCE = 'langchain_community.llms.Ollama'
        print('Using Ollama from langchain_community.llms')
    except Exception:
        try:
            from langchain.llms import Ollama
            LLM_CLASS = Ollama
            LLM_SOURCE = 'langchain.llms.Ollama (legacy)'
            print('Using Ollama from langchain.llms (legacy)')
        except Exception as e:
            LLM_CLASS = None
            print("Warning: No Ollama wrapper available. Install 'langchain-ollama' or 'langchain-community' and 'ollama'.", e)


def get_llm():
    if LLM_CLASS is None:
        raise RuntimeError(
            "Ollama LLM wrapper not available. Install 'langchain-ollama' or 'langchain-community' and 'ollama' python packages."
        )
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_MODEL", "phi4")
    try:
        return LLM_CLASS(model=model, base_url=base_url)
    except TypeError:
        try:
            return LLM_CLASS(model)
        except TypeError:
            try:
                return LLM_CLASS(base_url=base_url, model=model)
            except Exception as e:
                raise RuntimeError(f"Failed to construct Ollama LLM with available wrapper ({LLM_SOURCE}): {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to construct Ollama LLM: {e}")


def load_dataframe_to_sql(df: pd.DataFrame, table_name: str):
    """Load DataFrame to SQL with improved data normalization."""
    table_name_safe = re.sub(r"[^\w\d_]", "_", table_name).lower()
    
    # Clean the DataFrame first
    df = df.copy()
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all')  # Remove rows where all values are NaN
    df = df.loc[:, ~df.isnull().all()]  # Remove columns where all values are NaN
    
    # Handle empty DataFrame
    if df.empty:
        raise ValueError(f"File {table_name} contains no valid data after cleaning")
    
    # Clean column names
    orig_cols = list(df.columns)
    sanitized_cols = []
    seen = {}
    col_map: Dict[str, str] = {}
    
    for c in orig_cols:
        # Convert to string and clean
        c_str = str(c).strip()
        if not c_str or c_str.lower() in ['nan', 'none', 'null', 'unnamed']:
            c_str = "unnamed_column"
        
        # Sanitize column name for SQL
        s = re.sub(r"[^\w\d_]", "_", c_str)
        s = re.sub(r"_+", "_", s)  # Replace multiple underscores with single
        s = s.strip("_")  # Remove leading/trailing underscores
        
        if not s or s.isdigit():
            s = f"col_{s}" if s.isdigit() else "col"
        
        # Make sure column name is unique
        base = s
        i = 1
        while s in seen:
            s = f"{base}_{i}"
            i += 1
        seen[s] = True
        sanitized_cols.append(s)
        col_map[s] = c_str
    
    df.columns = sanitized_cols
    
    # Clean data types and values
    for col in df.columns:
        # Replace various null representations with actual NaN
        df[col] = df[col].replace(['', ' ', 'NULL', 'null', 'None', 'none', '#N/A', '#NULL!'], pd.NA)
        
        # Try to convert numeric columns
        if df[col].dtype == 'object':
            # Check if column looks numeric
            sample_non_null = df[col].dropna().head(10)
            if len(sample_non_null) > 0:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(sample_non_null, errors='coerce')
                if not numeric_series.isna().all():
                    # If most values can be converted to numeric, convert the whole column
                    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Write using pandas to_sql via SQLAlchemy engine for broader DB support
    engine = _init_engine()
    try:
        df.to_sql(table_name_safe, engine, if_exists="replace", index=False, method='multi')
    except Exception as e:
        # fallback to sqlite3 direct if SQLAlchemy write fails
        try:
            if conn is not None:
                df.to_sql(table_name_safe, conn, if_exists="replace", index=False)
            else:
                raise e
        except Exception as fallback_error:
            raise ValueError(f"Failed to save data to database: {fallback_error}")
    
    cols = list(df.columns)
    TABLE_COLUMNS[table_name_safe] = cols
    COLUMN_NAME_MAP[table_name_safe] = col_map
    
    print(f"Successfully loaded {len(df)} rows and {len(cols)} columns to table '{table_name_safe}'")
    return table_name_safe


def read_table_into_df(table: str, limit: Optional[int] = None) -> pd.DataFrame:
    engine = _init_engine()
    q = f"select * from \"{table}\""
    if limit:
        q = q + f" limit {limit}"
    try:
        return pd.read_sql_query(q, engine)
    except Exception:
        # fallback to sqlite3
        if conn is not None:
            return pd.read_sql_query(q, conn)
        raise


def drop_table(table: str):
    engine = _init_engine()
    try:
        with engine.connect() as c:
            c.execute(text(f"DROP TABLE IF EXISTS \"{table}\""))
    except Exception:
        # fallback to sqlite3
        if conn is not None:
            try:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            except Exception:
                pass
        else:
            raise


def initialize_data_folder():
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        try:
            if filename.lower().endswith(".csv"):
                df = pd.read_csv(path)
                tbl = os.path.splitext(filename)[0]
                table_name_safe = load_dataframe_to_sql(df, tbl)
                UPLOADED_FILES[filename] = table_name_safe
                ACTIVE_FILES[filename] = True  # Activate by default
                print(f"Loaded CSV: {filename} -> table {tbl}")
            elif filename.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
                tbl = os.path.splitext(filename)[0]
                table_name_safe = load_dataframe_to_sql(df, tbl)
                UPLOADED_FILES[filename] = table_name_safe
                ACTIVE_FILES[filename] = True  # Activate by default
                print(f"Loaded XLSX: {filename} -> table {tbl}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

        try:
            if CHROMA_AVAILABLE and filename in UPLOADED_FILES:
                tbl = UPLOADED_FILES[filename]
                try:
                    if filename.lower().endswith('.csv'):
                        df = pd.read_csv(os.path.join(DATA_DIR, filename))
                    else:
                        df = pd.read_excel(os.path.join(DATA_DIR, filename))
                    t = threading.Thread(target=index_dataframe_to_chroma, args=(df, filename, tbl), daemon=True)
                    t.start()
                except Exception:
                    pass
        except Exception:
            pass

# --- additional helpers exported for routes.py ---

# pandas-ai placeholders (actual imports are lazy in routes to avoid hard dependency at import time)
PandasAI = None
LangChain = None


def build_schema_description():
    """Build schema description for only active files."""
    active_files = get_active_files()
    if not active_files:
        return "NO_ACTIVE_TABLES"
    
    # Get tables from active files only
    active_tables = set(active_files.values())
    if not active_tables:
        return "NO_ACTIVE_TABLES"
    
    parts = []
    for t, cols in TABLE_COLUMNS.items():
        if t not in active_tables:
            continue  # Skip inactive tables
            
        col_texts = []
        mapping = COLUMN_NAME_MAP.get(t, {})
        for c in cols:
            orig = mapping.get(c)
            if orig and orig != c:
                col_texts.append(f"{c} (original: {orig})")
            else:
                col_texts.append(c)
        parts.append(f"Table `{t}` with columns: {', '.join(col_texts)}")
    
    return "\n".join(parts) if parts else "NO_ACTIVE_TABLES"


def normalize_sql_safely(sql: str) -> str:
    """Normalize SQL characters and fix common issues automatically."""
    s = sql.strip()
    
    # Self-healing character normalization
    char_fixes = {
        ''': "'",  # Smart quote to regular quote
        ''': "'",  # Smart quote to regular quote  
        '"': '"',  # Smart quote to regular quote
        '"': '"',  # Smart quote to regular quote
        '–': '-',  # En dash to regular dash
        '—': '-',  # Em dash to regular dash
        '…': '...',  # Ellipsis to dots
    }
    
    for bad_char, good_char in char_fixes.items():
        s = s.replace(bad_char, good_char)
    
    # Remove any remaining problematic characters but preserve SQL structure
    s = re.sub(r'[^\s\w\d\.\,\*\(\)=<>!\"\'%\-\+\/\[\]_`:]', '', s)
    
    return s

def validate_sql_safe(sql: str):
    """Validate and clean SQL for safety with self-healing."""
    # First normalize problematic characters
    s = normalize_sql_safely(sql)
    
    # Remove trailing semicolons (common LLM behavior)
    while s.endswith(';'):
        s = s[:-1].strip()
    
    # Remove any extra whitespace and newlines
    s = re.sub(r'\s+', ' ', s)
    
    s_lower = s.lower()
    
    if not s_lower.startswith("select"):
        raise HTTPException(status_code=400, detail="LLM must return a SELECT query only.")
    
    # Check for multiple statements (semicolons in middle)
    if ";" in s:
        raise HTTPException(status_code=400, detail="Multiple SQL statements not allowed.")
    
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "exec ", "execute "]
    if any(k in s_lower for k in forbidden):
        raise HTTPException(status_code=400, detail="Only read (SELECT) queries are allowed.")
    
    # More lenient character validation (allow more SQL characters including backticks and colons)
    if not re.match(r"^[\s\w\d\.\,\*\(\)=<>!\"'%\-\+\/\[\]_`:]+$", s):
        raise HTTPException(status_code=400, detail="SQL contains unexpected characters.")
    
    return s  # Return the cleaned and normalized SQL


def auto_quote_string_literals(sql: str) -> str:
    """Conservative sanitizer to quote unquoted string literals.

    Enhancements:
    - Quote items inside IN(...) lists when not quoted
    - Quote RHS for = and LIKE when the RHS looks like a textual value
    """
    if not sql:
        return sql

    def _quote_item_raw(v: str) -> str:
        v = v.strip()
        if not v:
            return v
        if v[0] in ("'", '"'):
            return v
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v):
            return v
        if v.lower() in ('null', 'true', 'false'):
            return v
        # escape single quotes
        safe = v.replace("'", "''")
        return f"'{safe}'"

    # 1) Handle IN lists: col IN (a, b, c) -> col IN ('a','b','c')
    try:
        pattern_in = re.compile(r"(\b[A-Za-z_][A-Za-z0-9_\.]*\b)\s+IN\s*\(([^\)]*?)\)", flags=re.IGNORECASE)

        def _in_replace(m):
            lhs = m.group(1)
            items = m.group(2)
            # split on commas that are not inside quotes (simple approach)
            parts = [p.strip() for p in re.split(r',', items)]
            quoted = []
            for p in parts:
                if not p:
                    continue
                quoted.append(_quote_item_raw(p))
            return f"{lhs} IN ({', '.join(quoted)})"

        sql = pattern_in.sub(_in_replace, sql)
    except Exception:
        pass

    # 2) Handle = and LIKE where RHS looks like textual value
    try:
        pattern = re.compile(r"(?P<lhs>\b[A-Za-z_][A-Za-z0-9_\.]*\b)\s*(?P<op>=|LIKE|like)\s*(?P<val>[^;\)]+)", flags=re.IGNORECASE)

        def _replace(m):
            lhs = m.group('lhs')
            op = m.group('op')
            val = m.group('val').strip()
            # stop at AND/OR if present
            val = re.split(r"\b(and|or)\b", val, flags=re.IGNORECASE)[0].strip()
            if not val:
                return m.group(0)
            if val[0] in ("'", '"'):
                return m.group(0)
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", val):
                return m.group(0)
            if val.lower() in ('null', 'true', 'false'):
                return m.group(0)
            # If RHS is a single token that matches a known column for this table, avoid quoting
            # (we can't always know table here; be conservative and only skip if it's an exact known column name anywhere)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*", val):
                if val.lower() in (c.lower() for cols in TABLE_COLUMNS.values() for c in cols):
                    return m.group(0)
            safe = val.replace("'", "''")
            prefix = m.string[m.start():m.start('op')]
            return f"{prefix}{op} { _quote_item_raw(val) }"

        sql = pattern.sub(_replace, sql)
    except Exception:
        pass

    return sql


SQL_PROMPT_TEMPLATE = '''You are an expert SQL generator for HR data queries. Generate precise SQL for business questions.

Available schema (tables and columns):
{schema}

INSTRUCTIONS (critical - follow exactly):
1. Produce a single, valid SQLite SELECT statement that answers the user's question.
2. Use the exact table names and column names shown above.
3. For counting questions, use COUNT(*) or COUNT(column_name).
4. For status/category filters, use exact string matching with proper quotes.
5. For aggregations, use appropriate functions (COUNT, SUM, AVG, MIN, MAX).
6. Handle common HR terms:
   - "how many" = COUNT(*)
   - "total value/cost" = SUM(value_column)
   - "in repair/active/disposed" = WHERE status = 'Status'
   - "by location/category" = GROUP BY column
7. CRITICAL - Search strategies:
   a) Names may be stored as "Last, Title First" format (e.g., "Howard, Mr. Benjamin")
      - For searching "John Smith", use: Name LIKE '%John%' AND Name LIKE '%Smith%'
      - For searching "Benjamin Howard", use: Name LIKE '%Benjamin%' AND Name LIKE '%Howard%'
   b) Asset tags, IDs, or codes should use exact matching or LIKE:
      - For "ASSET-VAF-HO-IV-124", use: Asset_TAG = 'ASSET-VAF-HO-IV-124' 
      - Or use: Asset_TAG LIKE '%ASSET-VAF-HO-IV-124%'
      - Note: After sanitation, column names use underscores instead of spaces
   c) Always use LIKE with % wildcards for partial text searches
8. Use LIKE for partial text matching when appropriate.
9. Output ONLY the SQL inside <SQL>...</SQL> tags, nothing else.

User question: "{question}"

Common query patterns:
- Count: SELECT COUNT(*) FROM table WHERE condition
- Sum: SELECT SUM(column) FROM table WHERE condition  
- Group: SELECT column, COUNT(*) FROM table GROUP BY column
- Filter: SELECT * FROM table WHERE column = 'value'
- Name search: SELECT * FROM table WHERE Name LIKE '%FirstName%' AND Name LIKE '%LastName%'
- Asset/ID search: SELECT * FROM table WHERE Asset_TAG = 'ASSET-123' OR Asset_TAG LIKE '%ASSET-123%'

<SQL>SELECT ...</SQL>'''


SUMMARY_PROMPT_TEMPLATE = """You are a concise summarizer. I will provide:
1) The original user question.
2) The SQL that was executed.
3) The resulting rows (as JSON array of objects). 

Task: Produce a short, factual answer to the user question based *only* on the rows provided. Do NOT add any facts not present in the rows. If the result is empty, reply: "No matching records found." Keep it 1-3 sentences max.

Format: return a JSON object with two keys:
- "text": the short answer as a single string.
- "table_preview": an array of up to 10 rows (objects) from the result to show to the user.

Input:
---
question: {question}
sql: {sql}
rows_json: {rows_json}
---
"""


def suggest_column_alternatives(invalid_column: str, table_name: str = None) -> List[str]:
    """AI-powered intelligent column suggestions."""
    suggestions = []
    
    # Get all available columns
    all_cols = []
    if table_name and table_name in TABLE_COLUMNS:
        all_cols = TABLE_COLUMNS[table_name]
    else:
        # Get columns from all tables
        for cols in TABLE_COLUMNS.values():
            all_cols.extend(cols)
    
    # Get original column names too
    if table_name and table_name in COLUMN_NAME_MAP:
        mapping = COLUMN_NAME_MAP[table_name]
        for sanitized, original in mapping.items():
            if original and original not in all_cols:
                all_cols.append(original)
    else:
        # Get original names from all tables
        for mapping in COLUMN_NAME_MAP.values():
            for sanitized, original in mapping.items():
                if original and original not in all_cols:
                    all_cols.append(original)
    
    # Smart matching with multiple strategies
    lower_col = invalid_column.lower()
    
    # Strategy 1: Exact partial matches
    exact_matches = [col for col in all_cols if lower_col in col.lower() or col.lower() in lower_col]
    suggestions.extend(exact_matches[:2])
    
    # Strategy 2: Fuzzy matching
    if len(suggestions) < 3:
        matches = difflib.get_close_matches(lower_col, [col.lower() for col in all_cols], n=3, cutoff=0.4)
        lower_to_orig = {col.lower(): col for col in all_cols}
        fuzzy_matches = [lower_to_orig[match] for match in matches if match in lower_to_orig]
        for match in fuzzy_matches:
            if match not in suggestions:
                suggestions.append(match)
    
    # Strategy 3: Enhanced smart semantic suggestions with context awareness
    if len(suggestions) < 3:
        semantic_map = {
            # Asset & Equipment terms (multiple variations)
            'asset': ['Asset_TAG', 'item_Name', 'Model_name'],
            'assets': ['Asset_TAG', 'item_Name', 'Model_name'],
            'tag': ['Asset_TAG', 'Serial_number'],
            'tags': ['Asset_TAG', 'Serial_number'],
            'equipment': ['item_Name', 'Category', 'Model_name', 'Manufacturer'],
            'device': ['item_Name', 'Category', 'Model_name', 'Manufacturer'],
            'devices': ['item_Name', 'Category', 'Model_name', 'Manufacturer'],
            
            # Manufacturer variations
            'manufacture': ['Manufacturer', 'Model_name', 'item_Name'],
            'manufacturer': ['Manufacturer', 'Model_name', 'item_Name'],
            'brand': ['Manufacturer', 'Model_name', 'item_Name'],
            'make': ['Manufacturer', 'Model_name', 'item_Name'],
            'made': ['Manufacturer', 'Model_name'],
            'company': ['Manufacturer', 'Company', 'Supplier'],
            
            # Location and company terms
            'location': ['Location', 'Company', 'Status'],
            'where': ['Location', 'Company', 'Status'],
            'place': ['Location', 'Company'],
            'office': ['Location', 'Company'],
            'building': ['Location', 'Company'],
            
            # Financial terms
            'cost': ['Purchase_Cost', 'Supplier'],
            'costs': ['Purchase_Cost', 'Supplier'],
            'price': ['Purchase_Cost', 'Supplier'],
            'value': ['Purchase_Cost'],
            'money': ['Purchase_Cost'],
            'bought': ['Purchase_Cost', 'Supplier'],
            'purchased': ['Purchase_Cost', 'Supplier'],
            
            # Status and condition terms
            'status': ['Status', 'Warranty', 'Location'],
            'condition': ['Status', 'Warranty'],
            'state': ['Status', 'Warranty'],
            'working': ['Status', 'Warranty'],
            'broken': ['Status', 'Warranty'],
            'active': ['Status', 'Warranty'],
            'warranty': ['Warranty', 'Status', 'Purchase Date'],
            
            # User and ownership terms
            'user': ['Username', 'Company', 'Location'],
            'owner': ['Username', 'Company', 'Location'],
            'person': ['Username', 'Company'],
            'employee': ['Username', 'Company', 'Location'],
            'staff': ['Username', 'Company', 'Location'],
            'who': ['Username', 'Company'],
            
            # Product and model terms
            'model': ['Model_name', 'Model_number', 'Manufacturer'],
            'product': ['item_Name', 'Model_name', 'Category'],
            'item': ['item_Name', 'Category', 'Model_name'],
            'name': ['item_Name', 'Model_name', 'Username'],
            'type': ['Category', 'item_Name', 'Model_name'],
            'category': ['Category', 'item_Name'],
            
            # Technical identifiers
            'serial': ['Serial_number', 'Model_number', 'Asset_TAG'],
            'number': ['Serial_number', 'Model_number', 'Asset_TAG', 'Order_Number'],
            'id': ['Asset_TAG', 'Serial_number', 'Model_number'],
            'identifier': ['Asset_TAG', 'Serial_number', 'Model_number'],
        }
        
        for term, cols in semantic_map.items():
            if term in lower_col:
                for col in cols:
                    if col in all_cols and col not in suggestions:
                        suggestions.append(col)
                        if len(suggestions) >= 3:
                            break
    
    return suggestions[:3]  # Return top 3 intelligent suggestions


def generate_intelligent_query_suggestion(original_query: str, failed_column: str, suggested_columns: List[str]) -> str:
    """Generate intelligent alternative query suggestions with advanced pattern recognition."""
    if not suggested_columns:
        return original_query
    
    # Smart query rewriting
    query_lower = original_query.lower()
    best_suggestion = suggested_columns[0]
    
    # Advanced pattern recognition for different writing styles
    writing_patterns = {
        # Casual/Informal patterns
        r'what\'s the (.+) of (.+)': r'what is the \1 of \2',
        r'who made (.+)': r'what is the manufacturer of \1',
        r'what brand is (.+)': r'what is the manufacturer of \1',
        r'where is (.+)': r'what is the location of \1',
        r'how much (.+) cost': r'what is the purchase cost of \1',
        r'when was (.+) bought': r'what is the purchase date of \1',
        
        # Technical/Formal patterns
        r'manufacture of (.+)': r'manufacturer of \1',
        r'asset tag (.+)': r'asset with tag \1',
        r'equipment (.+)': r'item named \1',
        r'device (.+)': r'item named \1',
        r'serial number (.+)': r'item with serial \1',
        r'model (.+)': r'model name \1',
        
        # Question word variations
        r'what\'s': 'what is',
        r'where\'s': 'where is',
        r'who\'s': 'who is',
        r'how\'s': 'how is',
        
        # Common misspellings/variations
        r'manufact\w*': 'manufacturer',
        r'equipement': 'equipment',
        r'assett?': 'asset',
        r'locati?on': 'location',
        r'purchas\w*': 'purchase',
    }
    
    # Apply pattern-based transformations
    improved_query = original_query
    for pattern, replacement in writing_patterns.items():
        improved_query = re.sub(pattern, replacement, improved_query, flags=re.IGNORECASE)
    
    # Context-aware column suggestions
    context_mappings = {
        'cost': ['Purchase Cost', 'Value ($)', 'Purchase Date'],
        'price': ['Purchase Cost', 'Value ($)', 'Purchase Date'],
        'location': ['Location', 'Company', 'Status'],
        'where': ['Location', 'Company', 'Status'],
        'manufacturer': ['Manufacturer', 'Model name', 'item Name'],
        'brand': ['Manufacturer', 'Model name', 'item Name'],
        'serial': ['Serial number', 'Model number', 'Asset TAG'],
        'tag': ['Asset TAG', 'Serial number', 'Model number'],
        'status': ['Status', 'Warranty', 'Location'],
        'warranty': ['Warranty', 'Status', 'Purchase Date'],
    }
    
    return improved_query


def sanitize_dataframe(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """
    Comprehensive data sanitation to prevent SQL issues and improve indexing.
    Returns sanitization report and cleaned DataFrame.
    """
    sanitation_report = {
        "filename": filename,
        "original_columns": list(df.columns),
        "cleaned_columns": [],
        "issues_found": [],
        "issues_fixed": [],
        "rows_processed": len(df),
        "data_changes": []
    }
    
    # Stage 1: Column Name Sanitation
    column_mapping = {}
    problematic_chars = {
        ' ': '_',           # Spaces to underscores
        ':': '_',           # Colons to underscores  
        ';': '_',           # Semicolons to underscores
        ',': '_',           # Commas to underscores
        '(': '_',           # Parentheses to underscores
        ')': '_',
        '[': '_',           # Brackets to underscores
        ']': '_',
        '{': '_',           # Braces to underscores
        '}': '_',
        '"': '',            # Remove quotes
        "'": '',            # Remove quotes
        '`': '',            # Remove backticks
        '%': 'pct',         # Percent to 'pct'
        '$': 'dollar',      # Dollar to 'dollar'
        '#': 'num',         # Hash to 'num'
        '@': 'at',          # At to 'at'
        '&': 'and',         # Ampersand to 'and'
        '+': 'plus',        # Plus to 'plus'
        '=': 'eq',          # Equals to 'eq'
        '<': 'lt',          # Less than to 'lt'
        '>': 'gt',          # Greater than to 'gt'
        '?': '',            # Remove question marks
        '!': '',            # Remove exclamation marks
        '*': 'star',        # Asterisk to 'star'
        '/': '_',           # Forward slash to underscore
        '\\': '_',          # Backslash to underscore
        '|': '_',           # Pipe to underscore
        '~': '_',           # Tilde to underscore
        '^': '_',           # Caret to underscore
        '-': '_',           # Dashes to underscores (except in data)
    }
    
    for original_col in df.columns:
        cleaned_col = original_col
        found_issues = []
        
        # Apply character replacements
        for bad_char, replacement in problematic_chars.items():
            if bad_char in cleaned_col:
                found_issues.append(f"'{bad_char}' → '{replacement}'")
                cleaned_col = cleaned_col.replace(bad_char, replacement)
        
        # Clean up multiple underscores
        cleaned_col = re.sub(r'_+', '_', cleaned_col)
        # Remove leading/trailing underscores
        cleaned_col = cleaned_col.strip('_')
        
        # Ensure column name starts with letter or underscore (SQL requirement)
        if cleaned_col and not cleaned_col[0].isalpha() and cleaned_col[0] != '_':
            cleaned_col = 'col_' + cleaned_col
            found_issues.append("Added 'col_' prefix (SQL requirement)")
        
        # Handle empty column names
        if not cleaned_col:
            cleaned_col = f'column_{df.columns.get_loc(original_col)}'
            found_issues.append("Empty name → assigned generic name")
        
        column_mapping[original_col] = cleaned_col
        
        if found_issues:
            sanitation_report["issues_found"].append({
                "type": "column_name",
                "original": original_col,
                "cleaned": cleaned_col,
                "issues": found_issues
            })
    
    # Apply column renaming
    df_cleaned = df.rename(columns=column_mapping)
    sanitation_report["cleaned_columns"] = list(df_cleaned.columns)
    
    # Stage 2: Data Value Sanitation
    data_issues = 0
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':  # String columns
            # Count problematic values before cleaning
            problematic_mask = df_cleaned[col].astype(str).str.contains(r'[;\'"`]', na=False)
            problematic_count = problematic_mask.sum()
            
            if problematic_count > 0:
                data_issues += problematic_count
                
                # Clean data values (less aggressive than column names)
                df_cleaned[col] = df_cleaned[col].astype(str).apply(lambda x: 
                    x.replace('"', "'")       # Double quotes to single quotes
                     .replace('`', "'")       # Backticks to single quotes  
                     .replace(';', ',')       # Semicolons to commas
                     .replace('\n', ' ')      # Newlines to spaces
                     .replace('\r', ' ')      # Carriage returns to spaces
                     .replace('\t', ' ')      # Tabs to spaces
                )
                
                sanitation_report["data_changes"].append({
                    "column": col,
                    "issues_fixed": int(problematic_count),
                    "description": "Cleaned quotes, semicolons, and whitespace"
                })
    
    # Stage 3: Generate Summary
    total_column_issues = len(sanitation_report["issues_found"])
    sanitation_report["issues_fixed"] = [
        f"{total_column_issues} column names sanitized",
        f"{data_issues} data values cleaned" if data_issues > 0 else "No data value issues found"
    ]
    
    # Remove empty data changes
    sanitation_report["data_changes"] = [dc for dc in sanitation_report["data_changes"] if dc["issues_fixed"] > 0]
    
    return {
        "cleaned_df": df_cleaned,
        "report": sanitation_report,
        "needs_cleaning": total_column_issues > 0 or data_issues > 0
    }


def reload_tables_from_database():
    """Reload table structure from existing database into memory."""
    if ENGINE is None and conn is None:
        return
    
    try:
        # Get all tables from database
        if ENGINE is not None:
            from sqlalchemy import text
            with ENGINE.connect() as cx:
                result = cx.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result.fetchall()]
        elif conn is not None:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        
        # For each table, get its columns and populate TABLE_COLUMNS
        for table_name in tables:
            if table_name.startswith('sqlite_'):  # Skip system tables
                continue
                
            try:
                if ENGINE is not None:
                    with ENGINE.connect() as cx:
                        result = cx.execute(text(f"PRAGMA table_info({table_name})"))
                        columns = [row[1] for row in result.fetchall()]  # Column name is at index 1
                elif conn is not None:
                    cursor = conn.cursor()
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in cursor.fetchall()]
                
                TABLE_COLUMNS[table_name] = columns
                
                # Try to find the original filename in data folder
                possible_files = [
                    f"{table_name}.csv",
                    f"cleaned_{table_name}.csv",
                    f"{table_name}.xlsx",
                ]
                
                for possible_file in possible_files:
                    file_path = os.path.join(DATA_DIR, possible_file)
                    if os.path.exists(file_path):
                        UPLOADED_FILES[possible_file] = table_name
                        ACTIVE_FILES[possible_file] = True  # Activate by default
                        break
                else:
                    # If no file found, create a placeholder
                    UPLOADED_FILES[f"{table_name}.csv"] = table_name
                    ACTIVE_FILES[f"{table_name}.csv"] = True
                        
                print(f"Reloaded table '{table_name}' with {len(columns)} columns")
                
            except Exception as e:
                print(f"Failed to reload table '{table_name}': {e}")
                
    except Exception as e:
        print(f"Failed to reload tables from database: {e}")


def get_active_files() -> Dict[str, str]:
    """Return only files that are currently active."""
    return {fname: tbl for fname, tbl in UPLOADED_FILES.items() if ACTIVE_FILES.get(fname, False)}


def score_candidate_tables(question: str) -> List[Dict[str, Any]]:
    q = (question or "").lower()
    candidates: List[Dict[str, Any]] = []
    # Only consider active files
    active_files = get_active_files()
    for fname, tbl in active_files.items():
        score = 0.0
        if tbl.lower() in q or os.path.splitext(fname)[0].lower() in q:
            score += 1.0
        cols = TABLE_COLUMNS.get(tbl, [])
        for c in cols:
            if c.lower() in q:
                score += 0.5
        candidates.append({"filename": fname, "table": tbl, "score": score, "active": True})

    if CHROMA_AVAILABLE and candidates:
        try:
            res = chroma_collection.query(query_texts=[question], n_results=3, include=['metadatas', 'distances'])
            hits = res.get('metadatas', [[]])[0]
            dists = res.get('distances', [[]])[0]
            for i, h in enumerate(hits):
                if isinstance(h, dict) and 'file' in h and 'table' in h:
                    for c in candidates:
                        if c['table'] == h['table']:
                            try:
                                c['score'] += max(0, 1.0 - float(dists[i]))
                            except Exception:
                                c['score'] += 0.5
        except Exception:
            pass

    # small fuzzy boost
    for c in candidates:
        base = os.path.splitext(c['filename'])[0]
        match = difflib.get_close_matches(base.lower(), [q], n=1, cutoff=0.6)
        if match:
            c['score'] += 0.4

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates


def generate_smart_suggestions(question: str, sql: str, inferred_tables: List[str]) -> List[Dict[str, Any]]:
    """
    Generate intelligent AI suggestions when no results are found.
    Analyzes user intent and actual data to provide helpful suggestions.
    """
    suggestions = []
    
    try:
        # Pattern 1: Asset tag queries
        asset_tag_match = re.search(r"asset.*tag.*([A-Z0-9-]+)", question, re.IGNORECASE)
        if asset_tag_match:
            searched_tag = asset_tag_match.group(1)
            
            # Find tables with Asset_TAG column
            asset_tables = []
            for table_name, columns in TABLE_COLUMNS.items():
                if 'Asset_TAG' in columns:
                    asset_tables.append(table_name)
            
            if asset_tables:
                target_table = asset_tables[0]  # Use first available asset table
                
                # Find similar asset tags in the database
                try:
                    all_tags = []
                    if ENGINE is not None:
                        with ENGINE.connect() as cx:
                            res = cx.execute(text(f'SELECT DISTINCT "Asset_TAG" FROM "{target_table}" WHERE "Asset_TAG" IS NOT NULL LIMIT 100'))
                            all_tags = [str(r[0]) for r in res.fetchall() if r[0] is not None]
                    elif conn is not None:
                        cur = conn.cursor()
                        cur.execute(f'SELECT DISTINCT "Asset_TAG" FROM "{target_table}" WHERE "Asset_TAG" IS NOT NULL LIMIT 100')
                        all_tags = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
                        
                    # Find close matches
                    if all_tags:
                        close_matches = difflib.get_close_matches(str(searched_tag), all_tags, n=3, cutoff=0.6)
                        for match in close_matches:
                            # Ensure all values are strings and JSON serializable
                            suggestions.append({
                                "type": "asset_tag_correction",
                                "original": str(searched_tag),
                                "suggested": str(match),
                                "label": f'Did you mean "{str(match)}"?',
                                "suggested_sqls": [f'SELECT * FROM "{target_table}" WHERE "Asset_TAG" = \'{str(match)}\''],
                                "icon": "🏷️"
                            })
                        
                        # If no close matches, suggest browsing all assets
                        if not close_matches:
                            suggestions.append({
                                "type": "browse_assets",
                                "label": "Browse all asset tags",
                                "suggested_sqls": [f'SELECT "Asset_TAG", "Manufacturer", "Model_name" FROM "{target_table}" ORDER BY "Asset_TAG"'],
                                "icon": "📋"
                            })
                            
                except Exception as e:
                    print(f"Error in asset tag suggestions: {e}")
                    # Fallback suggestion if database query fails
                    suggestions.append({
                        "type": "browse_assets",
                        "label": "Browse all asset tags",
                        "suggested_sqls": [f'SELECT "Asset_TAG", "Manufacturer", "Model_name" FROM "{target_table}" ORDER BY "Asset_TAG"'],
                        "icon": "📋"
                    })
        
        # Pattern 2: Manufacturer queries  
        manufacturer_match = re.search(r"(?:manufacturer|who made|made by)", question, re.IGNORECASE)
        if manufacturer_match and not suggestions:
            # Find tables with Manufacturer column
            mfg_tables = []
            for table_name, columns in TABLE_COLUMNS.items():
                if 'Manufacturer' in columns:
                    mfg_tables.append(table_name)
                    
            if mfg_tables:
                target_table = mfg_tables[0]
                
                # Suggest browsing manufacturers
                suggestions.append({
                    "type": "browse_manufacturers", 
                    "label": "Show all manufacturers",
                    "suggested_sqls": [f'SELECT DISTINCT "Manufacturer", COUNT(*) as count FROM "{target_table}" GROUP BY "Manufacturer" ORDER BY count DESC'],
                    "icon": "🏭"
                })
        
        # Pattern 3: General search term extraction
        if not suggestions:
            # Extract potential search terms from the question
            search_terms = re.findall(r'\b([A-Z]{2,}(?:-[A-Z0-9]+)*)\b', question)
            search_terms += re.findall(r'\b([A-Za-z]{3,})\b', question)
            
            # Remove common words
            common_words = {'what', 'is', 'the', 'of', 'this', 'that', 'are', 'where', 'how', 'many', 'who', 'when', 'why', 'which', 'asset', 'tag', 'manufacture', 'manufacturer'}
            search_terms = [term for term in search_terms if term.lower() not in common_words and len(term) > 2]
            
            if search_terms and TABLE_COLUMNS:
                # Use the first available table for general search
                target_table = next(iter(TABLE_COLUMNS.keys()))
                columns = TABLE_COLUMNS[target_table]
                
                # Build a general search query across text columns
                text_columns = [col for col in columns if any(word in col.lower() for word in ['name', 'model', 'manufacturer', 'tag', 'description', 'category'])]
                
                if text_columns and search_terms:
                    search_term = search_terms[0]
                    where_conditions = []
                    for col in text_columns[:3]:  # Limit to first 3 text columns
                        where_conditions.append(f'"{col}" LIKE \'%{search_term}%\'')
                    
                    if where_conditions:
                        where_clause = " OR ".join(where_conditions)
                        suggestions.append({
                            "type": "general_search",
                            "label": f'Search for "{search_term}"',
                            "suggested_sqls": [f'SELECT * FROM "{target_table}" WHERE {where_clause}'],
                            "icon": "🔍"
                        })
        
        # Fallback: Suggest exploring the data
        if not suggestions and TABLE_COLUMNS:
            target_table = next(iter(TABLE_COLUMNS.keys()))
            suggestions.append({
                "type": "explore_data",
                "label": "Show sample data", 
                "suggested_sqls": [f'SELECT * FROM "{target_table}" LIMIT 10'],
                "icon": "📊"
            })
                
    except Exception as e:
        print(f"Error generating smart suggestions: {e}")
    
    return suggestions
