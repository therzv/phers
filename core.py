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
    out = q
    for cre, repl in QUESTION_REPLACEMENTS:
        out = cre.sub(repl, out)
    try:
        cand_map = {}
        for tbl, mapping in COLUMN_NAME_MAP.items():
            cand_map[tbl.lower()] = tbl
            for san, orig in mapping.items():
                if not orig:
                    continue
                cand_map[orig.lower()] = san
                cand_map[orig.lower() + 's'] = san
                for w in re.split(r"\s+", orig.lower()):
                    if w:
                        cand_map[w] = san
                        cand_map[w + 's'] = san
        tokens = re.split(r"(\W+)", out)
        for i, tok in enumerate(tokens):
            lower = tok.lower()
            if lower and lower.isalpha() and len(lower) > 2:
                if lower in cand_map:
                    tokens[i] = cand_map[lower]
                else:
                    choices = list(cand_map.keys())
                    matches = difflib.get_close_matches(lower, choices, n=1, cutoff=0.8)
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

        # find mentioned tables (by sanitized table name)
        used_tables = [t for t in TABLE_COLUMNS.keys() if re.search(r"\b" + re.escape(t.lower()) + r"\b", txt)]
        if not used_tables:
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
            # not found -> gather suggestions
            suggestions = difflib.get_close_matches(ident, list(all_cols_map.keys()), n=3, cutoff=0.6)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise Exception(f"SQL references unknown column or identifier: {ident}.{suggestion_text}")

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
    table_name_safe = re.sub(r"[^\w\d_]", "_", table_name).lower()
    orig_cols = list(df.columns)
    sanitized_cols = []
    seen = {}
    col_map: Dict[str, str] = {}
    for c in orig_cols:
        s = re.sub(r"[^\w\d_]", "_", str(c)).strip()
        if not s:
            s = "col"
        base = s
        i = 1
        while s in seen:
            s = f"{base}_{i}"
            i += 1
        seen[s] = True
        sanitized_cols.append(s)
        col_map[s] = str(c)
    df = df.copy()
    df.columns = sanitized_cols
    # write using pandas to_sql via SQLAlchemy engine for broader DB support
    engine = _init_engine()
    try:
        df.to_sql(table_name_safe, engine, if_exists="replace", index=False, method='multi')
    except Exception:
        # fallback to sqlite3 direct if SQLAlchemy write fails
        if conn is not None:
            df.to_sql(table_name_safe, conn, if_exists="replace", index=False)
        else:
            raise
    cols = list(df.columns)
    TABLE_COLUMNS[table_name_safe] = cols
    COLUMN_NAME_MAP[table_name_safe] = col_map
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
                print(f"Loaded CSV: {filename} -> table {tbl}")
            elif filename.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
                tbl = os.path.splitext(filename)[0]
                table_name_safe = load_dataframe_to_sql(df, tbl)
                UPLOADED_FILES[filename] = table_name_safe
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
    if not TABLE_COLUMNS:
        return "NO_TABLES"
    parts = []
    for t, cols in TABLE_COLUMNS.items():
        col_texts = []
        mapping = COLUMN_NAME_MAP.get(t, {})
        for c in cols:
            orig = mapping.get(c)
            if orig and orig != c:
                col_texts.append(f"{c} (original: {orig})")
            else:
                col_texts.append(c)
        parts.append(f"Table `{t}` with columns: {', '.join(col_texts)}")
    return "\n".join(parts)


def validate_sql_safe(sql: str):
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise HTTPException(status_code=400, detail="LLM must return a SELECT query only.")
    if ";" in s:
        raise HTTPException(status_code=400, detail="Semicolons not allowed in SQL.")
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create "]
    if any(k in s for k in forbidden):
        raise HTTPException(status_code=400, detail="Only read (SELECT) queries are allowed.")
    if not re.match(r"^[\s\w\d\.\,\*\(\)=<>!\"'%-]+$", sql):
        raise HTTPException(status_code=400, detail="SQL contains unexpected characters.")
    return True


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


SQL_PROMPT_TEMPLATE = '''You are a strict SQL generator. The user will ask a question to query HR tables.
Available schema (tables and columns):
{schema}

INSTRUCTIONS (critical - follow exactly):
1. Produce a single, valid SQLite SELECT statement that answers the user's question.
2. Use the exact table names and column names shown above.
3. Do NOT invent columns, tables, or data.
4. Do NOT include any comments, explanation, or extra text.
5. Output ONLY the SQL inside a <SQL>...</SQL> tag, and nothing else.
6. The SQL must be a single SELECT statement (no semicolons).
7. Keep result columns concise (select only needed columns).

User question:
"""
{question}
"""

Remember: return ONLY:
<SQL>SELECT ...</SQL>
'''


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


def score_candidate_tables(question: str) -> List[Dict[str, Any]]:
    q = (question or "").lower()
    candidates: List[Dict[str, Any]] = []
    for fname, tbl in UPLOADED_FILES.items():
        score = 0.0
        if tbl.lower() in q or os.path.splitext(fname)[0].lower() in q:
            score += 1.0
        cols = TABLE_COLUMNS.get(tbl, [])
        for c in cols:
            if c.lower() in q:
                score += 0.5
        candidates.append({"filename": fname, "table": tbl, "score": score})

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
