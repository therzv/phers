"""Core helpers and global state for the HR Data Chat app.
This module holds DB connection, table state, indexing, LLM helper, and utility functions.
"""
from typing import Dict, Any, List, Optional
import sqlite3
import pandas as pd
import os
import io
import re
import json
import threading
import datetime
import math
import difflib

# Optional pandas-ai integration (fast prototype mode)
PANDAS_AI_AVAILABLE = False
PANDASAI_LANGCHAIN_AVAILABLE = False
try:
    from pandasai import PandasAI
    try:
        # adapter that accepts a LangChain LLM instance
        from pandasai.llm.langchain import LangChain
        PANDASAI_LANGCHAIN_AVAILABLE = True
    except Exception:
        LangChain = None
    PANDAS_AI_AVAILABLE = True
except Exception:
    PandasAI = None
    LangChain = None
    PANDAS_AI_AVAILABLE = False

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
DB_PATH = ":memory:"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    """Parse SQL and ensure it only references known tables and columns."""
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            raise ValueError("Could not parse SQL")
        # Remove contents of single- and double-quoted string literals so values like 'III' are not treated as identifiers
        try:
            cleaned = re.sub(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")", "", sql)
        except Exception:
            cleaned = sql
        text = cleaned.lower()
        used_tables = [t for t in TABLE_COLUMNS.keys() if t.lower() in text]
        if not used_tables:
            raise Exception("SQL references unknown tables.")
        identifiers = set(re.findall(r"\b[\w_]+\b", cleaned))
        keywords = {k.lower() for k in ['select','from','where','group','by','order','limit','and','or','as','on','join','left','right','inner','outer','having','count','sum','avg','min','max']}
        idents = [i for i in identifiers if i.lower() not in keywords]
        for ident in idents:
            found = False
            for t in used_tables:
                if ident in TABLE_COLUMNS.get(t, []):
                    found = True
                    break
            if not found and ident.lower() not in [t.lower() for t in TABLE_COLUMNS.keys()]:
                all_cols = []
                for cols in TABLE_COLUMNS.values():
                    all_cols.extend(cols)
                suggestions = difflib.get_close_matches(ident, all_cols, n=3, cutoff=0.6)
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise Exception(f"SQL references unknown column or identifier: {ident}.{suggestion_text}")
        return True
    except Exception as e:
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
    df.to_sql(table_name_safe, conn, if_exists="replace", index=False)
    cols = list(df.columns)
    TABLE_COLUMNS[table_name_safe] = cols
    COLUMN_NAME_MAP[table_name_safe] = col_map
    return table_name_safe


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
