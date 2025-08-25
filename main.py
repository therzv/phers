# main.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
from typing import Dict, Any, List, Optional
import uvicorn
from pydantic import BaseModel

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
    import sqlparse
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


def index_dataframe_to_chroma(df: pd.DataFrame, filename: str, table_name: str):
    """Index rows from a dataframe into chroma; store ids in CHROMA_IDS_BY_FILE."""
    if not CHROMA_AVAILABLE:
        return
    # register and mark start
    INDEXING_STATUS[filename] = {"status": "indexing", "started": datetime.datetime.now().isoformat(), "progress": 0.0, "message": ""}
    ids = []
    metadatas = []
    docs = []
    # chunk rows into documents of N rows each for better semantic grouping
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
            # update progress based on doc_count portion (reserve up to 60% for embedding)
            try:
                INDEXING_STATUS[filename]["progress"] = min(0.6, (doc_count / total_docs) * 0.6)
            except Exception:
                pass
    # remaining
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
        # compute embeddings (batch)
        INDEXING_STATUS[filename]["progress"] = max(INDEXING_STATUS[filename].get("progress", 0.0), 0.0)
        embs = embedding_model.encode(docs).tolist()
        # embedding done -> update progress to 0.9 before adding to chroma
        INDEXING_STATUS[filename]["progress"] = 0.9
        # sanitize metadatas: chroma expects primitive metadata values; convert lists/dicts to JSON strings
        try:
            sanitized_metas = []
            for m in metadatas:
                sm = {}
                for k, v in m.items():
                    if v is None or isinstance(v, (str, int, float, bool)):
                        sm[k] = v
                    else:
                        # fallback: stringify lists/dicts
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
    """Build regex replacements from COLUMN_NAME_MAP so user text maps to sanitized names."""
    global QUESTION_REPLACEMENTS
    repls = []
    for table, mapping in COLUMN_NAME_MAP.items():
        # allow matching table name (sanitized) as-is
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
                # allow optional plural 's' on the last word to match simple pluralization
                last = re.escape(words[-1]) + r"s?"
                pat = r"\b" + sep_pattern.join([re.escape(w) for w in words[:-1]] + [last]) + r"\b"
            else:
                # allow simple plural forms (word and word+s)
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
    # token-level fuzzy replacement for simple typos/plurals
    try:
        # build candidate map: lower-case original -> sanitized
        cand_map = {}
        for tbl, mapping in COLUMN_NAME_MAP.items():
            cand_map[tbl.lower()] = tbl
            for san, orig in mapping.items():
                if not orig:
                    continue
                cand_map[orig.lower()] = san
                cand_map[orig.lower() + 's'] = san
                # also split words
                for w in re.split(r"\s+", orig.lower()):
                    if w:
                        cand_map[w] = san
                        cand_map[w + 's'] = san
        # token replace
        tokens = re.split(r"(\W+)", out)
        for i, tok in enumerate(tokens):
            lower = tok.lower()
            if lower and lower.isalpha() and len(lower) > 2:
                # direct map
                if lower in cand_map:
                    tokens[i] = cand_map[lower]
                else:
                    # fuzzy match against candidate keys
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
        tokens = [t for stmt in parsed for t in stmt.tokens]
        text = sql.lower()
        # quick check: all table/column names used must be in TABLE_COLUMNS
        # naive approach: search for known table names and column names in the SQL text
        used_tables = [t for t in TABLE_COLUMNS.keys() if t.lower() in text]
        if not used_tables:
            raise HTTPException(status_code=400, detail="SQL references unknown tables.")
        # columns: ensure any identifier that looks like column is present in at least one used table
        # split by non-word chars and test
        identifiers = set(re.findall(r"\b[\w_]+\b", sql))
        # remove SQL keywords
        keywords = {k.lower() for k in ['select','from','where','group','by','order','limit','and','or','as','on','join','left','right','inner','outer','having','count','sum','avg','min','max']}
        idents = [i for i in identifiers if i.lower() not in keywords]
        for ident in idents:
            found = False
            for t in used_tables:
                if ident in TABLE_COLUMNS.get(t, []):
                    found = True
                    break
            if not found and ident.lower() not in [t.lower() for t in TABLE_COLUMNS.keys()]:
                # try fuzzy match suggestions
                all_cols = []
                for cols in TABLE_COLUMNS.values():
                    all_cols.extend(cols)
                suggestions = difflib.get_close_matches(ident, all_cols, n=3, cutoff=0.6)
                suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise HTTPException(status_code=400, detail=f"SQL references unknown column or identifier: {ident}.{suggestion_text}")
        return True
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL validation error: {e}")


# Ollama wrapper import order (try official langchain_ollama, then langchain_community, then legacy)
LLM_CLASS = None
LLM_SOURCE = None
try:
    # official wrapper (recommended)
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

DATA_DIR = "data"
DB_PATH = ":memory:"  # in-memory sqlite (fast, ephemeral). change to file path if persistence desired.

app = FastAPI(title="HR-Data Chat (Simple)")

# Serve index.html from the same directory
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Global sqlite connection and table list
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
TABLE_COLUMNS: Dict[str, list] = {}  # table -> [columns...]
UPLOADED_FILES: Dict[str, str] = {}  # original filename -> sanitized table name
COLUMN_NAME_MAP: Dict[str, Dict[str, str]] = {}  # table -> {sanitized_col: original_col}

# LLM wrapper helper (Ollama/phi4 via LangChain)
def get_llm():
    """Return an Ollama LLM instance.

    Notes:
    - Requires the Python package that provides the Ollama wrapper (langchain-community or compatible langchain).
    - Also requires the Ollama daemon/server to be running and the named model available.
    - Configure via env vars: OLLAMA_BASE_URL (default http://127.0.0.1:11434) and OLLAMA_MODEL (default phi4).
    """
    if LLM_CLASS is None:
        raise RuntimeError(
            "Ollama LLM wrapper not available. Install 'langchain-ollama' or 'langchain-community' and 'ollama' python packages."
        )
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_MODEL", "phi4")
    try:
        # Try common keyword args first
        return LLM_CLASS(model=model, base_url=base_url)
    except TypeError:
        # Try alternate constructor signatures
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
    # sanitize table name
    table_name_safe = re.sub(r"[^\w\d_]", "_", table_name).lower()
    # sanitize columns: map spaces/special chars to underscore so SQL identifiers are safe
    orig_cols = list(df.columns)
    sanitized_cols = []
    seen = {}
    col_map: Dict[str, str] = {}
    for c in orig_cols:
        s = re.sub(r"[^\w\d_]", "_", str(c)).strip()
        if not s:
            s = "col"
        # avoid duplicates
        base = s
        i = 1
        while s in seen:
            s = f"{base}_{i}"
            i += 1
        seen[s] = True
        sanitized_cols.append(s)
        col_map[s] = str(c)
    # rename dataframe columns to sanitized names for SQL
    df = df.copy()
    df.columns = sanitized_cols
    df.to_sql(table_name_safe, conn, if_exists="replace", index=False)
    cols = list(df.columns)
    TABLE_COLUMNS[table_name_safe] = cols
    COLUMN_NAME_MAP[table_name_safe] = col_map
    return table_name_safe

def initialize_data_folder():
    os.makedirs(DATA_DIR, exist_ok=True)
    # load CSV and XLSX from data dir
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
                # load first sheet
                df = pd.read_excel(path)
                tbl = os.path.splitext(filename)[0]
                table_name_safe = load_dataframe_to_sql(df, tbl)
                UPLOADED_FILES[filename] = table_name_safe
                print(f"Loaded XLSX: {filename} -> table {tbl}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

        # optionally index in chroma — run indexing in background thread to avoid blocking startup
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

# Call at startup
initialize_data_folder()
update_question_replacements()

# Simple schema builder for prompt
def build_schema_description():
    if not TABLE_COLUMNS:
        return "NO_TABLES"
    parts = []
    for t, cols in TABLE_COLUMNS.items():
        # show sanitized column names and original names (if available)
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
    # ✅ updated regex: allow letters, numbers, underscores, quotes, spaces, punctuation
    if not re.match(r"^[\s\w\d\.\,\*\(\)=<>!\"'%-]+$", sql):
        raise HTTPException(status_code=400, detail="SQL contains unexpected characters.")
    return True

# Prompt templates
SQL_PROMPT_TEMPLATE = """You are a strict SQL generator. The user will ask a question to query HR tables.
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
\"\"\"
{question}
\"\"\"

Remember: return ONLY:
<SQL>SELECT ...</SQL>
"""

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

class ChatRequest(BaseModel):
    question: str


class ExecuteRequest(BaseModel):
    sql: str
    question: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def index():
    # serve the index.html file in the same folder
    return FileResponse("index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    filename = file.filename
    contents = await file.read()
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv or .xlsx.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")
    # store to disk (so server restart persists) and also load into sqlite
    save_path = os.path.join(DATA_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(contents)
    table_name = load_dataframe_to_sql(df, os.path.splitext(filename)[0])
    # record uploaded file mapping
    UPLOADED_FILES[filename] = table_name
    # ensure indexing status is present so the UI shows progress/finished even when CHROMA is not available
    try:
        INDEXING_STATUS[filename] = {"status": "indexing", "started": datetime.datetime.now().isoformat(), "progress": 0.0, "message": ""}
    except Exception:
        pass
    # schedule chroma indexing in background (threaded function)
    try:
        if CHROMA_AVAILABLE:
            def _index():
                try:
                    index_dataframe_to_chroma(df, filename, table_name)
                except Exception:
                    pass
            t = threading.Thread(target=_index, daemon=True)
            t.start()
        else:
            # If chroma isn't available, mark indexing as done immediately (we still loaded into SQL)
            try:
                rows = len(df)
                chunk = int(os.environ.get("RAG_CHUNK_ROWS", "8"))
                docs = max(1, math.ceil(rows / chunk))
                INDEXING_STATUS[filename]["progress"] = 1.0
                INDEXING_STATUS[filename]["status"] = "done"
                INDEXING_STATUS[filename]["finished"] = datetime.datetime.now().isoformat()
                INDEXING_STATUS[filename]["count"] = docs
            except Exception:
                pass
    except Exception:
        pass

    return {"status": "ok", "table": table_name, "columns": TABLE_COLUMNS[table_name]}


def score_candidate_tables(question: str) -> List[Dict[str, Any]]:
    """Return a scored list of uploaded tables likely relevant to the question.
    Uses simple substring matching + chroma relevance (if available) + fuzzy scoring.
    """
    q = question.lower()
    candidates: List[Dict[str, Any]] = []
    for fname, tbl in UPLOADED_FILES.items():
        score = 0.0
        # exact/table name match
        if tbl.lower() in q or os.path.splitext(fname)[0].lower() in q:
            score += 1.0
        # column name overlap
        cols = TABLE_COLUMNS.get(tbl, [])
        for c in cols:
            if c.lower() in q:
                score += 0.5
        candidates.append({"filename": fname, "table": tbl, "score": score})

    # chroma boost
    if CHROMA_AVAILABLE and candidates:
        try:
            res = chroma_collection.query(query_texts=[question], n_results=3, include=['metadatas','distances'])
            hits = res.get('metadatas', [[]])[0]
            dists = res.get('distances', [[]])[0]
            for i, h in enumerate(hits):
                if isinstance(h, dict) and 'file' in h and 'table' in h:
                    for c in candidates:
                        if c['table'] == h['table']:
                            # higher boost for closer distances
                            try:
                                c['score'] += max(0, 1.0 - float(dists[i]))
                            except Exception:
                                c['score'] += 0.5
        except Exception:
            pass

    # fuzzy similarity boost (filename vs question)
    names = [os.path.splitext(c['filename'])[0] for c in candidates]
    for c in candidates:
        base = os.path.splitext(c['filename'])[0]
        match = difflib.get_close_matches(base.lower(), [q], n=1, cutoff=0.6)
        if match:
            c['score'] += 0.4

    # sort by score desc
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates


@app.post('/choose_file')
async def choose_file(req: ChatRequest):
    """Return ranked candidate files/tables for a given question to let the user pick when ambiguous."""
    q = req.question or ""
    candidates = score_candidate_tables(q)
    # only return the top few
    return {"candidates": candidates[:6]}


@app.get('/index_status')
async def index_status():
    """Return indexing status for all files."""
    # include files not yet indexed
    out = {}
    for fname in UPLOADED_FILES.keys():
        st = INDEXING_STATUS.get(fname)
        if not st:
            out[fname] = {"status": "pending", "progress": 0.0}
        else:
            out[fname] = st
    return {"status": out}


@app.get("/files")
async def list_files():
    """Return list of uploaded files and corresponding table/columns."""
    files = []
    for orig, tbl in UPLOADED_FILES.items():
        files.append({"filename": orig, "table": tbl, "columns": TABLE_COLUMNS.get(tbl, [])})
    return {"files": files}


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete uploaded file, remove SQL table and optional chroma entries."""
    # remove file from disk
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove file: {e}")
    # drop sql table
    tbl = UPLOADED_FILES.pop(filename, None)
    if tbl:
        try:
            conn.execute(f"DROP TABLE IF EXISTS \"{tbl}\"")
            TABLE_COLUMNS.pop(tbl, None)
        except Exception:
            pass
    # delete from chroma
    if CHROMA_AVAILABLE and filename in CHROMA_IDS_BY_FILE:
        try:
            chroma_collection.delete(ids=CHROMA_IDS_BY_FILE[filename])
        except Exception:
            pass
    CHROMA_IDS_BY_FILE.pop(filename, None)
    return {"ok": True}

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")
    if not TABLE_COLUMNS:
        return JSONResponse({"error": "No tables loaded. Upload CSV/XLSX files first."}, status_code=400)

    # Build prompt for SQL generation
    # normalize user text to replace known original column/table mentions with sanitized identifiers
    normalized_q = normalize_question_text(question)
    schema = build_schema_description()
    sql_prompt = SQL_PROMPT_TEMPLATE.format(schema=schema, question=normalized_q)

    # Generate SQL (preview only)
    try:
        llm = get_llm()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM not configured: {e}")

    llm_response = llm.predict(sql_prompt)
    if not llm_response:
        raise HTTPException(status_code=500, detail="Empty LLM response.")

    m = re.search(r"<SQL>(.*?)</SQL>", llm_response, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise HTTPException(status_code=500, detail="LLM did not return SQL within <SQL> tags.")
    sql = m.group(1).strip()

    # Basic safety checks for the preview
    validate_sql_safe(sql)

    # Collect lightweight suggestions (don't block preview)
    suggestions = []
    try:
        if SQLPARSE_AVAILABLE:
            # run validation but capture errors as suggestions rather than abort
            try:
                validate_sql_against_schema(sql)
            except HTTPException as he:
                suggestions.append(str(he.detail))
    except Exception:
        pass

    # Try to infer relevant table(s) to help the UI suggest which uploaded file(s) are relevant.
    def infer_tables_from_sql(sql_text: str) -> List[str]:
        txt = sql_text.lower()
        found = [t for t in TABLE_COLUMNS.keys() if t.lower() in txt]
        # fallback: try chroma relevance if available
        if not found and CHROMA_AVAILABLE:
            try:
                # ask chroma for the most relevant docs given the question
                results = chroma_collection.query(query_texts=[normalized_q], n_results=3, include=['metadatas'])
                hits = results['metadatas'][0]
                for h in hits:
                    if isinstance(h, dict) and 'table' in h:
                        if h['table'] not in found:
                            found.append(h['table'])
            except Exception:
                pass
        return found

    inferred_tables = infer_tables_from_sql(sql)

    return {
        "sql": sql,
        "suggestions": suggestions,
        "inferred_tables": inferred_tables,
    }


@app.post("/execute_sql")
async def execute_sql(req: ExecuteRequest):
    sql = req.sql.strip()
    question = (req.question or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="Empty SQL.")
    # final safety checks
    validate_sql_safe(sql)
    if SQLPARSE_AVAILABLE:
        validate_sql_against_schema(sql)

    # Execute SQL
    try:
        df_result = pd.read_sql_query(sql, conn)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution error: {e}")

    rows = df_result.to_dict(orient="records")

    # Map sanitized column names back to original for display
    display_rows = []
    if rows:
        tbl_candidates = list(TABLE_COLUMNS.keys())
        inferred_table = None
        for t in tbl_candidates:
            if t in sql.lower():
                inferred_table = t
                break
        col_map = COLUMN_NAME_MAP.get(inferred_table, {}) if inferred_table else {}
        for r in rows:
            mapped = {}
            for k, v in r.items():
                mapped[col_map.get(k, k)] = v
            display_rows.append(mapped)
    else:
        display_rows = rows

    # Summarize
    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
        question=question or sql,
        sql=sql,
        rows_json=json.dumps(rows[:200], default=str)
    )
    summary_text = ""
    table_preview = rows[:10]
    try:
        summary_llm = get_llm()
        summary_resp = summary_llm.predict(summary_prompt)
        try:
            parsed = json.loads(summary_resp)
            summary_text = parsed.get("text", "")
            table_preview = parsed.get("table_preview", rows[:10])
        except Exception:
            summary_text = summary_resp.strip()
    except Exception:
        summary_text = ""

    return {
        "sql": sql,
        "rows": rows,
        "summary": summary_text or (f"{len(rows)} rows returned."),
        "table_preview": table_preview,
        "display_rows": display_rows,
    }

if __name__ == "__main__":
    # Run with: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

