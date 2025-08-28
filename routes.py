from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi import BackgroundTasks
import os
import io
import json
import pandas as pd
from typing import List

from core import (
    DATA_DIR, UPLOADED_FILES, TABLE_COLUMNS, COLUMN_NAME_MAP,
    load_dataframe_to_sql, initialize_data_folder, index_dataframe_to_chroma,
    INDEXING_STATUS, CHROMA_AVAILABLE, CHROMA_IDS_BY_FILE, chroma_collection,
    score_candidate_tables, normalize_question_text, build_schema_description,
    SQL_PROMPT_TEMPLATE, validate_sql_safe, validate_sql_against_schema,
    get_llm, PANDAS_AI_AVAILABLE, PANDASAI_LANGCHAIN_AVAILABLE, PandasAI, LangChain,
    SQLPARSE_AVAILABLE, read_table_into_df, drop_table, suggest_column_alternatives
)
from core import conn, ENGINE, SUMMARY_PROMPT_TEMPLATE
from sqlalchemy import text
import threading
import re
import sys
import importlib

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("index.html")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    filename = file.filename
    contents = await file.read()
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    # Parse the file
    try:
        if filename.lower().endswith(".csv"):
            try:
                # Try different encodings for CSV
                df = pd.read_csv(io.BytesIO(contents))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv or .xlsx files.")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="File contains no data or has invalid format")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Check if DataFrame has data
    if df.empty:
        raise HTTPException(status_code=400, detail="File contains no valid data")
    
    # Save file to data directory
    save_path = os.path.join(DATA_DIR, filename)
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Load to SQL database
    try:
        table_name = load_dataframe_to_sql(df, os.path.splitext(filename)[0])
        UPLOADED_FILES[filename] = table_name
    except Exception as e:
        # Clean up saved file if database loading fails
        try:
            os.remove(save_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to process data: {str(e)}")
    try:
        if CHROMA_AVAILABLE:
            def _index():
                try:
                    index_dataframe_to_chroma(df, filename, table_name)
                except Exception:
                    pass
            t = threading.Thread(target=_index, daemon=True)
            t.start()
    except Exception:
        pass
    return {"status": "ok", "table": table_name, "columns": TABLE_COLUMNS[table_name]}


@router.post('/choose_file')
async def choose_file(req: dict):
    q = req.get('question','')
    candidates = score_candidate_tables(q)
    return {"candidates": candidates[:6]}


@router.get('/index_status')
async def index_status():
    out = {}
    for fname in UPLOADED_FILES.keys():
        st = INDEXING_STATUS.get(fname)
        if not st:
            out[fname] = {"status": "pending", "progress": 0.0}
        else:
            out[fname] = st
    return {"status": out}


@router.get('/files')
async def list_files():
    files = []
    for orig, tbl in UPLOADED_FILES.items():
        files.append({"filename": orig, "table": tbl, "columns": TABLE_COLUMNS.get(tbl, [])})
    return {"files": files}


@router.delete('/files/{filename}')
async def delete_file(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove file: {e}")
    tbl = UPLOADED_FILES.pop(filename, None)
    if tbl:
        try:
            drop_table(tbl)
            TABLE_COLUMNS.pop(tbl, None)
        except Exception:
            pass
    if CHROMA_AVAILABLE and filename in CHROMA_IDS_BY_FILE:
        try:
            chroma_collection.delete(ids=CHROMA_IDS_BY_FILE[filename])
        except Exception:
            pass
    CHROMA_IDS_BY_FILE.pop(filename, None)
    return {"ok": True}


@router.post('/chat')
async def chat(req: dict):
    question = (req.get('question') or '').strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")
    if not TABLE_COLUMNS:
        return JSONResponse({"error": "No tables loaded. Upload CSV/XLSX files first."}, status_code=400)

    use_pandas_ai = os.environ.get('USE_PANDAS_AI', '0') in ['1', 'true', 'True']
    if use_pandas_ai:
        # try lazy import of pandas-ai and adapters
        try:
            from pandasai import PandasAI
            try:
                from pandasai.llm.langchain import LangChain
                pandasai_langchain = True
            except Exception:
                LangChain = None
                pandasai_langchain = False
        except Exception:
            PandasAI = None
            pandasai_langchain = False

        if not PandasAI:
            raise HTTPException(status_code=500, detail="pandas-ai not installed in the running Python environment.")
        candidates = score_candidate_tables(question)
        chosen = candidates[0] if candidates else None
        if not chosen and len(UPLOADED_FILES) == 1:
            fname, tbl = next(iter(UPLOADED_FILES.items()))
            chosen = {"filename": fname, "table": tbl, "score": 0.0}
        if not chosen:
            raise HTTPException(status_code=400, detail="No suitable table found to query.")
        tbl = chosen['table']
        try:
            df = read_table_into_df(tbl, limit=5000)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load table {tbl}: {e}")
        try:
            adapter = None
            if pandasai_langchain:
                try:
                    llm_for_adapter = get_llm()
                    adapter = LangChain(llm_for_adapter)
                except Exception:
                    adapter = None
            if adapter is None:
                try:
                    from pandasai.llm.openai import OpenAI as PandasOpenAI
                    if os.environ.get('OPENAI_API_KEY'):
                        adapter = PandasOpenAI(api_token=os.environ.get('OPENAI_API_KEY'))
                except Exception:
                    adapter = None
            if adapter is None:
                raise HTTPException(status_code=500, detail='No usable LLM adapter for pandas-ai (enable Ollama or set OPENAI_API_KEY)')
            pandas_ai = PandasAI(adapter)
            summary_resp = pandas_ai.run(df, prompt=question)
            summary_text = str(summary_resp)
            table_preview = []
            try:
                df_result = pandas_ai.run(df, prompt=f"Return a pandas DataFrame (only data) with up to 10 rows answering: {question}")
                if hasattr(df_result, 'to_dict'):
                    table_preview = df_result.head(10).to_dict(orient='records')
            except Exception:
                pass
            return {"suggestions": [], "inferred_tables": [tbl], "summary": summary_text, "table_preview": table_preview}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pandas-AI error: {e}")

    # fallback SQL flow (use existing prompt and LLM)
    normalized_q = normalize_question_text(question)
    schema = build_schema_description()
    sql_prompt = SQL_PROMPT_TEMPLATE.format(schema=schema, question=normalized_q)
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
    # attempt to auto-quote unquoted textual literals (helps LLM outputs like: WHERE status = In Repair)
    try:
        from core import auto_quote_string_literals
        sql = auto_quote_string_literals(sql)
    except Exception:
        pass
    # Clean and validate SQL (returns cleaned version)
    sql = validate_sql_safe(sql)
    suggestions = []
    try:
        if SQLPARSE_AVAILABLE:
            try:
                validate_sql_against_schema(sql)
            except Exception as he:
                suggestions.append(str(he))
    except Exception:
        pass
    def infer_tables_from_sql(sql_text: str) -> List[str]:
        txt = sql_text.lower()
        found = [t for t in TABLE_COLUMNS.keys() if t.lower() in txt]
        if not found and CHROMA_AVAILABLE:
            try:
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
    return {"sql": sql, "suggestions": suggestions, "inferred_tables": inferred_tables}


@router.get('/health')
async def health():
    info = {}
    info['python_executable'] = sys.executable
    for mod in ('pandasai', 'pandasai.llm.langchain', 'pandasai.llm.openai', 'langchain'):
        try:
            importlib.import_module(mod)
            info[mod] = True
        except Exception:
            info[mod] = False
    try:
        llm_ok = False
        try:
            lm = get_llm()
            llm_ok = True
        except Exception as e:
            info['llm_error'] = str(e)
        info['llm_constructed'] = llm_ok
    except Exception as e:
        info['llm_constructed'] = False
        info['llm_error'] = str(e)
    info['use_pandas_ai_env'] = os.environ.get('USE_PANDAS_AI', '0')
    info['ollama_base_url'] = os.environ.get('OLLAMA_BASE_URL')
    info['ollama_model'] = os.environ.get('OLLAMA_MODEL')
    return info


@router.get('/debug_sql')
async def debug_sql(question: str):
    """Dev helper: returns the raw SQL produced by the LLM and the sanitized SQL we plan to run.

    Only intended for debugging in dev. Do not expose in production.
    """
    # build prompt
    try:
        normalized_q = normalize_question_text(question)
        schema = build_schema_description()
        sql_prompt = SQL_PROMPT_TEMPLATE.format(schema=schema, question=normalized_q)
        llm = get_llm()
        llm_response = llm.predict(sql_prompt)
        m = re.search(r"<SQL>(.*?)</SQL>", llm_response, flags=re.DOTALL | re.IGNORECASE)
        raw_sql = m.group(1).strip() if m else None
        try:
            from core import auto_quote_string_literals
            sanitized = auto_quote_string_literals(raw_sql) if raw_sql else None
        except Exception:
            sanitized = None
        return {"raw_sql": raw_sql, "sanitized_sql": sanitized, "llm_response": llm_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/execute_sql')
async def execute_sql(req: dict):
    sql = (req.get('sql') or '').strip()
    question = (req.get('question') or '').strip()
    if not sql:
        raise HTTPException(status_code=400, detail="Empty SQL.")
    # quick check: ensure the SQL references at least one known uploaded table
    used_tables = [t for t in TABLE_COLUMNS.keys() if re.search(r"\b" + re.escape(t) + r"\b", sql, flags=re.IGNORECASE)]
    if not used_tables:
        # offer candidate files/tables to the UI so it can suggest choices
        candidates = score_candidate_tables(question or sql)
        # return a friendly error with candidates (UI will render suggestions)
        raise HTTPException(status_code=400, detail="No known table referenced in the SQL.", headers={"X-Candidates": json.dumps(candidates)})
    # safety checks and cleaning
    sql = validate_sql_safe(sql)
    if SQLPARSE_AVAILABLE:
        validate_sql_against_schema(sql)

    try:
        if ENGINE is not None:
            # pandas accepts SQLAlchemy engine
            df = pd.read_sql_query(sql, ENGINE)
        elif conn is not None:
            df = pd.read_sql_query(sql, conn)
        else:
            raise Exception('No database engine available')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution error: {e}")

    # Convert DataFrame to records with proper JSON serialization
    try:
        rows = df.to_dict(orient='records')
        # Clean up any NaN or infinity values that can't be JSON serialized
        for row in rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif isinstance(value, float) and (value != value or abs(value) == float('inf')):
                    row[key] = None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query results: {e}")

    # Summarize the rows with the LLM if available
    summary_text = ""
    table_preview = rows[:10]
    try:
        # Safely serialize to JSON
        rows_json = json.dumps(rows[:200], default=str, ensure_ascii=False)
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(question=question or sql, sql=sql, rows_json=rows_json)
        summary_llm = get_llm()
        summary_resp = summary_llm.predict(summary_prompt)
        try:
            parsed = json.loads(summary_resp)
            summary_text = parsed.get('text', '')
            table_preview = parsed.get('table_preview', table_preview)
        except Exception:
            summary_text = summary_resp.strip()
    except Exception:
        summary_text = ""

    # Map sanitized column names back to original for display
    display_rows = []
    if rows:
        inferred_table = None
        for t in TABLE_COLUMNS.keys():
            if t.lower() in sql.lower():
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

    # If zero rows, try to suggest close matches for RHS literals in equality expressions
    suggestions = []
    try:
        if not rows:
            # find simple equality patterns like: column = 'value'
            # handle double or single quotes
            eqs = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*=\s*'(.*?)'", sql, flags=re.IGNORECASE)
            eqs += re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b\s*=\s*"(.*?)"', sql, flags=re.IGNORECASE)
            # try for IN lists: col IN ('a','b')
            in_matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s+IN\s*\((.*?)\)", sql, flags=re.IGNORECASE)
            # infer table name from SQL text (best-effort)
            inferred_table = None
            for t in TABLE_COLUMNS.keys():
                if t.lower() in sql.lower():
                    inferred_table = t
                    break
            for col, val in eqs:
                if not inferred_table:
                    continue
                # fetch distinct values for column
                candidates = []
                try:
                    q = f'SELECT DISTINCT "{col}" as v FROM "{inferred_table}" WHERE "{col}" IS NOT NULL LIMIT 500'
                    if ENGINE is not None:
                        with ENGINE.connect() as cx:
                            res = cx.execute(text(q))
                            candidates = [ (r[0] if isinstance(r, (list, tuple)) else r) for r in res.fetchall() ]
                    elif conn is not None:
                        cur = conn.cursor()
                        cur.execute(q)
                        candidates = [r[0] for r in cur.fetchall()]
                except Exception:
                    candidates = []
                # normalize and match
                cand_strs = [str(c) for c in candidates if c is not None]
                lower_map = {s.lower(): s for s in cand_strs}
                import difflib as _dif
                matches = _dif.get_close_matches(val.lower(), [s.lower() for s in cand_strs], n=5, cutoff=0.6)
                matched = [lower_map[m] for m in matches if m in lower_map]
                # also include exact case-insensitive matches
                if val.lower() in lower_map and lower_map[val.lower()] not in matched:
                    matched.insert(0, lower_map[val.lower()])
                if matched:
                    suggested_sqls = []
                    for mv in matched:
                        # create a suggested SQL by replacing the first occurrence of the literal (keep quotes)
                        safe_mv = mv.replace("'","''")
                        pattern = re.compile(r"(\b" + re.escape(col) + r"\b\s*=\s*)(?:'[^']*'|\"[^\"]*\")", flags=re.IGNORECASE)
                        def _repl(m):
                            return m.group(1) + "'" + safe_mv + "'"
                        sug = pattern.sub(_repl, sql, count=1)
                        # if the regex didn't replace (different quoting), try simple replace of the original value
                        if sug == sql:
                            sug = sql.replace("'"+val+"'", "'"+safe_mv+"'")
                        suggested_sqls.append(sug)
                    suggestions.append({"column": col, "original": val, "candidates": matched, "suggested_sqls": suggested_sqls})
            # handle IN-list suggestions (simple: if any list item fuzzy-matches known values)
            for col, list_body in in_matches:
                if not inferred_table:
                    continue
                list_items = re.findall(r"'([^']*)'|\"([^\"]*)\"", list_body)
                # flatten tuples
                items = [a or b for a,b in list_items]
                # fetch candidates as above
                candidates = []
                try:
                    q = f'SELECT DISTINCT "{col}" as v FROM "{inferred_table}" WHERE "{col}" IS NOT NULL LIMIT 500'
                    if ENGINE is not None:
                        with ENGINE.connect() as cx:
                            res = cx.execute(text(q))
                            candidates = [ (r[0] if isinstance(r, (list, tuple)) else r) for r in res.fetchall() ]
                    elif conn is not None:
                        cur = conn.cursor()
                        cur.execute(q)
                        candidates = [r[0] for r in cur.fetchall()]
                except Exception:
                    candidates = []
                cand_strs = [str(c) for c in candidates if c is not None]
                import difflib as _dif
                matched_map = {}
                for it in items:
                    matches = _dif.get_close_matches(it.lower(), [s.lower() for s in cand_strs], n=3, cutoff=0.6)
                    matched = [ { 'original': it, 'matches': [ { 'value': (lambda m: ( [s for s in cand_strs if s.lower()==m][0] if any(s.lower()==m for s in cand_strs) else m ))(m) } for m in matches ] } ]
                    if matched:
                        matched_map[it] = matched
                if matched_map:
                    # produce a suggested SQL by replacing each original item with first matched candidate
                    sug_sql = sql
                    for it, ml in matched_map.items():
                        first = ml[0]['matches'][0]['value']
                        safe_mv = first.replace("'","''")
                        sug_sql = sug_sql.replace("'"+it+"'", "'"+safe_mv+"'")
                    suggestions.append({"column": col, "original_in_list": items, "candidates_map": matched_map, "suggested_sqls": [sug_sql]})
    except Exception:
        suggestions = []

    return {"sql": sql, "rows": rows, "summary": summary_text or (f"{len(rows)} rows returned."), "table_preview": table_preview, "display_rows": display_rows, "suggestions": suggestions}


@router.get('/db_info')
async def db_info():
    """Return DB file and list of tables (if using sqlite)."""
    db_path = os.path.join(DATA_DIR, 'phers.db')
    tables = []
    try:
        if ENGINE is not None:
            with ENGINE.connect() as cx:
                res = cx.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [r[0] for r in res.fetchall()]
        elif conn is not None:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
    except Exception:
        tables = []
    return {"db_path": db_path, "tables": tables}
