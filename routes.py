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
    SQLPARSE_AVAILABLE, read_table_into_df, drop_table
)
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
    save_path = os.path.join(DATA_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(contents)
    table_name = load_dataframe_to_sql(df, os.path.splitext(filename)[0])
    UPLOADED_FILES[filename] = table_name
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
    validate_sql_safe(sql)
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
