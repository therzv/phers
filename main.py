from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core import initialize_data_folder, update_question_replacements
from routes import router


app = FastAPI(title="HR-Data Chat (Simple)")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.include_router(router)

# initialize on import
initialize_data_folder()
update_question_replacements()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
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

    # Optionally use pandas-ai for direct DataFrame querying (preferred when enabled)
    use_pandas_ai = os.environ.get('USE_PANDAS_AI', '0') in ['1', 'true', 'True']
    if use_pandas_ai and PANDAS_AI_AVAILABLE:
        # choose best candidate table
        candidates = score_candidate_tables(question)
        chosen = None
        if candidates:
            chosen = candidates[0]
        if not chosen and len(UPLOADED_FILES) == 1:
            fname, tbl = next(iter(UPLOADED_FILES.items()))
            chosen = {"filename": fname, "table": tbl, "score": 0.0}

        if not chosen:
            raise HTTPException(status_code=400, detail="No suitable table found to query.")

        tbl = chosen['table']
        try:
            df = pd.read_sql_query(f"select * from \"{tbl}\" limit 5000", conn)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load table {tbl}: {e}")

        # build pandas-ai LLM adapter: prefer LangChain adapter with local Ollama LLM
        try:
            adapter = None
            if PANDASAI_LANGCHAIN_AVAILABLE:
                try:
                    llm_for_adapter = get_llm()
                    adapter = LangChain(llm_for_adapter)
                except Exception:
                    adapter = None

            if adapter is None:
                # fallback: try pandasai OpenAI adapter if env key present
                try:
                    from pandasai.llm.openai import OpenAI as PandasOpenAI
                    if os.environ.get('OPENAI_API_KEY'):
                        adapter = PandasOpenAI(api_token=os.environ.get('OPENAI_API_KEY'))
                except Exception:
                    adapter = None

            if adapter is None:
                raise HTTPException(status_code=500, detail='No usable LLM adapter for pandas-ai (enable Ollama or set OPENAI_API_KEY)')

            pandas_ai = PandasAI(adapter)
            # run summary
            summary_resp = pandas_ai.run(df, prompt=question)
            summary_text = str(summary_resp)
            table_preview = []
            # try to explicitly request a DataFrame result
            try:
                df_result = pandas_ai.run(df, prompt=f"Return a pandas DataFrame (only data) with up to 10 rows answering: {question}")
                if hasattr(df_result, 'to_dict'):
                    table_preview = df_result.head(10).to_dict(orient='records')
            except Exception:
                pass

            return {"sql": None, "suggestions": [], "inferred_tables": [tbl], "summary": summary_text, "table_preview": table_preview}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pandas-AI error: {e}")

    # Default legacy SQL path (unchanged)
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

