from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi import BackgroundTasks
import os
import io
import json
import pandas as pd
from typing import List

from core import (
    DATA_DIR, UPLOADED_FILES, TABLE_COLUMNS, COLUMN_NAME_MAP, ACTIVE_FILES,
    load_dataframe_to_sql, initialize_data_folder, index_dataframe_to_chroma,
    INDEXING_STATUS, CHROMA_AVAILABLE, CHROMA_IDS_BY_FILE, chroma_collection,
    score_candidate_tables, normalize_question_text, build_schema_description,
    SQL_PROMPT_TEMPLATE, validate_sql_safe, validate_sql_against_schema,
    get_llm, PANDAS_AI_AVAILABLE, PANDASAI_LANGCHAIN_AVAILABLE, PandasAI, LangChain,
    SQLPARSE_AVAILABLE, read_table_into_df, drop_table, suggest_column_alternatives,
    get_active_files, generate_smart_suggestions, SANITIZATION_AVAILABLE, COLUMN_INTELLIGENCE_AVAILABLE
)

# Import sanitization modules
if SANITIZATION_AVAILABLE:
    from sanitization import data_sanitizer
    from sql_security import sql_security
    print("Sanitization modules loaded in routes.py")
else:
    print("Warning: Sanitization modules not available in routes.py")

# Import column intelligence modules  
if COLUMN_INTELLIGENCE_AVAILABLE:
    from column_intelligence import column_intelligence
    from dynamic_mapping import dynamic_mapper
    print("Column intelligence modules loaded in routes.py")
else:
    print("Warning: Column intelligence modules not available in routes.py")

# Import Phase 2: Advanced Query Intelligence modules
QUERY_INTELLIGENCE_AVAILABLE = False
try:
    from query_intelligence import query_intelligence
    from multi_table_intelligence import multi_table_intelligence  
    from context_engine import context_engine
    QUERY_INTELLIGENCE_AVAILABLE = True
    print("Phase 2: Advanced Query Intelligence modules loaded")
except ImportError as e:
    print(f"Warning: Phase 2 modules not available: {e}")
    QUERY_INTELLIGENCE_AVAILABLE = False

# Import Phase 3: AI-Powered Query Generation modules
AI_GENERATION_AVAILABLE = False
try:
    from ai_query_generator import ai_query_generator
    from query_refinement import query_refinement
    from conversation_manager import conversation_manager
    AI_GENERATION_AVAILABLE = True
    print("Phase 3: AI-Powered Query Generation modules loaded")
except ImportError as e:
    print(f"Warning: Phase 3 modules not available: {e}")
    AI_GENERATION_AVAILABLE = False

# Import Phase 4: Performance Optimization modules
PERFORMANCE_OPTIMIZATION_AVAILABLE = False
try:
    from intelligent_cache import get_intelligent_cache
    from database_optimizer import get_database_optimizer
    from performance_monitor import get_performance_monitor
    from memory_manager import get_memory_manager
    from scalability_manager import get_scalability_manager
    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
    print("Phase 4: Performance Optimization modules loaded")
except ImportError as e:
    print(f"Warning: Phase 4 modules not available: {e}")
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False

# Import Natural Language Response modules
NATURAL_RESPONSE_AVAILABLE = False
try:
    from data_cleansing_pipeline import enterprise_cleaner
    from natural_response_generator import natural_response_generator, QueryContext, ResponseContext
    from smart_suggestion_engine import smart_suggestion_engine, SuggestionContext
    NATURAL_RESPONSE_AVAILABLE = True
    print("Natural Language Response modules loaded")
except ImportError as e:
    print(f"Warning: Natural Language Response modules not available: {e}")
    NATURAL_RESPONSE_AVAILABLE = False
from core import conn, ENGINE, SUMMARY_PROMPT_TEMPLATE, reload_tables_from_database
from sqlalchemy import text
import threading
import re
import time


def _generate_dynamic_summary(question: str, row: dict) -> str:
    """
    Generate intelligent summary using dynamic column detection.
    
    Args:
        question: Original user question
        row: Single result row
        
    Returns:
        Contextual summary based on detected column roles
    """
    if not COLUMN_INTELLIGENCE_AVAILABLE:
        return "Found 1 matching record."
    
    question_lower = question.lower()
    
    # Analyze what the user is asking about
    if any(term in question_lower for term in ['asset', 'tag', 'id', 'identifier']):
        # Look for identifier-type columns in the row
        for col_name, col_value in row.items():
            if col_value and any(pattern in col_name.lower() for pattern in ['tag', 'id', 'asset', 'code']):
                return f"The identifier is: {col_value}"
    
    elif any(term in question_lower for term in ['manufacturer', 'brand', 'make', 'company']):
        # Look for manufacturer-type columns
        for col_name, col_value in row.items():
            if col_value and any(pattern in col_name.lower() for pattern in ['manufacturer', 'brand', 'company', 'make']):
                return f"The manufacturer is: {col_value}"
    
    elif any(term in question_lower for term in ['user', 'name', 'person', 'employee']):
        # User query - show multiple relevant fields
        info_parts = []
        
        # Find identifier
        for col_name, col_value in row.items():
            if col_value and any(pattern in col_name.lower() for pattern in ['tag', 'id', 'asset']):
                info_parts.append(f"ID: {col_value}")
                break
        
        # Find manufacturer/device
        for col_name, col_value in row.items():
            if col_value and any(pattern in col_name.lower() for pattern in ['manufacturer', 'brand']):
                info_parts.append(f"Device: {col_value}")
                break
        
        # Find product/item
        for col_name, col_value in row.items():
            if col_value and any(pattern in col_name.lower() for pattern in ['item', 'product', 'model']):
                info_parts.append(f"Item: {col_value}")
                break
        
        if info_parts:
            return f"Found record for user. {', '.join(info_parts)}"
    
    elif any(term in question_lower for term in ['product', 'item', 'model', 'device']):
        # Look for product-type columns
        for col_name, col_value in row.items():
            if col_value and any(pattern in col_name.lower() for pattern in ['item', 'product', 'model', 'name']):
                return f"The item is: {col_value}"
    
    elif any(term in question_lower for term in ['location', 'where', 'place']):
        # Look for location columns
        for col_name, col_value in row.items():
            if col_value and any(pattern in col_name.lower() for pattern in ['location', 'place', 'site', 'building']):
                return f"The location is: {col_value}"
    
    elif any(term in question_lower for term in ['price', 'cost', 'value', 'money']):
        # Look for money columns
        for col_name, col_value in row.items():
            if col_value and any(pattern in col_name.lower() for pattern in ['price', 'cost', 'value', 'amount']):
                return f"The cost is: {col_value}"
    
    # Default fallback
    return "Found 1 matching record."
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
    
    # Data Sanitation Step
    from core import sanitize_dataframe
    sanitation_report = None
    try:
        sanitation_result = sanitize_dataframe(df, filename)
        df_cleaned = sanitation_result["cleaned_df"]
        sanitation_report = sanitation_result["report"]
        needs_cleaning = sanitation_result["needs_cleaning"]
        
        # If sanitation found issues, save both versions and return report
        if needs_cleaning:
            # Use the cleaned DataFrame for processing
            df = df_cleaned
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data sanitation failed: {str(e)}")

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
        ACTIVE_FILES[filename] = True  # Activate new files by default
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
    # Ensure columns are JSON serializable (convert numpy types to Python types)
    columns = TABLE_COLUMNS[table_name]
    safe_columns = []
    for col in columns:
        if isinstance(col, str):
            safe_columns.append(col)
        else:
            safe_columns.append(str(col))
    
    response = {"status": "ok", "table": table_name, "columns": safe_columns}
    
    # Include sanitation report if cleaning was performed  
    if sanitation_report and sanitation_report.get("issues_found"):
        # Ensure sanitation report is also JSON serializable
        safe_sanitation_report = {}
        for key, value in sanitation_report.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                safe_sanitation_report[key] = value
            else:
                safe_sanitation_report[key] = str(value)
        response["sanitation_report"] = safe_sanitation_report
        response["sanitized"] = True
        
    return response


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
        files.append({
            "filename": orig, 
            "table": tbl, 
            "columns": TABLE_COLUMNS.get(tbl, []),
            "active": ACTIVE_FILES.get(orig, False)
        })
    return {"files": files}


@router.post('/files/{filename}/toggle')
async def toggle_file_activation(filename: str):
    """Toggle file activation status."""
    if filename not in UPLOADED_FILES:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Toggle the active status
    current_status = ACTIVE_FILES.get(filename, False)
    ACTIVE_FILES[filename] = not current_status
    
    return {
        "filename": filename,
        "active": ACTIVE_FILES[filename],
        "message": f"File {'activated' if ACTIVE_FILES[filename] else 'deactivated'}"
    }


@router.post('/files/activate-all')
async def activate_all_files():
    """Activate all uploaded files."""
    for filename in UPLOADED_FILES.keys():
        ACTIVE_FILES[filename] = True
    
    active_count = len(UPLOADED_FILES)
    return {"message": f"Activated {active_count} files", "active_files": list(UPLOADED_FILES.keys())}


@router.post('/files/deactivate-all')
async def deactivate_all_files():
    """Deactivate all uploaded files."""
    for filename in UPLOADED_FILES.keys():
        ACTIVE_FILES[filename] = False
    
    return {"message": "Deactivated all files", "active_files": []}


@router.get('/files/active')
async def get_active_files_info():
    """Get information about currently active files."""
    active_files = get_active_files()
    active_info = []
    for filename, table in active_files.items():
        active_info.append({
            "filename": filename,
            "table": table,
            "columns": TABLE_COLUMNS.get(table, [])
        })
    
    return {
        "active_files": active_info,
        "count": len(active_info),
        "schema": build_schema_description()
    }


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
    # Clean up activation status
    ACTIVE_FILES.pop(filename, None)
    if CHROMA_AVAILABLE and filename in CHROMA_IDS_BY_FILE:
        try:
            chroma_collection.delete(ids=CHROMA_IDS_BY_FILE[filename])
        except Exception:
            pass
    CHROMA_IDS_BY_FILE.pop(filename, None)
    return {"ok": True}


@router.post('/chat')
async def chat(req: dict):
    # Step 1: Sanitize user input immediately
    raw_question = (req.get('question') or '').strip()
    
    if not raw_question:
        raise HTTPException(status_code=400, detail="Empty question.")
    
    # Step 2: Apply input sanitization if available
    if SANITIZATION_AVAILABLE:
        question = data_sanitizer.sanitize_user_input(raw_question, "general")
        if not question:
            raise HTTPException(status_code=400, detail="Question became empty after security sanitization.")
        # Log if significant changes were made
        if len(question) < len(raw_question) * 0.7:
            print(f"Warning: User input heavily sanitized. Original: '{raw_question}' -> Sanitized: '{question}'")
    else:
        # Basic sanitization fallback
        question = re.sub(r'[<>"|\\;]', '', raw_question)
        question = question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question contains invalid characters.")
    
    if not TABLE_COLUMNS:
        return JSONResponse({"error": "No tables loaded. Upload CSV/XLSX files first."}, status_code=400)
    
    # Check if any files are active
    active_files = get_active_files()
    if not active_files:
        return JSONResponse({"error": "No files are currently active. Please activate at least one file to query data."}, status_code=400)
    
    # Phase 2: Advanced Query Intelligence
    query_analysis = None
    query_context = None
    multi_table_analysis = None
    
    if QUERY_INTELLIGENCE_AVAILABLE:
        # Step 1: Get context for query interpretation
        query_context = context_engine.get_context_for_query(question)
        print(f"Phase 2: Context confidence: {query_context.get('context_confidence', 0.0):.2f}")
        
        # Step 2: Analyze query intent with advanced intelligence
        query_analysis = query_intelligence.analyze_query_intent(question)
        print(f"Phase 2: Intent '{query_analysis['intent']}' with confidence {query_analysis['confidence']:.2f}")
        
        # Step 3: Analyze table relationships if multi-table query needed
        if query_analysis['complexity_score'] > 5:
            # Get table information for relationship analysis
            table_info = {}
            for table_name in TABLE_COLUMNS.keys():
                if table_name in dynamic_mapper.table_analyses:
                    table_info[table_name] = dynamic_mapper.table_analyses[table_name]
            
            if len(table_info) > 1:
                multi_table_analysis = multi_table_intelligence.analyze_table_relationships(table_info)
                print(f"Phase 2: Found {len(multi_table_analysis['direct_relationships'])} table relationships")
        
        # Add context hints to response for debugging/transparency
        if query_context.get('disambiguation_hints'):
            print(f"Phase 2: Disambiguation hints: {query_context['disambiguation_hints']}")
        if query_context.get('query_suggestions'):
            print(f"Phase 2: Query suggestions: {query_context['query_suggestions']}")
    else:
        print("Phase 2: Advanced Query Intelligence not available, using basic processing")
    
    # Phase 3: AI-Powered Query Generation
    conversation_result = None
    ai_generated_sql = None
    query_optimization = None
    
    if AI_GENERATION_AVAILABLE and QUERY_INTELLIGENCE_AVAILABLE:
        try:
            # Step 1: Process conversational context
            conversation_result = conversation_manager.process_conversational_query(
                question, query_analysis, None  # previous_results would come from session
            )
            print(f"Phase 3: Conversation type: {conversation_result.get('conversation_type', 'initial')}")
            
            # Step 2: Generate enhanced AI query
            table_info = {}
            for table_name in TABLE_COLUMNS.keys():
                if table_name in dynamic_mapper.table_analyses:
                    table_info[table_name] = dynamic_mapper.table_analyses[table_name]
            
            if table_info:
                schema = build_schema_description()
                ai_result = ai_query_generator.generate_enhanced_query(
                    question=conversation_result.get('enhanced_query', question),
                    query_analysis=query_analysis,
                    table_info=table_info,
                    schema=schema,
                    context=query_context,
                    multi_table_plan=multi_table_analysis if 'multi_table_analysis' in locals() else None
                )
                
                ai_generated_sql = ai_result.get('sql', '')
                print(f"Phase 3: AI generated SQL with {ai_result.get('confidence', 0.0):.2f} confidence")
                
                # Step 3: Optimize the generated query if needed
                if ai_generated_sql:
                    query_optimization = query_refinement.optimize_query(
                        ai_generated_sql, 
                        performance_metrics=None,  # Would come from execution
                        table_info=table_info
                    )
                    print(f"Phase 3: Applied {len(query_optimization.get('optimizations_applied', []))} optimizations")
        
        except Exception as phase3_error:
            print(f"Phase 3: Error in AI generation: {phase3_error}")
            # Continue with fallback processing
    else:
        print("Phase 3: AI-Powered Query Generation not available, using standard processing")

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
    
    # Phase 4: Performance Optimization - Initialize components
    phase4_cache_key = None
    phase4_performance_start = time.time()
    
    if PERFORMANCE_OPTIMIZATION_AVAILABLE:
        try:
            # Initialize Phase 4 components
            intelligent_cache = get_intelligent_cache()
            performance_monitor = get_performance_monitor()
            
            # Check cache first
            phase4_cache_key = f"query_{hash(question + str(active_files))}"
            cached_result = intelligent_cache.get(phase4_cache_key, category="query_results")
            
            if cached_result:
                print("Phase 4: Cache HIT - Returning cached result")
                performance_monitor.track_query_performance(
                    question, 0.001, cache_hits=1, cache_misses=0
                )
                return JSONResponse(cached_result)
            else:
                print("Phase 4: Cache MISS - Processing query")
                
        except Exception as e:
            print(f"Phase 4: Cache initialization error: {e}")
    
    # Phase 3: Use AI-generated SQL if available and confident
    sql = None
    
    if AI_GENERATION_AVAILABLE and ai_generated_sql and ai_result.get('confidence', 0.0) > 0.7:
        sql = query_optimization.get('optimized_sql', ai_generated_sql) if query_optimization else ai_generated_sql
        print(f"Phase 3: Using AI-generated SQL (confidence: {ai_result.get('confidence', 0.0):.2f})")
        
        # Add Phase 3 metadata to response
        phase3_metadata = {
            'ai_generated': True,
            'ai_confidence': ai_result.get('confidence', 0.0),
            'optimizations_applied': query_optimization.get('optimizations_applied', []) if query_optimization else [],
            'conversation_type': conversation_result.get('conversation_type', 'initial') if conversation_result else 'initial'
        }
    else:
        # Fallback to pattern matching and LLM as before
        print("Phase 3: AI confidence too low or not available, using pattern matching + LLM")
        phase3_metadata = {'ai_generated': False}
    
    # Add simple pattern matching for common queries before calling slow LLM (if not using AI SQL)
    if sql is None:
        # Pattern: "what is the manufacture/manufacturer of asset tag X"
        asset_tag_match = re.search(r"(?:manufacture|manufacturer).*asset.*tag.*([A-Z0-9-]+)", question, re.IGNORECASE)
        if asset_tag_match:
            raw_asset_tag = asset_tag_match.group(1)
        
        # Sanitize the asset tag input
        if SANITIZATION_AVAILABLE:
            asset_tag = data_sanitizer.sanitize_user_input(raw_asset_tag, "asset_tag")
        else:
            asset_tag = re.sub(r'[^A-Za-z0-9\-_]', '', raw_asset_tag)
        
        if not asset_tag:
            raise HTTPException(status_code=400, detail="Invalid asset tag format.")
        
        # Use dynamic column detection to find tables with identifier columns
        target_table = None
        target_column = None
        
        if COLUMN_INTELLIGENCE_AVAILABLE:
            # Find table with identifier role (asset tags, IDs, etc.)
            identifier_tables = dynamic_mapper.suggest_tables_for_role('identifier')
            if identifier_tables:
                best_match = identifier_tables[0]  # Highest confidence
                target_table = best_match['table_name']
                target_column = best_match['column_name']
                print(f"Dynamic detection: Using {target_table}.{target_column} for identifier (confidence: {best_match['confidence']:.2f})")
        
        # Fallback to legacy hardcoded approach
        if not target_table:
            for table_name, columns in TABLE_COLUMNS.items():
                if 'Asset_TAG' in columns:
                    target_table = table_name
                    target_column = 'Asset_TAG'
                    print(f"Legacy fallback: Using {target_table}.{target_column}")
                    break
        
        if target_table and target_column:
            # Create secure parameterized query using dynamic column detection
            if SANITIZATION_AVAILABLE:
                sql_template, params = sql_security.build_safe_select_query(
                    table_name=target_table,
                    where_conditions={target_column: asset_tag}
                )
                sql = sql_template  # This will be executed safely later
                # Store params for later use in execute_sql
                setattr(chat, '_secure_params', params)
                setattr(chat, '_is_parameterized', True)
            else:
                # Fallback with manual escaping using dynamic column
                safe_asset_tag = asset_tag.replace("'", "''")
                sql = f'SELECT * FROM "{target_table}" WHERE "{target_column}" = \'{safe_asset_tag}\''
                
            print(f"Generated secure SQL for asset tag: {asset_tag}")
    
    # Pattern: "manufacturer of X" or "who made X"  
    if not sql:
        manufacturer_match = re.search(r"(?:manufacturer|made|who made).*?([A-Z0-9-]+)", question, re.IGNORECASE)
        if manufacturer_match:
            raw_search_term = manufacturer_match.group(1)
            
            # Sanitize the search term
            if SANITIZATION_AVAILABLE:
                search_term = data_sanitizer.sanitize_user_input(raw_search_term, "manufacturer")
            else:
                search_term = re.sub(r'[^A-Za-z0-9\s\-_]', '', raw_search_term)
            
            if not search_term:
                raise HTTPException(status_code=400, detail="Invalid search term.")
            
            # Use dynamic column detection to find relevant tables
            target_table = None
            identifier_column = None
            manufacturer_column = None
            
            if COLUMN_INTELLIGENCE_AVAILABLE:
                # Find tables with both identifier and manufacturer roles
                identifier_tables = dynamic_mapper.suggest_tables_for_role('identifier')
                manufacturer_tables = dynamic_mapper.suggest_tables_for_role('manufacturer')
                
                # Find common table with both roles
                identifier_table_names = {t['table_name'] for t in identifier_tables}
                manufacturer_table_names = {t['table_name'] for t in manufacturer_tables}
                common_tables = identifier_table_names.intersection(manufacturer_table_names)
                
                if common_tables:
                    target_table = list(common_tables)[0]
                    # Get column names for this table
                    identifier_column = dynamic_mapper.get_column_for_role(target_table, 'identifier')
                    manufacturer_column = dynamic_mapper.get_column_for_role(target_table, 'manufacturer')
                    print(f"Dynamic detection: Using {target_table} with {identifier_column} and {manufacturer_column}")
            
            # Fallback to legacy approach
            if not target_table:
                for table_name, columns in TABLE_COLUMNS.items():
                    if 'Asset_TAG' in columns:
                        target_table = table_name
                        identifier_column = 'Asset_TAG'
                        manufacturer_column = 'Manufacturer'
                        print(f"Legacy fallback: Using {target_table} with hardcoded columns")
                        break
            
            if target_table and identifier_column and manufacturer_column:
                if SANITIZATION_AVAILABLE:
                    # Build secure LIKE query with dynamic columns
                    safe_search = f"%{search_term}%"
                    sql = f'SELECT * FROM "{target_table}" WHERE "{identifier_column}" LIKE ? OR "{manufacturer_column}" LIKE ?'
                    setattr(chat, '_secure_params', [safe_search, safe_search])
                    setattr(chat, '_is_parameterized', True)
                else:
                    # Fallback with manual escaping using dynamic columns
                    safe_search_term = search_term.replace("'", "''")
                    sql = f'SELECT * FROM "{target_table}" WHERE "{identifier_column}" LIKE \'%{safe_search_term}%\' OR "{manufacturer_column}" LIKE \'%{safe_search_term}%\''
    
    # If no pattern matched, use LLM
    if not sql:
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
    
    # Add Phase 3 metadata to response
    response = {"sql": sql, "suggestions": suggestions, "inferred_tables": inferred_tables}
    
    if AI_GENERATION_AVAILABLE and 'phase3_metadata' in locals():
        response.update({
            "phase3_metadata": phase3_metadata,
            "conversation_suggestions": conversation_result.get('suggested_followups', []) if conversation_result else [],
            "query_optimizations": query_optimization.get('suggestions', []) if query_optimization else []
        })
    
    return response


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
    
    # Phase 4: Performance monitoring start
    execution_start_time = time.time()
    
    # Sanitize question input
    if question and SANITIZATION_AVAILABLE:
        question = data_sanitizer.sanitize_user_input(question, "general")
    
    if not sql:
        raise HTTPException(status_code=400, detail="Empty SQL.")
    
    # Check if this is a parameterized query from chat endpoint
    secure_params = getattr(req, '_secure_params', None) or req.get('_secure_params', None)
    is_parameterized = getattr(req, '_is_parameterized', False) or req.get('_is_parameterized', False)
    # quick check: ensure the SQL references at least one known uploaded table
    import re
    used_tables = [t for t in TABLE_COLUMNS.keys() if re.search(r"\b" + re.escape(t) + r"\b", sql, flags=re.IGNORECASE)]
    if not used_tables:
        # offer candidate files/tables to the UI so it can suggest choices
        candidates = score_candidate_tables(question or sql)
        # return a friendly error with candidates (UI will render suggestions)
        import json
        raise HTTPException(status_code=400, detail="No known table referenced in the SQL.", headers={"X-Candidates": json.dumps(candidates)})
    # safety checks and cleaning
    sql = validate_sql_safe(sql)
    if SQLPARSE_AVAILABLE:
        try:
            validate_sql_against_schema(sql)
        except Exception as schema_err:
            # Log the schema validation error but don't crash the request
            print(f"Schema validation warning: {schema_err}")
            # Continue execution - the SQL might still work

    auto_fixed = False  # Track if we auto-fixed any issues
    try:
        # Use secure parameterized execution if available
        if is_parameterized and SANITIZATION_AVAILABLE and secure_params:
            print(f"Executing parameterized query with {len(secure_params)} parameters")
            try:
                # Handle different parameter formats
                if isinstance(secure_params, dict):
                    param_dict = secure_params
                elif isinstance(secure_params, list):
                    param_dict = {f"param_{i}": p for i, p in enumerate(secure_params)}
                else:
                    param_dict = {"param_0": secure_params}
                
                if ENGINE is not None:
                    df = sql_security.execute_safe_query(sql, param_dict, ENGINE)
                elif conn is not None:
                    df = sql_security.execute_safe_query(sql, param_dict, conn)
                else:
                    raise Exception('No database engine available')
            except Exception as secure_error:
                print(f"Secure execution failed: {secure_error}, falling back to traditional")
                # Fall back to traditional execution
                if ENGINE is not None:
                    df = pd.read_sql_query(sql, ENGINE)
                elif conn is not None:
                    df = pd.read_sql_query(sql, conn)
                else:
                    raise Exception('No database engine available')
        else:
            # Traditional execution (with validation)
            if ENGINE is not None:
                df = pd.read_sql_query(sql, ENGINE)
            elif conn is not None:
                df = pd.read_sql_query(sql, conn)
            else:
                raise Exception('No database engine available')
    except Exception as e:
        error_msg = str(e)
        
        # Self-healing: Check if it's a column not found error
        if "no such column" in error_msg.lower():
            # Extract the problematic column name
            import re
            column_match = re.search(r"no such column[:\s]+([`\w\s]+)", error_msg, re.IGNORECASE)
            if column_match:
                problematic_col = column_match.group(1).strip('`\'\"')
                
                # Get all available columns from all tables
                all_columns = []
                for table_cols in TABLE_COLUMNS.values():
                    all_columns.extend(table_cols)
                
                # Find the closest matching column
                from core import suggest_column_alternatives
                suggestions = suggest_column_alternatives(problematic_col)
                
                if suggestions:
                    best_match = suggestions[0]
                    # Try to auto-fix the SQL by replacing the problematic column
                    fixed_sql = sql
                    
                    # Replace various forms of the problematic column with the best match
                    replacements = [
                        f"`{problematic_col}`",
                        f"[{problematic_col}]", 
                        f'"{problematic_col}"',
                        f"'{problematic_col}'",
                        problematic_col
                    ]
                    
                    for old_col in replacements:
                        if old_col in sql:
                            fixed_sql = fixed_sql.replace(old_col, f"`{best_match}`")
                            break
                    
                    # Try executing the fixed SQL
                    try:
                        if ENGINE is not None:
                            df = pd.read_sql_query(fixed_sql, ENGINE)
                        elif conn is not None:
                            df = pd.read_sql_query(fixed_sql, conn)
                        
                        # If successful, continue with the rest of the function using the fixed result
                        auto_fixed = True
                        
                    except Exception:
                        # Auto-fix failed, return helpful error with suggestions
                        suggestion_text = f" Did you mean: {', '.join(suggestions[:3])}?"
                        raise HTTPException(status_code=400, detail=f"SQL execution error: Column '{problematic_col}' not found.{suggestion_text}")
                else:
                    # No good suggestions found
                    raise HTTPException(status_code=400, detail=f"SQL execution error: {error_msg}")
            else:
                # Couldn't parse the column name, return original error
                raise HTTPException(status_code=400, detail=f"SQL execution error: {error_msg}")
        else:
            # Not a column error, return original error
            raise HTTPException(status_code=400, detail=f"SQL execution error: {error_msg}")

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

    # Generate natural language summary using Phi4 LLM integration
    summary_text = ""
    table_preview = rows[:10]
    
    if NATURAL_RESPONSE_AVAILABLE:
        try:
            # Build context for natural response generation
            available_columns = []
            column_data_samples = {}
            
            # Get column information from active tables
            for table_name in active_files:
                if table_name in TABLE_COLUMNS:
                    available_columns.extend(TABLE_COLUMNS[table_name])
            
            # Get sample data from the results or table
            if rows:
                # Sample from actual results
                sample_data = rows[:5] if len(rows) >= 5 else rows
                for col in sample_data[0].keys() if sample_data else []:
                    column_data_samples[col] = [str(row.get(col, '')) for row in sample_data if row.get(col) is not None]
            else:
                # No results, get samples from table for suggestions
                for table_name in active_files:
                    try:
                        sample_df = read_table_into_df(table_name, limit=5)
                        for col in sample_df.columns:
                            column_data_samples[col] = sample_df[col].astype(str).dropna().tolist()[:5]
                    except:
                        continue
            
            # Create suggestion context for smart suggestions
            suggestion_context = SuggestionContext(
                available_columns=list(set(available_columns)),
                column_data_samples=column_data_samples,
                data_domain='general',  # Could be enhanced to detect domain
                user_query_history=[],  # Could be enhanced with session history
                common_search_patterns={},
                dataset_size=len(rows),
                column_types={}  # Could be enhanced with type detection
            )
            
            # Build query context
            query_context = QueryContext(
                original_query=question,
                sql_query=sql,
                intent_analysis=query_analysis if 'query_analysis' in locals() else None,
                suggested_tables=active_files,
                column_mappings=COLUMN_NAME_MAP,
                query_confidence=1.0,
                expected_result_type='data_retrieval'
            )
            
            # Build response context
            response_context = ResponseContext(
                results=rows,
                result_count=len(rows),
                query_success=len(rows) > 0,
                execution_time=0.0,  # Could be enhanced with actual timing
                suggested_actions=[],
                data_insights={}
            )
            
            # Generate natural language response
            if len(rows) > 0:
                # Successful query with results
                natural_response = natural_response_generator.generate_response(
                    query_context, response_context, suggestion_context
                )
                summary_text = natural_response
                print(f"Natural Language Response: Generated conversational summary")
            else:
                # No results - generate suggestions with fuzzy matching
                similar_values = smart_suggestion_engine.find_similar_values(
                    question, suggestion_context
                )
                
                if similar_values:
                    # Found similar matches - generate helpful response
                    suggestion_text = natural_response_generator.generate_no_results_response(
                        query_context, similar_values[:3], suggestion_context
                    )
                    summary_text = suggestion_text
                    print(f"Natural Language Response: Generated suggestion response with {len(similar_values)} matches")
                else:
                    # No similar matches - generate exploration response
                    exploration_response = natural_response_generator.generate_exploration_response(
                        query_context, suggestion_context
                    )
                    summary_text = exploration_response
                    print(f"Natural Language Response: Generated exploration response")
                    
        except Exception as natural_response_error:
            print(f"Natural Language Response: Error generating response: {natural_response_error}")
            # Fallback to basic response
            if len(rows) == 1:
                summary_text = f"Found 1 matching record."
            elif len(rows) > 1:
                summary_text = f"Found {len(rows)} matching records."
            else:
                summary_text = "No matching records found."
    else:
        # Fallback when natural language modules not available
        if len(rows) == 1:
            row = rows[0]
            summary_text = f"Found 1 matching record."
            
            # Smart summary based on the original question using dynamic column detection
            if COLUMN_INTELLIGENCE_AVAILABLE:
                # Use dynamic column detection for intelligent summaries
                summary_text = _generate_dynamic_summary(question, row)
        elif len(rows) > 1:
            summary_text = f"Found {len(rows)} matching records."
        else:
            summary_text = "No matching records found."
    
    # COMMENTED OUT: LLM summary was causing JSON errors - disabled temporarily

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

    # If zero rows, generate intelligent AI suggestions based on user intent and actual data
    suggestions = []
    try:
        if not rows:
            # Generate smart suggestions by analyzing the original question
            # Get inferred tables from SQL or use empty list
            inferred_from_sql = []
            for t in TABLE_COLUMNS.keys():
                if t.lower() in sql.lower():
                    inferred_from_sql.append(t)
            suggestions = generate_smart_suggestions(question, sql, inferred_from_sql)
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        suggestions = []

    # Ensure all response data is JSON serializable
    try:
        # Ensure suggestions are JSON serializable
        safe_suggestions = []
        for s in suggestions:
            try:
                safe_suggestion = {}
                for key, value in s.items():
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        safe_suggestion[key] = value
                    else:
                        safe_suggestion[key] = str(value)
                safe_suggestions.append(safe_suggestion)
            except Exception:
                # Skip problematic suggestion
                continue
        
        response_data = {
            "sql": str(sql), 
            "rows": rows, 
            "summary": str(summary_text or f"{len(rows)} rows returned."), 
            "table_preview": table_preview, 
            "display_rows": display_rows, 
            "suggestions": safe_suggestions, 
            "auto_fixed": bool(auto_fixed if 'auto_fixed' in locals() else False)
        }
        
        # Test JSON serialization before returning
        import json
        json.dumps(response_data, default=str, ensure_ascii=False)
        
        # Phase 2: Record query results for context learning
        if QUERY_INTELLIGENCE_AVAILABLE and 'query_analysis' in locals():
            try:
                # Prepare execution result for context learning
                execution_result = {
                    'success': len(rows) > 0 or auto_fixed,
                    'rows_returned': len(rows),
                    'tables_used': inferred_tables if 'inferred_tables' in locals() else [],
                    'summary_generated': bool(summary_text),
                    'auto_fixed': auto_fixed,
                    'performance': {
                        'result_size': len(rows),
                        'suggestions_count': len(safe_suggestions)
                    }
                }
                
                # Add to conversation history for learning
                original_question = req.get('question', '')
                if original_question and query_analysis:
                    context_engine.add_query_to_history(
                        query=original_question,
                        query_analysis=query_analysis,
                        execution_result=execution_result
                    )
                    print(f"Phase 2: Query recorded in context engine")
                
            except Exception as context_error:
                print(f"Phase 2: Error recording query context: {context_error}")
        
        # Phase 4: Performance tracking and caching
        if PERFORMANCE_OPTIMIZATION_AVAILABLE:
            try:
                execution_time = time.time() - execution_start_time
                
                # Track performance
                performance_monitor = get_performance_monitor()
                performance_monitor.track_query_performance(
                    sql, execution_time, len(rows), cache_hits=0, cache_misses=1
                )
                
                # Analyze query for optimization opportunities
                database_optimizer = get_database_optimizer()
                database_optimizer.analyze_query_execution(sql, execution_time, len(rows))
                
                # Cache successful results (not errors or empty results)
                if len(rows) > 0 and not auto_fixed:
                    intelligent_cache = get_intelligent_cache()
                    cache_key = f"sql_{hash(sql)}"
                    intelligent_cache.set(cache_key, response_data, category="query_results")
                    print(f"Phase 4: Cached query result ({len(rows)} rows)")
                
                # Add performance metadata to response
                response_data['phase4_performance'] = {
                    'execution_time': execution_time,
                    'rows_processed': len(rows),
                    'cached': False  # First execution, not from cache
                }
                
                print(f"Phase 4: Query executed in {execution_time:.3f}s, {len(rows)} rows")
                
            except Exception as phase4_error:
                print(f"Phase 4: Performance tracking error: {phase4_error}")
        
        return response_data
    except Exception as json_err:
        # Log the error for debugging
        import traceback
        print(f"JSON serialization error in execute_sql: {json_err}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"Problematic suggestions: {suggestions}")
        
        # Fallback response if JSON serialization fails
        return {
            "sql": str(sql),
            "rows": [],
            "summary": f"Query executed successfully. Found {len(rows)} rows.",
            "table_preview": [],
            "display_rows": [],
            "suggestions": [],
            "auto_fixed": bool(auto_fixed),
            "serialization_error": str(json_err)
        }


@router.post('/reload_tables')
async def reload_tables():
    """Reload table structure from database into memory."""
    try:
        reload_tables_from_database()
        return {"status": "success", "message": f"Reloaded {len(TABLE_COLUMNS)} tables", "tables": list(TABLE_COLUMNS.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload tables: {e}")

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
