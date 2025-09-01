"""
PHERS Routes - Clean 6-Step Flow Implementation
Upload → Profile → AI Clean → Index → Chat → Results
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import pandas as pd
import io
from pathlib import Path
from typing import Dict, Any

from core import (
    DataProfiler, AIDataCleaner, DataCleaner, ChatSession, NLProcessor,
    get_dataset_id, UPLOAD_FOLDER, redis_client
)

router = APIRouter()

# Active datasets in memory
ACTIVE_DATASETS = {}

@router.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI"""
    return FileResponse("index.html")

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Step 1: Upload messy dataset"""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Validate file type
    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
    
    try:
        # Read file into DataFrame
        contents = await file.read()
        
        if file.filename.lower().endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(contents))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="File contains no data")
        
        # Generate dataset ID
        dataset_id = get_dataset_id(file.filename)
        
        # Save file
        file_path = UPLOAD_FOLDER / file.filename
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        # Store in memory for processing
        ACTIVE_DATASETS[dataset_id] = {
            'filename': file.filename,
            'dataframe': df,
            'uploaded_at': pd.Timestamp.now().isoformat(),
            'status': 'uploaded'
        }
        
        # Auto-run the 6-step process for frontend compatibility
        try:
            # Step 2: Profile data
            profile = DataProfiler.profile_dataset(df)
            ACTIVE_DATASETS[dataset_id]['profile'] = profile
            ACTIVE_DATASETS[dataset_id]['status'] = 'profiled'
            
            # Step 3: Get AI cleaning suggestions
            suggestions = AIDataCleaner.suggest_cleaning_operations(profile)
            ACTIVE_DATASETS[dataset_id]['cleaning_suggestions'] = suggestions
            ACTIVE_DATASETS[dataset_id]['status'] = 'suggestions_ready'
            
            # Step 4: Clean and index data
            df_clean = DataCleaner.apply_cleaning_operations(df, suggestions)
            DataCleaner.index_data(dataset_id, df_clean)
            ACTIVE_DATASETS[dataset_id]['clean_dataframe'] = df_clean
            ACTIVE_DATASETS[dataset_id]['status'] = 'indexed'
            
            # Return success with processing info
            processing_info = {
                "sanitized": len(suggestions) > 0,
                "sanitation_report": {
                    "rows_processed": df.shape[0],
                    "issues_found": [
                        {
                            "original": f"Column: {s['column']}", 
                            "issues": [s['operation'].replace('_', ' ').title()],
                            "cleaned": s['explanation']
                        }
                        for s in suggestions
                    ],
                    "original_columns": df.columns.tolist(),
                    "cleaned_columns": df_clean.columns.tolist()
                }
            }
            
        except Exception as e:
            # If processing fails, just return upload success
            print(f"⚠️ Auto-processing failed for {file.filename}: {e}")
            processing_info = {}
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            **processing_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/profile/{dataset_id}")
async def profile_dataset(dataset_id: str):
    """Step 2: Profile data and identify issues"""
    
    if dataset_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = ACTIVE_DATASETS[dataset_id]['dataframe']
        
        # Generate profile
        profile = DataProfiler.profile_dataset(df)
        
        # Update dataset status
        ACTIVE_DATASETS[dataset_id]['profile'] = profile
        ACTIVE_DATASETS[dataset_id]['status'] = 'profiled'
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "profile": profile
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")

@router.post("/suggest-cleaning/{dataset_id}")
async def suggest_cleaning(dataset_id: str):
    """Step 3: AI suggests cleaning operations with explanations"""
    
    if dataset_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = ACTIVE_DATASETS[dataset_id]
    if 'profile' not in dataset:
        raise HTTPException(status_code=400, detail="Dataset must be profiled first")
    
    try:
        # Get AI suggestions
        suggestions = AIDataCleaner.suggest_cleaning_operations(dataset['profile'])
        
        # Update dataset
        ACTIVE_DATASETS[dataset_id]['cleaning_suggestions'] = suggestions
        ACTIVE_DATASETS[dataset_id]['status'] = 'suggestions_ready'
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "suggestions": suggestions,
            "explanation": f"Found {len(suggestions)} cleaning operations to improve your data quality."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI suggestion failed: {str(e)}")

@router.post("/clean-and-index/{dataset_id}")
async def clean_and_index(dataset_id: str, apply_suggestions: Dict[str, bool] = None):
    """Step 4: Clean data and index it"""
    
    if dataset_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = ACTIVE_DATASETS[dataset_id]
    if 'cleaning_suggestions' not in dataset:
        raise HTTPException(status_code=400, detail="No cleaning suggestions available")
    
    try:
        df_original = dataset['dataframe']
        
        # Filter suggestions based on user selection
        if apply_suggestions:
            operations_to_apply = [
                op for op in dataset['cleaning_suggestions'] 
                if apply_suggestions.get(f"{op['column']}_{op['operation']}", True)
            ]
        else:
            operations_to_apply = dataset['cleaning_suggestions']
        
        # Apply cleaning operations
        df_clean = DataCleaner.apply_cleaning_operations(df_original, operations_to_apply)
        
        # Index the cleaned data
        DataCleaner.index_data(dataset_id, df_clean)
        
        # Update dataset
        ACTIVE_DATASETS[dataset_id]['clean_dataframe'] = df_clean
        ACTIVE_DATASETS[dataset_id]['status'] = 'indexed'
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "operations_applied": len(operations_to_apply),
            "shape_before": df_original.shape,
            "shape_after": df_clean.shape,
            "message": "Data cleaned and indexed successfully! You can now start chatting with your data."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleaning and indexing failed: {str(e)}")

@router.post("/start-chat/{dataset_id}")
async def start_chat_session(dataset_id: str):
    """Step 5: Create natural language chat session"""
    
    if dataset_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if ACTIVE_DATASETS[dataset_id]['status'] != 'indexed':
        raise HTTPException(status_code=400, detail="Dataset must be indexed before chatting")
    
    try:
        session_id = ChatSession.create_session(dataset_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "dataset_id": dataset_id,
            "message": "Chat session created! Ask me anything about your data in natural language."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.post("/chat/{session_id}")
async def chat_with_data(session_id: str, question: Dict[str, str]):
    """Step 6: Convert natural language to code and return conversational results"""
    
    user_question = question.get('question', '').strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        # Get session info
        session = ChatSession.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        dataset_id = session['dataset_id']
        
        # Process the question using NLP + AI
        response = NLProcessor.process_question(dataset_id, user_question)
        
        if not response.get('success'):
            return JSONResponse(content=response, status_code=400)
        
        # Add to session history
        ChatSession.add_message(session_id, user_question, response)
        
        return {
            "success": True,
            "session_id": session_id,
            "question": user_question,
            "answer": response.get('answer'),
            "data": response.get('data', {}),
            "explanation": response.get('explanation'),
            "dataset_info": response.get('dataset_info')
        }
        
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "message": "Sorry, I couldn't process your question. Please try rephrasing it."
        }, status_code=500)

@router.get("/datasets")
async def list_datasets():
    """Get list of active datasets"""
    datasets = {}
    for dataset_id, info in ACTIVE_DATASETS.items():
        datasets[dataset_id] = {
            'filename': info['filename'],
            'status': info['status'],
            'uploaded_at': info['uploaded_at'],
            'shape': info['dataframe'].shape if 'dataframe' in info else None
        }
    return {"datasets": datasets}

@router.get("/health")
async def health_check():
    """System health check"""
    from core import redis_client, mysql_conn
    
    health = {
        "status": "healthy",
        "components": {
            "redis": "connected" if redis_client else "disconnected",
            "mysql": "connected" if mysql_conn else "disconnected",
            "active_datasets": len(ACTIVE_DATASETS)
        }
    }
    
    return health

# =============================================================================
# COMPATIBILITY ENDPOINTS FOR EXISTING FRONTEND
# =============================================================================

@router.get("/files")
async def list_files():
    """Compatibility endpoint for frontend file list"""
    files = []
    for dataset_id, info in ACTIVE_DATASETS.items():
        files.append({
            'filename': info['filename'],
            'table': f"dataset_{dataset_id}",
            'active': info['status'] == 'indexed',
            'status': info['status']
        })
    
    return {"files": files}

@router.post("/files/{filename}/toggle")
async def toggle_file_status(filename: str):
    """Compatibility endpoint for file activation toggle"""
    # Find dataset by filename
    dataset_id = None
    for did, info in ACTIVE_DATASETS.items():
        if info['filename'] == filename:
            dataset_id = did
            break
    
    if not dataset_id:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Toggle between indexed and uploaded status
    current_status = ACTIVE_DATASETS[dataset_id]['status']
    if current_status == 'indexed':
        ACTIVE_DATASETS[dataset_id]['status'] = 'uploaded'
        message = f"Deactivated {filename}"
    else:
        # If not indexed, run through the full 6-step process
        try:
            # Auto-run the 6-step flow for activation
            if current_status == 'uploaded':
                # Step 2: Profile
                df = ACTIVE_DATASETS[dataset_id]['dataframe']
                profile = DataProfiler.profile_dataset(df)
                ACTIVE_DATASETS[dataset_id]['profile'] = profile
                
                # Step 3: Get cleaning suggestions  
                suggestions = AIDataCleaner.suggest_cleaning_operations(profile)
                ACTIVE_DATASETS[dataset_id]['cleaning_suggestions'] = suggestions
                
                # Step 4: Clean and index
                df_clean = DataCleaner.apply_cleaning_operations(df, suggestions)
                DataCleaner.index_data(dataset_id, df_clean)
                ACTIVE_DATASETS[dataset_id]['clean_dataframe'] = df_clean
                ACTIVE_DATASETS[dataset_id]['status'] = 'indexed'
                
            message = f"Activated {filename}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Activation failed: {str(e)}")
    
    return {"success": True, "message": message}

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """Compatibility endpoint for file deletion"""
    # Find and remove dataset by filename
    dataset_id = None
    for did, info in ACTIVE_DATASETS.items():
        if info['filename'] == filename:
            dataset_id = did
            break
    
    if not dataset_id:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Remove from active datasets
    del ACTIVE_DATASETS[dataset_id]
    
    # Clean up Redis data
    if redis_client:
        redis_client.delete(f"dataset:{dataset_id}:metadata")
        redis_client.delete(f"dataset:{dataset_id}:sample") 
        redis_client.delete(f"dataset:{dataset_id}:dataframe")
    
    return {"success": True, "message": f"Deleted {filename}"}

@router.post("/files/activate-all")
async def activate_all_files():
    """Compatibility endpoint to activate all files"""
    activated = 0
    errors = []
    
    for dataset_id, info in ACTIVE_DATASETS.items():
        if info['status'] != 'indexed':
            try:
                # Run 6-step process for each dataset
                df = info['dataframe']
                profile = DataProfiler.profile_dataset(df)
                suggestions = AIDataCleaner.suggest_cleaning_operations(profile)
                df_clean = DataCleaner.apply_cleaning_operations(df, suggestions)
                DataCleaner.index_data(dataset_id, df_clean)
                
                info['profile'] = profile
                info['cleaning_suggestions'] = suggestions
                info['clean_dataframe'] = df_clean
                info['status'] = 'indexed'
                activated += 1
            except Exception as e:
                errors.append(f"{info['filename']}: {str(e)}")
    
    message = f"Activated {activated} files"
    if errors:
        message += f". Errors: {'; '.join(errors)}"
    
    return {"success": True, "message": message}

@router.post("/files/deactivate-all")
async def deactivate_all_files():
    """Compatibility endpoint to deactivate all files"""
    deactivated = 0
    
    for dataset_id, info in ACTIVE_DATASETS.items():
        if info['status'] == 'indexed':
            info['status'] = 'uploaded'
            deactivated += 1
    
    return {"success": True, "message": f"Deactivated {deactivated} files"}

@router.post("/chat")
async def chat_compatibility(request: Dict[str, str]):
    """Compatibility endpoint for direct chat (auto-manages sessions)"""
    question = request.get('question', '').strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Find the first indexed dataset
    dataset_id = None
    for did, info in ACTIVE_DATASETS.items():
        if info['status'] == 'indexed':
            dataset_id = did
            break
    
    if not dataset_id:
        return {
            "success": False,
            "error": "No active datasets available",
            "message": "Please upload and activate a dataset first"
        }
    
    try:
        # Create or get existing session for this dataset
        session_key = f"auto_session_{dataset_id}"
        if session_key not in globals():
            session_id = ChatSession.create_session(dataset_id)
            globals()[session_key] = session_id
        else:
            session_id = globals()[session_key]
        
        # Process question
        response = NLProcessor.process_question(dataset_id, question)
        
        if not response.get('success'):
            return response
        
        # Add to session history
        ChatSession.add_message(session_id, question, response)
        
        # Format response for frontend compatibility
        return {
            "success": True,
            "question": question,
            "summary": response.get('answer', 'No response generated'),
            "table_preview": response.get('data', {}).get('preview', []),
            "explanation": response.get('explanation', ''),
            "suggestions": []  # Frontend expects this field
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Chat processing failed: {str(e)}"
        }

@router.get("/db_info") 
async def database_info():
    """Compatibility endpoint for database information"""
    return {
        "db_path": "Redis + MySQL (PHERS 2.0)",
        "tables": [f"dataset_{did}" for did in ACTIVE_DATASETS.keys() if ACTIVE_DATASETS[did]['status'] == 'indexed']
    }

@router.get("/index_status")
async def indexing_status():
    """Compatibility endpoint for indexing status"""
    status = {}
    for dataset_id, info in ACTIVE_DATASETS.items():
        filename = info['filename']
        if info['status'] == 'indexed':
            status[filename] = {
                "status": "done",
                "progress": 1.0,
                "finished": info.get('indexed_at', 'recently'),
                "count": info['dataframe'].shape[0] if 'dataframe' in info else 0
            }
        else:
            status[filename] = {
                "status": "pending", 
                "progress": 0.5 if info['status'] != 'uploaded' else 0.0
            }
    
    return {"status": status}