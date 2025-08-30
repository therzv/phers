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
    get_dataset_id, UPLOAD_FOLDER
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
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist()
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