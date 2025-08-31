# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PHERS (Personal HR Data Chat System) is a streamlined 6-step data chat application that enables natural language querying of HR datasets:

**6-Step Flow**: Upload → Profile → AI Clean → Index → Chat → Results

## Development Commands

### Running the Application
```bash
# Activate virtual environment (if using)
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start the development server
python main.py
# Server runs at http://localhost:8000
```

### Testing
- No specific test framework configured
- Manual testing through web interface at http://localhost:8000

### Dependencies Management
- Use `pip freeze > requirements.txt` to update dependencies
- Virtual environment in `venv/` directory (gitignored)

## Architecture Overview

### Core Components

1. **main.py**: FastAPI application entry point with basic routing setup
2. **routes.py**: RESTful API endpoints implementing the 6-step workflow
3. **core.py**: Business logic and data processing classes
4. **index.html**: Frontend UI (served as static file)

### Data Flow Architecture

```
Upload → Profile → AI Clean → Index → Chat → Results
  ↓         ↓         ↓         ↓       ↓        ↓
File     Issues   Suggestions Storage  NLP   Response
Read   Detection   (AI-based)  (Redis  (Phi-4) (JSON)
              
```

### Key Classes in core.py

- **DataProfiler**: Analyzes uploaded data for issues (Step 2)
- **AIDataCleaner**: Generates AI-suggested cleaning operations (Step 3)  
- **DataCleaner**: Applies cleaning operations and indexes data (Step 4)
- **ChatSession**: Manages user chat sessions with Redis (Step 5)
- **NLProcessor**: Processes natural language questions (Step 6)

### Storage Architecture

- **In-memory**: Active datasets stored in `ACTIVE_DATASETS` dict in routes.py
- **Redis**: Session management, metadata caching, quick data access
- **MySQL**: Persistent storage of cleaned datasets
- **File system**: Original uploaded files in `./data/` directory

### Configuration

Environment variables (optional, with defaults):
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
- `OLLAMA_BASE_URL` (default: http://127.0.0.1:11434)
- `OLLAMA_MODEL` (default: phi4)
- `UPLOAD_FOLDER` (default: ./data)
- `SESSION_TIMEOUT` (default: 3600 seconds)

### API Workflow

1. `POST /upload` - Upload CSV/Excel file
2. `POST /profile/{dataset_id}` - Generate data profile with issues
3. `POST /suggest-cleaning/{dataset_id}` - Get AI cleaning suggestions  
4. `POST /clean-and-index/{dataset_id}` - Clean data and index it
5. `POST /start-chat/{dataset_id}` - Create chat session
6. `POST /chat/{session_id}` - Natural language query processing

### Technology Stack

- **Backend**: FastAPI + uvicorn
- **AI/ML**: PandasAI + Ollama (Phi-4 model)
- **Data**: Pandas + NumPy
- **Storage**: Redis + MySQL + SQLite fallback
- **Frontend**: Vanilla HTML/CSS/JS with TailwindCSS

### File Handling

- Supports CSV (.csv) and Excel (.xlsx, .xls) files
- Handles encoding issues with fallback to 'latin-1'
- Files saved to `./data/` directory with original names
- Dataset IDs generated as MD5 hash of filename (8 chars)

### Error Handling Patterns

- HTTP exceptions with descriptive error messages
- Try-catch blocks around database operations
- Graceful degradation when Redis/MySQL unavailable
- User-friendly error responses in JSON format