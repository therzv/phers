# HR Data Chat - Chat with Your HR Documents

A modern web application that allows HR teams to chat naturally with their structured data using AI. Built with FastAPI, Ollama, and Phi4 for natural language processing.

## Features

✨ **Natural Language Queries**: Ask questions in plain English like "How many assets are in repair status?"

🤖 **AI-Powered**: Uses Ollama + Phi4 for intelligent SQL generation and natural responses  

📊 **Smart Data Processing**: 
- Automatic column name normalization
- Fuzzy matching for typos and variations
- Intelligent suggestions when columns don't exist

🎯 **HR-Optimized**: Built specifically for HR data with pre-configured terminology mapping

🔍 **Advanced Search**: RAG (Retrieval Augmented Generation) using ChromaDB and sentence transformers

💡 **User-Friendly Interface**: 
- Modern, responsive UI with example questions
- Real-time loading states and error handling
- Data preview and visualization

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed and running
- Phi4 model downloaded (`ollama pull phi4`)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd phers
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Start the application:
```bash
python main.py
```

4. Open http://localhost:8000 in your browser

### Usage

1. **Upload Data**: Upload CSV or Excel files with your HR data
2. **Ask Questions**: Use natural language like:
   - "How many assets are in repair status?"
   - "What is the total value of all machinery?"
   - "Show me all assets in Seattle"
   - "Count assets by category"

## Sample Data

The app includes sample HR assets data (`data/hr_assets.csv`) with:
- Asset ID, Category, Value, Location, Status
- 15 sample records across different asset types
- Perfect for testing the natural language capabilities

## Configuration

### Environment Variables
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://127.0.0.1:11434)
- `OLLAMA_MODEL`: Model to use (default: phi4)
- `DATABASE_URL`: Custom database connection string
- `USE_PANDAS_AI`: Enable pandas-ai integration (0/1)

### Supported File Types
- CSV files (.csv)
- Excel files (.xlsx, .xls)

## Architecture

- **Backend**: FastAPI with SQLAlchemy
- **AI**: Ollama + Phi4 for natural language processing
- **Search**: ChromaDB + SentenceTransformers for RAG
- **Database**: SQLite (default) or MySQL/PostgreSQL
- **Frontend**: Modern HTML/CSS/JS with TailwindCSS

## Key Improvements Made

1. **Enhanced SQL Generation**: Better prompt engineering for HR-specific queries
2. **Smart Column Mapping**: Handles common HR terminology and typos
3. **Intelligent Suggestions**: Provides alternatives when queries fail
4. **Modern UI**: Clean, responsive interface with example questions
5. **Better Error Handling**: User-friendly error messages with suggestions
6. **Real-time Feedback**: Loading states and progress indicators

## Deployment

For production deployment:

1. Set environment variables for your database and Ollama server
2. Use a production WSGI server like gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## Troubleshooting

**Common Issues:**
- Ensure Ollama is running and phi4 model is downloaded
- Check that uploaded files have proper headers
- Verify database permissions for SQLite file creation

**Performance:**
- For large datasets, consider using PostgreSQL or MySQL
- Adjust RAG chunk size with `RAG_CHUNK_ROWS` environment variable

## License

MIT License - see LICENSE file for details
