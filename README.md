Quick start (Ollama + LangChain wiring)

1) Install dependencies (global or in your environment):

```bash
python3 -m pip install -r requirements.txt
```

2) Ensure Ollama daemon/CLI is installed and running, and you have a model available (e.g., `phi4`):

```bash
# list models
ollama list
# pull model if needed
ollama pull phi4
```

3) Optionally set env vars to match your Ollama host/model:

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="phi4"
```

4) Run the app:

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

If you see ImportErrors related to langchain wrappers, install `langchain-ollama` or `langchain-community` as shown in `requirements.txt`.
