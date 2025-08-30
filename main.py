from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routes import router

app = FastAPI(
    title="PHERS - Personal Data Chat System",
    description="6-Step Flow: Upload → Profile → AI Clean → Index → Chat → Results",
    version="2.0.0"
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.include_router(router)

# Print startup message
print("🚀 PHERS 2.0 - Streamlined Data Chat System")
print("📋 6-Step Flow: Upload → Profile → AI Clean → Index → Chat → Results")
print("🔧 Tech Stack: FastAPI + MySQL + Redis + PandasAI + Phi-4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

