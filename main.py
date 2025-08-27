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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

