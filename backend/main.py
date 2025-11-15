from fastapi import FastAPI
from api.rag_routes import router

app = FastAPI()
app.include_router(router)
