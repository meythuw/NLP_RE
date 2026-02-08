from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
import uvicorn

from app.api.ner import ner_routes
from app.api.re import re_routes
from app.db.mongo import init_indexes
from app.api.train_svm import train_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_indexes()
    yield
app = FastAPI(lifespan=lifespan)

    
app.include_router(re_routes.router)
app.include_router(ner_routes.router)
app.include_router(train_routes.router)
if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
