from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from api.api.endpoints import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



app = FastAPI()


app.add_middleware(CORSMiddleware,
                    allow_origins=["*"], 
                    allow_credentials = True,
                    allow_methods=["*"], 
                    allow_headers=["*"])




app.include_router(api_router)

