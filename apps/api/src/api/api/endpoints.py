from fastapi import APIRouter, Request
from pydantic import BaseModel
from api.api.models import RAGRequest, RAGResponse, RAGUsedContext
import logging

from api.agents.retrieval_generation import rag_pipeline_wraper


logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s ")

logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")

def chat(request: Request,
         payload: RAGRequest
) -> RAGResponse:
    result = rag_pipeline_wraper(payload.query)
    return RAGResponse(answer=result["answer"], used_context= [RAGUsedContext(**item) for item in result['used_context']])

api_router =APIRouter()

api_router.include_router(rag_router, prefix = "/rag",tags = ["rag"])