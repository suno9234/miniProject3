import logging
from fastapi import APIRouter
from pydantic import BaseModel
from services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter()
agent_service = AgentService()

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    response: str

@router.post("/process-text", response_model=TextResponse)
async def process_text(request: TextRequest):
    logger.info(f"Received text request: {request.text[:50]}...")
    result = await agent_service.process_text(request.text)
    logger.info(f"Processed response: {result[:50]}...")
    
    return TextResponse(response=result)