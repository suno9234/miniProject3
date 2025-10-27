import logging
from fastapi import APIRouter
from app.models import TextRequest, TextResponse
from app.nodes.graph import build_graph

logger = logging.getLogger(__name__)

router = APIRouter()
graph = build_graph()

@router.post("/process-text", response_model=TextResponse)
async def process_text(request: TextRequest):
    logger.info(f"Received text request: {request.text[:50]}...")
    
    # state 초기화
    initial_state = {"user_query": request.text}
    
    # 그래프 실행
    result = graph.invoke(initial_state)
    
    # generation 필드에서 응답 추출
    response = result.get("generation", "응답을 생성할 수 없습니다.")
    
    logger.info(f"Processed response: {response[:50]}...")
    
    return TextResponse(response=response)