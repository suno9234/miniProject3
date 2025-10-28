from pydantic import BaseModel
from typing import Literal

# 애플리케이션에서 사용하는 모델 정의

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    response: str

class AgentRoute(BaseModel):
    """플래너가 선택한 에이전트 타입"""
    agent_type: Literal["search", "inventory", "reservation"]
    reasoning: str  # 선택 이유