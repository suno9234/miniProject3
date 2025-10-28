# 유저 쿼리 분석해서 적절한 에이전트 타입 선택
from app.nodes.state import AppState
from app.services.llm import llm_with_agent_route
from langchain_core.prompts import ChatPromptTemplate

def planner_node(state: AppState) -> AppState:
    """유저 쿼리를 분석해서 에이전트 타입을 결정하는 노드"""
    user_query = state["user_query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            다음 사용자 쿼리를 분석하여 적절한 에이전트를 선택하세요:

            에이전트 옵션:
            1. search: 내부/외부 데이터 기반 검색 및 질문 답변
            2. inventory: 특정 물품의 재고 확인
            3. reservation: 회의실/기구/시설 등 예약
        """),
        ("user", "{query}")
    ])
    
    chain = prompt | llm_with_agent_route
    result = chain.invoke({"query": user_query})
    return {"agent_type": result.agent_type}