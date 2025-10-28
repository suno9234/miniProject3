# search_agent 서브그래프 정의
from langgraph.graph import StateGraph, END
from app.nodes.state import AppState

# TODO: 각 노드 함수들 임포트
from app.nodes.search_agent.sensitive_info_detector import QueryNormalizer, SensitiveInfoDetector
from app.nodes.search_agent.external_search import sensitive_branch_answer
from app.nodes.search_agent.internal_search import internal_searcher
from app.nodes.search_agent.response_generator import response_generator_node

def build_search_agent_graph():
    """검색 에이전트 서브그래프 빌드"""
    graph = StateGraph(AppState)
    
    # TODO: 노드 추가
    graph.add_node("query_normalize", QueryNormalizer().apply)
    graph.add_node("sensitive_info_detect", SensitiveInfoDetector().apply)
    
    graph.add_node("external_search", sensitive_branch_answer)
    graph.add_node("internal_search", internal_searcher)
    graph.add_node("response_generator", response_generator_node)
    
    # TODO: 엣지 연결 (순차 또는 병렬 처리)
    graph.set_entry_point("query_normalize")
    graph.add_edge("query_normalize", "sensitive_info_detect")
    graph.add_edge("sensitive_info_detect", "external_search")
    graph.add_edge("external_search", "internal_search")
    graph.add_edge("internal_search", "response_generator")
    graph.add_edge("response_generator", END)
    
    return graph.compile()

# 컴파일된 서브그래프
search_agent_graph = build_search_agent_graph()