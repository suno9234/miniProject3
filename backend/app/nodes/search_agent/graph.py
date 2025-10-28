# search_agent 서브그래프 정의
from langgraph.graph import StateGraph, END
from app.nodes.state import AppState

# TODO: 각 노드 함수들 임포트
# from .sensitive_info_detector import sensitive_info_node
# from .external_search import external_search_node
# from .internal_search import internal_search_node
# from .response_generator import response_generator_node

def build_search_agent_graph():
    """검색 에이전트 서브그래프 빌드"""
    graph = StateGraph(AppState)
    
    # TODO: 노드 추가
    # graph.add_node("sensitive_info", sensitive_info_node)
    # graph.add_node("external_search", external_search_node)
    # graph.add_node("internal_search", internal_search_node)
    # graph.add_node("response_generator", response_generator_node)
    
    # TODO: 엣지 연결 (순차 또는 병렬 처리)
    # graph.set_entry_point("sensitive_info")
    # graph.add_edge("sensitive_info", "external_search")
    # graph.add_edge("sensitive_info", "internal_search")  # 병렬 처리
    # graph.add_edge(["external_search", "internal_search"], "response_generator")
    # graph.add_edge("response_generator", END)
    
    return graph.compile()

# 컴파일된 서브그래프
search_agent_graph = build_search_agent_graph()