# reservation_agent 서브그래프 정의
from langgraph.graph import StateGraph, END
from app.nodes.state import AppState

# TODO: 노드 함수들 임포트
# from .reservation_node import reservation_check_node, reservation_process_node

def build_reservation_agent_graph():
    """예약 처리 에이전트 서브그래프 빌드"""
    graph = StateGraph(AppState)
    
    # TODO: 노드 추가
    # graph.add_node("reservation_check", reservation_check_node)
    # graph.add_node("reservation_process", reservation_process_node)
    
    # TODO: 엣지 연결
    # graph.set_entry_point("reservation_check")
    # graph.add_edge("reservation_check", "reservation_process")
    # graph.add_edge("reservation_process", END)
    
    return graph.compile()

# 컴파일된 서브그래프
reservation_agent_graph = build_reservation_agent_graph()