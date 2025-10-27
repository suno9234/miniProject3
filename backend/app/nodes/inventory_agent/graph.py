# inventory_agent 서브그래프 정의
from langgraph.graph import StateGraph, END
from app.nodes.state import AppState

# TODO: 노드 함수들 임포트
# from .inventory_node import inventory_check_node, inventory_response_node

def build_inventory_agent_graph():
    """재고 확인 에이전트 서브그래프 빌드"""
    graph = StateGraph(AppState)
    
    # TODO: 노드 추가 (단순한 경우 하나의 노드로도 가능)
    # graph.add_node("inventory_check", inventory_check_node)
    # graph.add_node("inventory_response", inventory_response_node)
    
    # TODO: 엣지 연결
    # graph.set_entry_point("inventory_check")
    # graph.add_edge("inventory_check", "inventory_response")
    # graph.add_edge("inventory_response", END)
    
    return graph.compile()

# 컴파일된 서브그래프
inventory_agent_graph = build_inventory_agent_graph()