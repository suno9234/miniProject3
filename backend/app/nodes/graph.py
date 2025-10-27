from langgraph.graph import StateGraph, END
from app.nodes.state import AppState
from app.nodes.planner import planner_node

# TODO: 각 에이전트 노드 임포트 추가
# from app.nodes.search_agent.response_generator import search_agent_graph
# from app.nodes.inventory_agent.inventory_node import inventory_agent_node
# from app.nodes.reservation_agent.reservation_node import reservation_agent_node

def route_to_agent(state: AppState) -> str:
    """에이전트 타입에 따라 라우팅"""
    agent_type = state.get("agent_type")
    if agent_type == "search":
        return "search_agent"
    elif agent_type == "inventory":
        return "inventory_agent"
    elif agent_type == "reservation":
        return "reservation_agent"
    else:
        return END

def build_graph():
    graph = StateGraph(AppState)

    # 노드 추가
    graph.add_node("planner", planner_node)
    
    # TODO: 각 에이전트 노드 추가
    # graph.add_node("search_agent", search_agent_graph)
    # graph.add_node("inventory_agent", inventory_agent_node)
    # graph.add_node("reservation_agent", reservation_agent_node)

    # 엣지 연결
    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_to_agent,
        {
            "search_agent": "search_agent",
            "inventory_agent": "inventory_agent", 
            "reservation_agent": "reservation_agent"
        }
    )
    
    # TODO: 각 에이전트에서 END로 연결
    # graph.add_edge("search_agent", END)
    # graph.add_edge("inventory_agent", END)
    # graph.add_edge("reservation_agent", END)

    return graph.compile()