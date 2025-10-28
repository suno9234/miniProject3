from typing import Literal, TypedDict, Annotated, Dict, Any
from langgraph.graph import add_messages

def add_message_with_limit(
    state: Dict[str, Any],
    new_message: Any,
    max_messages: int = 20,
    messages_key: str = "chat_history",
) -> Dict[str, Any]:
    """
    chat_history에 새 메시지를 추가하고,
    전체 길이가 max_messages를 초과하면 가장 오래된 메시지를 삭제.
    """
    history: list[Any] = list(state.get(messages_key, []))
    history.append(new_message)
    # 초과분 잘라내기 (앞에서부터 제거)
    if len(history) > max_messages:
        history = history[-max_messages:]
    return {**state, messages_key: history}

class AppState(TypedDict, total=False):
    user_query: Annotated[str, "사용자의 입력값"]
    agent_type: Annotated[Literal["search", "inventory", "reservation"], "선택된 에이전트 타입"]
    is_sensative: Annotated[bool, "민감한 정보 포함 여부"]
    masked_user_query: Annotated[str, "마스킹된 사용자 입력값"]
    internal_documents: Annotated[list[str], "내부 문서 검색 결과 리스트"]
    external_search_result: Annotated[str, "외부 검색 결과"]
    internal_search_result: Annotated[str, "내부 검색 결과"]
    generation: Annotated[str, "생성된 응답"]
    chat_history: Annotated[list[tuple], "UI 채팅창에 표시될 대화 기록", add_message_with_limit]

