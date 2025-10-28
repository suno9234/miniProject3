from typing import Literal, TypedDict, Annotated

class AppState(TypedDict, total=False):
    user_query: Annotated[str, "사용자의 입력값"]
    agent_type: Annotated[Literal["search", "inventory", "reservation"], "선택된 에이전트 타입"]
    is_sensitive: Annotated[bool, "민감한 정보 포함 여부"]
    masked_user_query: Annotated[str, "마스킹된 사용자 입력값"]
    internal_documents: Annotated[list[str], "내부 문서 검색 결과 리스트"]
    external_search_result: Annotated[str, "외부 검색 결과"]
    generation: Annotated[str, "생성된 응답"]