"""
검색 에이전트 - 내부/외부 결과 종합 후 보고서 출력 (간결 자연어 답변 포함)
출력 형식:
[일일 대화 보고서]
...
─────────────────────────────────────────────
질문: ...
답변: ...
─────────────────────────────────────────────
"""

from __future__ import annotations
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.nodes.state import AppState
from app.services.llm import llm
from app.services.db import db_service

# ==================================
# 구조화 출력 스키마
# ==================================
class SynthSummary(BaseModel):
    key_points: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)
    recommended_steps: List[str] = Field(default_factory=list)

# ==================================
# LLM 프롬프트 정의
# ==================================
_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "너는 내부 매뉴얼과 외부 검색 결과를 종합하여 구조화된 분석 보고서를 작성하는 전문가이다. "
        "핵심 요지, 주의/제약, 권장 절차를 명확히 구분하여 제시하라."
    ),
    (
        "user",
        "다음 정보를 종합하여 구조화된 요약을 만들어라.\n\n"
        "[사용자 질문]\n{query}\n\n"
        "[내부 결과]\n{internal}\n\n"
        "[외부 결과]\n{external}\n"
    ),
])
# 자연스러운 전체 답변 생성
_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "너는 사용자의 질문에 대한 응답을 사용자에게 제공하는 전문가이다. "
        "핵심 요지, 주의사항, 권장 절차의 모든 내용을 종합하여 자연스럽고 구체적인 문장으로 설명하라. "
        "모든 정보를 반드시 포함하되, 문장체로 작성하라. "
        "불릿(-)이나 번호는 사용하지 말고, '또한', '따라서', '특히' 등의 연결어를 활용하라."
    ),
    (
        "user",
        "다음 정보를 종합하여 사용자에게 설명하라.\n\n"
        "[핵심 요지]\n{key_points}\n\n"
        "[주의/제약]\n{cautions}\n\n"
        "[권장 절차]\n{recommended_steps}"
    ),
])
# 핵심 문장만 요약하는 프롬프트
_SHORT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "너는 외부 검색 결과/내부 매뉴얼 검색 결과를 핵심 두 문장 이내로 간결하게 요약하여 사용자에게 설명하는 전문가이다. "
        "가장 중요한 행동 지침만 남기고, 부가적인 세부 설명은 생략하라."
    ),
    (
        "user",
        "다음 문단의 핵심 내용만 1~2문장으로 요약하라.\n\n{paragraph}"
    ),
])
# LLM 체인 생성
_summary_chain = _SUMMARY_PROMPT | llm.with_structured_output(SynthSummary)
_answer_chain = _ANSWER_PROMPT | llm
_short_chain = _SHORT_ANSWER_PROMPT | llm
# ==================================
# 보고서 생성 함수
# ==================================
def _build_report(state: AppState, summary: SynthSummary) -> str:
    """일일 대화 보고서 본문 생성"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    result_text = []
    if summary.key_points:
        result_text.append("- 핵심 요지\n  - " + "\n  - ".join(summary.key_points))
    if summary.cautions:
        result_text.append("- 주의/제약\n  - " + "\n  - ".join(summary.cautions))
    if summary.recommended_steps:
        result_text.append("- 권장 절차\n  - " + "\n  - ".join(summary.recommended_steps))
    combined = "\n".join(result_text)
    return f"""
        [일일 대화 보고서]
        생성 시각 : {now}
        사용자 질의 : {state.get('user_query', '')}
        민감 여부 : {"Yes" if state.get("is_sensative") else "No"}
        [외부 결과]
        {state.get('external_search_result', '(없음)')}
        [내부 결과]
        {state.get('internal_documents', '(없음)')}
        [결과 종합 및 도출]
        {combined}
        """.strip()

# ==================================
# 메인 노드
# ==================================
async def response_generator_node(state: AppState) -> AppState:
    """보고서 + 간결한 자연어 답변 생성"""
    query = (state.get("masked_user_query") or state.get("user_query") or "").strip()
    internal_docs = state.get("internal_documents") or []
    if not isinstance(internal_docs, list):
        internal_docs = [str(internal_docs)]
    joined_internal = "\n".join(internal_docs)
    external_raw = state.get("external_search_result") or ""
    # :일: 구조화된 요약 생성
    summary: SynthSummary = await _summary_chain.ainvoke({
        "query": query,
        "internal": joined_internal,
        "external": external_raw,
    })
    # :둘: 자연어 전체 설명 생성
    answer_msg = await _answer_chain.ainvoke({
        "key_points": "\n".join(summary.key_points),
        "cautions": "\n".join(summary.cautions),
        "recommended_steps": "\n".join(summary.recommended_steps),
    })
    full_answer = answer_msg.content.strip()
    # :셋: 핵심 요약 (1~2문장)
    short_msg = await _short_chain.ainvoke({"paragraph": full_answer})
    concise_answer = short_msg.content.strip()
    # :넷: DB 저장용 (첫 문장)
    short_answer = concise_answer.split(".")[0].strip() + "."
    # :다섯: 출력
    print(_build_report(state, summary))
    print("\n─────────────────────────────────────────────")
    print(f"질문: {query}")
    print(f"답변: {concise_answer}")
    print("─────────────────────────────────────────────")
    # :여섯: DB 저장
    await db_service.save_daily_report(state, short_answer)
    return state
