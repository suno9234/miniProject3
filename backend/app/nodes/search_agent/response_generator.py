# 최종 응답 생성 및 보고서 작성 노드
# external_search_result와 internal_documents 정보 종합
# 사용자에게 제공할 최종 답변 생성
# 검색 로그 및 보고서 DB에 저장
# state의 generation 필드 업데이트


# app/nodes/search_agent/response_generator.py
"""
검색 에이전트 - 최종 응답 생성 & 일일 대화 보고서 저장

[동작 개요]
- 내부(RAG, open-weight LLM 결과)와 외부(closed-weight LLM 결과)를 종합하여 최종 응답 생성
- 사용자 대화 기록(chat_history)에 메시지를 추가
- 하루치 대화 로그를 기반으로 보고서를 생성해 DB에 저장
"""

from __future__ import annotations

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from typing import List, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# === 내부 서비스 모듈 import ===
from app.nodes.state import AppState              # 상태 딕셔너리 (LangGraph 전체 상태 공유)
from app.services.llm import llm                   # LLM 인스턴스 (LangChain 기반)
from app.services.db import db_service             # DB 서비스 (일일 보고서 저장 기능 포함)


# ==================================
# 유틸 함수
# ==================================
def _append_history(state: AppState, role: str, text: str) -> None:
    """
    state.chat_history에 (role, text) 튜플을 안전하게 추가한다.
    - role: "user" 또는 "assistant"
    - text: 대화 내용
    """
    hist: List[Tuple[str, str]] = list(state.get("chat_history", []))  # 기존 히스토리 가져오기 (없으면 빈 리스트)
    hist.append((role, text))                                          # 새 메시지 추가
    state["chat_history"] = hist                                       # state에 다시 저장


# ==================================
# LLM이 출력할 구조화 요약 스키마
# ==================================
class SynthSummary(BaseModel):
    """
    LLM이 생성할 최종 요약 결과의 구조를 정의
    - key_points: 핵심 요지 목록
    - cautions: 주의점 또는 제약사항
    - recommended_steps: 실행 또는 권장 절차
    """
    key_points: List[str] = Field(default_factory=list, description="핵심 요지")
    cautions: List[str] = Field(default_factory=list, description="주의/제약")
    recommended_steps: List[str] = Field(default_factory=list, description="권장 절차(순서가 있다면 순서대로)")


# ==================================
# 프롬프트 정의 (내부 + 외부 결과 종합)
# ==================================
_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        # 시스템 역할 지정: 모델이 따라야 할 지침 정의
        "system",
        "너는 내부(open-weight LLM+RAG 결과)와 외부(closed-weight LLM 결과)를 종합하여 "
        "최종 의사결정용 요약과 답변을 생성하는 분석가이다. "
        "내부 내용이 조직 표준이며, 외부는 참고 근거로만 보조하라. "
        "출력은 반드시 JSON 스키마(SynthSummary)에 정확히 맞춰라. "
        "추가적인 설명 문장은 포함하지 마라."
    ),
    (
        # 사용자 메시지로 실제 입력 데이터 전달
        "user",
        "다음 정보를 종합하여 요약하라.\n\n"
        "[사용자 질의]\n{query}\n\n"             # 사용자의 실제 질문
        "[내부 결과(RAG)]\n{internal}\n\n"       # 내부 LLM + RAG 결과
        "[외부 결과(LLM 응답)]\n{external}\n"    # 외부 LLM 응답 결과
    ),
])

# LLM 호출 시 SynthSummary 스키마에 맞게 구조화된 출력을 강제
_summary_chain = _SUMMARY_PROMPT | llm.with_structured_output(SynthSummary)


# ==================================
# SynthSummary → 사용자용 텍스트 변환
# ==================================
def _render_generation_from_summary(s: SynthSummary) -> str:
    """
    SynthSummary 객체를 사람이 읽기 쉬운 문장 형태로 변환한다.
    - 섹션별 최대 항목 개수를 제한 (너무 길어지는 것 방지)
    - 비어있는 항목은 '(해당 없음)'으로 표시
    """
    def bullets(title: str, items: List[str], limit: int) -> str:
        items = [i for i in (items or []) if i.strip()]  # 공백 제거
        items = items[:limit]                            # 항목 수 제한
        if not items:
            return f"● {title}\n- (해당 없음)"
        return f"● {title}\n- " + "\n- ".join(items)     # bullet 포맷 생성

    # 섹션별로 조합
    parts = [
        bullets("핵심 요지", s.key_points, 8),
        bullets("주의/제약", s.cautions, 8),
        bullets("권장 절차", s.recommended_steps, 10),
    ]
    return "\n\n".join(parts)  # 최종 문자열 반환


# ==================================
# 일일 보고서 생성
# ==================================
def _build_daily_report(state: AppState) -> str:
    """
    하루 동안의 대화(chat_history)를 기반으로 일일 보고서 텍스트를 생성한다.
    - 생성 시각, 질의 내용, 민감 여부, 전체 대화 로그, 최종 응답 포함
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")  # 현재 시각 문자열
    # 대화 내역을 포맷팅 ([USER]/[ASSISTANT] 형태)
    chats = "\n".join([f"[{role.upper()}] {text}" for role, text in state.get("chat_history", [])])

    return f"""[일일 대화 보고서]
생성 시각 : {now}
사용자 질의 : {state.get('user_query','')}
민감 여부 : {"Yes" if state.get("is_sensative") else "No"}
마스킹 질의 : {state.get('masked_user_query','')}

[전체 대화 로그]
{chats or '(대화 내역 없음)'}

[최종 응답]
{state.get('generation','')}
"""


# ==================================
# LangGraph 노드: 최종 응답 생성기
# ==================================
async def response_generator_node(state: AppState) -> AppState:
    """
    내부 RAG 결과(internal_documents)와 외부 LLM 결과(external_search_result)를 종합하여
    최종 답변을 생성하고, 대화 기록 및 보고서를 DB에 저장한다.
    """
    # --- (1) 사용자 질의 정리 ---
    # 마스킹된 질의가 있으면 우선 사용, 없으면 원본 질의 사용
    query = (state.get("user_query")).strip()

    # 내부 문서(RAG 결과)는 리스트 형태로 강제 변환
    internal_docs = state.get("internal_documents") or []
    if not isinstance(internal_docs, list):
        internal_docs = [str(internal_docs)]
    joined_internal = "\n".join(internal_docs)  # 여러 문서를 하나의 문자열로 합침

    print("외부 검색 전")
    # 외부 검색 결과 (closed-weight LLM 응답)
    external_raw = state.get("external_search_result") or ""

    # --- (2) LLM 호출 (내부+외부 종합 요약 생성) ---
    summary: SynthSummary = await _summary_chain.ainvoke({
        "query": query,
        "internal": joined_internal,
        "external": external_raw,
    })

    print("외부 검색 후")

    # --- (3) 구조화 요약을 텍스트로 렌더링 후 state에 저장 ---
    generation = _render_generation_from_summary(summary)
    state["generation"] = generation

    # --- (4) 대화 기록(chat_history)에 추가 ---
    _append_history(state, "user", state.get("user_query", ""))   # 사용자의 질의 기록
    _append_history(state, "assistant", generation)               # LLM 응답 기록

    # # --- (5) 일일 대화 보고서 생성 및 DB 저장 ---
    # report_text = _build_daily_report(state)
    # try:
    #     # 보고서 저장 (DB layer에 정의된 save_daily_report 사용)
    #     await db_service.save_daily_report(state, report_text)
    # except Exception:
    #     # DB 저장 실패 시에도 전체 프로세스 중단 없이 무시
    #     pass

    # --- (6) state 반환 (LangGraph 노드 연결용) ---
    return state
