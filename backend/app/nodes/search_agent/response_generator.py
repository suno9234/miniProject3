"""
검색 에이전트 - 내부/외부 결과 종합 후 보고서 출력 (간결 자연어 답변 포함)

출력 형식(콘솔 프린트):
[간단 보고서]
생성 시각 : YYYY-MM-DD HH:MM
사용자 질의 : ...
내부 길이/외부 길이 : ...
소스 선택 : internal|external|both
─────────────────────────────────────────────
질문: ...
답변: ...
─────────────────────────────────────────────
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from app.nodes.state import AppState
from transformers.pipelines import TextGenerationPipeline
from app.services.llm import internal_pipe  # Qwen TextGenerationPipeline

# 내부 LLM 파이프라인
llm_pipe: TextGenerationPipeline = internal_pipe
PAD_ID = getattr(getattr(llm_pipe, "tokenizer", None), "eos_token_id", None)
if PAD_ID is None:
    PAD_ID = 0

# ---------------------------------------------------------------------
# 단일 프롬프트: 내부/외부/둘다 중 무엇을 사용할지 LLM이 스스로 결정
# - 반드시 아래 태그를 포함하여 반환:
#   <FINAL_ANSWER>...최종 사용자 답변...</FINAL_ANSWER>
# - 내부가 충분하면 내부를 우선하되, 불충분한 부분만 외부로 보완
# - 추측 금지, 내부/외부에 근거 없는 내용 금지
# ---------------------------------------------------------------------

ANSWER_PROMPT_TEMPLATE = """당신은 기술 지원 에이전트입니다. 아래 정보를 바탕으로 사용자 질문에 가장 정확하고 실행 가능한 답변을 작성하세요.
    내부 자료가 질문에 충분한 근거를 제공하면 내부 자료만 사용하세요. 내부에 빈칸이 있을 때만 외부 자료로 보완하세요.
    추측이나 일반 상식만으로 채우지 말고, 제공된 내부/외부 텍스트에 근거가 있을 때만 서술하세요.
    답변은 한국어 문장체로, 사용자가 바로 행동할 수 있도록 단계적으로 간결하게 작성하세요.

    [사용자 질문]
    {user_query}

    [내부 검색 결과]
    {internal}

    [외부 검색 결과]
    {external}

    다음의 태그만 포함하여 출력하라. 다른 텍스트는 절대 추가하지 말 것.
    <FINAL_ANSWER>
    ...최종 사용자 답변...
    </FINAL_ANSWER>
    """.strip()

_RE_ANS = re.compile(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", re.I | re.S)


def _call_internal_llm(prompt: str, max_new_tokens: int = 500) -> str:
    """Qwen(TextGenerationPipeline) 호출 래퍼"""
    outputs = llm_pipe(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        pad_token_id=PAD_ID,
        return_full_text=False,

    )
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return outputs[0].get("generated_text", "")
    if isinstance(outputs, str):
        return outputs
    return str(outputs)


def _parse_llm_output(raw: str) -> str:
    """
    <FINAL_ANSWER>...</FINAL_ANSWER> 파싱.
    실패 시 합리적 기본값으로 폴백.
    """
    m_ans = _RE_ANS.search(raw)
    final_answer = (m_ans.group(1).strip() if m_ans else raw.strip())
    return final_answer


def _build_report(
    user_query: str,
    internal_text: str,
    external_text: str,
    final_answer: str,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
        [간단 보고서]
        생성 시각 : {now}
        사용자 질의 : {user_query}
        내부 길이 / 외부 길이 : {len(internal_text)} / {len(external_text)}

        ─────────────────────────────────────────────
        질문: {user_query}
        답변: {final_answer}
        ─────────────────────────────────────────────
        """.strip()


def response_generator_node(state: AppState) -> Dict[str, Any]:
    """
    단일 LLM 프롬프트로 내부/외부/둘다 중 소스 선택을 LLM에게 맡기고,
    최종 답변(generation)을 생성하여 state에 저장.
    """
    user_query = state.get("user_query").strip()

    internal_docs = state.get("internal_documents") or []
    if not isinstance(internal_docs, list):
        internal_docs = [str(internal_docs)]
    internal_text = "\n".join(internal_docs)

    external_text = state.get("external_search_result") or ""

    # ---- 디버깅: 입력 프린트
    print("========== [response_generator_node] ==========")
    print(f"[Query]\n{user_query}\n")
    print(f"[Internal(len={len(internal_text)})]\n{internal_text[:800]}{'...' if len(internal_text) > 800 else ''}\n")
    print(f"[External(len={len(external_text)})]\n{external_text[:800]}{'...' if len(external_text) > 800 else ''}\n")

    prompt = ANSWER_PROMPT_TEMPLATE.format(
        user_query=user_query,
        internal=internal_text if internal_text else "(없음)",
        external=external_text if external_text else "(없음)",
    )
    raw = _call_internal_llm(prompt)
    final_answer = _parse_llm_output(raw)

    # 형식 미준수 시 1회 재시도(권장)
    if not _RE_ANS.search(raw):
        repair_prompt = (
            prompt
            + "\n\n주의: 형식을 지키지 않았습니다. 아래 형식만 포함하여 다시 출력하세요.\n"
              "<FINAL_ANSWER>\n...최종 사용자 답변...\n</FINAL_ANSWER>"
        )
        raw2 = _call_internal_llm(repair_prompt)
        if _RE_ANS.search(raw2):
            final_answer = _parse_llm_output(raw2)
            raw = raw2  # 디버깅 표시를 위해 교체

    # ---- 디버깅: 출력 프린트
    print("[RAW OUTPUT]\n" + raw[:1200] + ("..." if len(raw) > 1200 else ""))
    print(f"[PARSED] final_answer={final_answer[:400]}{'...' if len(final_answer) > 400 else ''}")

    # 간단 보고서 프린트
    report = _build_report(user_query, internal_text, external_text, final_answer)
    print("\n" + report + "\n")

    # state 업데이트
    return {
        **state,
        "generation": final_answer
    }
