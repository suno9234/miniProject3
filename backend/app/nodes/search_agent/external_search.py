# Closed-weight LLM (GPT API) 외부 검색 노드
# GPT API를 사용해서 사용자 질문에 대한 일반적인 답변 생성
# 외부 지식 기반 정보 제공
# state의 external_search_result 필드 업데이트

import os
from ..state import AppState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# ------------------------------------------------
# .env 로드 (external_search.py → ../../.. → backend/.env)
# ------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# LLM (API 키는 .env에서 로드)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("TEMPERATURE", "0.1")),
    max_tokens=600,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# (민감 True일 때만) 마스킹-친화 정제
masking_prompt = PromptTemplate.from_template(
    (
        "다음 문장은 한국어로 작성된 현장 질의이며, 일부 구체 정보(제품명, 장비명, 수치, 코드명, 고객사명 등)가 "
        "기밀로 간주되어 마스킹([***]) 처리되어 있을 수 있습니다.\n"
        "목표: 마스킹은 그대로 두고, 문맥이 자연스럽게 이어지도록 다듬으세요.\n"
        "규칙:\n"
        "- [***], [제품명], [코드], [수치] 등은 복원/추정 금지.\n"
        "- 새로운 기밀정보(회사명, 구체 수치, 내부 용어) 추가 금지.\n"
        "- 한국어로만 출력, 부가 설명 금지.\n\n"
        "입력 문장:\n{masked_user_query}"
    )
)
masking_chain = masking_prompt | llm | StrOutputParser()

# 최종 답변 프롬프트
answer_prompt = PromptTemplate.from_template(
    (
        "너는 현장에서의 문제를 진단하고 해결책을 제시하는 AI 현장 전문가다.\n"
        "사용자의 질문을 이해하고 상황을 추론하여, 간결하고 실용적인 답변을 제시하라.\n"
        "답변은 바로 행동에 옮길 수 있는 형태로, 불필요한 설명은 생략한다.\n\n"
        "정보가 충분하지 않다면 빈 문자열을 전달할 것\n"
        "질문: {final_query}\n"
    )
)
answer_chain = answer_prompt | llm | StrOutputParser()

def sensitive_branch_answer(state: AppState) -> AppState:
    #민감 여부에 따라 마스킹된 쿼리 또는 일반 쿼리를 LLM에 전달
    is_sensitive = bool(state.get("is_sensitive", False))

    if is_sensitive:
        masked_q = (state.get("masked_user_query") or "").strip()
        if not masked_q:
            return {"external_search_result": ""}  
        final_query = masking_chain.invoke({"masked_user_query": masked_q}).strip()
    else:
        user_q = (state.get("user_query") or "").strip()
        if not user_q:
            return {"external_search_result": ""}  
        final_query = user_q

    answer = answer_chain.invoke({"final_query": final_query}).strip()

    return {"external_search_result": answer}