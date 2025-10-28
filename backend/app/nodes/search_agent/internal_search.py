"""
내부 DB 내용으로 답변 생성
=============
1. prompt build 함수
2. agent 기능 함수 : internal_searcher
"""

# 라이브러리
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.services.internal_retriever import Retriever
from app.services.faiss_store import FaissVectorStore
from app.nodes.state import AppState
from typing import Dict, Any
from app.services.llm import internal_pipe



# agent Node

def _build_prompt(query: str, contexts: list[str], chat_history: list) -> str:
    context_str = "\n".join([f"[출처: {i}] {c}" for i, c in enumerate(contexts, 1)]) if contexts else "검색된 근거 없음."

    # 튜플과 dict 모두 처리
    history_strs = []
    for msg in chat_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            history_strs.append(f"{msg['role']}: {msg['content']}")
        elif isinstance(msg, (tuple, list)) and len(msg) == 2:
            history_strs.append(f"{msg[0]}: {msg[1]}")
        else:
            history_strs.append(str(msg))
    history_str = "\n".join(history_strs)

    prompt = f"""
[이전 대화]
{history_str}

[검색된 근거]
{context_str}

[규칙]
1. "검색된 근거"에 있는 내용만을 바탕으로 답변을 생성하세요.
2. 절대로 "검색된 근거"에 없는 내용을 추론하거나 지어내지 마세요.
3. 답변에 "검색된 근거"의 내용을 포함할 경우, 문장 끝에 [출처: N] 형식으로 출처를 반드시 인용하세요.

[질문]
{query}

[답변]
"""
    return prompt

# --- 3. 핵심 노드 함수 (LangGraph가 호출할 함수) ---
def internal_searcher(state: AppState, retriever: Retriever, internal_pipeline : internal_pipe) -> Dict[str, Any]:
    """
    LangGraph의 '노드' 역할을 하는 함수입니다.
    Retriever와 LLM Pipeline은 외부에서 '주입(injected)'받습니다.
    
    Args:
        state (AppState): LangGraph로부터 전달받은 현재 상태.
        retriever (Retriever): main 앱에서 초기화된 Retriever 객체.
        internal_pipe (Pipeline): main 앱에서 초기화된 Transformers Pipeline 객체.

    Returns:
        Dict[str, Any]: AppState를 업데이트할 키와 값.
    """
    print("--- [노드 실행] internal_search_node ---")
    
    # 1. State에서 필요한 정보(쿼리, 대화기록)를 가져옵니다.
    user_query = state.get("user_query", "")
    chat_history = state.get("chat_history", [])
    
    # 2. Retriever를 사용해 RAG 파이프라인(RRF+MMR -> Rerank)을 실행합니다.
    print(f"'{user_query}'에 대한 검색을 시작합니다...")
    # (참고: retriever.retrieve()의 세부 파라미터는 state에서 받아오거나 하드코딩할 수 있습니다.)
    results = retriever.retrieve(query=user_query, top_k=3, candidate_k=5)
    
    retrieved_ids = [idx for idx, score in results]
    contexts = retriever.get_documents_by_ids(retrieved_ids)
    
    # 3. 검색된 근거, 쿼리, 대화 기록으로 프롬프트를 생성합니다.
    prompt = _build_prompt(user_query, contexts, chat_history)
    
    # 4. 주입받은 Internal LLM 파이프라인으로 답변을 생성합니다.
    print("검색된 컨텍스트로 답변 생성을 시작합니다...")
    try:
        response = internal_pipeline(prompt)
        answer = response[0]['generated_text'].split("[답변]\n")[-1].strip()
    except Exception as e:
        print(f"LLM 답변 생성 중 오류 발생: {e}")
        answer = "답변 생성 중 오류가 발생했습니다."

    # 5. LangGraph가 State를 업데이트할 수 있도록 결과를 딕셔너리로 반환합니다.
    return {
        "internal_documents": contexts,
        "internal_search_result": answer
    }


    