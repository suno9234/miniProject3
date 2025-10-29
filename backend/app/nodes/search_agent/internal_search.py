from typing import Dict, Any, List
from transformers.pipelines import TextGenerationPipeline
from app.services.internal_retriever import retriever_instance
from app.nodes.state import AppState
from app.services.llm import internal_pipe

def _build_prompt(query: str, contexts: List[str], chat_history: list) -> str:
    context_str = "\n".join([f"[출처: {i}] {c}" for i, c in enumerate(contexts, 1)]) if contexts else "검색된 근거 없음."

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
4. **반드시 아래 포맷 그대로 출력하세요. 출력 시작은 [답변]이어야 하며, [답변] 다음 줄부터 내용만 작성하세요.**
    예시:
    [답변]
    <여기에 최종 답변>

[질문]
{query}

[답변]
"""
    return prompt.strip()


def internal_searcher(state: AppState) -> Dict[str, Any]:
    """
    RRF+MMR+Rerank 기반 내부 검색 후, internal LLM으로 근거 인용 답변 생성
    """
    print("--- [노드 실행] internal_search_node ---")

    # 0) LLM 파이프라인 확인
    assert internal_pipe is not None, "internal_pipe가 초기화되지 않았습니다."
    llm_pipe: TextGenerationPipeline = internal_pipe

    # 1) 입력
    user_query = state.get("user_query", "") or ""
    chat_history = state.get("chat_history", []) or []

    # 2) 검색 (예외 흡수 + 폴백)
    contexts: List[str] = []
    if retriever_instance is None:
        print("[Retriever] retriever_instance가 None입니다. 컨텍스트 없이 진행합니다.")
    else:
        try:
            results = retriever_instance.retrieve(query=user_query, top_k=3, candidate_k=5)
            retrieved_ids = [idx for idx, _ in results] if results else []
            if retrieved_ids:
                try:
                    contexts = retriever_instance.get_documents_by_ids(retrieved_ids) or []
                except Exception as e:
                    print(f"[Retriever] get_documents_by_ids 예외: {e}")
                    contexts = []
            else:
                print("[Retriever] 검색 결과가 비어있습니다.")
        except Exception as e:
            print(f"[Retriever] retrieve 예외: {e}  -> 컨텍스트 없이 진행")

    # 3) 프롬프트 생성
    prompt = _build_prompt(user_query, contexts, chat_history)

    # 4) LLM 호출 (pad_token_id/파싱 방어)
    print("검색된 컨텍스트로 답변 생성을 시작합니다...")
    try:
        pad_id = getattr(getattr(llm_pipe, "tokenizer", None), "eos_token_id", None)
        response = llm_pipe(
            prompt,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=300,
            pad_token_id=pad_id,
            return_full_text=False,
        )
        gen = response[0].get("generated_text", "") if response and isinstance(response, list) else ""
        answer = gen.split("[답변]\n", 1)[-1].strip() if "[답변]" in gen else (gen or "답변을 생성하지 못했습니다.").strip()
    except Exception as e:
        print(f"LLM 답변 생성 중 오류 발생: {e}")
        answer = "답변 생성 중 오류가 발생했습니다."

    # 5) 반환
    return {
        "internal_documents": contexts,
        "internal_search_result": answer
    }