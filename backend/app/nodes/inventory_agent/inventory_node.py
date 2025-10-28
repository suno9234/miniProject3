# 재고 확인 서브그래프
# 사용자 쿼리에서 물품명 추출
# DB에서 해당 물품의 재고 정보 조회
# 재고 현황을 사용자에게 제공할 형태로 가공
# state의 generation 필드 업데이트

"""
단일 노드: STT 텍스트 → (LLM 전담) 물품 추출 → 더미 DB 조회 → 응답 생성
- LLM 주입: state.context["llm"] (필수에 가깝게 사용)
- LLM 응답 형식: {"item": "<string 또는 null>"}
- Fallback: 카탈로그(키 목록) 기반 유사도 매칭(difflib), 정규식 사용 안함
"""

# app/nodes/inventory_agent/inventory_node.py
# ======================================================
# 재고 확인 서브그래프 (DB 연동 버전)
# STT 텍스트 → LLM 물품 추출 → DB 조회 → 응답 생성
# ======================================================

from __future__ import annotations
import json
from typing import Any, Dict, Optional, List
from difflib import SequenceMatcher, get_close_matches

from app.nodes.state import AppState
from app.services.llm import llm
from app.services.db import db_service

__all__ = ["inventory_agent_node"]

# ======================================================
# 유틸
# ======================================================

def _last_user_query(state: AppState) -> str:
    """STT 텍스트 추출 """
    if getattr(state, "user_query", None):
        return str(getattr(state, "user_query")).strip()
    if isinstance(state, dict) and state.get("user_query"):
        return str(state["user_query"]).strip()
    return ""


# ======================================================
# LLM 추출
# ======================================================


def extract_item_via_llm(text: str) -> Optional[str]:
    """LLM을 통해 재고 물품명 추출"""
    if not text:
        return None
    prompt = f"""
    너는 STT로 전사된 문장에서 '재고 확인이 필요한 물품명'을 한 개만 추출한다.
    JSON 형식으로만 응답하라: {{"item": "<string 또는 null>"}}
    입력: "{text}"
    """
    try:
        out = llm.invoke(prompt)
        if hasattr(out, "content"):
            out = out.content
        start, end = out.find("{"), out.rfind("}")
        payload = out[start : end + 1] if start != -1 and end != -1 else out
        obj = json.loads(payload)
        item = obj.get("item")
        print("[LLM OUTPUT]:", out)
        if item and isinstance(item, str) and item.strip():
            return item.strip()
    except Exception as e:
        print("[LLM ERROR]", e)
    return None


# ======================================================
# DB 조회
# ======================================================


async def _lookup_from_db(item_name: str) -> Dict[str, Any]:
    """MariaDB item 테이블 조회"""
    async with db_service.get_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT name, quantity, location, sku
                FROM item
                WHERE name LIKE %s
                LIMIT 1;
                """,
                (f"%{item_name}%",),
            )
            row = await cursor.fetchone()

    if not row:
        return {"found": False, "data": {"item_name": item_name}}

    name, quantity, location, sku = row
    return {
        "found": True,
        "data": {
            "item_name": name,
            "quantity": quantity,
            "location": location,
            "sku": sku,
        },
    }


# ======================================================
# 단일 노드 (비동기)
# ======================================================


async def inventory_agent_node(state: AppState) -> Dict[str, Any]:
    """
    STT 텍스트 → LLM 추출 → DB 조회 → 응답 생성
    """
    print("========== INVENTORY AGENT ==========")
    print(state)

    user_text = _last_user_query(state)
    item = extract_item_via_llm(user_text)

    if not item:
        reply = "확인할 물품명을 파악하지 못했습니다. 예) '맥북 프로 14 재고 있나요'처럼 모델명을 포함해 말씀해 주세요."
        return {
            "generation": {"inventory": {"text": reply}},
            "messages": [{"role": "assistant", "content": reply}],
        }

    # DB 조회
    inv = await _lookup_from_db(item)
    found = inv["found"]
    data = inv["data"]
    item_name = data.get("item_name") or item
    qty = data.get("quantity", 0)
    loc = data.get("location", "확인 중")
    sku = data.get("sku", "-")

    # 응답
    if found and qty > 0:
        reply = f"'{item_name}' 재고가 {qty}개 확인되었습니다. 출고지: {loc} (SKU: {sku}). 픽업/배송 중 원하는 방법을 알려 주세요."
    elif found and qty == 0:
        reply = f"죄송합니다. '{item_name}'는 현재 재고가 없습니다. 입고 알림을 설정할까요? 확인된 출고지: {loc} (SKU: {sku})."
    else:
        reply = f"'{item_name}'에 대한 재고 정보를 찾지 못했습니다. 정확한 모델명이나 다른 명칭을 알려 주시면 재확인하겠습니다."

    return {
        "generation": {
            "inventory": {
                "text": reply,
                "raw": {"query": user_text, "item": item, "lookup": inv},
            }
        },
    }
