import re
import requests
from typing import Dict, Any
from app.nodes.state import AppState

# normalizer #
class QueryNormalizer:
    """
    STT로부터 받은 텍스트를 LLM을 사용하여 정제
    
    책임:
    - filler 제거 ("어", "음", "그니까", "있잖아" 등)
    - 반복된 구절 축약
    - 비문을 간단한 요청문 형태로 마무리
    - AppState.user_query 에 저장
    """

    def __init__(self, llm_url: str = "http://localhost:8001/chat"):
        self.llm_url = llm_url
        
    def get_normalization_prompt(self, raw_text: str) -> str:
        """쿼리 정제를 위한 프롬프트 생성"""
        return f"""
다음은 STT(Speech-to-Text)로 변환된 사용자 입력입니다. 이를 깔끔하고 명확한 질문으로 정제해주세요.

원본 텍스트: "{raw_text}"

정제 규칙:
1. 말버릇/필러 제거: "어", "음", "그니까", "있잖아", "일단" 등
2. 반복 구문 축약: "단가 알려줘 단가 알려줘" → "단가 알려줘"
3. 문법 오류 수정: 부자연스러운 표현을 자연스럽게
4. 질문 형태 완성: 명확한 질문문으로 마무리
5. 전문용어 정제: 줄임말이나 오타 수정

정제된 결과만 한 줄로 출력하세요.
앞이나 뒤에 설명, 따옴표, 라벨을 붙이지 마세요.
출력은 오직 최종 문장 그 자체만이어야 합니다:
"""

    def normalize_with_llm(self, raw_text: str) -> str:
        """LLM을 사용하여 텍스트 정제 (동기)"""
        prompt = self.get_normalization_prompt(raw_text)
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", raw_text).strip()
            else:
                print(f"LLM 요청 실패: {response.status_code}")
                return raw_text
                
        except Exception as e:
            print(f"LLM 정제 중 오류: {e}")
            return raw_text

    def apply(self, state: AppState) -> AppState:
        """
            입력: AppState (state['user_query']에 STT 그대로/원문이 있다고 가정)
            출력: AppState (state['user_query']를 정제된 텍스트로 덮어쓴 복사본)
        """
        raw = state.get("user_query", "") or ""

        normalized = self.normalize_with_llm(raw)

        if not normalized or normalized.strip() == "":
            normalized = raw

        new_state: AppState = {**state}
        new_state["user_query"] = normalized
        return new_state


# SensitiveInfoDetector #
import json
import requests

class SensitiveInfoDetector:
    def __init__(self, llm_url: str = "http://localhost:8002/chat") -> None:
        """
        llm_url:
            - 민감정보 감지는 반드시 내부/사내용 안전 모델을 사용한다는 전제를 가진다.
            - 외부 오픈 모델을 쓰면 안 된다.
        """
        self.llm_url = llm_url

    def get_sensitivity_detection_prompt(self, text: str) -> str:
        """
        LLM에게 민감 여부와 마스킹된 쿼리를 JSON으로만 반환하게 강제하는 프롬프트.
        """
        return f"""
너는 민감정보 보안 필터다.
아래 사용자의 질의에 대해 두 가지를 판단해 JSON으로만 답하라.

1. is_sensitive:
   - true  : 민감한 정보(개인 식별 정보, 내부 단가/원가, 고객사/파트너 실명, 비공개 일정/로드맵 등)가 포함된 경우
   - false : 민감한 정보가 전혀 없는 경우

2. safe_query:
   - 외부(오픈 모델 / 외부 검색 엔진 등)에 그대로 노출해도 되는 안전한 버전의 쿼리
   - 민감한 부분(전화번호, 이메일, 주민등록번호, 고객사명, 내부 단가/원가, 비공개 일정 등)은 "###"로 마스킹한다
   - 민감하지 않다면 원문을 그대로 사용한다

반드시 아래 JSON 한 줄만 출력하고, 그 외 설명/마크다운/코드블록은 출력하지 마라.

예시 출력 (예시는 설명일 뿐 그대로 복사하지 마라):
{"is_sensitive": true, "safe_query": "고객 ### 단가 알려줘?"}

사용자 질의:
"{text}"
""".strip()

    def detect_and_mask_with_llm(self, text: str) -> tuple[bool, str]:
        """
        text(정제된 user_query)를 LLM에 보내서
        - 민감 여부(bool)
        - 외부로 내보내도 되는 마스킹된 쿼리(safe_query)
        를 받아온다.

        반환:
            (is_sensitive, masked_text)

        보안상 중요한 동작:
        - LLM 출력은 반드시 JSON이라고 가정하고 json.loads()로만 읽는다.
        - 파싱 실패나 통신 에러 시에는 보수적으로:
            is_sensitive = True
            masked_text  = "###"
          로 리턴한다.
        """
        prompt = self.get_sensitivity_detection_prompt(text)

        try:
            response = requests.post(
                self.llm_url,
                json={
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=10.0,
            )

            # API 레벨 오류 (예: 500, 404 등)
            if response.status_code != 200:
                # 운영시엔 logger.warning(...)으로 바꾸는 게 더 낫다
                print(f"[SensitiveInfoDetector] LLM 요청 실패: {response.status_code}")
                # 보수적 fallback
                return True, "###"

            result_json = response.json()

            # 모델 응답 텍스트 추출 (일반적인 chat-style 응답 가정)
            content = (
                result_json
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()

            # JSON 파싱 시도
            try:
                data = json.loads(content)
                is_sensitive = bool(data.get("is_sensitive", False))
                safe_query  = data.get("safe_query", text)

            except Exception:
                # 모델이 JSON 형식 안 지킨 경우: 가장 안전한 디폴트
                print("[SensitiveInfoDetector] LLM 응답 JSON 파싱 실패. 보수적 fallback 사용.")
                is_sensitive = True
                safe_query = "###"

            # 방어로직: safe_query가 비정상적으로 비었으면 최소한 마스킹된 형태로 막아
            if not safe_query or safe_query.strip() == "":
                is_sensitive = True
                safe_query = "###"

            return is_sensitive, safe_query

        except Exception as e:
            # 통신 자체 실패도 민감한 걸로 취급
            print(f"[SensitiveInfoDetector] LLM 민감정보 감지 중 예외 발생: {e}")
            return True, "###"


    def apply(self, state: AppState) -> AppState:
        user_query = state.get("user_query", "") or ""
        has_sensitive, masked_query = self.detect_and_mask_with_llm(user_query)

        new_state: AppState = {**state}
        new_state["is_sensative"] = has_sensitive
        new_state["masked_user_query"] = masked_query
        return new_state


