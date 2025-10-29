import re, json
from app.nodes.state import AppState
from app.services.llm import internal_pipe
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

# Fallback
# LLM호출 실패 시 기본 정규식 기반 탐지를 통해 최소한의 민감데이터 마스킹

# normalizer #
class QueryNormalizer:
    def __init__(self, pipe=internal_pipe):
        if pipe is None:
            raise ValueError("QueryNormalizer needs a valid pipeline")
        self.pipe = pipe

    def _build_prompt(self, raw_query: str) -> str:
        return f"""
당신은 사용자의 음성/STT 질의를 다음 규칙에 따라서 자연스럽고 간결한 텍스트 쿼리로 정제하는 도우미다.

사용자 질의:
{raw_query}

규칙:
- '어', '음', '그니까' 같은 말버릇 삭제
- 중복 표현 정리
- 의미 왜곡 금지 (정보 추가 금지)
- 불완전하면 자연스러운 질문형으로 마무리
- 정제된 문장만 한 줄로 출력. 설명/따옴표 금지.
- 수치는 절대 민감정보가 아니기 때문에 주의나 경고 없이 모두 그대로 출력한다.

예시
사용자 질의: '어.. 그러니까 내일 회의 10시'

정제된 질의: '내일 회의가 10시에 있나요?'
""".strip()

    def _call_model(self, prompt: str) -> str:
        # internal_pipe는 HF pipeline("text-generation")이라고 가정
        out = self.pipe(
            prompt,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
        )
        # pipeline 출력은 보통 [{"generated_text": "..."}] 형태
        full = out[0]["generated_text"]

        # 모델이 prompt까지 복사해서 같이 줄 수 있으니까 마지막 줄만 뽑는 식으로 정리
        cleaned = full.splitlines()[-1].strip()
        first_line = cleaned.splitlines()[0].strip()
        return first_line

    def apply(self, state: AppState) -> AppState:
        raw = state.get("user_query", "") or ""

        prompt = self._build_prompt(raw)
        try:
            normalized = self._call_model(prompt)
        except Exception as e:
            print(f"[QueryNormalizer] local model failed ({e}), fallback to raw.")
            normalized = raw

        if not normalized.strip():
            normalized = raw

        new_state: AppState = {**state}
        new_state["user_query"] = normalized
        print("Normalize 완료")
        print(normalized)
        return new_state


# SensitiveInfoDetector #
class SensitiveInfoDetector:
    def __init__(self, pipe=internal_pipe):
        if pipe is None:
            raise ValueError("SensitiveInfoDetector needs a valid pipeline")
        self.pipe = pipe

        # 정규식 기반 fallback 준비. 최소한의 마스킹 규칙
        self.sensitive_patterns = [
            re.compile(r"(\d{6})[- ]?(\d{7})\b"),  # 주민번호 유사
            re.compile(r"(\d{2,4})[- ]?(\d{3,4})[- ]?(\d{4})\b"),  # 전화번호
        ]
        risky_keywords = [
            r"(단가|가격|원가|비용|할인|마진|수익)",
            r"(고객사|파트너|협력사)",
            r"(로드맵|출시일|내부|기밀)"
        ]
        self.risky_patterns = [re.compile(p, re.IGNORECASE) for p in risky_keywords]

    def _build_prompt(self, text: str) -> str:
        return f"""
너는 민감정보 보안 필터다.
아래 사용자의 질의를 분석해 두 가지를 결정하고 JSON으로만 한 줄로 출력하라.

1. is_sensitive:
   - true  : 개인 식별 정보(전화번호, 주민번호 등) 또는
             회사 내부 단가/원가/마진/수익/가격조건, 고객사명, 파트너명,
             비공개 로드맵/출시일 등 민감 정보가 포함됨
   - false : 위 민감 정보가 전혀 없음

2. safe_query:
   - 외부에 보내도 되는 안전한 쿼리
   - 민감한 부분은 전부 "###" 치환
   - 민감하지 않으면 원문 그대로

JSON 한 줄만 출력하라. 마크다운/설명 금지.
예: {{"is_sensitive": true, "safe_query": "고객 ### 단가 알려줘?"}}

사용자 질의:
{text}
""".strip()

    def _call_model_for_mask(self, text: str) -> tuple[bool, str] | None:
        prompt = self._build_prompt(text)
        try:
            out = self.pipe(
                prompt,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
            )
        except Exception as e:
            print(f"[SensitiveInfoDetector] local model failed: {e}")
            return None

        full = out[0]["generated_text"]
        last_line = full.splitlines()[-1].strip()

        try:
            data = json.loads(last_line)
        except Exception:
            print("[SensitiveInfoDetector] model output not valid JSON, fallback.")
            return None

        is_sensitive = bool(data.get("is_sensitive", False))
        safe_query = data.get("safe_query", text)

        if not isinstance(safe_query, str) or not safe_query.strip():
            return (True, "###")

        return (is_sensitive, safe_query)

    def _fallback_detect_and_mask(self, text: str) -> tuple[bool, str]:
        is_sensitive = False
        masked = text

        # 주민번호 마스킹
        masked = re.sub(r"(\d{6})[- ]?(\d{7}\b)", r"\1-*******", masked)

        # 전화번호 마스킹
        masked = re.sub(r"(01[0-9]-?\d{3,4}-?)(\d{4})", r"\1****", masked)

        # risky 키워드 감지 → 민감 플래그 True, 텍스트 치환
        for p in self.risky_patterns:
            if p.search(text):
                is_sensitive = True
                masked = p.sub("###", masked)

        # 개인정보 패턴도 감지되면 민감 True
        for p in self.sensitive_patterns:
            if p.search(text):
                is_sensitive = True

        if is_sensitive and not masked.strip():
            masked = "###"

        if not is_sensitive:
            return (False, text)
        return (True, masked)

    def apply(self, state: AppState) -> AppState:
        user_query = state.get("user_query", "") or ""

        result = self._call_model_for_mask(user_query)
        if result is not None:
            has_sensitive, safe_query = result
        else:
            has_sensitive, safe_query = self._fallback_detect_and_mask(user_query)

        new_state: AppState = {**state}
        new_state["is_sensitive"] = has_sensitive
        new_state["masked_user_query"] = safe_query
        print("마스킹 처리 완료")
        print(safe_query)
        return new_state