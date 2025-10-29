import re, json
from app.nodes.state import AppState
from app.services.llm import internal_pipe
import sys


# -----------------------------
# 공통: 패턴 정의
# -----------------------------
# 주민번호: 엄격/느슨 + 키워드 후행 숫자
SSN_STRICT = re.compile(r'(?<!\d)(\d{6})[- ]?(\d{7})(?!\d)')
SSN_LOOSE  = re.compile(r'(?<!\d)(\d{6})[- ]?(\d{3,7})(?!\d)')
SSN_AFTER_KW = re.compile(
    r'((?:주민등록?번호|주민번호|주번)\s*[:은는이가]?\s*["“”]?\s*)([\d\- ]{6,})',
    re.IGNORECASE
)
# 전화번호
PHONE = re.compile(r'(01[0-9]-?\d{3,4}-?)(\d{4})(?!\d)')

# 태그 파서
NORM_TAG_RE = re.compile(r"<norm>(.*?)</norm>", re.DOTALL)
JSON_TAG_RE = re.compile(r"<json>(.*?)</json>", re.DOTALL)

# -----------------------------
# normalizer
# -----------------------------
class QueryNormalizer:
    def __init__(self, pipe=internal_pipe):
        if pipe is None:
            raise ValueError("QueryNormalizer needs a valid pipeline")
        self.pipe = pipe

    def _build_prompt(self, raw_query: str) -> str:
        return f"""
당신은 사용자의 음성/STT 질의를 간결한 텍스트 쿼리로 '정제'만 하는 도우미다.

# 규칙
- 말버릇(예: 어, 음, 그러니까) 삭제
- 중복 표현/군더더기 제거
- 의미 왜곡/추가 금지 (없는 정보 만들지 말 것)
- 문장의 끝맺음이 불완전하면 자연스러운 '질문형'으로 마무리한다.
- 수치/기호/날짜/코드/번호는 그대로 둔다. 주의·경고·면책문을 절대 추가하지 말 것.
- 출력은 반드시 <norm> ... </norm> 형태의 태그 한 줄만. 태그 밖의 글자/공백/설명 금지.

# 예시
사용자 질의: "어... 그니까 내일 10시에 회의"
정제된 질의: <norm>내일 10시 회의가 맞나요?</norm>

사용자 질의: "음 가격 129,000원으로 다시 할 수 있어?"
정제된 질의: <norm>가격을 129,000원으로 다시 적용할 수 있나요?</norm>

사용자 질의: "{raw_query}"
정제된 질의:
""".strip()

    def _extract_norm(self, generated: str, raw_query: str) -> str:
        # 1) <norm>...</norm> 우선
        m = NORM_TAG_RE.search(generated)
        if m:
            txt = m.group(1).strip()
        else:
            # 2) 태그 없으면 마지막 비어있지 않은 줄
            lines = [l.strip() for l in generated.splitlines() if l.strip()]
            txt = lines[-1] if lines else ""

        # 3) 경고/주의 접두 제거
        if txt.startswith(("주의:", "(주의", "경고:", "(경고")):
            lines = [l.strip() for l in generated.splitlines() if l.strip()]
            txt = lines[-2] if len(lines) >= 2 else ""

        # 4) 최종 방어
        if not txt:
            txt = raw_query
        return txt

    def _call_model(self, prompt: str, raw_query: str) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=128,
            temperature=0.0,   # 결정적
            do_sample=False,
        )
        full = out[0]["generated_text"]
        return self._extract_norm(full, raw_query)

    def apply(self, state: AppState) -> AppState:
        raw = state.get("user_query", "") or ""
        prompt = self._build_prompt(raw)
        try:
            normalized = self._call_model(prompt, raw)
        except Exception as e:
            print(f"[QueryNormalizer] local model failed ({e}), fallback to raw.")
            normalized = raw

        if not normalized.strip():
            normalized = raw

        new_state: AppState = {**state}
        new_state["user_query"] = normalized
        print("Normalize 완료")
        print(json.dumps(normalized, ensure_ascii=False))  
        return new_state


# -----------------------------
# SensitiveInfoDetector
# -----------------------------
class SensitiveInfoDetector:
    def __init__(self, pipe=internal_pipe):
        if pipe is None:
            raise ValueError("SensitiveInfoDetector needs a valid pipeline")
        self.pipe = pipe

        # 비즈 민감 키워드
        risky_keywords = [
            r"(단가|가격|원가|비용|할인|마진|수익)",
            r"(고객사|파트너|협력사)",
            r"(로드맵|출시일|내부|기밀)"
        ]
        self.risky_patterns = [re.compile(p, re.IGNORECASE) for p in risky_keywords]

    def _build_prompt(self, text: str) -> str:
        # 치환 규칙을 명시하여 모델이 일관되게 마스킹하도록 유도
        return f"""
너는 민감정보 보안 필터다. 아래 규칙과 출력 양식을 반드시 지켜라.

[치환 규칙]
- 숫자 형태(예: 12, 763, 970904-1234567, 9709041234567, 970904-1180 등 숫자 뭉치)는 전부 #로 치환한다.
- 변환 예시(예: ##, ###, ######-#######, #############, ######-####)
- 다음 단어들에 해당하는 정보(보통 단어 뒤)는 모두 "###"로 치환한다: 단가, 가격, 원가, 마진, 수익, 고객사, 협력사, 파트너, 로드맵, 출시일, 내부, 기밀

[출력 규격]
- 오직 한 줄의 JSON만 출력한다.
- JSON은 <json>와 </json> 사이에만 둔다.
- 키: is_sensitive(boolean), safe_query(string)
- 경고/설명/마크다운 금지. 태그 밖 출력 금지.

[예시]
입력: "우리 고객사 A사의 단가 13,900원 알려줘"
<json>{{"is_sensitive": true, "safe_query": "우리 고객사 ##의 단가 ##,###원 알려줘"}}</json>

[입력]
{text}

[응답(반드시 한 줄)]
<json>{{"is_sensitive": <true|false>, "safe_query": "<여기에 결과>"}}</json>
""".strip()

    def _parse_json_one_line(self, generated: str, fallback_text: str):
        # 1) 태그 내부 우선
        m = JSON_TAG_RE.search(generated)
        cand = m.group(1).strip() if m else generated.strip()

        # 2) 본문 내 마지막 {...} 블록 추출
        if not (cand.startswith("{") and cand.endswith("}")):
            m2 = re.search(r"\{.*\}", cand, re.DOTALL)
            if m2:
                cand = m2.group(0).strip()

        # 3) 경고 접두 제거
        if cand.startswith(("주의:", "(주의", "경고:", "(경고")):
            cand = re.sub(r'^(\(*주의|\(*경고)\s*[:：)]\s*', '', cand).strip()

        # 4) 경량 보정: 작은따옴표 JSON 보정(가능한 안전하게)
        if "'" in cand and '"' not in cand:
            cand = re.sub(r"\'([A-Za-z0-9_]+)\'\s*:", r'"\1":', cand)   # 키
            cand = re.sub(r":\s*\'([^']*)\'", r': "\1"', cand)          # 값

        # 5) True/False/None 보정
        cand = re.sub(r"\bTrue\b", "true", cand)
        cand = re.sub(r"\bFalse\b", "false", cand)
        cand = re.sub(r"\bNone\b", "null", cand)

        try:
            data = json.loads(cand)
            is_sensitive = bool(data.get("is_sensitive", False))
            safe_query = data.get("safe_query", fallback_text)
            if not isinstance(safe_query, str) or not safe_query.strip():
                return True, "###"
            return is_sensitive, safe_query
        except Exception:
            return None

    def _call_model_for_mask(self, text: str) -> tuple[bool, str] | None:
        prompt = self._build_prompt(text)
        try:
            out = self.pipe(
                prompt,
                max_new_tokens=256,
                temperature=0.0,   # 결정적
                do_sample=False,
            )
        except Exception as e:
            print(f"[SensitiveInfoDetector] local model failed: {e}")
            return None

        full = out[0]["generated_text"]
        parsed = self._parse_json_one_line(full, fallback_text=text)
        if parsed is None:
            print("[SensitiveInfoDetector] model output not valid JSON, fallback.")
            return None
        return parsed

    def _fallback_detect_and_mask(self, text: str) -> tuple[bool, str]:
        masked = text
        flagged = False

        # 키워드 뒤 숫자 뭉치 → ### (주민번호 언급 후 숫자)
        before = masked
        masked = SSN_AFTER_KW.sub(lambda m: m.group(1) + "###", masked)
        flagged = flagged or (masked != before)

        # 주민번호 (엄격/느슨)
        before = masked
        masked = SSN_STRICT.sub("###", masked)
        masked = SSN_LOOSE.sub("###", masked)
        flagged = flagged or (masked != before)

        # 전화번호
        before = masked
        masked = PHONE.sub(r"\1****", masked)
        flagged = flagged or (masked != before)

        # 비즈 민감 키워드
        for p in self.risky_patterns:
            if p.search(text):
                masked = p.sub("###", masked)
                flagged = True

        return (flagged, masked if flagged else text)

    def apply(self, state: AppState) -> AppState:
        user_query = state.get("user_query", "") or ""

        result = self._call_model_for_mask(user_query)
        if result is not None:
            has_sensitive, safe_query = result
        else:
            has_sensitive, safe_query = self._fallback_detect_and_mask(user_query)

        # 추가 검증: is_sensitive=True인데 safe_query에 ###/****가 전혀 없으면 보수적 마킹
        if has_sensitive and ("###" not in safe_query and "****" not in safe_query):
            # 최소한 키워드 마스킹이라도 적용
            _, safe_query = self._fallback_detect_and_mask(user_query)

        new_state: AppState = {**state}
        new_state["is_sensitive"] = has_sensitive
        new_state["masked_user_query"] = safe_query
        print("마스킹 처리 완료")
        print(json.dumps(safe_query, ensure_ascii=False))  
        return new_state
