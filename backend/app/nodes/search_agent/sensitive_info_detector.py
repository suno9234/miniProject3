import re, json
from app.nodes.state import AppState
from app.services.llm import internal_pipe
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

# ============================================================
# 공통 유틸
# ============================================================

_CODEFENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", flags=re.DOTALL)

def _strip_codefence(s: str) -> str:
    """백틱 코드펜스(``` 혹은 ```json ...)가 있으면 제거"""
    if s.strip().startswith("```"):
        return _CODEFENCE_RE.sub("", s).strip()
    return s

def _truncate_to_first_json_obj(s: str) -> str:
    """문자열에서 첫 번째 '{'부터 짝이 맞는 '}'까지 잘라서 반환(단순 절단)."""
    start = s.find("{")
    if start == -1:
        return s
    # 첫 '}'까지 자르기(간단/보수적). 복잡한 중괄호 중첩은 _extract_json이 처리 시도.
    end = s.find("}", start)
    if end != -1:
        return s[start:end+1]
    return s[start:]

def _extract_json(s: str) -> dict | None:
    """
    생성 텍스트에서 첫 번째 JSON 오브젝트를 찾아 파싱.
    코드펜스/잡설이 섞여도 중괄호 블록을 스캔하여 안정화.
    """
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        return None

def _get_tokenizer():
    """
    internal_pipe에서 tokenizer를 확보.
    (pipeline 생성 시 tokenizer를 주입했으므로 존재해야 함)
    """
    tk = getattr(internal_pipe, "tokenizer", None)
    if tk is None:
        raise RuntimeError("internal_pipe.tokenizer 를 찾을 수 없습니다.")
    return tk

# ============================================================
# QueryNormalizer
# ============================================================

class QueryNormalizer:
    def __init__(self, pipe=internal_pipe):
        if pipe is None:
            raise ValueError("QueryNormalizer needs a valid pipeline")
        self.pipe = pipe
        self.tokenizer = _get_tokenizer()

    def _build_messages(self, raw_query: str):
        """
        Qwen chat template용 메시지 구성.
        - 인사말/추가문구/설명 금지
        - 한 줄만 출력
        - 의미 보존 + 군더더기 제거
        - few-shot 예시로 행동 구속
        """
        system_txt = (
            "역할: 너는 사용자의 음성/STT 질의를 의미를 그대로 유지한 채, 불필요한 말을 제거하여 "
            "간결한 텍스트 쿼리로 정제하는 도우미다.\n"
            "절대 규칙:\n"
            "1) 한 줄만 출력한다.\n"
            "2) 인사말/추가 설명/경고/코드블록/따옴표를 절대 쓰지 않는다.\n"
            "3) 의미를 왜곡하거나 새로운 정보를 추가하지 않는다.\n"
            "4) 수치(시간, 날짜, 숫자 등)는 그대로 유지한다.\n"
            "5) 불완전하면 자연스러운 질문형으로 마무리한다."
        )

        examples = [
            {"role": "user", "content": "어.. 그러니까 내일 회의 10시"},
            {"role": "assistant", "content": "내일 회의가 10시에 있나요?"},
            {"role": "user", "content": "음 그 파일 최신 버전 어디 있어"},
            {"role": "assistant", "content": "그 파일의 최신 버전은 어디에 있나요?"},
            {"role": "user", "content": "그니까 스마트싱스 등록 언제까지야"},
            {"role": "assistant", "content": "스마트싱스 등록 마감일이 언제인가요?"},
            # 숫자/전화번호 보존 예시
            {"role": "user", "content": "010-1234-5678로 연락해줘"},
            {"role": "assistant", "content": "010-1234-5678로 연락해 줄 수 있나요?"},
        ]

        messages = [{"role": "system", "content": system_txt}]
        messages.extend(examples)
        messages.append({"role": "user", "content": raw_query})
        return messages

    def _call_model(self, raw_query: str) -> str:
        messages = self._build_messages(raw_query)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = self.pipe(
            prompt,
            max_new_tokens=96,
            do_sample=False,
            return_full_text=False,  # 프롬프트 에코 제거
        )
        gen = out[0]["generated_text"].strip()
        gen = _strip_codefence(gen)

        # 한 줄만 요구: 첫 줄만 반영
        line = gen.splitlines()[0].strip()
        # 혹시 따옴표로 감싸면 제거
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()
        return line

    def apply(self, state: AppState) -> AppState:
        raw = state.get("user_query", "") or ""
        try:
            normalized = self._call_model(raw)
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

# ============================================================
# SensitiveInfoDetector
# ============================================================

class SensitiveInfoDetector:
    def __init__(self, pipe=internal_pipe):
        if pipe is None:
            raise ValueError("SensitiveInfoDetector needs a valid pipeline")
        self.pipe = pipe
        self.tokenizer = _get_tokenizer()

        # 정규식 기반 fallback 준비. 최소한의 마스킹 규칙
        self.sensitive_patterns = [
            re.compile(r"(\d{6})[- ]?(\d{7})\b"),                  # 주민번호 유사
            re.compile(r"(\d{2,4})[- ]?(\d{3,4})[- ]?(\d{4})\b"),  # 전화번호
        ]
        risky_keywords = [
            r"(단가|가격|원가|비용|할인|마진|수익)",
            r"(고객사|파트너|협력사)",
            r"(로드맵|출시일|내부|기밀)"
        ]
        self.risky_patterns = [re.compile(p, re.IGNORECASE) for p in risky_keywords]

    def _build_messages(self, text: str):
        """
        Qwen chat template용 메시지 구성.
        - 출력 형식: JSON 한 줄
        - 코드블록/마크다운/설명 금지
        - few-shot 예시를 '별도 턴'으로 제공
        """
        system_txt = (
            "너는 민감정보 보안 필터다. 아래 사용자 질의를 분석하여 다음 JSON '한 줄'만 출력하라. "
            "다른 문자는 절대 출력하지 마라. 코드블록/마크다운 금지.\n\n"
            '출력 형식: {"is_sensitive": true|false, "safe_query": "<문장>"}\n\n'
            "판정 규칙:\n"
            "- is_sensitive=true: 개인 식별 정보(전화번호, 주민번호 등) 또는 "
            "회사 내부 단가/원가/마진/수익/가격조건, 고객사명, 파트너명, "
            "비공개 로드맵/출시일 등 민감 정보가 포함됨\n"
            "- is_sensitive=false: 위 민감 정보가 전혀 없음\n\n"
            "safe_query 규칙:\n"
            '- 외부에 보내도 되는 안전한 쿼리. 민감한 부분은 전부 "###"로 치환\n'
            "- 민감하지 않으면 원문 그대로\n"
        )

        fewshots = [
            # 전화번호
            {"role": "user", "content": "010-1234-5678로 연락주세요"},
            {"role": "assistant", "content": '{"is_sensitive": true, "safe_query": "###로 연락주세요"}'},

            # 주민번호
            {"role": "user", "content": "주민번호 123456-7123456 확인해줘"},
            {"role": "assistant", "content": '{"is_sensitive": true, "safe_query": "주민번호 ### 확인해줘"}'},

            # 고객사/마진
            {"role": "user", "content": "고객사 ABC회사 마진율 알려줘"},
            {"role": "assistant", "content": '{"is_sensitive": true, "safe_query": "고객사 ### 마진율 알려줘"}'},

            # 비민감 예시
            {"role": "user", "content": "오늘 서울 날씨 어때?"},
            {"role": "assistant", "content": '{"is_sensitive": false, "safe_query": "오늘 서울 날씨 어때?"}'},
        ]

        user_turn = {"role": "user", "content": text}

        messages = [{"role": "system", "content": system_txt}]
        messages.extend(fewshots)
        messages.append(user_turn)
        return messages

    def _call_model_for_mask(self, text: str) -> tuple[bool, str] | None:
        messages = self._build_messages(text)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            out = self.pipe(
                prompt,
                max_new_tokens=128,
                do_sample=False,
                return_full_text=False  # 프롬프트 에코 제거
            )
        except Exception as e:
            print(f"[SensitiveInfoDetector] local model failed: {e}")
            return None


        gen = out[0]["generated_text"].strip()
        gen = _strip_codefence(gen)
        gen = _truncate_to_first_json_obj(gen)

        data = _extract_json(gen)
        if not data:
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


if __name__ == "__main__":
    # 테스트 케이스
    test_cases = [
        # "010-1234-5678로 연락주세요",
        # "주민번호 123456-7123456 확인해줘",
        # "오늘 날씨 어때?",
        "고객사 ABC회사 마진율 알려줘"
    ]

    # 현재 테스트 흐름은 Normalizer -> Detector 이지만,
    # 민감정보를 원문 기준으로 먼저 지우고 싶다면 상위 그래프에서 순서를 바꾸십시오.
    normalizer = QueryNormalizer()
    detector = SensitiveInfoDetector()

    for i, query in enumerate(test_cases, 1):
        # print(f"\n=== 테스트 {i} ===")
        print(f"입력: {query}")

        st: AppState = {"user_query": query}
        # st = normalizer.apply(st)  # 정규화
        st = detector.apply(st)    # 마스킹

        print(f"민감정보 여부: {st.get('is_sensitive')}")
        print(f"마스킹된 쿼리: {st.get('masked_user_query')}")
