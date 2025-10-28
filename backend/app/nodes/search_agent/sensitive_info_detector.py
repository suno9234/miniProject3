import re
from app.nodes.state import AppState

# normalizer #
class QueryNormalizer:
    """
    사용자의 입력(user_query)을 LLM/검색에 쓰기 적합한 형태로 다듬는다.
    - 말버릇/군더더기 제거
    - 반복된 단어 축약
    - 도메인 용어 표준화 (placeholder)
    - 문장 마무리 보정
    결과는 다시 state["user_query"]에 덮어쓴다.
    """

    # filler / 말버릇 / 생각 중 멈춤 등
    FILLER_PATTERNS = [
        r"\b어\b",
        r"\b음\b",
        r"\b어음\b",
        r"\b아 그러니까\b",
        r"\b그니까\b",
        r"\b있잖아\b",
        r"\b있잖아요\b",
        r"\b그 뭐냐\b",
        r"\b일단\b",
    ]

    def remove_fillers(self, text: str) -> str:
        """어, 음, 그니까... 같은 말버릇 제거"""
        cleaned = text
        for pat in self.FILLER_PATTERNS:
            cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
        # 중복 공백 정리
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def squash_repeats(self, text: str) -> str:
        """
        너무 가까이 반복된 토큰 제거.
        예: "단가 알려줘 단가 알려줘 그 단가" -> "단가 알려줘 그 단가"
        """
        tokens = text.split()
        result = []
        last = None
        for tok in tokens:
            if tok == last:
                continue
            result.append(tok)
            last = tok
        return " ".join(result)

    def normalize_domain_terms(self, text: str) -> str:
        """
        도메인/내부 용어를 표준화하거나 애매한 표현을 명확하게 바꾸는 자리.
        예:
        - "텀" -> "더미 로드"
        - "캘킷" -> "캘리브레이션 키트"
        """
        # TODO: 사내 도메인 용어 매핑 룰 적용 예정
        return text

    def finalize_request_style(self, text: str) -> str:
        """
        문장 마무리가 애매하면 간단한 요청/질문형으로 정리.
        짧은 쿼리에만 '?'를 붙여 읽기 쉽게 만든다.
        """
        t = text.strip()
        if not re.search(r"[.?!]$", t):
            if len(t) <= 80:
                t = t + "?"
        return t

    def apply(self, state: AppState) -> AppState:
        """
        입력: AppState (state["user_query"]가 있어야 함)
        출력: AppState (state["user_query"]를 정제된 텍스트로 덮어쓴 사본)
        """
        original = state.get("user_query", "") or ""
        normalized = original

        # 1) filler 제거
        normalized = self.remove_fillers(normalized)

        # 2) 반복된 단어/구절 축약
        normalized = self.squash_repeats(normalized)

        # 3) 도메인 용어 표준화 
        normalized = self.normalize_domain_terms(normalized)

        # 4) 문장형으로 마무리
        normalized = self.finalize_request_style(normalized)

        # 불변 패턴 유지: 복사 후 덮어쓰기
        new_state: AppState = {**state}
        new_state["user_query"] = normalized
        return new_state


# SensitiveInfoDetector #
class SensitiveInfoDetector:
    def __init__(self) -> None:
        self.sensitive_patterns = [
            re.compile(r"\b(\d{6})[- ]?(\d{7})\b"),  # 주민번호
            re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),  # 이메일
            re.compile(r"\b(\d{2,4})[- ]?(\d{3,4})[- ]?(\d{4})\b"),  # 전화번호
        ]

        risky_keywords = [
            r"\b(?:CPU|GPU|메모리|RAM|스토리지|용량|성능|스펙|사양)\b",
            r"\b(?:가격|비용|원가|단가|할인|할인율|마진|수익)\b",
            r"\b(?:출시일|런칭|발표|공개|출시)\b",
            r"\b(?:기밀|비밀|내부|문서|자료|데이터|정보)\b",
            r"\b(?:전략|계획|로드맵|비전|목표)\b",
            r"\b(?:고객|클라이언트|파트너|협력사)\b",
            r"\b(?:알고리즘|코드|소스|프로그램|시스템)\b",
            r"\b(?:특허|지적재산권|IP|라이선스)\b",
            r"\b(?:매출|수익|손익|재무|회계|예산)\b",
            r"\b(?:투자|펀딩|자금|자본|주식)\b",
        ]
        self.risky_patterns = [re.compile(p, re.IGNORECASE) for p in risky_keywords]

    def detect_sensitive_info(self, text: str) -> bool:
        return any(p.search(text) for p in self.sensitive_patterns)

    def detect_risky_keywords(self, text: str) -> bool:
        return any(p.search(text) for p in self.risky_patterns)

    def detect_any_sensitive_content(self, text: str) -> bool:
        return self.detect_sensitive_info(text) or self.detect_risky_keywords(text)

    def mask_sensitive_info(self, text: str) -> str:
        # 주민번호 -> 뒤자리 가림
        text = re.sub(r"(\b\d{6})-(\d{7}\b)", r"\1-*******", text)

        # 전화번호 -> 마지막 4자리 가림
        text = re.sub(r"(01[0-9]-?\d{3,4}-?)(\d{4}\b)", r"\1****", text)

        # 이메일 -> 로컬파트 절반만 남기고 나머지 '*'
        def _mask_email(m):
            full = m.group(0)
            local, domain = full.split("@", 1)
            keep = max(1, len(local) // 2)
            return local[:keep] + "*" * (len(local) - keep) + "@" + domain

        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", _mask_email, text)
        return text

    def mask_risky_keywords(self, text: str) -> str:
        masked = text
        for pat in self.risky_patterns:
            masked = pat.sub("###", masked)
        return masked

    def mask_all_sensitive_content(self, text: str) -> str:
        return self.mask_risky_keywords(self.mask_sensitive_info(text))

    def apply(self, state: AppState) -> AppState:
        user_query = state.get("user_query", "") or ""
        has_sensitive = self.detect_any_sensitive_content(user_query)

        masked_query = (
            self.mask_all_sensitive_content(user_query)
            if has_sensitive
            else user_query
        )

        new_state: AppState = {**state}
        new_state["is_sensative"] = has_sensitive
        new_state["masked_user_query"] = masked_query
        return new_state