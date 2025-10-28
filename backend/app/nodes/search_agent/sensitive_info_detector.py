import re
from app.nodes.state import AppState

# normalizer #
class QueryNormalizer:
    """
    STT로부터 받은 텍스트를 정제

    책임:
    - filler 제거 ("어", "음", "그니까", "있잖아" 등)
    - 반복된 구절 축약
    - 비문을 간단한 요청문 형태로 마무리
    - AppState.user_query 에 저장
    """

    # 1) 말버릇 / filler / 잡음 후보
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
        cleaned = text
        for pat in self.FILLER_PATTERNS:
            cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)
        # 공백 정리
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def squash_repeats(self, text: str) -> str:
        """
        너무 가까이 반복되는 단어나 구문을 줄여준다.
        예: "단가 알려줘 단가 알려줘 그거 단가" -> "단가 알려줘 그거 단가"
        아주 단순하게만 처리.
        """
        tokens = text.split()
        result = []
        last = None
        for tok in tokens:
            if tok == last:
                # 직전 토큰과 동일하면 스킵
                continue
            result.append(tok)
            last = tok
        return " ".join(result)

    # 전문/기술 용어 정제
    def normalize_domain_terms(self, text: str) -> str:
        """
            domain 내용 정제
            예: '텀 -> 더미 로드', '캘 -> 캘리브레이션'
        """
        return text

    def finalize_request_style(self, text: str) -> str:
        """
        문장 끝이 너무 깨져 있으면 간단한 요청형으로 마무리.
        예: "카니발 다음주 재고 있나 예약"
          -> "카니발 다음주 재고 있나요? 예약 가능한가요?"
        이건 규칙 기반으로는 완벽할 수 없으니까 최소만 보정.
        """
        t = text.strip()

        # 끝이 명사/동사로 딱 끊기면 물음표 하나 붙여준다.
        if not re.search(r"[.?!]$", t):
            # 너무 공격적으로 붙이지 말고 짧은 쿼리면만 붙이자
            if len(t) <= 80:
                t = t + "?"
        return t


    def apply(self, state: AppState) -> AppState:
        """
        입력: AppState (user_query_raw 가 있다고 가정)
        출력: AppState (user_query 가 추가/갱신된 복사본)
        """
        raw = state.get("user_query_raw", "") or ""
        normalized = raw

        normalized = self.remove_fillers(normalized)
        normalized = self.squash_repeats(normalized)
        normalized = self.normalize_domain_terms(normalized)
        normalized = self.finalize_request_style(normalized)

        new_state: AppState = {**state}
        new_state["user_query"] = normalized
        return new_state


# SensitiveInfoDetector #
class SensitiveInfoDetector:
    def __init__(self) -> None:
        # 주민등록번호, 이메일, 전화번호 등
        self.sensitive_patterns = [
            re.compile(r"\b(\d{6})[- ]?(\d{7})\b"),  # 주민번호
            re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),  # 이메일
            re.compile(r"\b(\d{2,4})[- ]?(\d{3,4})[- ]?(\d{4})\b"),  # 전화번호
        ]

        # 리스키 키워드 패턴 원문 리스트
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

    # --- 감지 로직 ---
    def detect_sensitive_info(self, text: str) -> bool:
        return any(p.search(text) for p in self.sensitive_patterns)

    def detect_risky_keywords(self, text: str) -> bool:
        return any(p.search(text) for p in self.risky_patterns)

    def detect_any_sensitive_content(self, text: str) -> bool:
        return self.detect_sensitive_info(text) or self.detect_risky_keywords(text)

    # --- 마스킹 로직 ---
    def mask_sensitive_info(self, text: str) -> str:
        import re

        # 주민번호: 뒷자리 전부 감춤
        text = re.sub(r"(\b\d{6})-(\d{7}\b)", r"\1-*******", text)

        # 전화번호: 마지막 4자리 감춤
        text = re.sub(r"(01[0-9]-?\d{3,4}-?)(\d{4}\b)", r"\1****", text)

        # 이메일: 로컬파트 일부만 남기고 나머지 *
        def _mask_email(m):
            full = m.group(0)  # 전체 매치 "local@domain"
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

    # --- 💡 새 노드 진입점 ---
    def apply(self, state: AppState) -> AppState:
        """
        그래프 노드에서 호출:
        입력: 현재 AppState (최소 user_query는 있다고 가정)
        출력: is_sensative / masked_user_query 를 채워 넣은 새로운 AppState
        """
        user_query = state.get("user_query", "") or ""

        has_sensitive = self.detect_any_sensitive_content(user_query)

        masked_query = (
            self.mask_all_sensitive_content(user_query)
            if has_sensitive
            else user_query
        )

        # 불변성 유지: 복사본 만들고 필드만 추가/갱신
        new_state: AppState = {**state}
        new_state["is_sensative"] = has_sensitive  # 팀 정의 필드명 그대로 사용
        new_state["masked_user_query"] = masked_query

        return new_state

