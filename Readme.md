# AI Agent API

## 환경 설정

1. `.env.example`을 복사해서 `.env` 파일 생성:
```bash
cp .env.example .env
```

2. `.env` 파일에서 비밀번호 설정

## 실행

```bash
cd backend
docker-compose up --build
```

- API: http://localhost:8000
- DB: localhost:3306

## 📁전체 디렉터리 구조

```
backend/
├── app/
│   ├── models.py                    # 전역 모델 정의
│   ├── api/
│   │   └── agent_router.py         # API 엔드포인트
│   ├── services/                   # 전역 설정 및 서비스
│   │   ├── db.py                   # MariaDB 연결 관리
│   │   ├── llm.py                  # LLM 설정 및 structured output
│   │   └── faiss.py                # FAISS 벡터 검색 설정
│   └── nodes/                      # 그래프 노드 및 에이전트
│       ├── state.py                # 전체 상태 정의
│       ├── planner.py              # 라우팅 플래너
│       ├── graph.py                # 메인 그래프 구조
│       ├── search_agent/           # 검색 에이전트
│       │   ├── __init__.py
│       │   ├── graph.py            # 서브그래프 정의
│       │   ├── sensitive_info_detector.py  # 민감정보 판별 (우찬민)
│       │   ├── external_search.py          # GPT API 외부 검색 (고서아)
│       │   ├── internal_search.py          # FAISS 내부 검색 (이재휘)
│       │   └── response_generator.py       # 최종 응답 생성 (고은렬)
│       ├── inventory_agent/        # 재고 확인 에이전트 (신순호)
│       │   ├── __init__.py
│       │   ├── graph.py            # 서브그래프 정의
│       │   └── inventory_node.py   # 재고 확인 로직
│       └── reservation_agent/      # 예약 처리 에이전트 (신수민)
│           ├── __init__.py
│           ├── graph.py            # 서브그래프 정의
│           └── reservation_node.py # 예약 처리 로직
├── .env                           # 환경변수 설정
├── .env.example                   # 환경변수 예시
├── requirements.txt               # Python 의존성
└── docker-compose.yml            # Docker 설정
```

## 🔄 전체 플로우
API 요청 → agent_router.py

플래너 실행 → planner.py (LLM으로 에이전트 타입 결정)

에이전트 라우팅 → graph.py (조건부 분기)

서브그래프 실행 → 각 에이전트의 graph.py

최종 응답 → state.generation 필드 반환
