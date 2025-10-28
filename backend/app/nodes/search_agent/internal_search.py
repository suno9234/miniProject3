# Open-weight LLM + FAISS 내부 검색 노드
# 사용자 쿼리를 임베딩으로 변환
# FAISS에서 유사도 검색으로 관련 문서 찾기
# 검색된 내부 문서들을 기반으로 답변 생성
# state의 internal_documents 필드 업데이트