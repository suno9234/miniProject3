"""
FAISS 설정 코드
=============

0. 환경 변수 설정 
1. FAISS INDEX 생성 및 임베딩, 저장, 추가, 검색 기능 추가

"""
#라이브러리
from __future__ import annotations
import os
from typing import Optional
import numpy as np
from dotenv import load_dotenv

# [수정] 'faiss' 라이브러리 자체를 임포트해야 합니다.
import faiss 
from sentence_transformers import SentenceTransformer

load_dotenv()

# 1. FAISS 
class FaissVectorStore:
    """
    Faiss 벡터 스토어 클래스.
    - 역할: 인코딩, 인코더 및 인덱스 초기화
    """
    def __init__(self):
        """
        환경 변수에서 설정을 로드하고 인코더를 초기화합니다.
        """
        # 1. 환경 변수에서 설정값 불러오기
        self.model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
        self.metric = os.getenv("FAISS_METRIC", "ip").lower()
        self.index_path = os.getenv("FAISS_INDEX_PATH", "./faiss.index")
        
        # 2. 인코더 및 인덱스 초기화
        self.encoder = SentenceTransformer(self.model_name)
        # [수정] faiss_store.Index -> faiss.Index
        self.index: Optional[faiss.Index] = None

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        주어진 텍스트 목록을 벡터(임베딩)로 변환합니다.
        """
        normalize = (self.metric == 'ip')
        return self.encoder.encode(
            texts, 
            normalize_embeddings=normalize,
            convert_to_numpy=True
        ).astype("float32")

    def build(self, documents: list[str]):
        """
        문서 목록으로부터 Faiss 인덱스를 빌드합니다.
        """
        embeddings = self.encode(documents)
        dimension = embeddings.shape[1]

        if self.metric == 'ip':
            # [수정] faiss_store.IndexFlatIP -> faiss.IndexFlatIP
            self.index = faiss.IndexFlatIP(dimension)
        elif self.metric == 'l2':
            # [수정] faiss_store.IndexFlatL2 -> faiss.IndexFlatL2
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError("FAISS_METRIC 환경 변수는 'ip' 또는 'l2'여야 합니다.")

        self.index.add(embeddings)
        print(f"인덱스 빌드 완료. 총 {self.index.ntotal}개의 벡터가 저장되었습니다.")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        벡터를 입력받아 Faiss 인덱스에서 검색을 수행합니다. (가장 단순한 형태)
        """
        if self.index is None:
            raise RuntimeError("인덱스가 빌드되거나 로드되지 않았습니다.")
        
        # 쿼리 벡터가 1D 배열일 경우 2D로 변경
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("쿼리 벡터는 (1, dimension) 형태의 2차원 배열이어야 합니다.")
        
        # [수정] search 함수가 결과를 반환(return)하도록 수정
        return self.index.search(query_vector.astype("float32"), top_k)

    def save(self, path: Optional[str] = None):
        """
        빌드된 인덱스를 파일에 저장합니다.
        """
        save_path = path or self.index_path
        if self.index is None:
            raise RuntimeError("저장할 인덱스가 존재하지 않습니다.")
        
        # [수정] faiss_store.write_index -> faiss.write_index
        faiss.write_index(self.index, save_path)
        print(f"인덱스를 '{save_path}' 경로에 저장했습니다.")

    def load(self, path: Optional[str] = None):
        """
        파일에서 인덱스를 로드합니다.
        """
        load_path = path or self.index_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"'{load_path}' 경로에 인덱스 파일이 없습니다.")
            
        # [수정] faiss_store.read_index -> faiss.read_index
        self.index = faiss.read_index(load_path)
        print(f"'{load_path}'에서 인덱스를 로드했습니다. (총 {self.index.ntotal}개 벡터)")

