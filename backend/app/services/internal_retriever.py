
"""
search 기능 구현
=============

0.
1. sparse search, hybrid search 기능 함수 구현
2. RRF 및 MMR 기능 함수 구현
(변경) rerank(크로스 인코더) 제거
"""

# 라이브러리
import numpy as np
import os
from elasticsearch import Elasticsearch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from app.services.faiss_store import FaissVectorStore
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200").strip()


# 기능 구현 class
class Retriever:
    def __init__(self, vector_store: FaissVectorStore, documents=None):
        self.documents = documents or []
        self.vector_store = vector_store
        self._all_doc_embeddings = None
        self.es_client = None
        self.es_index_name = "documents"

        if Elasticsearch is not None:
            try:
                self.es_client = Elasticsearch(ELASTICSEARCH_HOST)
                if self.es_client.info():
                    print(f"Elasticsearch 연결 성공 ({ELASTICSEARCH_HOST})")
                else:
                    print(f"Elasticsearch 연결 실패 ({ELASTICSEARCH_HOST})")
                    self.es_client = None
            except Exception as e:
                print(f"Elasticsearch 연결 실패 ({ELASTICSEARCH_HOST}): {e}")

        # (삭제) CrossEncoder / reranker 관련 초기화 전부 제거

    @property
    def all_doc_embeddings(self) -> np.ndarray:
        """
        문서 임베딩 계산 및 캐시
        """
        if self._all_doc_embeddings is None:
            print("문서 임베딩 계산 중...")
            self._all_doc_embeddings = self.vector_store.encode(self.documents)
        return self._all_doc_embeddings

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_vector = self.vector_store.encode([query])
        distances, indices = self.vector_store.search(query_vector, top_k=top_k)

        if len(indices) == 0 or len(indices[0]) == 0:
            return []

        return [(int(idx), 1.0 - float(dist)) for idx, dist in zip(indices[0], distances[0])]

    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.es_client is None:
            print("[ES] 클라이언트 없음 → sparse_search 건너뜀")
            return []

        try:
            if not self.es_client.indices.exists(index=self.es_index_name):
                print(f"[ES] 인덱스 미존재: {self.es_index_name} → sparse_search 건너뜀")
                return []
        except Exception as e:
            print(f"[ES] indices.exists 예외: {e} → sparse_search 건너뜀")
            return []

        fields_to_search = ["content.nori", "content.ngram", "content.compact"]
        try:
            response = self.es_client.search(
                index=self.es_index_name,
                query={"multi_match": {"query": query, "fields": fields_to_search}},
                _source=["original_index"],
                size=top_k,
            )
        except Exception as e:
            print(f"[ES] search 예외: {e} → sparse_search 건너뜀")
            return []

        results = []
        for hit in response.get("hits", {}).get("hits", []):
            if "_score" not in hit or "_source" not in hit or "original_index" not in hit["_source"]:
                print(f"[ES] 결과 필드 누락 (id={hit.get('_id')})")
                continue
            results.append((hit["_source"]["original_index"], hit["_score"]))
        return results

    def hybrid_search_rrf(
        self,
        query: str,
        top_k: int = 10,
        k_val: int = 60,
        threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Dense + Sparse 결과를 Reciprocal Rank Fusion (RRF)로 결합.
        """
        k_retrieve = max(top_k * 5, 20)
        dense_results = self.dense_search(query, top_k=k_retrieve)
        sparse_results = self.sparse_search(query, top_k=k_retrieve)

        dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        sparse_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_results)}

        rrf_scores = {}
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        for doc_id in all_doc_ids:
            dense_rank = dense_ranks.get(doc_id, 0)
            sparse_rank = sparse_ranks.get(doc_id, 0)

            rrf_score = 0.0
            if dense_rank > 0:
                rrf_score += 1.0 / (k_val + dense_rank)
            if sparse_rank > 0:
                rrf_score += 1.0 / (k_val + sparse_rank)

            rrf_scores[doc_id] = rrf_score

        sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        filtered_results = [(doc_id, score) for doc_id, score in sorted_rrf if score >= threshold]
        return filtered_results[:top_k]

    def retrieve_mmr(
        self,
        query: str,
        top_k: int = 10,
        lambda_mult: float = 0.5,
        relevance_threshold: float = 0.0,
    ) -> List[int]:
        """
        Maximal Marginal Relevance(MMR)로 다양성 반영.
        """
        k_retrieve = max(top_k * 5, 20)
        initial_results = self.dense_search(query, top_k=k_retrieve)

        initial_results_filtered = [(idx, score) for idx, score in initial_results if score >= relevance_threshold]
        if not initial_results_filtered:
            return []

        candidate_ids = [doc_id for doc_id, _ in initial_results_filtered]
        relevance_scores = {doc_id: score for doc_id, score in initial_results_filtered}

        _ = self.all_doc_embeddings  # ensure cached
        selected_ids: List[int] = []
        remaining_ids = list(candidate_ids)

        if not remaining_ids:
            return []

        # 첫 문서: 가장 높은 관련성
        best_id = max(remaining_ids, key=lambda id: relevance_scores.get(id, 0))
        selected_ids.append(best_id)
        remaining_ids.remove(best_id)

        while len(selected_ids) < top_k and remaining_ids:
            selected_embeds = self.all_doc_embeddings[selected_ids]
            mmr_scores = {}

            for cand_id in remaining_ids:
                relevance = relevance_scores[cand_id]
                cand_embed = self.all_doc_embeddings[cand_id].reshape(1, -1)
                similarity_to_selected = cosine_similarity(cand_embed, selected_embeds)[0]
                max_similarity = float(np.max(similarity_to_selected))
                mmr_score = (lambda_mult * relevance) - ((1 - lambda_mult) * max_similarity)
                mmr_scores[cand_id] = mmr_score

            best_id = max(mmr_scores, key=mmr_scores.get)
            selected_ids.append(best_id)
            remaining_ids.remove(best_id)

        return selected_ids

    # (삭제) rerank 함수 전체 제거

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k_val: int = 60,
        rrf_threshold: float = 0.0,
        mmr_lambda: float = 0.5,
        mmr_threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        (업데이트) rerank 제거 버전:
        1) RRF로 후보군과 RRF 점수 확보
        2) MMR로 다양성 후보 확보
        3) 후보 합집합을 만들고, RRF 점수(없으면 0) 기준으로 정렬하여 top_k 반환
        """
        # 1. RRF 후보 (점수 포함)
        rrf_candidates = self.hybrid_search_rrf(
            query, top_k=candidate_k, k_val=rrf_k_val, threshold=rrf_threshold
        )  # List[(id, score)]
        rrf_ids = {idx for idx, _ in rrf_candidates}
        rrf_score_map = {idx: score for idx, score in rrf_candidates}

        # 2. MMR 후보 (점수 없음)
        mmr_candidate_ids = self.retrieve_mmr(
            query, top_k=candidate_k, lambda_mult=mmr_lambda, relevance_threshold=mmr_threshold
        )
        mmr_ids = set(mmr_candidate_ids)

        combined_ids = list(rrf_ids | mmr_ids)
        if not combined_ids:
            print("RRF와 MMR에서 후보군을 찾지 못했습니다.")
            return []

        print(f"RRF({len(rrf_ids)}) + MMR({len(mmr_ids)}) = {len(combined_ids)}개의 고유 후보를 RRF 점수 기준으로 정렬합니다.")

        # 3. RRF 점수 기준 정렬 (MMR에서만 온 문서는 0점)
        scored = [(doc_id, rrf_score_map.get(doc_id, 0.0)) for doc_id in combined_ids]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

        return scored_sorted[:top_k]

    def get_documents_by_ids(self, doc_ids: List[int]) -> List[str]:
        """
        인덱스 리스트 → 실제 문서 텍스트 리스트 변환
        """
        docs = []
        total = len(self.documents) if self.documents else 0
        for i in doc_ids:
            if 0 <= i < total:
                docs.append(self.documents[i])
            else:
                docs.append(f"[문서없음: {i}]")
        return docs


def load_documents_from_vdb():
    """VDB 구축 시 사용된 문서들을 로드"""
    print("경고: load_documents_from_vdb() 함수가 구현되지 않았습니다.")
    return ["임시 문서입니다."]


def create_retriever_instance():
    """Retriever 인스턴스 생성"""
    try:
        vector_store = FaissVectorStore()

        index_path = os.getenv("FAISS_INDEX_PATH", "./faiss.index")
        if os.path.exists(index_path):
            vector_store.load(index_path)
            print(f"FAISS 인덱스 로드 완료: {vector_store.index.ntotal}개 벡터")
        else:
            print(f"경고: FAISS 인덱스 파일을 찾을 수 없습니다: {index_path}")
            print("vdb_manual.py를 실행하여 인덱스를 먼저 생성하세요.")
            return None

        documents = load_documents_from_vdb()
        return Retriever(vector_store=vector_store, documents=documents)

    except Exception as e:
        print(f"Retriever 인스턴스 생성 실패: {e}")
        return None


# 전역 Retriever 인스턴스 생성
retriever_instance = create_retriever_instance()


def get_documents_by_ids(doc_ids: List[int]) -> List[str]:
    """
    인덱스 리스트 → 실제 문서 텍스트 리스트 변환
    """
    if retriever_instance is None:
        return ["Retriever 인스턴스가 초기화되지 않았습니다."]
    return retriever_instance.get_documents_by_ids(doc_ids)

