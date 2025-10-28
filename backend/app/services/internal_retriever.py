"""
search 기능 구현
=============

0. 
1. sparse search, hybrid search 기능 함수 구현
2. RRF 및 MMR 기능 함수 구현

"""

#라이브러리
import numpy as np
import os
from elasticsearch import Elasticsearch
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Tuple
from sentence_transformers import CrossEncoder
from project.miniProject3.backend.app.services.faiss_store import FaissVectorStore
from dotenv import load_dotenv

load_dotenv()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
# [수정] .env에서 ELASTICSEARCH_HOST 값을 읽어옵니다. (기본값은 localhost)
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")


# 기능 구현 class
class Retriever:
    def __init__(self, documents: list[str], vector_store: FaissVectorStore):
        self.documents = documents
        self.vector_store = vector_store
        self._all_doc_embeddings = None
        self.es_client = None
        self.es_index_name = "documents"

        if Elasticsearch is not None:
            try:
                # [수정] 하드코딩된 주소 대신, 위에서 읽어온 ELASTICSEARCH_HOST 변수를 사용합니다.
                self.es_client = Elasticsearch(ELASTICSEARCH_HOST)
                if self.es_client.info():
                    print(f"Elasticsearch 연결 성공 ({ELASTICSEARCH_HOST})")
                else:
                    print(f"Elasticsearch 연결 실패 ({ELASTICSEARCH_HOST})")
                    self.es_client = None
            except Exception as e:
                print(f"Elasticsearch 연결 실패 ({ELASTICSEARCH_HOST}): {e}")
        
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"CrossEncoder 로드 실패: {e}")
    
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
        
        # 검색 결과가 있는 경우에만 처리
        if len(indices) == 0 or len(indices[0]) == 0:
            return []

        return [(int(idx), 1.0 - float(dist)) for idx, dist in zip(indices[0], distances[0])]
    
    def sparse_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.es_client is None:
            raise RuntimeError("Elasticsearch 클라이언트가 초기화되지 않았습니다.")
        
        fields_to_search = ["content.nori", "content.ngram", "content.compact"]

        es_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields_to_search
                }
            },
            "_source": ["original_index"], # [중요] RRF/MMR을 위해 원본 인덱스 번호를 가져옵니다.
            "size": top_k
        }

        response = self.es_client.search(index=self.es_index_name, body=es_query)

        results = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            if '_score' not in hit or 'original_index' not in hit['_source']:
                print(f"경고 : ES 결과에 ORIGINAL_INDEX 필드가 없습니다. (ID: {hit['_id']})")
                continue
            original_index = hit['_source']['original_index']
            results.append((original_index, score))
        return results

    def hybrid_search_rrf(self, query: str, top_k: int = 10, k_val: int = 60, threshold: float = 0.0) -> List[Tuple[int, float]]:
        """
        Dense + Sparse 검색 결과를 Reciprocal Rank Fusion (RRF)로 결합합니다.
        'threshold' (임계치)를 설정하여 일정 점수 이하의 결과는 필터링합니다. 
        
        Args:
            k_val (int): RRF 하이퍼파라미터 (기본값 60)
            threshold (float): RRF 점수 임계값
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
        
        sorted_rrf = sorted(rrf_scores.items(), key=lambda item:item[1], reverse=True)
        filtered_results = [(doc_id, score) for doc_id, score in sorted_rrf if score >= threshold]
        return filtered_results[:top_k]
    
    def retrieve_mmr(self, query: str, top_k: int = 10, lambda_mult: float = 0.5, relevance_threshold: float=0.0) -> List[int]:
        """
        Maximal Marginal Relevance (MMR)를 사용하여 결과를 재정렬합니다.
        'relevance_threshold' (임계치)로 초기 후보군을 필터링합니다. 
        
        Args:
            lambda_mult (float): 다양성/관련성 조절 (0.0 ~ 1.0). 기본값 0.5.
            relevance_threshold (float): 초기 Dense 검색 결과의 최소 점수 임계값.
            results_filtered 에서 걸려지고 candidate 들로 embedding 
        """
        
        k_retrieve = max(top_k * 5, 20)
        initial_results = self.dense_search(query, top_k=k_retrieve)

        initial_results_filtered = [
            (idx, score) for idx, score in initial_results if score >= relevance_threshold
        ]

        if not initial_results_filtered:
            return []
        
        candidate_ids = [doc_id for doc_id, _ in initial_results_filtered]
        relevance_scores = {doc_id : score for doc_id, score in initial_results_filtered}

        candidate_embeds = self.all_doc_embeddings[candidate_ids]
        
        selected_ids: List[int] = []
        remaining_ids = list(candidate_ids)
        
        if not remaining_ids:
            return []

        # 첫 번째 문서는 무조건 관련성이 가장 높은 문서를 선택
        best_id = max(remaining_ids, key=lambda id: relevance_scores.get(id, 0))
        selected_ids.append(best_id)
        remaining_ids.remove(best_id)

        while len(selected_ids) < top_k and remaining_ids:
            selected_embeds = self.all_doc_embeddings[selected_ids]
            mmr_scores = {}
            
            for cand_id in remaining_ids:
                relevance = relevance_scores[cand_id]
                cand_embed = self.all_doc_embeddings[cand_id].reshape(1, -1)
                
                # 이미 선택된 문서들과의 유사도 계산
                similarity_to_selected = cosine_similarity(cand_embed, selected_embeds)[0]
                max_similarity = np.max(similarity_to_selected)
                
                # MMR 점수 계산
                mmr_score = (lambda_mult * relevance) - ((1 - lambda_mult) * max_similarity)
                mmr_scores[cand_id] = mmr_score
            
            # MMR 점수가 가장 높은 문서를 선택
            best_id = max(mmr_scores, key=mmr_scores.get)
            
            selected_ids.append(best_id)
            remaining_ids.remove(best_id)
            
        return selected_ids
    
    def rerank(self, query: str, candidate_ids: list[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Cross-Encoder를 사용하여 후보 문서들을 재정렬합니다.
        """
        if self.reranker is None:
            print("Reranker가 초기화되지 않았습니다. Reranking을 건너뜁니다.")
            return [(idx, 0.0) for idx in candidate_ids[:top_k]] 
            
        if not candidate_ids:
            return []

        candidate_docs = [self.documents[i] for i in candidate_ids]
        pairs = [(query, doc) for doc in candidate_docs]

        scores = self.reranker.predict(pairs)

        reranked_results = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        candidate_k: int = 20,
        rrf_k_val: int = 60,
        rrf_threshold: float = 0.0,
        mmr_lambda: float = 0.5,
        mmr_threshold: float = 0.0
    ) -> list[tuple[int, float]]:
        """
        [업데이트] 고정된 파이프라인으로 검색을 수행합니다:
        1. RRF로 후보군(candidate_k) 확보 (관련성 중심)
        2. MMR로 후보군(candidate_k) 확보 (다양성 중심)
        3. 두 후보군을 합쳐(중복 제거) Reranker로 재정렬 -> 최종 top_k 반환
        """
        
        # 1. RRF로 후보군 확보 
        rrf_candidates = self.hybrid_search_rrf(
            query, top_k=candidate_k, k_val=rrf_k_val, threshold=rrf_threshold
        )
        # [오류 수정] hybrid_search_rrf가 (인덱스, 점수) 튜플을 반환하므로 정상 동작
        rrf_ids = {idx for idx, _ in rrf_candidates} 
        
        # 2. MMR로 후보군 확보 
        mmr_candidate_ids = self.retrieve_mmr(
            query, top_k=candidate_k, lambda_mult=mmr_lambda, relevance_threshold=mmr_threshold
        )
        mmr_ids = set(mmr_candidate_ids) # set
        
        combined_candidate_ids = list(rrf_ids | mmr_ids)
        
        if not combined_candidate_ids:
            print("RRF와 MMR에서 후보군을 찾지 못했습니다.")
            return []

        print(f"RRF({len(rrf_ids)}) + MMR({len(mmr_ids)}) = {len(combined_candidate_ids)}개의 고유 후보를 재정렬합니다.")
        return self.rerank(query, combined_candidate_ids, top_k=top_k)



