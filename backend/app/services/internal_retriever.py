"""
Hybrid Retrieval (Dense + Sparse) with RRF + MMR
- vdb_manual.py가 저장한 chunks.jsonl을 로드
- FAISS 인덱스와 문서 개수 정합성 체크/보정
- Cross-encoder rerank 제거
"""

import os
import json
from typing import List, Tuple, Dict, Any

import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sklearn.metrics.pairwise import cosine_similarity

from app.services.faiss_store import FaissVectorStore

load_dotenv()

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200").strip()
ES_INDEX_NAME = "documents"

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(os.path.dirname(__file__), "faiss.index"))
VDB_DOC_PATH = os.getenv("VDB_DOC_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/chunks.jsonl"))


# ---------------------------------
# 문서 로더
# ---------------------------------
def load_documents_from_vdb() -> List[str]:
    """
    vdb_manual.py가 저장한 jsonl(chunks.jsonl)에서 문서를 로드한다.
    리스트 인덱스 == FAISS 벡터 id가 되도록 정렬하여 반환한다.
    """
    docs_by_id: Dict[int, str] = {}
    path = VDB_DOC_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                did = obj.get("id")
                txt = obj.get("content", "")
                if isinstance(did, int):
                    docs_by_id[did] = txt

        if not docs_by_id:
            print(f"[VDB] '{path}'에서 로드된 문서가 없습니다.")
            return []

        max_id = max(docs_by_id.keys())
        docs = [docs_by_id.get(i, "") for i in range(max_id + 1)]
        print(f"[VDB] 문서 {len(docs)}개 로드 완료 (id 0..{max_id})")
        return docs

    except FileNotFoundError:
        print(f"[VDB] 파일을 찾을 수 없습니다: {path}")
    except Exception as e:
        print(f"[VDB] 로드 중 예외: {e}")

    return []


# ---------------------------------
# Retriever
# ---------------------------------
class Retriever:
    def __init__(self, vector_store: FaissVectorStore, documents: List[str] = None):
        self.documents = documents or []
        self.vector_store = vector_store
        self._all_doc_embeddings = None
        self.es_client = None
        self.es_index_name = ES_INDEX_NAME

        try:
            self.es_client = Elasticsearch(ELASTICSEARCH_HOST)
            if self.es_client.info():
                print(f"Elasticsearch 연결 성공 ({ELASTICSEARCH_HOST})")
            else:
                print(f"Elasticsearch 연결 실패 ({ELASTICSEARCH_HOST})")
                self.es_client = None
        except Exception as e:
            print(f"Elasticsearch 연결 실패 ({ELASTICSEARCH_HOST}): {e}")
            self.es_client = None

    @property
    def all_doc_embeddings(self) -> np.ndarray:
        """
        문서 임베딩(=FAISS 인덱스에 저장된 벡터)을 얻어 캐시한다.
        주의: 여기서는 VectorStore.encode(docs)가 아닌, 이미 저장된 인덱스를 사용한다.
        """
        if self._all_doc_embeddings is None:
            # VectorStore가 제공하는 방식에 맞게 로드
            # 여기서는 encode(docs)를 사용하지 않고, 내부 인덱스 벡터를 바로 가져온다고 가정
            # 만약 FaissVectorStore가 벡터를 직접 꺼내는 메소드가 없다면 encode(self.documents)로 대체
            try:
                self._all_doc_embeddings = self.vector_store.get_all_embeddings()
            except AttributeError:
                # get_all_embeddings()가 없다면 encode(docs)로 재계산
                print("[경고] get_all_embeddings() 미구현 -> documents로부터 임베딩 재계산")
                self._all_doc_embeddings = self.vector_store.encode(self.documents)
        return self._all_doc_embeddings

    # ---------------- Dense ----------------
    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_vector = self.vector_store.encode([query])
        distances, indices = self.vector_store.search(query_vector, top_k=top_k)

        if len(indices) == 0 or len(indices[0]) == 0:
            return []
        return [(int(idx), 1.0 - float(dist)) for idx, dist in zip(indices[0], distances[0])]

    # ---------------- Sparse (ES) ----------------
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

        results: List[Tuple[int, float]] = []
        for hit in response.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            if "_score" not in hit or "original_index" not in src:
                continue
            results.append((int(src["original_index"]), float(hit["_score"])))
        return results

    # ---------------- RRF ----------------
    def hybrid_search_rrf(
        self,
        query: str,
        top_k: int = 10,
        k_val: int = 60,
        threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Dense + Sparse 결과를 Reciprocal Rank Fusion(RRF)으로 결합
        """
        k_retrieve = max(top_k * 5, 20)
        dense_results = self.dense_search(query, top_k=k_retrieve)
        sparse_results = self.sparse_search(query, top_k=k_retrieve)

        dense_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        sparse_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_results)}

        rrf_scores: Dict[int, float] = {}
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        for doc_id in all_doc_ids:
            score = 0.0
            d_rank = dense_ranks.get(doc_id, 0)
            s_rank = sparse_ranks.get(doc_id, 0)
            if d_rank > 0:
                score += 1.0 / (k_val + d_rank)
            if s_rank > 0:
                score += 1.0 / (k_val + s_rank)
            if score >= threshold:
                rrf_scores[doc_id] = score

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_rrf[:top_k]

    # ---------------- MMR ----------------
    def retrieve_mmr(
        self,
        query: str,
        top_k: int = 10,
        lambda_mult: float = 0.5,
        relevance_threshold: float = 0.0,
    ) -> List[int]:
        """
        Maximal Marginal Relevance로 다양성 반영
        """
        k_retrieve = max(top_k * 5, 20)
        initial_results = self.dense_search(query, top_k=k_retrieve)
        initial_results = [(i, s) for i, s in initial_results if s >= relevance_threshold]
        if not initial_results:
            return []

        num_docs = len(self.documents)
        if num_docs == 0:
            print("[MMR] documents 비어있음 -> 건너뜀")
            return []

        # 범위 내 id만 유지
        initial_results = [(i, s) for i, s in initial_results if 0 <= i < num_docs]
        if not initial_results:
            return []

        # 임베딩 캐시 및 정합성 확인
        _ = self.all_doc_embeddings
        if self._all_doc_embeddings.shape[0] != num_docs:
            print(f"[MMR] 임베딩({self._all_doc_embeddings.shape[0]}) != 문서수({num_docs}) -> MMR 건너뜀")
            # relevance 상위 1개라도 반환해 폴백
            return [max(initial_results, key=lambda x: x[1])[0]]

        candidate_ids = [i for i, _ in initial_results]
        relevance_scores = {i: s for i, s in initial_results}

        selected: List[int] = []
        remaining = list(candidate_ids)

        # 1개 seed
        best = max(remaining, key=lambda i: relevance_scores[i])
        selected.append(best)
        remaining.remove(best)

        while len(selected) < top_k and remaining:
            valid_selected = [i for i in selected if 0 <= i < num_docs]
            if not valid_selected:
                break
            selected_embeds = self._all_doc_embeddings[valid_selected]

            mmr_scores: Dict[int, float] = {}
            for cand in list(remaining):
                if not (0 <= cand < num_docs):
                    remaining.remove(cand)
                    continue
                relevance = relevance_scores[cand]
                cand_embed = self._all_doc_embeddings[cand].reshape(1, -1)
                sim = cosine_similarity(cand_embed, selected_embeds)[0]
                diversity = float(np.max(sim))
                mmr_scores[cand] = (lambda_mult * relevance) - ((1 - lambda_mult) * diversity)

            if not mmr_scores:
                break

            best = max(mmr_scores, key=mmr_scores.get)
            selected.append(best)
            remaining.remove(best)

        return selected

    # ---------------- Orchestrate ----------------
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
        rerank 제거 버전 파이프라인
        1) RRF로 후보/점수 확보
        2) MMR로 다양성 반영 후보 확보
        3) 합집합을 RRF 점수 기준 정렬
        """
        rrf_candidates = self.hybrid_search_rrf(
            query, top_k=candidate_k, k_val=rrf_k_val, threshold=rrf_threshold
        )  # List[(id, score)]
        rrf_ids = {i for i, _ in rrf_candidates}
        rrf_score = {i: s for i, s in rrf_candidates}

        mmr_ids = set(self.retrieve_mmr(
            query, top_k=candidate_k, lambda_mult=mmr_lambda, relevance_threshold=mmr_threshold
        ))

        combined_ids = list(rrf_ids | mmr_ids)

        # 범위 필터
        num_docs = len(self.documents)
        combined_ids = [i for i in combined_ids if 0 <= i < num_docs]
        if not combined_ids:
            print("RRF와 MMR 후 유효 후보가 없습니다. (문서/인덱스 불일치 가능)")
            return []

        print(f"RRF({len(rrf_ids)}) + MMR({len(mmr_ids)}) = {len(combined_ids)}개의 고유 후보를 RRF 점수 기준으로 정렬합니다.")
        scored = [(i, rrf_score.get(i, 0.0)) for i in combined_ids]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        return scored_sorted[:top_k]

    # ---------------- Utils ----------------
    def get_documents_by_ids(self, doc_ids: List[int]) -> List[str]:
        docs: List[str] = []
        total = len(self.documents)
        for i in doc_ids:
            if 0 <= i < total:
                docs.append(self.documents[i])
            else:
                docs.append(f"[문서없음: {i}]")
        return docs


# ---------------------------------
# 인스턴스 생성
# ---------------------------------
def create_retriever_instance():
    try:
        vs = FaissVectorStore()
        if os.path.exists(FAISS_INDEX_PATH):
            vs.load(FAISS_INDEX_PATH)
            print(f"FAISS 인덱스 로드 완료: {vs.index.ntotal}개 벡터 (path={FAISS_INDEX_PATH})")
        else:
            print(f"경고: FAISS 인덱스 파일을 찾을 수 없습니다: {FAISS_INDEX_PATH}")
            print("vdb_manual.py를 먼저 실행해 인덱스를 생성하세요.")
            return None

        documents = load_documents_from_vdb()
        ntotal = vs.index.ntotal
        ndocs = len(documents)
        print(f"[Diag] FAISS ntotal={ntotal}, docs={ndocs}")

        # 임시 보정(급한 불 끄기): 불일치 시 최소 길이로 자르기
        if ndocs == 0:
            print("[경고] 문서가 비어 있습니다. VDB를 다시 구축하세요.")
            return None
        if ntotal != ndocs:
            min_len = min(ntotal, ndocs)
            documents = documents[:min_len]
            print(f"[조정] 문서 수를 {min_len}로 절단하여 정합성 맞춤 (임시 조치)")

        return Retriever(vector_store=vs, documents=documents)
    except Exception as e:
        print(f"Retriever 인스턴스 생성 실패: {e}")
        return None


# 전역 인스턴스
retriever_instance = create_retriever_instance()


def get_documents_by_ids(doc_ids: List[int]) -> List[str]:
    if retriever_instance is None:
        return ["Retriever 인스턴스가 초기화되지 않았습니다."]
    return retriever_instance.get_documents_by_ids(doc_ids)
