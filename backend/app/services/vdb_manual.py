"""
매뉴얼 vdb 저장 (Faiss + Elasticsearch 동시 구축)
===========
1. PDF 로드 및 분할
2. Faiss 인덱스 빌드 및 저장
3. Elasticsearch 인덱스 빌드 및 업로드
4. (추가) 분할 청크 jsonl 저장 -> internal_retriever가 그대로 로드
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 'backend' 루트 경로 추가 ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../backend/app/services
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)               # .../backend/app
sys.path.append(PROJECT_ROOT)

# --- FaissVectorStore 임포트 ---
try:
    from app.services.faiss_store import FaissVectorStore
except ImportError:
    print("Error: 'app.services.faiss_store' 임포트 실패")
    FaissVectorStore = None

# --- .env ---
load_dotenv()

# --- 경로/설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # .../backend/app/services
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "../docs")

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200").strip()
ES_INDEX_NAME = "documents"

# 저장 경로: 환경변수 없으면 기본값 사용
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(BASE_DIR, "faiss.index"))
VDB_DOC_PATH = os.getenv("VDB_DOC_PATH", os.path.join(PROJECT_ROOT, "data/chunks.jsonl"))


# --------------------------
# 1. PDF 로드 및 분할
# --------------------------
def load_and_split_pdfs(directory_path: str) -> List[str]:
    """
    pypdf로 PDF 텍스트 추출 후 청크 분할
    """
    all_chunks: List[str] = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    print(f"'{directory_path}' 폴더에서 PDF 파일을 스캔합니다...")

    if not os.path.exists(directory_path):
        print(f"경고: '{directory_path}' 폴더를 찾을 수 없습니다.")
        return []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"  - 처리 중: {filename}")

            try:
                reader = PdfReader(file_path)
                full_text = ""
                for page in reader.pages:
                    full_text += (page.extract_text() or "")

                chunks = text_splitter.split_text(full_text)
                all_chunks.extend(chunks)
                print(f"    -> {filename}에서 {len(chunks)}개 청크 추출")
            except Exception as e:
                print(f"    -> Error: {filename} 처리 중 오류: {e}")

    return all_chunks


# --------------------------
# 1.5 청크 저장 (추가)
# --------------------------
def save_chunks_jsonl(documents: List[str], out_path: str):
    """
    문서 청크를 jsonl로 저장 (id == enumerate index)
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for i, text in enumerate(documents):
            f.write(json.dumps({"id": i, "content": text}, ensure_ascii=False) + "\n")
    print(f"[VDB] 청크 {len(documents)}개를 '{out_path}'에 저장했습니다.")


# --------------------------
# 2. Elasticsearch 인덱스
# --------------------------
def create_es_index(es_client: Elasticsearch, index_name: str):
    """
    Nori/Ngram/Compact 필드가 있는 매핑 생성
    """
    if es_client.indices.exists(index=index_name):
        print(f"ES 인덱스 '{index_name}' 존재 -> 삭제 후 재생성")
        es_client.indices.delete(index=index_name)

    index_mapping: Dict[str, Any] = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "nori_analyzer": {"type": "custom", "tokenizer": "nori_tokenizer"},
                    "ngram_analyzer": {"type": "custom", "tokenizer": "ngram_tokenizer"},
                    "compact_analyzer": {"type": "custom", "tokenizer": "whitespace", "filter": ["compact_filter"]},
                },
                "tokenizer": {
                    "ngram_tokenizer": {"type": "ngram", "min_gram": 2, "max_gram": 3, "token_chars": ["letter", "digit"]}
                },
                "filter": {
                    "compact_filter": {"type": "pattern_replace", "pattern": "\\s+", "replacement": ""}
                },
            }
        },
        "mappings": {
            "properties": {
                "original_index": {"type": "integer"},
                "content": {
                    "type": "text",
                    "fields": {
                        "nori": {"type": "text", "analyzer": "nori_analyzer"},
                        "ngram": {"type": "text", "analyzer": "ngram_analyzer"},
                        "compact": {"type": "text", "analyzer": "compact_analyzer"},
                    },
                },
            }
        },
    }

    try:
        es_client.indices.create(index=index_name, body=index_mapping)
        print(f"ES 인덱스 '{index_name}' 생성 완료")
    except Exception as e:
        print(f"ES 인덱스 생성 실패: {e}")
        print("Elasticsearch 'analysis-nori' 플러그인 설치 여부를 확인하세요.")
        raise e


def generate_es_actions(documents: List[str], index_name: str):
    for i, doc_text in enumerate(documents):
        yield {
            "_index": index_name,
            "_source": {
                "original_index": i,
                "content": doc_text,
            },
        }


# --------------------------
# 3. 메인 빌드
# --------------------------
def build_all_vdbs():
    # 1) PDF -> 청크
    documents = load_and_split_pdfs(PDF_SOURCE_DIR)
    if not documents:
        print("VDB를 빌드할 문서가 없습니다. PDF_SOURCE_DIR을 확인하세요.")
        return

    # 1.5) 청크 저장 (internal_retriever가 사용)
    save_chunks_jsonl(documents, VDB_DOC_PATH)

    print(f"\n총 {len(documents)}개 청크로 VDB 빌드를 시작합니다...")

    # 2) FAISS
    print("\n--- [Part 1] Faiss VDB 빌드 ---")
    if FaissVectorStore is None:
        print("FaissVectorStore 임포트 실패 -> Faiss 빌드 건너뜀")
    else:
        try:
            vs = FaissVectorStore()
            vs.build(documents)
            # 경로를 명시적으로 설정하고 저장
            vs.save(FAISS_INDEX_PATH)
            print(f"[성공] Faiss VDB 저장 완료 -> {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"[오류] Faiss VDB 빌드 중 오류: {e}")

    # 3) Elasticsearch
    print("\n--- [Part 2] Elasticsearch VDB 빌드 ---")
    try:
        es_client = Elasticsearch(ELASTICSEARCH_HOST)
        if not es_client.info():
            raise RuntimeError("Elasticsearch 서버 정보를 가져올 수 없습니다.")
        print(f"Elasticsearch 연결 성공 ({ELASTICSEARCH_HOST})")

        create_es_index(es_client, ES_INDEX_NAME)

        print(f"{len(documents)}개 문서를 Elasticsearch에 업로드합니다...")
        success, failed = bulk(
            es_client,
            generate_es_actions(documents, ES_INDEX_NAME),
            raise_on_error=False,
        )
        print(f"업로드 성공: {success}, 실패: {failed}")
        if failed:
            print("[오류] 일부 문서 업로드 실패. ES 로그 확인 요망.")
        else:
            print("[성공] Elasticsearch VDB 빌드 완료")
    except Exception as e:
        print(f"[오류] Elasticsearch VDB 빌드 중 오류: {e}")
        print("Elasticsearch 실행 여부 및 nori 플러그인 설치를 확인하세요.")


if __name__ == "__main__":
    build_all_vdbs()
