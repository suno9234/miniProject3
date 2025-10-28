"""
매뉴얼 vdb 저장 (Faiss + Elasticsearch 동시 구축)
===========
1. PDF 로드 및 분할
2. Faiss 인덱스 빌드 및 저장
3. Elasticsearch 인덱스 빌드 및 업로드
"""

# 라이브러리
import os
import sys
# import fitz  # PyMuPDF  <- [수정] fitz 라이브러리 제거
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- [신규] Elasticsearch 관련 라이브러리 추가 ---
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# --- [기존] Langchain/Faiss 관련 라이브러리 ---
from pypdf import PdfReader # <- [수정] 이 라이브러리를 사용합니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- [기존] 'backend' 폴더를 파이썬 경로에 강제로 추가 ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR) # 'backend'
sys.path.append(PROJECT_ROOT)
# ---------------------------------------------------

# --- [기존] FaissVectorStore 클래스 임포트 ---
try:
    from app.services.faiss_store import FaissVectorStore
except ImportError:
    print(f"Error: 'app.services.faiss_store.py'를 임포트할 수 없습니다.")
    FaissVectorStore = None # 임시 정의

# --- .env 로드 ---
load_dotenv()

# --- [기존] Faiss 및 PDF 경로 설정 ---
# BASE_DIR => D:\...\backend\app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PDF_SOURCE_DIR => D:\...\backend\app\docs
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "../docs")

# --- [신규] Elasticsearch 설정 (build_es_vdb.py에서 가져옴) ---
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ES_INDEX_NAME = "documents" # internal_retriever.py와 동일한 이름


# --- 1. PDF 로드 및 분할 함수 (공통 사용) ---
def load_and_split_pdfs(directory_path: str) -> List[str]:
    """
    [수정] pypdf (PdfReader)를 사용하여 PDF 텍스트를 추출하고 분할합니다.
    """
    all_chunks = []
    
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
            
            # --- [수정] fitz 대신 PdfReader 사용 ---
            try:
                # 1. PdfReader로 PDF 열기
                reader = PdfReader(file_path)
                full_text = ""
                
                # 2. 모든 페이지의 텍스트 추출 및 결합
                for page in reader.pages:
                    # extract_text()가 None을 반환할 경우를 대비해 'or ""' 추가
                    full_text += (page.extract_text() or "")
                
                # 3. 텍스트 스플리터로 분할
                chunks = text_splitter.split_text(full_text)
                all_chunks.extend(chunks)
                print(f"    -> {filename}에서 {len(chunks)}개의 청크를 추출했습니다.")

            except Exception as e:
                print(f"    -> Error: {filename} 처리 중 오류 발생: {e}")
            # --- [수정] 완료 ---

    return all_chunks

# --- 2. Elasticsearch 헬퍼 함수 (build_es_vdb.py에서 가져옴) ---

def create_es_index(es_client: Elasticsearch, index_name: str):
    """
    Nori, Ngram, Compact 필드를 가진 Elasticsearch 인덱스 매핑을 생성합니다.
    """
    if es_client.indices.exists(index=index_name):
        print(f"ES 인덱스 '{index_name}'가 이미 존재합니다. 삭제 후 재생성합니다.")
        es_client.indices.delete(index=index_name)

    index_mapping = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "nori_analyzer": {"type": "custom", "tokenizer": "nori_tokenizer"},
                    "ngram_analyzer": {"type": "custom", "tokenizer": "ngram_tokenizer"},
                    "compact_analyzer": {"type": "custom", "tokenizer": "whitespace", "filter": ["compact_filter"]}
                },
                "tokenizer": {
                    "ngram_tokenizer": {"type": "ngram", "min_gram": 2, "max_gram": 3, "token_chars": ["letter", "digit"]}
                },
                "filter": {
                    "compact_filter": {"type": "pattern_replace", "pattern": "\\s+", "replacement": ""}
                }
            }
        },
        "mappings": {
            "properties": {
                "original_index": { "type": "integer" }, # Faiss 인덱스와 매칭
                "content": {
                    "type": "text",
                    "fields": {
                        "nori": {"type": "text", "analyzer": "nori_analyzer"},
                        "ngram": {"type": "text", "analyzer": "ngram_analyzer"},
                        "compact": {"type": "text", "analyzer": "compact_analyzer"}
                    }
                }
            }
        }
    }
    
    try:
        es_client.indices.create(index=index_name, body=index_mapping)
        print(f"ES 인덱스 '{index_name}'를 성공적으로 생성했습니다.")
    except Exception as e:
        print(f"ES 인덱스 생성 실패: {e}")
        print("Elasticsearch에 'analysis-nori' 플러그인이 설치되어 있는지 확인하세요.")
        raise e

def generate_es_actions(documents: List[str], index_name: str) -> List[Dict[str, Any]]:
    """
    Elasticsearch Bulk API에 맞게 문서를 포맷팅합니다.
    """
    for i, doc_text in enumerate(documents):
        yield {
            "_index": index_name,
            "_source": {
                "original_index": i, # Faiss의 인덱스 번호(i)와 동일하게 저장
                "content": doc_text
            }
        }

# --- 3. 메인 VDB 구축 함수 (Faiss + ES) ---

def build_all_vdbs():
    """
    하나의 문서 목록으로 Faiss와 Elasticsearch VDB를 모두 빌드합니다.
    """
    
    # --- 공통: 1. PDF 로드 및 분할 ---
    documents = load_and_split_pdfs(PDF_SOURCE_DIR)
    
    if not documents:
        print("VDB를 빌드할 문서가 없습니다. PDF_SOURCE_DIR 폴더를 확인하세요.")
        return
    
    print(f"\n총 {len(documents)}개의 문서 청크로 VDB 빌드를 시작합니다...")

    # --- Part 1: FAISS VDB 빌드 ---
    print("\n--- [Part 1] Faiss VDB 빌드 시작 ---")
    if FaissVectorStore is None:
        print("FaissVectorStore가 임포트되지 않아 Faiss 빌드를 건너뜁니다.")
    else:
        try:
            vector_store = FaissVectorStore()
            vector_store.build(documents)
            vector_store.save()
            print("[성공] Faiss VDB 빌드 및 저장이 완료되었습니다.")
        except Exception as e:
            print(f"[오류] Faiss VDB 빌드 중 오류: {e}")

    # --- Part 2: Elasticsearch VDB 빌드 ---
    print("\n--- [Part 2] Elasticsearch VDB 빌드 시작 ---")
    try:
        es_client = Elasticsearch(ELASTICSEARCH_HOST)
        if not es_client.info():
            raise RuntimeError("Elasticsearch 서버 정보를 가져올 수 없습니다.")
        print(f"Elasticsearch 연결 성공 ({ELASTICSEARCH_HOST})")
        
        # 2-1. ES 인덱스 생성
        create_es_index(es_client, ES_INDEX_NAME)
        
        # 2-2. ES에 Bulk 업로드
        print(f"{len(documents)}개의 문서를 Elasticsearch에 업로드합니다...")
        success, failed = bulk(
            es_client, 
            generate_es_actions(documents, ES_INDEX_NAME),
            raise_on_error=False
        )
        print(f"업로드 성공: {success}, 실패: {failed}")
        if failed:
            print("[오류] ES 업로드 중 일부 실패. ES 로그를 확인하세요.")
        else:
            print("[성공] Elasticsearch VDB 빌드가 완료되었습니다.")
            
    except Exception as e:
        print(f"[오류] Elasticsearch VDB 빌드 중 오류: {e}")
        print("Elasticsearch 서버가 실행 중인지, Nori 플러그인이 설치되었는지 확인하세요.")


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 이 스크립트(vdb_manual.py)를 실행하면 Faiss와 ES VDB가 모두 생성됩니다.
    
    # [사전 확인]
    # 1. Faiss: .env 파일에 FAISS_INDEX_PATH, EMBED_MODEL 설정
    # 2. ES: .env 파일에 ELASTICSEARCH_HOST 설정
    # 3. ES: Elasticsearch 서버 실행 및 'analysis-nori' 플러그인 설치
    
    build_all_vdbs()

