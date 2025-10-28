"""
매뉴얼 vdb 저장
===========
"""

# 라이브러리
import os
# [수정] pypdf의 PdfReader를 임포트
from pypdf import PdfReader
# [삭제] from dotenv import load_dotenv (하드코딩으로 변경)
# [수정] 임포트 경로 변경 (langchain.text_splitter -> langchain_text_splitters)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from faiss_store import FaissVectorStore


# [삭제] .env 파일 로드 (하드코딩으로 변경)
# load_dotenv()

# [수정] __file__ (현재 스크립트 파일)을 기준으로 절대 경로 생성
# __file__ => D:\...\backend\app\vdb_manual.py
# BASE_DIR => D:\...\backend\app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PDF_SOURCE_DIR => D:\...\backend\app\docs
PDF_SOURCE_DIR = os.path.join(BASE_DIR, "../docs")


# vdb 저장 
def load_and_split_pdfs(directory_path: str) -> list[str]:
    """
    지정된 디렉터리에서 모든 PDF 파일을 읽어 텍스트로 변환하고,
    langchain의 TextSplitter를 사용해 청크(Chunk) 리스트로 반환합니다.

    Args:
        directory_path (str): PDF 파일이 있는 폴더 경로

    Returns:
        List[str]: 텍스트 청크(문서 조각)의 리스트
    """
    all_chunks = []
    
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""] 
    )

    # [수정] 이제 directory_path는 절대 경로(D:\...\app\docs)입니다.
    print(f"'{directory_path}' 폴더에서 PDF 파일을 스캔합니다...")
    
    if not os.path.exists(directory_path):
        print(f"경고: '{directory_path}' 폴더를 찾을 수 없습니다.")
        return []

    for filename in os.listdir(directory_path):
        # [수정] "[manual] "로 시작하고 ".pdf"로 끝나는 파일만 처리
        if filename.lower().startswith("[manual]") and filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"  - 처리 중: {filename}")
            
            try:
                # 1. [수정] pypdf의 PdfReader로 PDF 열기
                reader = PdfReader(file_path)
                full_text = ""
                
                # 2. 모든 페이지의 텍스트 추출 및 결합
                for page in reader.pages:
                    full_text += page.extract_text() or "" # None 방지
                
                # [수정] pypdf는 close()가 필요 없음
                
                # 3. 텍스트 스플리터로 분할
                chunks = text_splitter.split_text(full_text)
                
                all_chunks.extend(chunks)
                
                print(f"    -> {filename}에서 {len(chunks)}개의 청크를 추출했습니다.")

            except Exception as e:
                print(f"    -> Error: {filename} 처리 중 오류 발생: {e}")
        else:
            print(f"  - 스킵: {filename} (이름 형식이 '[manual]*.pdf'가 아님)")

    return all_chunks

def build_vdb():
    """
    PDF에서 텍스트를 로드하고, FaissVectorStore를 사용해 VDB를 빌드 및 저장합니다.
    """
    # 1. PDF 로드 및 분할 (절대 경로인 PDF_SOURCE_DIR 사용)
    documents = load_and_split_pdfs(PDF_SOURCE_DIR)
    
    if not documents:
        print(f"VDB를 빌드할 문서가 없습니다. '{PDF_SOURCE_DIR}' 폴더를 확인하세요.")
        return
    
    if FaissVectorStore is None:
        print("FaissVectorStore가 임포트되지 않아 VDB 빌드를 중단합니다.")
        return

    print(f"\n총 {len(documents)}개의 문서 청크로 VDB 빌드를 시작합니다...")
    
    try:
        # 2. FaissVectorStore 객체 생성 (환경 변수는 자동으로 로드됨)
        vector_store = FaissVectorStore()
        
        # 3. 인덱스 빌드 (벡터로 변환 및 Faiss 인덱스에 추가)
        vector_store.build(documents)
        
        # 4. 인덱스 파일 저장 (FAISS_INDEX_PATH에 지정된 경로)
        #    저장 경로는 .env 또는 faiss_store.py의 기본값(./faiss.index)을 따릅니다.
        #    이 경로는 스크립트 실행 위치(backend) 기준이 되니 주의하세요.
        vector_store.save()
        
        print(f"\n[성공] VDB 인덱스 빌드 및 저장이 완료되었습니다. (저장 위치: {os.path.abspath(vector_store.index_path)})")
        
    except Exception as e:
        print(f"\n[오류] VDB 빌드 중 오류가 발생했습니다: {e}")


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # .env 파일에 다음을 설정하세요 (FaissVectorStore가 사용):
    # FAISS_INDEX_PATH=./my_faiss.index (실행 위치 기준, 예: backend/my_faiss.index)
    # EMBED_MODEL=intfloat/multilingual-e5-small
    # PDF_SOURCE_DIR은 이제 코드에 하드코딩되었습니다.
    
    build_vdb()

