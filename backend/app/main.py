import logging
import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.agent_router import router as agent_router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)


# ✅ 서버 시작 시 vdb_manual.py 실행
@app.on_event("startup")
async def startup_event():
    logging.info("=== 서버 시작: vdb_manual.py 실행 ===")
    try:
        subprocess.run(
            ["python", "app/services/vdb_manual.py"], check=True
        )
        logging.info("vdb_manual.py 실행 완료")
    except subprocess.CalledProcessError as e:
        logging.error(f"vdb_manual.py 실행 실패: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)