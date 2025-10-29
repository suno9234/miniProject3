# main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.agent_router import router as agent_router

# ⬇️ build_all_vdbs 함수 직접 import
from app.services.vdb_manual import build_all_vdbs

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

# 서버 시작 시 직접 함수 실행
@app.on_event("startup")
async def startup_event():
    logging.info("=== 서버 시작: VDB 빌드 시작 ===")
    build_all_vdbs()
    logging.info("=== VDB 빌드 완료 ===")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)