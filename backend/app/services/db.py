import aiomysql
import os
from contextlib import asynccontextmanager
from typing import Optional

# MariaDB 연결 설정 및 관리

async def ensure_tables():
    """데이터베이스 테이블 생성"""
    conn = await aiomysql.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT')),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        db=os.getenv('DB_NAME'),
        charset='utf8mb4'
    )
    
    try:
        async with conn.cursor() as cursor:
            # TODO: 필요한 테이블들 CREATE TABLE IF NOT EXISTS 구문 추가
            # 예시: inventory, reservations, search_logs 등
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_table(
                id INT AUTO_INCREMENT PRIMARY KEY
            )
            """)
        await conn.commit()
    finally:
        await conn.ensure_closed()

class DatabaseService:
    def __init__(self):
        self.pool: Optional[aiomysql.Pool] = None
    
    async def connect(self):
        self.pool = await aiomysql.create_pool(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            db=os.getenv('DB_NAME', 'agent_db'),
            charset='utf8mb4'
        )
        await ensure_tables()
    
    @asynccontextmanager
    async def get_connection(self):
        async with self.pool.acquire() as conn:
            yield conn
    
    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

# 전역 인스턴스
db_service = DatabaseService()