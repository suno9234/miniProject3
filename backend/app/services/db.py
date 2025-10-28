import aiomysql
import os
from dotenv import load_dotenv
import asyncio
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

            # user 테이블
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS user (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(50) NOT NULL COMMENT '사용자 이름',
                employee_id VARCHAR(20) NOT NULL UNIQUE COMMENT '사번',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # item 테이블
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS item (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL COMMENT '품명',
                quantity INT NOT NULL DEFAULT 0 COMMENT '수량',
                location VARCHAR(100) COMMENT '장소',
                sku VARCHAR(50) UNIQUE COMMENT 'SKU 코드',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
        await conn.commit()
        
        # ✅ 더미데이터 삽입 (조건: 유저 2명 미만 / 아이템 5개 미만)
        async with conn.cursor() as cursor:
            # 현재 유저/아이템 수 조회
            await cursor.execute("SELECT COUNT(*) FROM user;")
            user_count = (await cursor.fetchone())[0]

            await cursor.execute("SELECT COUNT(*) FROM item;")
            item_count = (await cursor.fetchone())[0]

            # 더미 유저
            users_seed = [
                ("홍길동", "E1001"),
                ("김영희", "E1002"),
            ]

            # 더미 아이템
            items_seed = [
                ("맥북 프로 14", 3, "용인1센터", "MBP14-2023-BASE"),
                ("맥북 에어 13", 1, "분당센터", "MBA13-2022-M2"),
                ("아이패드 에어", 0, "서울역점", "IPAIR-64-GRY"),
                ("갤럭시 탭 S9", 7, "강남물류", "GTABS9-128"),
                ("AA 건전지", 52, "서초물류", "AA-ENEL-4P"),
            ]

            # 유저가 2명 미만이면 삽입
            if user_count < 2:
                await cursor.executemany(
                    "INSERT IGNORE INTO user (name, employee_id) VALUES (%s, %s);",
                    users_seed,
                )
                print(f"[SEED] user inserted ({2 - user_count} missing rows filled)")

            # 아이템이 5개 미만이면 삽입
            if item_count < 5:
                await cursor.executemany(
                    "INSERT IGNORE INTO item (name, quantity, location, sku) VALUES (%s, %s, %s, %s);",
                    items_seed,
                )
                print(f"[SEED] item inserted ({5 - item_count} missing rows filled)")

        await conn.commit()
        print("[DB INIT] 기본 데이터 삽입 완료 ✅")
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


async def _print_snapshot(conn):
    """현재 user / item 테이블 스냅샷을 출력"""
    async with conn.cursor() as cur:
        await cur.execute("SELECT COUNT(*) FROM user;")
        user_cnt = (await cur.fetchone())[0]
        await cur.execute("SELECT COUNT(*) FROM item;")
        item_cnt = (await cur.fetchone())[0]
        print(f"[SNAPSHOT] users={user_cnt}, items={item_cnt}")

        # 간단히 상위 몇 개만 확인
        if user_cnt:
            await cur.execute("SELECT id, name, employee_id, created_at FROM user ORDER BY id ASC LIMIT 5;")
            users = await cur.fetchall()
            print("[USERS TOP 5]")
            for row in users:
                print("  ", row)

        if item_cnt:
            await cur.execute("SELECT id, name, quantity, location, sku FROM item ORDER BY id ASC LIMIT 5;")
            items = await cur.fetchall()
            print("[ITEMS TOP 5]")
            for row in items:
                print("  ", row)

async def _main():
    # .env 로드 (환경변수 사용 시)
    load_dotenv()

    # 풀 생성 + 테이블 보장 + 시드(조건부)
    await db_service.connect()

    # 연결 하나 빌려서 스냅샷 출력
    async with db_service.get_connection() as conn:
        await _print_snapshot(conn)

    # 풀 정리
    await db_service.close()

if __name__ == "__main__":
    # Windows/Unix 공통: asyncio.run으로 실행
    asyncio.run(_main())