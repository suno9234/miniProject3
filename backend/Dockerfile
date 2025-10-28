FROM python:3.11-slim
WORKDIR /backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# backend 폴더 전체를 /backend 로 복사 (app/ 포함)
COPY backend .
# 선택: 안전하게 PYTHONPATH를 고정
ENV PYTHONPATH=/backend
EXPOSE 8000
# uvicorn을 python -m으로 실행 → 현재 WORKDIR가 sys.path에 확실히 포함
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]