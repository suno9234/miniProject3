FROM python:3.11-slim

WORKDIR /backend/app

COPY requirements.txt /backend/requirements.txt
RUN pip install --no-cache-dir -r /backend/requirements.txt

COPY . /backend/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]