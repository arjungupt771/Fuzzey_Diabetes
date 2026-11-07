FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "diabetes_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
