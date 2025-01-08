FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN python -m ensurepip --upgrade && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "app.py"]
