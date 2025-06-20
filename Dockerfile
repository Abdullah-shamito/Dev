FROM python:3.10-slim

WORKDIR /app

COPY . /app

# Add PyTorch CPU wheel index for compatibility
RUN pip install --no-cache-dir --upgrade pip \
    && pip config set global.extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
