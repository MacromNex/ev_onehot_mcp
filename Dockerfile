FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    git gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install --ignore-installed fastmcp

FROM python:3.10-slim AS runtime

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY repo/ ./repo/
RUN mkdir -p tmp/inputs tmp/outputs

ENV PYTHONPATH=/app

CMD ["python", "src/server.py"]
