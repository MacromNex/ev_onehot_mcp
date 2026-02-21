FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install --ignore-installed fastmcp

FROM python:3.10-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
RUN chmod -R a+r /app/src/
COPY repo/ ./repo/
RUN mkdir -p tmp/inputs tmp/outputs /tmp/.cache && \
    chmod -R 1777 /app/tmp /tmp/.cache

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache

CMD ["python", "src/server.py"]
