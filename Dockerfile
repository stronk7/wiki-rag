FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./pyproject.toml /app/pyproject.toml
RUN --mount=source=.git,target=.git,type=bind \
    pip install --no-cache-dir -e .

FROM python:3.12-slim AS runner

WORKDIR /app/

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONPATH="/app"

COPY ./LICENSE /app/
COPY ./*.md /app/
COPY ./wiki_rag /app/wiki_rag

CMD ["wr-server"]
