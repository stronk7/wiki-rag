FROM python:3.12-slim

WORKDIR /usr/src/app

ARG VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WIKI_RAG=$VERSION

ENV PYTHONPATH="/usr/src/app"

COPY pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -e .

COPY ./wiki_rag ./wiki_rag

CMD ["wr-server"]
