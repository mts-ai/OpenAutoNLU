FROM python:3.12-slim

ARG EXTRA=cuda
ARG UV_VERSION=0.8.22

WORKDIR /app

RUN pip install --no-cache-dir uv==${UV_VERSION}

COPY . .

RUN uv sync --extra ${EXTRA} --no-cache

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

