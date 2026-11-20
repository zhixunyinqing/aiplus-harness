ARG BASE_PYTHON=python:3.12-slim
FROM ${BASE_PYTHON}

ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 BASELINE_CPU_THREADS=2

WORKDIR /app

COPY sdk /opt/contest-sdk
RUN pip install --no-cache-dir -i ${PIP_INDEX} numpy==1.26.4 /opt/contest-sdk && rm -rf /opt/contest-sdk

COPY app /app
RUN chmod -R a-w /app

CMD ["python3", "/app/harness.py", "describe"]
