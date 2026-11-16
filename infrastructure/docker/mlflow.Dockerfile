FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir \
      "mlflow==3.18.0" \
      "psycopg[binary]>=3.2,<4" \
      "boto3>=1.35,<2"

RUN groupadd --gid 10001 mlflow \
    && useradd \
      --uid 10001 \
      --gid 10001 \
      --create-home \
      --shell /usr/sbin/nologin \
      mlflow

USER mlflow
WORKDIR /home/mlflow

EXPOSE 5000

CMD ["mlflow", "server"]
