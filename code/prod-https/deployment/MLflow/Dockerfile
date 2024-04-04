FROM python:3.8.3

ARG MLFLOW_VERSION=2.11.1
ARG PSYCOPG2_VERSION=2.9.2
ARG PYMYSQL_VERSION=1.0.2
ARG MYSQLCLIENT_VERSION=2.1.0

RUN apt-get update
RUN apt-get -y install default-libmysqlclient-dev

RUN pip install --upgrade pip setuptools wheel && pip install mlflow==${MLFLOW_VERSION} psycopg2==${PSYCOPG2_VERSION} protobuf==3.20.0 boto3==1.15.16

RUN pip install mysqlclient==${MYSQLCLIENT_VERSION}

RUN pip install google-cloud-storage

WORKDIR /mlflow

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB=mlflow_db

ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123
ENV MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
ENV MLFLOW_S3_IGNORE_TLS=true
ENV MLFLOW_SQL_ALCHEMY_CONN=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT}/${PG_DATABASE}

EXPOSE 5000

CMD mlflow server --backend-store-uri=${MLFLOW_SQL_ALCHEMY_CONN} \
                  --default-artifact-root=/mlflow/mlruns/ \
                  --artifacts-destination=${MLFLOW_S3_ENDPOINT_URL} \
                  --no-serve-artifacts --host=0.0.0.0 --port=5000
