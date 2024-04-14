# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ENV CENTRAL_ADDRESS=127.0.0.1
ENV CENTRAL_PORT=7500

ENV WORKER_PORT=7501
ENV WORKER_ADDRESS=127.0.0.1
ENV WORKER_SYSTEM_MONITOR=0

ENV MINIO_ENDPOINT=127.0.0.1:9000
ENV AWS_ACCESS_KEY_ID=23034opsdjhksd
ENV AWS_SECRET_ACCESS_KEY=sdkl3slömdm

ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000

WORKDIR /worker

COPY requirements.txt .

COPY run.py .

COPY app.py .

COPY routes /worker/routes

COPY functions /worker/functions

COPY templates /worker/templates

RUN pip3 install -r requirements.txt

EXPOSE 7501

CMD ["python", "run.py"]