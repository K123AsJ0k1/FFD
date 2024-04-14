# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ENV MINIO_ENDPOINT=127.0.0.1:9000
ENV AWS_ACCESS_KEY_ID=23034opsdjhksd
ENV AWS_SECRET_ACCESS_KEY=sdkl3slömdm

ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000

WORKDIR /central

COPY requirements.txt .

COPY run.py .

COPY app.py .

COPY routes /central/routes

COPY functions /central/functions

COPY templates /central/templates

RUN pip install -r requirements.txt 

EXPOSE 7500

CMD ["python", "run.py"]