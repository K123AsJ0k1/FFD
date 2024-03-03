# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

WORKDIR /worker

COPY config.py .

COPY run.py .

COPY app.py .

COPY templates /worker/templates

COPY data /worker/data

COPY logs /worker/logs

COPY models /worker/models

COPY tensors /worker/tensors

COPY routes /worker/routes

COPY functions /worker/functions

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 7500

CMD ["python", "run.py"]