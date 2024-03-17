FROM docker.io/minio/minio:RELEASE.2024-03-15T01-07-19Z

ENV MINIO_ROOT_USER=minio
ENV MINIO_ROOT_PASSWORD=minio123

EXPOSE 9000
EXPOSE 9001

RUN mkdir /data

RUN minio server /data & \
    server_pid=$!; \
    until mc alias set local http://localhost:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}; do \
        sleep 1; \
    done; \
    mc mb local/mlflow; \
    kill $server_pid

CMD ["minio", "server", "/data","--address",":9000", "--console-address", ":9001"]