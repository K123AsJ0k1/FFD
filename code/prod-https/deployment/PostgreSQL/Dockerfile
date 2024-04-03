FROM docker.io/postgres:13

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV CONTAINER_STORAGE=/data

RUN mkdir ${CONTAINER_STORAGE}

ENV PGDATA=${CONTAINER_STORAGE}

COPY --chmod=0755 init.sql /docker-entrypoint-initdb.d/init.sql

EXPOSE 5432

CMD ["postgres"]