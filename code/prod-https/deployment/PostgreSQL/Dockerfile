FROM docker.io/postgres:13

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

RUN mkdir /data

ENV PGDATA=/data

COPY --chmod=0755 init.sql /docker-entrypoint-initdb.d/init.sql

EXPOSE 5432

CMD ["postgres"]