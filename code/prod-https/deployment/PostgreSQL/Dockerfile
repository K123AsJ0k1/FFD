FROM docker.io/postgres:13

COPY --chmod=0755 postgresql.conf /etc/postgresql/postgresql.conf
COPY --chmod=0755 pg_hba.conf /etc/postgresql/pg_hba.conf

RUN mkdir /data

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=mlflow_db
ENV PGDATA=/data/pgdata

EXPOSE 5432

CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]