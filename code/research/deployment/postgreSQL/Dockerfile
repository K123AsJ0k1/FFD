FROM docker.io/postgres:16.2

RUN mkdir /postgres 
RUN chown -R postgres:postgres /postgres && chmod -R 700 /postgres
RUN mkdir /postgres/config 
RUN chown -R postgres:postgres /postgres/config && chmod -R 700 /postgres/config
RUN mkdir /postgres/logs
RUN chown -R postgres:postgres /postgres/logs && chmod -R 700 /postgres/logs

COPY --chmod=0755 postgresql.conf /postgres/config/postgresql.conf

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=mlflow_db

EXPOSE 5432

CMD ["postgres", "-c", "config_file=/postgres/config/postgresql.conf"]