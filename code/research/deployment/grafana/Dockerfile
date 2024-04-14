FROM docker.io/grafana/grafana:10.0.12

ENV PROMETHEUS_URL=http://127.0.0.1:9090

COPY --chmod=0755 datasource.yml /etc/grafana/provisioning/datasources/datasource.yml

EXPOSE 3000