FROM docker.io/prom/prometheus:v2.50.1

ENV SIGN='$'
ENV PROMETHEUS_URL=127.0.0.1:9090
ENV PUSHGATEWAY_URL=127.0.0.1:9091

COPY --chmod=0755 prometheus.yml /etc/prometheus/prometheus.yml

EXPOSE 9090