FROM docker.io/prom/prometheus:v2.50.1

COPY --chmod=0755 prometheus.yml /etc/prometheus/prometheus.yml

ENV PROMETHEUS_TARGET="'127.0.0.1:9090'"
ENV PUSHGATEWAY_TARGET="'127.0.0.1:9091'"
ENV CENTRAL_TARGET="'127.0.0.1:7600'"
ENV WORKER_TARGET="'127.0.0.1:7500'"

RUN sed -i "s/\[1\]/[$PROMETHEUS_TARGET]/" /etc/prometheus/prometheus.yml
RUN sed -i "s/\[2\]/[$PUSHGATEWAY_TARGET]/" /etc/prometheus/prometheus.yml
RUN sed -i "s/\[3\]/[$CENTRAL_TARGET]/" /etc/prometheus/prometheus.yml
RUN sed -i "s/\[4\]/[$WORKER_TARGET]/" /etc/prometheus/prometheus.yml

EXPOSE 9090