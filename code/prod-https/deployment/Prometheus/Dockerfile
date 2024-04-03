FROM docker.io/prom/prometheus:v2.50.1

ENV PROMETHEUS_TARGET="'127.0.0.1:9090'"
ENV PUSHGATEWAY_TARGET="'127.0.0.1:9091'"
ENV CENTRAL_TARGET="'127.0.0.1:7600'"
ENV WORKER_TARGET="'127.0.0.1:7500'"

COPY --chmod=0755 prometheus.yml /etc/prometheus/prometheus.yml

COPY --chmod=0755 target_setter.sh /usr/local/bin/

EXPOSE 9090

ENTRYPOINT ["/usr/local/bin/target_setter.sh"]

CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.console.libraries=/usr/share/prometheus/console_libraries", \
     "--web.console.templates=/usr/share/prometheus/consoles"]