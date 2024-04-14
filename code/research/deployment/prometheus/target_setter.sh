#!/bin/sh

sed -i "s/\[1\]/[$PROMETHEUS_TARGET]/" /etc/prometheus/prometheus.yml

sed -i "s/\[2\]/[$PUSHGATEWAY_TARGET]/" /etc/prometheus/prometheus.yml

sed -i "s/\[3\]/[$CENTRAL_TARGET]/" /etc/prometheus/prometheus.yml

sed -i "s/\[4\]/[$WORKER_TARGET]/" /etc/prometheus/prometheus.yml

exec prometheus "$@"