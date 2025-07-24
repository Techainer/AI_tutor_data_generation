#!/bin/sh

http_port=11000
grpc_port=11001
metrics_port=11002

docker run --gpus=0 -p $http_port:$http_port -p $grpc_port:$grpc_port -p $metrics_port:$metrics_port --net=host \
            -v $PWD/models:/models techainer1t/base_docker:tritonserver_full_2208-93dada6e tritonserver \
            --model-repository=/models --strict-model-config=false --exit-on-error=false \
            --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port