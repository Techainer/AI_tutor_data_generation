#syntax=docker/dockerfile:experimental
FROM minio/mc:latest

WORKDIR /home/techainer_docker

# Now copy the models
COPY --chown=techainer_docker models /home/techainer_docker/code/models
WORKDIR /home/techainer_docker/code/

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["/usr/bin/mc alias set s3 ${AWS_S3_ENDPOINT_URL} ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY} --api S3v4; /usr/bin/mc mb s3/${AWS_BUCKET} -p; /usr/bin/mc mirror --overwrite /home/techainer_docker/code/models s3/${AWS_BUCKET}/models;"]