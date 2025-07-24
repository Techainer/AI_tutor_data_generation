FROM minio/mc:RELEASE.2022-07-29T19-17-16Z

# RUN addgroup -S techainer_docker && adduser -S techainer_docker -G techainer_docker
# USER techainer_docker

WORKDIR /home/techainer_docker

# Now copy the models
COPY models /home/techainer_docker/code/models
WORKDIR /home/techainer_docker/code/

# Alias to s3 and 
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["/usr/bin/mc alias set s3 ${AWS_S3_ENDPOINT_URL} ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY} --api S3v4; /usr/bin/mc mb s3/${AWS_BUCKET} -p; /usr/bin/mc mirror --overwrite /home/techainer_docker/code/models s3/${AWS_BUCKET}/models;"]