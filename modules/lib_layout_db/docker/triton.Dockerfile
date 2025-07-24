#syntax=docker/dockerfile:experimental
FROM techainer1t/base_docker:paddle_2-93dada6e

# First install the requirements only
COPY requirements.txt /tmp
RUN --mount=type=cache,target=/github-actions/.cache/pip pip install -r /tmp/requirements.txt

# Then copy the rest of the code
RUN --mount=target=/ctx rsync -r --chown='techainer_docker' \
    --exclude='models' \
    --exclude='tests' \
    /ctx/ /home/techainer_docker/code/
WORKDIR /home/techainer_docker/code/

CMD mlchain serve mlchain run -c mlconfig.yaml -m triton