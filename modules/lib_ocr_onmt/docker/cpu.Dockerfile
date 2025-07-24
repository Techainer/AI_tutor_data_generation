# syntax = docker/dockerfile:experimental
FROM techainer1t/base_docker:pytorch_111-93dada6e

# First install the requirements only
COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/github-actions/.cache/pip pip install --no-cache-dir -r /tmp/requirements.txt

# Now copy the models
COPY --chown=techainer_docker models /home/techainer_docker/code/models

# Then copy the rest of the code
RUN --mount=target=/ctx rsync -r --chown='techainer_docker' \
                                --exclude='tests' \
                                --exclude='models' \
                                /ctx/ /home/techainer_docker/code/
WORKDIR /home/techainer_docker/code/

CMD mlchain serve mlchain run -c mlconfig.yaml -m prod