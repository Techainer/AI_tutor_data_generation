#!/bin/bash
if grep -sq 'docker\|lxc' /proc/1/cgroup; then
    # Go to proper path inside the container
    cd /home/techainer_docker/code/
fi

python3 -m coverage run --source=. tests/run_test.py
if [ $? -ne 0 ]; then
    exit 1;
fi
python3 -m coverage report
if grep -sq 'docker\|lxc' /proc/1/cgroup; then
    # Export coverage report in SonarQube format during CI
    python3 -m coverage xml -o tests/coverage.xml
    # Set everything inside the directory to be 777 so it can
    # be delete by the host easily
    chmod -R 777 /home/techainer_docker/code/tests/results
fi
