#!/bin/bash
echo "BUILD START"
orange=$(tput setaf 3)
reset_color=$(tput sgr0)

if command -v nvidia-smi &> /dev/null
then
    echo "Building for ${orange}nvidia${reset_color} hardware"
    DOCKERFILE=Dockerfile.nvidia
else
    echo "Building for ${orange}intel${reset_color} hardware: nvidia driver not found"
    DOCKERFILE=Dockerfile.intel
fi

DOCKER_BUILDKIT=1 docker build . \
    -f $DOCKERFILE \
    --build-arg UID=${UID} \
    --build-arg GID=${UID} \
    -t segmentator:latest
