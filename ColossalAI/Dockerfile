FROM registry.cnbita.com:5000/wuyiqiang/python3.9_cuda11.3_pytorch1.12:v1
# FROM registry.cnbita.com:5000/official-user-image-dev/ascend-base:cann5.1.rc2-py39-ubuntu18.04-arm64

WORKDIR /app

RUN apt update \
    && apt install vim -y \
    && apt install iproute2 -y \
    && apt install -y openssh-server -y \
    && apt install bind9-host -y

RUN pip install colossalai

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# COPY getIP.sh getIP.sh

# COPY train.py train.py

CMD ["/bin/bash"]

# registry.cnbita.com:5000/wuyiqiang/python3.9_cuda11.3_colossalai:v1