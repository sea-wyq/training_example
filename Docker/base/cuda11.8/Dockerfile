FROM registry.cnbita.com:5000/wuyiqiang/pytorch:2.3.0-cuda11.8-cudnn8-runtime  
# 上面镜像是自带conda环境的,基础镜像是pytorch提供，上面只是换了tag

WORKDIR /app

USER root

# 配置 ubuntu apt 源为中科大镜像源
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# 安装系统依赖、常用工具、修改时区、修改字符编码
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV TZ=Asia/Shanghai
ENV SHELL=/bin/bash
ENV NCCL_DEBUG=INFO
ENV TORCH_CPP_LOG_LEVEL=INFO
ENV TORCH_DISTRIBUTED_DEBUG=INFO

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-utils build-essential ca-certificates software-properties-common \
        wget curl vim git openssh-server tmux htop iputils-ping iproute2 net-tools unzip tzdata locales && \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    locale-gen en_US.UTF-8 && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# # 用 miniconda 安装 Python
# RUN wget --quiet -O ~/miniconda.sh https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm -f ~/miniconda.sh

# ENV PATH=/opt/conda/bin:$PATH
# ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
# RUN mv /opt/conda/lib/libtinfo.so.6 /opt/conda/lib/libtinfo.so.6.bak && \
#     mv /opt/conda/lib/libtinfo.so.6.3 /opt/conda/lib/libtinfo.so.6.3.bak

# 设置清华 Anaconda 镜像
# 参考 https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
RUN conda config --add channels defaults && \
    conda config --set show_channel_urls yes && \
    conda config --append default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --append default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r && \
    conda config --append default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 && \
    conda config --set custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.msys2 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.bioconda https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.menpo https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.pytorch https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.pytorch-lts https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.simpleitk https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.deepmodeling https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/

# registry.cnbita.com:5000/wuyiqiang/python3.8_cuda11.8_pytorch2.3.0:v1