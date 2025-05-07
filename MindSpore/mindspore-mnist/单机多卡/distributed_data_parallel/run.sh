#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "=============================================================================================================="

EXEC_PATH=$(pwd)

if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export GLOG_v=1 
export DATA_PATH=/home/MNIST_Data/train/ 
mpirun -n 4 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout python train_ddp.py


在设置RANK_TABLE_FILE  环境变量的情况下,使用mpirun的方式去启动分布式训练,结果是训练任务无法正常执行。


exit
# 单机八卡
# 脚本执行正常。可以正常进行训练。


docker run -it --ipc=host --rm -u root \
               --device=/dev/davinci0 \
               --device=/dev/davinci1 \
               --device=/dev/davinci2 \
               --device=/dev/davinci3 \
               --device=/dev/davinci4 \
               --device=/dev/davinci5 \
               --device=/dev/davinci6 \
               --device=/dev/davinci7 \
               --device=/dev/davinci_manager \
               --device=/dev/devmm_svm --device=/dev/hisi_hdc \
               -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
               -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
               -v /usr/local/sbin:/usr/local/sbin:ro \
               -v ./MNIST_Data:/home/MNIST_Data  \
               -v ~/ascend/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
               -v ~/ascend/log/npu/slog/:/var/log/npu/slog -v ~/ascend/log/npu/profiling/:/var/log/npu/profiling \
               -v ~/ascend/log/npu/dump/:/var/log/npu/dump -v ~/ascend/log/npu/:/usr/slog ${docker_image} \
               /bin/bash


# 多机多卡

教程连接： https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0/parallel/mpirun.html
