#!/bin/bash

set -x
ROLE=master
JOBNAME=$1
DATAMOUNTPATH=/data/LLAMA1

pip uninstall llama-recipes -y
cd $DATAMOUNTPATH/llama-recipes
pip install -e . 
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.470.74 /usr/lib/x86_64-linux-gnu/libcuda.so.1   

cd $DATAMOUNTPATH/llama-recipes/recipes/finetuning

export SAMSUM_PATH=$DATAMOUNTPATH/samsum
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size mb:32" # 减少显存碎片


if [ "$ROLE" == "master" ]; then
   bash $DATAMOUNTPATH/find.sh master $JOBNAME
   torchrun --nnodes 2 --nproc_per_node 1 --node-rank 0  --master-addr $POD_IP  --master-port 12345 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name  /data/LLAMA1/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 1
elif [ "$ROLE" == "worker" ]; then
    bash $DATAMOUNTPATH/find.sh worker  $JOBNAME
    export MASTER_IP=$(cat  $JOBNAME.env)
    torchrun --nnodes 2 --nproc_per_node 1 --node-rank 1  --master-addr $MASTER_IP  --master-port 12345 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name  /data/LLAMA1/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 1
else
    echo "[$current_time] Invalid type parameter."
    exit 1
fi

# 弹性训练的方式启动（rank会自动进行分配）
# torchrun --nnodes 2 --local-addr=$POD_IP --nproc_per_node 1 --rdzv-backend=c10d  --rdzv-endpoint=$MASTER_IP:12345 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name   $DATAMOUNTPATH/Llama-2-7b-hf --output_dir  /output --batch_size_training 1