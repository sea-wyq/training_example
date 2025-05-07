#!/bin/bash

set -x


export TASKDATAPATH=/input/DemoTraing/llama2
export SAMSUM_PATH=$TASKDATAPATH/samsum
export DEBIAN_FRONTEND=noninteractive

cd $TASKDATAPATH/llama-recipes
pip uninstall llama-recipes -y
pip install -e . 

cd $TASKDATAPATH/llama-recipes/recipes/finetuning

torchrun --nnodes 2 --nproc_per_node 1 --node-rank $VC_TASK_INDEX  --master-addr `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`  --master-port 12345 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name  $TASKDATAPATH/Llama-2-7b-hf --output_dir /output/DemoTraing/llama2/output --batch_size_training 1

torchrun --nnodes 2 --local-addr=$POD_IP --nproc_per_node 1 --rdzv-backend=c10d  --rdzv-endpoint=10.244.58.97:12345 ./train.py
# 单节点运行