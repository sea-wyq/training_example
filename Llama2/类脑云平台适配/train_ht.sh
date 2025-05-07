#!/bin/bash
# 超参调优
set -x

GAMMA=$1

export TASKDATAPATH=/input/llma2/llama2
export SAMSUM_PATH=$TASKDATAPATH/samsum
export DEBIAN_FRONTEND=noninteractive

cd $TASKDATAPATH/llama-recipes
pip uninstall llama-recipes -y
pip install -e . 

cd $TASKDATAPATH/llama-recipes/recipes/finetuning

python finetuning.py --gamma GAMMA --use_peft --peft_method lora --quantization --use_fp16 --model_name $TASKDATAPATH/Llama-2-7b-hf --output_dir /output/llama2/llama2/output --batch_size_training 1

# 单节点运行