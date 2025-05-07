#!/bin/bash

set -x


export TASKDATAPATH=/input/llma2/llama2
export SAMSUM_PATH=$TASKDATAPATH/samsum
export DEBIAN_FRONTEND=noninteractive

cd $TASKDATAPATH/llama-recipes
pip uninstall llama-recipes -y
pip install -e . 
pip install tensorboard 

cd $TASKDATAPATH/llama-recipes/recipes/finetuning

python finetuning.py --use_tensorboard  --tb_output_dir /output/llma2/llama2/output/tensorboard --use_peft --peft_method lora --quantization --use_fp16 --model_name $TASKDATAPATH/Llama-2-7b-hf --output_dir /output/llma2/llama2/output --batch_size_training 1

# 单节点运行