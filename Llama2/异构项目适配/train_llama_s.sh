#!/bin/bash

set -x

DATAMOUNTPATH=/data/LLAMA1

pip uninstall llama-recipes -y
cd $DATAMOUNTPATH/llama-recipes
pip install -e . 
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.470.74 /usr/lib/x86_64-linux-gnu/libcuda.so.1   

cd $DATAMOUNTPATH/llama-recipes/recipes/finetuning

export SAMSUM_PATH=$DATAMOUNTPATH/samsum

python finetuning.py --use_peft --peft_method lora --quantization --use_fp16  --model_name $DATAMOUNTPATH/Llama-2-7b-hf --output_dir /output --batch_size_training 1

# python3 /data/LLAMA1/train.py
# python finetuning.py --use_peft --peft_method lora --quantization --use_fp16 --dataset alpaca_dataset --model_name $DATAMOUNTPATH/Llama-2-7b-hf --output_dir /output --batch_size_training 1
# 单节点运行