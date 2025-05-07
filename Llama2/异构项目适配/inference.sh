#!/bin/bash

set -x

DATAMOUNTPATH=/data/LLAMA1

pip uninstall llama-recipes -y
cd $DATAMOUNTPATH/llama-recipes
pip install -e . 

ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.470.74 /usr/lib/x86_64-linux-gnu/libcuda.so.1   

export SAFE_PATH=$DATAMOUNTPATH/safety-flan-t5-base

python3 /root/inference.py


# 多节点运行







