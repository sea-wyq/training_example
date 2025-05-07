# 用户手册

## 前置条件
（1）下载模型数据Llama-2-7b-hf  
（2）安装llama-recipes
（3）资源要求：单机单卡（16c,32G）,多级多卡（16c.32G）

### 数据集获取

因为模型数据集比较大15G左右，在类脑云环境中，可在集群节点/leinaostorage/wyq/Llama2的这个目录下获取完整的数据集。

```bash
[root@heros-suanfa-100-211 Llama2]# ls /leinaostorage/wyq/Llama2
Llama-2-7b-hf  llama-recipes  llm.md  pod.yaml  safety-flan-t5-base  samsum  train_llama.sh  vj.yaml
```

### llama-recipes安装

- 仓库安装

    不建议使用该方式安装，因为数据是通过huggingface进行拉取的，正常环境是无法连接外网的。
    ```bash
    pip install llama-recipes
    ```

- 本地安装
    
    因为源码做了一些修改，建议通过此方式安装，后续训练推理流程都是基于此方式安装的llama-recipes来进行执行的。

    1. 克隆仓库
    ```
    git clone https://gitlab.bitahub.com/leinaoyun-models/llama2.git
    ```
    2. 进入llama2/llama-recipes目录
    ```
    wyq@DESKTOP-PVL03Q3 MINGW64 /d/GoPro/src/llama2 (main)
    $ ls
    README.md             inference.py       llama-recipes/
    huggingface_train.py  inference_peft.py  samsum/

    cd llama2/llama-recipes
    ```
    注：samsum 目录保存的是微调的数据集

    
    3. 安装llama-recipes
    ```bash
    pip install -e . 
    ```

### 卸载llama-recipes

```bash
pip uninstall llama-recipes -y    
```

## 基于llama2 进行微调训练

### 单机单卡微调模型流程

1） 可以使用llama-recipes目录下的脚本进行执行(需要修改加载的模型路径和输出路径)
```
export SAMSUM_PATH=/path/samsum  #需要将samsum数据集的挂载位置设置到SAMSUM_PATH环境变量中
cd llama-recipes/recipes/finetuning
python finetuning.py --use_peft --peft_method lora --quantization --use_fp16 --model_name /path/Llama-2-7b-hf --output_dir /path/model --batch_size_training 1
```

2） 使用huggingface_train.py脚本进行执行(需要修改加载的模型路径和输出路径)
```
python3 huggingface_train.py
```

### 单机多卡微调模型流程(单机二卡)

```
export SAMSUM_PATH=/path/samsum
cd llama-recipes/recipes/finetuning
torchrun --nnodes 1 --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint MASTER_IP:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /path/Llama-2-7b-hf --output_dir /path/model --batch_size_training 3
```

### 多机多卡微调模型流程(二机二卡)
```
cd llama-recipes/recipes/finetuning
torchrun --nnodes 2 --nproc_per_node 2 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 1
```

## 基于llama2 进行推理 (暂时只支持单卡推理)

### 基于源码模型文件进行推理
```
python3 inference.py
```

### 基于lora方式微调后的模型进行推理

```
python3 inference_peft.py
```