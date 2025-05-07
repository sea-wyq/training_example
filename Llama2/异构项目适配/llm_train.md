
## 实验结果

| nodes | GPUs | batchsize | sec/iter | samples/s | epoch time | scaling efficiency | speedup |
| ----- | ---- | --------- | -------- | --------- | ---------- | ------------------ | ------- |
| 1     | 1    | 1         | 7.75     | 0.1290    | 1:40:09    | -                  | -       |
| 1     | 2    | 6         | 10.18    | 0.5894    | 0:21:53    | 228.39%            | 4.5678  |
| 1     | 4    | 16        | 12.48    | 1.2521    | 0:09:59    | 248.40%            | 9.9359  |
| 1     | 4    | 20        | 15.33    | 1.3046    | 0:09:42    | 252.77%            | 10.1109 |
| 1     | 8    | 32        | 14.01    | 2.2841    | 0:05:36    | 221.27%            | 17.7016 |
| 1     | 8    | 40        | 16.49    | 2.4257    | 0:05:13    | 234.99%            | 18.7993 |
| 2     | 4    | 4         | 23.86    | 0.1676    | 1:16:45    | 32.48%             | 1.2992  |
| 2     | 4    | 8         | 25.09    | 0.3189    | 0:40:08    | 61.78%             | 2.4711  |
| 2     | 4    | 16        | 28.79    | 0.5557    | 0:23:02    | 107.68%            | 4.3071  |
| 2     | 8    | 40        | 33.65    | 1.1887    | 0:10:39    | 115.16%            | 9.2125  |
| 2     | 16   | 80        | 40.10    | 1.9950    | 0:05:20    | 96.63%             | 15.4613 |

> 由于训练集较小（775样本），DataLoader drop_last=True，所以计算加速比没有使用 epoch time，而是使用 sec/iter。2种方式除了2nodes16GPUs差异大，其他几乎无差异。


## 本地开发调试

```shell
conda create --prefix /Volumes/YpcSD1/conda-envs/llama-recipes python=3.10
source activate /Volumes/YpcSD1/conda-envs/llama-recipes
pip install -U pip setuptools
pip install -e .[tests,auditnlg,vllm]

# jupyter
conda install ipykernel
python -m ipykernel install --user --name llama-recipes --display-name llama-recipes
pip install matplotlib ipywidgets
```

## Docker 调试单机单卡

```shell
docker run -it -d --name ypc-llama-recipes \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -v /leinaostorage/bitahub-dev/projects/vne0a5ca25ae7e4648a572939c6c2c049f/llms/llama-recipes:/root/llama-recipes \
  -v /leinaostorage/bitahub-dev/models/vne0a5ca25ae7e4648a572939c6c2c049f/huggingface/meta-llama/Llama-2-7b-hf:/models/huggingface/meta-llama/Llama-2-7b-hf \
  -v /leinaostorage/bitahub-dev/datasets/vne0a5ca25ae7e4648a572939c6c2c049f/huggingface/samsum:/datasets/huggingface/samsum \
  registry.cnbita.com:5000/yangpengcheng/llama-recipes:v0.1 \
  bash

docker exec -it ypc-llama-recipes bash

# 改 samsum 数据集为本地路径 /datasets/huggingface/samsum
vi ~/llama-recipes/src/llama_recipes/datasets/samsum_dataset.py
# 改 samsum 数据集脚本数据位置为本地路径而不是 hf 的链接 /datasets/huggingface/samsum/data/corpus.7z
vi /datasets/huggingface/samsum/samsum.py

cd /root/llama-recipes/recipes/finetuning

# batch_size_training=4 设置 max_split_size_mb 没有效果，最后每次需要申请的内存大小不会更小了，说明是计算本身需要的最小内存
# PYTORCH_CUDA_ALLOC_CONF see https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256" python finetuning.py --use_peft --peft_method lora --quantization --use_fp16 --model_name /models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /root/yangpengcheng/PEFT/model

# batch_size_training=2 设置 max_split_size_mb 没有效果。随着调小（1024->64），每次要申请的内存没变，但当前 free 的内存增长并不明显
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64" python finetuning.py --use_peft --peft_method lora --quantization --use_fp16 --model_name /models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /root/yangpengcheng/PEFT/model --batch_size_training 2

torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.13 GiB. GPU 0 has a total capacty of 23.49 GiB of which 7.47 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 15.59 GiB is allocated by PyTorch, and 98.64 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

# batch_size_training=1 才不会 cuda OOM，7.73s/it
python finetuning.py --use_peft --peft_method lora --quantization --use_fp16 --model_name /models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /root/yangpengcheng/PEFT/model --batch_size_training 1
```

```
Training Epoch: 1/3, step 44/775 completed (loss: 1.0650898218154907):   6%|█▍                      | 45/775 [05:56<1:34:05,  7.73s/it]

# GPU 利用率在 76% - 96% 之间波动，大部分时间在 90% 以上

Thu Apr 18 10:17:44 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.146.02             Driver Version: 535.146.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:13:00.0 Off |                  N/A |
| 35%   66C    P2             331W / 350W |  21584MiB / 24576MiB |     79%	    Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage	    |
|=======================================================================================|
|    0   N/A  N/A   3698640      C   python                                    21576MiB |
+---------------------------------------------------------------------------------------+

# CPU 利用率在 163% - 187% 之间波动

CONTAINER ID   NAME                CPU %     MEM USAGE / LIMIT     MEM %     NET I/O           BLOCK I/O     PIDS
956aa95bbcfd   ypc-llama-recipes   183.76%   3.949GiB / 30.85GiB   12.80%    1.82kB / 4.01kB   21.2GB / 0B   32
```

训练日志
```
root@956aa95bbcfd:~/llama-recipes/recipes/finetuning# python finetuning.py --use_peft --peft_method lora --quantization --use_fp16 --model_name /models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /root/yangpengcheng/PEFT/model --batch_size_training 1
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████| 2/2 [01:45<00:00, 52.75s/it]
--> Model /models/huggingface/meta-llama/Llama-2-7b-hf

--> /models/huggingface/meta-llama/Llama-2-7b-hf has 262.41024 Million params

trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06220594176090199
Reusing dataset samsum (/root/.cache/huggingface/datasets/samsum/samsum/0.0.0/8d8c8cda2e7f628c0d93f72317e7a4aae56981e9e7b7fb0a90533628425f6492)
Parameter 'function'=<function get_preprocessed_samsum.<locals>.apply_prompt_template at 0x7f18d45ea4c0> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
Loading cached processed dataset at /root/.cache/huggingface/datasets/samsum/samsum/0.0.0/8d8c8cda2e7f628c0d93f72317e7a4aae56981e9e7b7fb0a90533628425f6492/cache-1c80317fa3b1799d.arrow
Loading cached processed dataset at /root/.cache/huggingface/datasets/samsum/samsum/0.0.0/8d8c8cda2e7f628c0d93f72317e7a4aae56981e9e7b7fb0a90533628425f6492/cache-bdd640fb06671ad1.arrow
--> Training Set Length = 14732
Reusing dataset samsum (/root/.cache/huggingface/datasets/samsum/samsum/0.0.0/8d8c8cda2e7f628c0d93f72317e7a4aae56981e9e7b7fb0a90533628425f6492)
Loading cached processed dataset at /root/.cache/huggingface/datasets/samsum/samsum/0.0.0/8d8c8cda2e7f628c0d93f72317e7a4aae56981e9e7b7fb0a90533628425f6492/cache-3eb13b9046685257.arrow
Loading cached processed dataset at /root/.cache/huggingface/datasets/samsum/samsum/0.0.0/8d8c8cda2e7f628c0d93f72317e7a4aae56981e9e7b7fb0a90533628425f6492/cache-23b8c1e9392456de.arrow
--> Validation Set Length = 818
Preprocessing dataset: 100%|███████████████████████████████████████████████████████████████████| 14732/14732 [00:06<00:00, 2260.31it/s]
Preprocessing dataset: 100%|███████████████████████████████████████████████████████████████████████| 818/818 [00:00<00:00, 2563.97it/s]
/opt/conda/lib/python3.8/site-packages/torch/cuda/memory.py:329: FutureWarning: torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats, which resets /all/ peak memory stats.
  warnings.warn(
Training Epoch: 1:   0%|                                                                                       | 0/775 [00:00<?, ?it/s]/opt/conda/lib/python3.8/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
/opt/conda/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:316: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
Training Epoch: 1/3, step 774/775 completed (loss: 0.9421892762184143): 100%|██████████████████████| 775/775 [1:40:09<00:00,  7.75s/it]
Max CUDA memory allocated was 19 GB
Max CUDA memory reserved was 20 GB
Peak active CUDA memory was 19 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
evaluating Epoch: 100%|████████████████████████████████████████████████████████████████████████████████| 42/42 [01:40<00:00,  2.40s/it]
 eval_ppl=tensor(2.7169, device='cuda:0') eval_epoch_loss=tensor(0.9995, device='cuda:0')
we are about to save the PEFT modules
/opt/conda/lib/python3.8/site-packages/peft/utils/save_and_load.py:154: UserWarning: Could not find a config file in /models/huggingface/meta-llama/Llama-2-7b-hf - will assume that the vocabulary was not modified.
  warnings.warn(
PEFT modules are saved in /root/yangpengcheng/PEFT/model directory
best eval loss on epoch 1 is 0.9994827508926392
Epoch 1: train_perplexity=2.8461, train_epoch_loss=1.0459, epoch time 6010.529560308001s
Training Epoch: 2/3, step 774/775 completed (loss: 0.8937536478042603): 100%|██████████████████████| 775/775 [1:39:51<00:00,  7.73s/it]
Max CUDA memory allocated was 19 GB
Max CUDA memory reserved was 20 GB
Peak active CUDA memory was 19 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
evaluating Epoch: 100%|████████████████████████████████████████████████████████████████████████████████| 42/42 [01:41<00:00,  2.41s/it]
 eval_ppl=tensor(2.7101, device='cuda:0') eval_epoch_loss=tensor(0.9970, device='cuda:0')
we are about to save the PEFT modules
PEFT modules are saved in /root/yangpengcheng/PEFT/model directory
best eval loss on epoch 2 is 0.9969720840454102
Epoch 2: train_perplexity=2.6733, train_epoch_loss=0.9833, epoch time 5991.841651730996s
Training Epoch: 3/3, step 774/775 completed (loss: 0.8481307029724121): 100%|██████████████████████| 775/775 [1:39:41<00:00,  7.72s/it]
Max CUDA memory allocated was 19 GB
Max CUDA memory reserved was 20 GB
Peak active CUDA memory was 19 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
evaluating Epoch: 100%|████████████████████████████████████████████████████████████████████████████████| 42/42 [01:43<00:00,  2.46s/it]
 eval_ppl=tensor(2.7498, device='cuda:0') eval_epoch_loss=tensor(1.0115, device='cuda:0')
Epoch 3: train_perplexity=2.5360, train_epoch_loss=0.9306, epoch time 5982.376789279006s
Key: avg_train_prep, Value: 2.6851089000701904
Key: avg_train_loss, Value: 0.9866079886754354
Key: avg_eval_prep, Value: 2.725571393966675
Key: avg_eval_loss, Value: 0.9978089729944865
Key: avg_epoch_time, Value: 5994.916000439334
Key: avg_checkpoint_time, Value: 0.2049053819985905
```


## 单机单卡

```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
python finetuning.py --use_peft --peft_method lora --quantization --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 1
```

## 单机2卡

batchsize=3*2

```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 1 --nproc_per_node 2 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 3
```

日志
```
Training Epoch: 1/3, step 128/129 completed (loss: 0.9865875244140625): 100%|████████████████████████| 129/129 [21:53<00:00, 10.18s/it]
Max CUDA memory allocated was 21 GB
Max CUDA memory reserved was 23 GB
Peak active CUDA memory was 21 GB
CUDA Malloc retries : 321
CPU Total Peak Memory consumed during the train (max): 2 GB
```

## 单机4卡

batchsize=4*4
```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 1 --nproc_per_node 4 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 4
```

日志
```
Training Epoch: 1/3, step 47/48 completed (loss: 1.062833547592163): 100%|█████████████████████████████| 48/48 [09:59<00:00, 12.48s/it]
Max CUDA memory allocated was 17 GB
Max CUDA memory reserved was 20 GB
Peak active CUDA memory was 17 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
```

batchsize=5*4
```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 1 --nproc_per_node 4 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 5
```

日志
```
Training Epoch: 1/3, step 37/38 completed (loss: 1.077970027923584): 100%|█████████████████████████████| 38/38 [09:42<00:00, 15.33s/it]
Max CUDA memory allocated was 19 GB
Max CUDA memory reserved was 22 GB
Peak active CUDA memory was 20 GB
CUDA Malloc retries : 1
CPU Total Peak Memory consumed during the train (max): 2 GB
```

## 单机8卡

batchsize=4*8
```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 1 --nproc_per_node 8 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 4
```

日志
```
Training Epoch: 1/3, step 23/24 completed (loss: 1.0850074291229248): 100%|█████████████████████████████████████| 24/24 [05:36<00:00, 14.01s/it]
Max CUDA memory allocated was 14 GB
Max CUDA memory reserved was 17 GB
Peak active CUDA memory was 14 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
```

batchsize=5*8
```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 1 --nproc_per_node 8 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 5
```

日志
```
Training Epoch: 1/3, step 18/19 completed (loss: 1.1089292764663696): 100%|█████████████████████████████████████| 19/19 [05:13<00:00, 16.49s/it]
Max CUDA memory allocated was 16 GB
Max CUDA memory reserved was 21 GB
Peak active CUDA memory was 17 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
```

## 类脑云2机4卡

### batch_size_training=1

```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 2 --nproc_per_node 2 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 1
```

c10d The IPv6 network addresses of  cannot be retrieved (gai error:  - Name or service not known).

```
Training Epoch: 1/3, step 192/193 completed (loss: 0.9798988699913025): 100%|██████████████████████| 193/193 [1:16:45<00:00, 23.86s/it]
Max CUDA memory allocated was 9 GB
Max CUDA memory reserved was 10 GB
Peak active CUDA memory was 10 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
evaluating Epoch: 100%|████████████████████████████████████████████████████████████████████████████████| 10/10 [01:52<00:00, 11.29s/it]
 eval_ppl=tensor(2.7410, device='cuda:0') eval_epoch_loss=tensor(1.0083, device='cuda:0')
we are about to save the PEFT modules
PEFT modules are saved in /input/projects/llama-recipes/outputs/PEFT/model directory
best eval loss on epoch 1 is 1.0083307027816772
Epoch 1: train_perplexity=2.9046, train_epoch_loss=1.0663, epoch time 4606.386543892324s
```

### batch_size_training=2

```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 2 --nproc_per_node 2 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 2
```

日志
```
Training Epoch: 1/3, step 95/96 completed (loss: 1.074712872505188): 100%|█████████████████████████████| 96/96 [40:08<00:00, 25.09s/it]
Max CUDA memory allocated was 12 GB
Max CUDA memory reserved was 14 GB
Peak active CUDA memory was 12 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
```

### batch_size_training=4

```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 2 --nproc_per_node 2 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 4
```

日志

```
Training Epoch: 1/3, step 47/48 completed (loss: 1.0658249855041504): 100%|████████████████████████████| 48/48 [23:02<00:00, 28.79s/it]
Max CUDA memory allocated was 17 GB
Max CUDA memory reserved was 20 GB
Peak active CUDA memory was 17 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
```

## 2机8卡

batch_size_training=5

```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 5
```

日志
```
Training Epoch: 1/3, step 18/19 completed (loss: 1.2273048162460327): 100%|█████████████████████████████████████| 19/19 [10:39<00:00, 33.65s/it]
Max CUDA memory allocated was 16 GB
Max CUDA memory reserved was 21 GB
Peak active CUDA memory was 17 GB
CUDA Malloc retries : 0
CPU Total Peak Memory consumed during the train (max): 2 GB
```

## 2机16卡

batch_size_training=6

```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 6
```

日志
```
Training Epoch: 1/3, step 7/8 completed (loss: 1.238161563873291): 100%|█████████████████████████████████| 8/8 [05:20<00:00, 40.10s/it]
Max CUDA memory allocated was 17 GB
Max CUDA memory reserved was 23 GB
Peak active CUDA memory was 18 GB
CUDA Malloc retries : 3
CPU Total Peak Memory consumed during the train (max): 2 GB
```

batch strategy
```shell
rm -rf /root/llama-recipes/
ln -sf /input/projects/llama-recipes /root/llama-recipes
cd /root/llama-recipes/recipes/finetuning
torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id ypc001 --rdzv_backend c10d --rdzv_endpoint `echo $VC_TASK1_HOSTS | awk -F , '{print $1}'`:21024 ./finetuning.py  --enable_fsdp --use_peft --peft_method lora --use_fp16 --model_name /input/models/huggingface/meta-llama/Llama-2-7b-hf --output_dir /input/projects/llama-recipes/outputs/PEFT/model --batch_size_training 6 --batching_strategy padding
```
