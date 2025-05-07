# mindspore-mnist

与rank table启动不同的是，在Ascend硬件平台上通过OpenMPI的mpirun命令运行脚本，用户**不需要配置RANK_TABLE_FILE环境变量**。

mpirun启动支持Ascend和GPU，此外还同时支持PyNative模式和Graph模式。
mpirun启动命令如下，其中DEVICE_NUM是所在机器的GPU数量：
```bash
mpirun -n DEVICE_NUM python net.py
```
mpirun还可以配置以下参数，更多配置可以参考mpirun文档：

- --output-filename log_output：将所有进程的日志信息保存到log_output目录下，不同卡上的日志会按rank_id分别保存在log_output/1/路径下对应的文件中。
- --merge-stderr-to-stdout：合并stderr到stdout的输出信息中。
- --allow-run-as-root：如果通过root用户执行脚本，则需要加上此参数。
- -mca orte_abort_on_non_zero_status 0：当一个子进程异常退出时，OpenMPI会默认abort所有的子进程，如果不想自动abort子进程，可以加上此参数。
- -bind-to none：OpenMPI会默认给拉起的子进程指定可用的CPU核数，如果不想限制进程使用的核数，可以加上此参数。

OpenMPI启动时会设置若干OPMI_*的环境变量，用户应避免在脚本中手动修改这些环境变量。

## 实验环境准备

mindspore官方镜像连接：https://www.hiascend.com/developer/ascendhub/detail/9de02a1a179b4018a4bf8e50c6c2339e
mindspore官方文档连接 ：https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0/parallel/dynamic_cluster.html#%E6%A6%82%E8%BF%B0

该仓库代码是基于mindspore 2.3版本镜像进行测试验证。通过mpirun的方式启动分布式训练任务。


## 实验流程


## cpu 环境执行

```bash
python3 train-cpu.py
```

## 单机多卡

```bash
cd distributed_data_parallel
bash run.sh 
```

## 多机多卡

1. 保证每个节点上都有相同的OpenMPI、NCCL、Python以及MindSpore版本。
2. 配置主机间免密登陆
3. 配置成功后，就可以通过mpirun指令启动多机任务，目前有两种方式启动多机训练任务：
   1. 通过mpirun -H方式。启动脚本如下：
   ```bash
    export DATA_PATH=./MNIST_Data/train/
    mpirun -n 16 -H DEVICE1_IP:8,DEVICE2_IP:8 --output-filename log_output --merge-stderr-to-stdout python net.py
   ```
    表示在ip为DEVICE1_IP和DEVICE2_IP的机器上分别起8个进程运行程序。在其中一个节点执行：
   ```bash
   bash run_mpirun_1.sh
   ```
   2. 通过mpirun --hostfile方式。为方便调试，建议用这种方法来执行多机多卡脚本。首先需要构造hostfile文件如下：
   ```bash
    DEVICE1 slots=8
    192.168.0.1 slots=8
   ```
   每一行格式为[hostname] slots=[slotnum]，hostname可以是ip或者主机名。上例表示在DEVICE1上有8张卡；ip为192.168.0.1的机器上也有8张卡。
    2机16卡的执行脚本如下，需要传入变量HOSTFILE，表示hostfile文件的路径：
    ```bash
    export DATA_PATH=./MNIST_Data/train/
    HOSTFILE=$1
    mpirun -n 16 --hostfile $HOSTFILE --output-filename log_output --merge-stderr-to-stdout python net.py
    ```
    在其中一个节点执行：
    ```bash
    bash run_mpirun_2.sh ./hostfile
    ```
