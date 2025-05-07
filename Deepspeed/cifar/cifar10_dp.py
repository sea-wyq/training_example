import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_moe_param_groups(model):
    parameters = {
        'params': [p for p in model.parameters()],
        'name': 'parameters'
    }
    return split_params_into_different_moe_groups_for_optimizer(parameters)

def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--with_cuda', default=False, action='store_true', help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False,  action='store_true', help='whether use exponential moving average')
    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs',  default=30,  type=int, help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--log-interval', type=int, default=2000, help="output logging information at a given interval")
    parser.add_argument('--moe', default=False, action='store_true', help='use deepspeed mixture of experts (moe)')
    parser.add_argument('--ep-world-size', default=1, type=int, help='(moe) expert parallel world size')
    parser.add_argument('--num-experts', type=int, nargs='+', default=[ 1, ], help='number of experts list, MoE related.')
    parser.add_argument( '--mlp-type', type=str, default='standard', help= 'Only applicable when num-experts > 1, accepts [standard, residual]')
    parser.add_argument('--top-k', default=1, type=int, help='(moe) gating top 1 and 2 supported')
    parser.add_argument( '--min-capacity', default=0,  type=int,  help= '(moe) minimum capacity of an expert regardless of the capacity_factor' )
    parser.add_argument( '--noisy-gate-policy', default=None, type=str, help= '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'  )
    parser.add_argument( '--moe-param-group', default=False, action='store_true', help= '(moe) create separate moe param groups, required when using ZeRO w. MoE' )
    parser.add_argument( '--dtype', default='fp16', type=str, choices=['bf16', 'fp16', 'fp32'], help= 'Datatype used for training' )
    parser.add_argument( '--stage', default=0, type=int, choices=[0, 1, 2, 3], help= 'Datatype used for training' )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = add_argument()

    deepspeed.init_distributed()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if torch.distributed.get_rank() != 0:
        torch.distributed.barrier()
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    
    if torch.distributed.get_rank() == 0:
        torch.distributed.barrier()

    trainloader = torch.utils.data.DataLoader(trainset,batch_size=16,shuffle=True,num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
    net = Net()

    parameters = filter(lambda p: p.requires_grad, net.parameters())
        
    ds_config = {
    "train_batch_size": 16,
    "steps_per_print": 2000,
    "optimizer": {
        "type": "Adam",
        "params": {
        "lr": 0.001,
        "betas": [
            0.8,
            0.999
        ],
        "eps": 1e-8,
        "weight_decay": 3e-7
        }
    },
    "elasticity": {
        "enabled": True,
        "max_train_batch_size": 200,
        "micro_batch_sizes": [16,32],
        "min_gpus": 1,
        "max_gpus": 2,
        "min_time": 0,
        "version": 0.1,
        "ignore_non_elastic_batch_info": True,
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 0.001,
        "warmup_num_steps": 1000
        }
    },
    "gradient_clipping": 1.0,
    "prescale_gradients": False,
    "bf16": {
        "enabled": args.dtype == "bf16"
    },
    "fp16": {
        "enabled": args.dtype == "fp16",
        "fp16_master_weights_and_grads": False,
        "loss_scale": 0,
        "loss_scale_window": 500,
        "hysteresis": 2,
        "min_loss_scale": 1,
        "initial_scale_power": 15
    },
    
    "wall_clock_breakdown": False,
    "zero_optimization": {
        "stage": args.stage,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 50000000,
        "reduce_bucket_size": 50000000,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "cpu_offload": False
    }
    }

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=trainset, config=ds_config)

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype=torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype=torch.half

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(local_device), data[1].to(local_device)
            if target_dtype != None:
                inputs = inputs.to(target_dtype)
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()
            if local_rank == 0 and i % args.log_interval == (
                    args.log_interval -1):  
                print('step:%d, [%d, %5d] loss: %.3f' %
                    (len(trainloader),epoch + 1, i + 1, running_loss / args.log_interval))
                running_loss = 0.0
                
    print('Finished Training')
