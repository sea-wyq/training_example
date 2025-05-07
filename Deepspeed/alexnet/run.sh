#!/bin/bash

export  NCCL_SHM_DISABLE=1; export NCCL_SOCKET_IFNAME=eth0; export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; export NCCL_DEBUG=INFO deepspeed --include="localhost:2,3" train.py --deepspeed_config=ds_config.json -p 2 --steps=200