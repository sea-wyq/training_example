FROM registry.cnbita.com:5000/official-user-image-dev/pytorch:1.13.1-cuda11.6-cudnn8-py310-ubuntu18.04

RUN pip install deepspeed==0.12.2

WORKDIR /app

COPY ./ ./

