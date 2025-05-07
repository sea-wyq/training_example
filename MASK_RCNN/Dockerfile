FROM registry.cnbita.com:5000/yangpengcheng/llama-recipes:nightly-20240426

WORKDIR /workspace

RUN ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.470.74 /usr/lib/x86_64-linux-gnu/libcuda.so.1 
RUN pip3 install "fastapi[standard]" 
RUN pip3 install minio
RUN pip3 install torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install pycocotools

COPY ./ ./

