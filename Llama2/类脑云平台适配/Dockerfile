FROM registry.cnbita.com:5000/wuyiqiang/python3.8_cuda11.8_pytorch2.3.0:v1

WORKDIR /root

COPY llama-recipes llama-recipes
RUN cd llama-recipes && pip install -e . 

# RUN cd /root/llama-recipes && \
#     pip install -U pip setuptools && \
#     pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118 && \
#     pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .[tests,auditnlg,vllm]