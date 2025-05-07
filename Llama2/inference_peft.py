import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_Path = "/data/LLAMA1/Llama-2-7b-hf"
peft_Model_Path = "/data/LLAMA1/outputs/model"
tokenizer = AutoTokenizer.from_pretrained(model_Path)

model = AutoModelForCausalLM.from_pretrained(
        model_Path,
        return_dict=True,
        load_in_8bit=True,    # 增加模型量化，防止OOM kill
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation=None,
        torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(model,peft_Model_Path)
model.eval()

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# 该脚本测试通过，可以正常训练llama2微调训练（单卡3090环境测试）