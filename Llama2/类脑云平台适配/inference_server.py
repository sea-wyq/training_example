import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fastapi import FastAPI
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    text: str

app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the base model')
parser.add_argument('--peft_model_path', type=str, required=True, help='Path to the PEFT model')
args = parser.parse_args()


model_Path = args.model_path
peft_Model_Path = args.peft_model_path

tokenizer = AutoTokenizer.from_pretrained(model_Path)

model = AutoModelForCausalLM.from_pretrained(
    model_Path,
    return_dict=True,
    load_in_8bit=True,  # 增加模型量化，防止OOM kill
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation=None,
    torch_dtype=torch.float16,
)

model = PeftModel.from_pretrained(model, peft_Model_Path)
model.eval()

@app.post("/generate/")
async def generate_text(request: InferenceRequest):
    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12345)

# python model_service.py --model_path /input/llma2/llama2/Llama-2-7b-hf --peft_model_path /model/llama2/output
# curl -X POST "https://hero-dev.cnbita.com/inf-app/a13073335257657344897706/generate/" -H "Content-Type: application/json" -d '{"text": "Hello my name is"}'