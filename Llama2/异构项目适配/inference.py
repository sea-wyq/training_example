from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class InferenceService():
    def __init__(self, **kargs):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        self.model_path = "/root/Llama-2-7b-hf"   # 原模型挂载的位置（内置在镜像中）
        self.peft_Model_Path = "/model/LLAMA2"    # lora模型挂载的位置

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                return_dict=True,
                load_in_8bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation=None,
        )

        self.model = PeftModel.from_pretrained(self.model,self.peft_Model_Path)

    def predict(self, X):
        inputs = self.tokenizer(X, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        return {"jsonData": { 'result': [{'generated_text': self.tokenizer.decode(outputs[0], skip_special_tokens=True)}]},"meta": {}}
    

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
class DataModel(BaseModel):
    strData: str
app = FastAPI()

inference_service = InferenceService()

@app.post("/image")
def inference(data: DataModel): 
    prediction = inference_service.predict(data.strData)
    return prediction

uvicorn.run(app, host="0.0.0.0", port=5000)