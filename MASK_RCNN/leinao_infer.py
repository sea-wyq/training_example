from PIL import Image
from io import BytesIO
import os
import sys
import numpy as np
import torch
from torchvision import tv_tensors
from pathlib import Path
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from torchvision.transforms import v2 as T


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class InferenceService:
    def __init__(self, **kargs):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        print("Initializing")
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')
        self.model_path = "/input/mask_rcnn/output/maskrcnn.pth" 
        self.model = torch.load(self.model_path,map_location=torch.device('cpu'))

    def predict(self, X):
        image = Image.open(BytesIO(X))
        eval_transform = get_transform(train=False)
        self.model.eval()
        try:
            with torch.no_grad():
                image = np.array(image)
                image = np.transpose(image, (2, 0, 1)) 
                image = tv_tensors.Image(image)
                x = eval_transform(image)
                x = x[:3, ...].to( self.device)
                predictions = self.model([x, ])
                pred = predictions[0]
                formatted_results = []
                pred_boxes = pred["boxes"].long()
                print(pred_boxes)
                for i in range(len(pred_boxes)):
                    formatted_results.append({
                        "box": {
                            "xmax": pred_boxes[i][2].item(),
                            "xmin": pred_boxes[i][0].item(),
                            "ymax": pred_boxes[i][3].item(),
                            "ymin": pred_boxes[i][1].item()
                        },
                        "label":  "pedestrian",
                        "score":  round(pred["scores"][i].item(),3)
                    })
                
                print(f'formatted_results: {formatted_results}')
                return {"data":{"result": formatted_results}}
        except Exception as e:
            return {"jsonData":{"result": e}}
            
    
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
class DataModel(BaseModel):
    strData: str
app = FastAPI()

inference_service = InferenceService()

@app.post("/predict")
def inference(files: UploadFile = File(...)): 
    # 调用推理服务进行预测
    prediction = inference_service.predict(files.file.read())
    return prediction

uvicorn.run(app, host="0.0.0.0", port=5000)

# 类脑云推理服务适配：url路径为/predict, 传入的文件参数名称是files。