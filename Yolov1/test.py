import os
from my_arguments import Args
import torch
from torch.utils.data import DataLoader

from model import MyNet
from data import MyDataset
from util import labels2bbox
from prepare_data import GL_CLASSES
import torchvision.transforms as transforms
from PIL import Image
import cv2


COLOR = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250),
         (255,125,125),(255,125,250),(125,125,0),(0,255,125),(255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(0,0,0),]  # 用来标识20个类别的bbox颜色，可自行设定


class TestInterface(object):
    def __init__(self, opts):
        self.opts = opts
        print("=======================Start inferring.=======================")

    def main(self):
        opts = self.opts
        img_list = os.listdir(opts.dataset_dir)
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        model = torch.load(opts.weight_path)
        if opts.use_GPU:
            model.to(opts.GPU_id)
        for img_name in img_list:
            img_path = os.path.join(opts.dataset_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = trans(img)
            img = torch.unsqueeze(img, dim=0)
            print(img_name, img.shape)
            if opts.use_GPU:
                img = img.to(opts.GPU_id)
            preds = torch.squeeze(model(img), dim=0).detach().cpu()
            preds = preds.permute(1,2,0)
            bbox = labels2bbox(preds)
            draw_img = cv2.imread(img_path)
            self.draw_bbox(draw_img, bbox)

    def draw_bbox(self, img, bbox):
        h, w = img.shape[0:2]
        n = bbox.shape[0]
        for i in range(n):
            confidence = bbox[i, 4]
            if confidence<0.2:
                continue
            p1 = (int(w * bbox[i, 0]), int(h * bbox[i, 1]))
            p2 = (int(w * bbox[i, 2]), int(h * bbox[i, 3]))
            cls_name = GL_CLASSES[int(bbox[i, 5])]
            print(cls_name, p1, p2)
            cv2.rectangle(img, p1, p2, COLOR[int(bbox[i, 5])])
            cv2.putText(img, cls_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(img, str(confidence), (p1[0],p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow("bbox", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    args = Args()
    args.set_test_args() 
    test_interface = TestInterface(args.get_opts())
    test_interface.main() 