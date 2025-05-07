# -*- coding: utf-8 -*-
"""
TorchVision Object Detection Finetuning Tutorial
====================================================
"""
import os
import torch
import sys
import utils
import numpy as np
from PIL import Image
import io
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from minio import Minio
from torchvision.transforms import v2 as T
from engine import train_one_epoch, evaluate
from torchvision.ops.boxes import masks_to_boxes
from typing import  Tuple
from minio.error import S3Error

# os.environ["BUCKET_NAME"] = "pennfudan"
# os.environ["MINIO_URL"] = "10.0.102.61:32000"
# os.environ["MINIO_ACCESS_KEY"] = "LEINAOYUNOS"
# os.environ["MINIO_SECRET_KEY"] = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYLEINAOYUNKEY"

os.environ["MINIO_SECURE"] = "false"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["USE_MINIO"] = "false"
os.environ["MINIO_URL"] = "obs.cn-south-222.ai.pcl.cn/"
os.environ["MINIO_ACCESS_KEY"] = "8B9WZNOZTRXM9CFEZZJV"
os.environ["MINIO_SECRET_KEY"] = "jbR2ysrJa2jb0fzagiYRY0oo1zNXJvG3XcQHM3EM"
os.environ["BUCKET_NAME"] = "fortest-3d3b637712f649ebb6dbc838f644ab59"

def get_minio_credentials() -> Tuple[str]:
    endpoint = os.environ['MINIO_URL']
    access_key = os.environ['MINIO_ACCESS_KEY']
    secret_key = os.environ['MINIO_SECRET_KEY']
    if os.environ['MINIO_SECURE']=='true': secure = True 
    else: secure = False 
    return (endpoint, access_key, secret_key, secure)


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, bucket_name, transforms):
        self.transforms = transforms
        url, access_key, secret_key, secure = get_minio_credentials()
        self.bucket_name = bucket_name
        self.minio_client = Minio(url, 
                        access_key,  
                        secret_key, 
                        secure=secure)
        self.imgs = []
        self.masks = []

        try:
            img_objects = self.minio_client.list_objects(self.bucket_name, "PennFudanPed/PNGImages/", recursive=True)
            self.imgs = [obj.object_name for obj in img_objects]
            mask_objects = self.minio_client.list_objects(self.bucket_name, "PennFudanPed/PedMasks/", recursive=True)
            self.masks = [obj.object_name for obj in mask_objects]
        except S3Error as e:
            print(f"Error listing objects from Minio: {e}")
            raise

    def __getitem__(self, idx):
        img_object_name = self.imgs[idx]
        mask_object_name = self.masks[idx]
        try:
            img_response = self.minio_client.get_object(self.bucket_name, img_object_name)
            img_data = img_response.read()
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1)) 

            mask_response = self.minio_client.get_object(self.bucket_name, mask_object_name)
            mask_data = mask_response.read()
            mask = Image.open(io.BytesIO(mask_data)).convert('L')
            mask = np.array(mask)
            mask = np.expand_dims(mask, axis=0)
        except S3Error as e:
            print(f"Error fetching data from Minio: {e}")
            raise
        mask = torch.from_numpy(mask)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        
        # d, h, w = mask.shape
        # masks = torch.full((d,h, w), False, dtype=torch.bool)
        # for n in range(d):
        #     for i in range(h):
        #         for j in range(w):
        #             masks[n,i, j] = mask[n, i, j] == obj_ids[0]
        # masks = masks.to(dtype=torch.uint8)
        masks = (mask.eq(obj_ids[:, None, None])).to(dtype=torch.uint8)
        
        boxes = masks_to_boxes(masks)

        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



def get_model_instance_segmentation(num_classes):
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model = torch.load("model/maskrcnn.pth")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


if __name__ == "__main__":
    sys.stdout.flush()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    num_epochs = 2
    dataset = PennFudanDataset(os.environ["BUCKET_NAME"], get_transform(train=True))
    dataset_test = PennFudanDataset(os.environ["BUCKET_NAME"], get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=2)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
        sys.stdout.flush()
    torch.save(model, "/output/maskrcnn.pth")
    print("That's it!")

    