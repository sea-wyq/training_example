import os
import torch
import utils
import time
from engine import train_one_epoch, evaluate
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.transforms import v2 as T
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        
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
    model = torch.load("/home/training_example/MASK_RCNN/maskrcnn.pth")
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

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    num_classes = 2
    dataset_train = PennFudanDataset("/home/training_example/MASK_RCNN/data/PennFudanPed", get_transform(train=True))
    dataset_test = PennFudanDataset("/home/training_example/MASK_RCNN/data/PennFudanPed", get_transform(train=False))
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                num_replicas=world_size,
                                                                rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test,
                                                                num_replicas=world_size,
                                                                rank=rank)
    
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=50,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=utils.collate_fn,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                               batch_size=50,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=utils.collate_fn,
                                               sampler=test_sampler)

    # create model and move it to GPU with id rank
    model = get_model_instance_segmentation(num_classes).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    params = [p for p in ddp_model.parameters() if p.requires_grad]
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

    epochs = 10
    start =  time.time()
    for epoch in range(epochs):
        train_one_epoch(ddp_model, optimizer, train_loader, rank, epoch, print_freq=2)
        lr_scheduler.step()
        # evaluate(ddp_model, test_loader, device=rank)
        torch.cuda.empty_cache()
    end = time.time()
    print("train time: ",(end-start) / epochs )
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus 
    run_demo(demo_basic, world_size)
    print("finished")

# python train-ddp.py  单机双卡环境测试完成