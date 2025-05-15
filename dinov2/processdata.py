from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/home/train/imagenet", extra="/home/train/extra")
    dataset.dump_extra()

