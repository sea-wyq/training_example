import torch
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    load_path = "./model/maskrcnn.pth"
    model = torch.load(load_path)
    model = model.to(device)

    image = read_image("data/PennFudanPed/PNGImages/FudanPed00046.png")
    eval_transform = get_transform(train=False)
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    print(pred["boxes"].long())
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")
    output_image_np = output_image.detach().cpu().numpy().transpose(1, 2, 0)
    
    from PIL import Image
    pil_image = Image.fromarray(output_image_np)
    save_path = "output_image.png"
    pil_image.save(save_path)