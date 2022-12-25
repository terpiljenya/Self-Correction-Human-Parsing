import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import networks
from datasets.simple_extractor_dataset import SimpleFolderDataset
from utils.transforms import get_affine_transform, transform_logits


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


class Extractor:
    def __init__(self) -> None:
        # num_classes = 20
        num_classes = 18
        # self.input_size = [473, 473]
        self.input_size = [512, 512]
        self.model = networks.init_model("resnet101", num_classes=num_classes, pretrained=None)
        self.palette = get_palette(num_classes)
        # state_dict = torch.load("checkpoints/final.pth")["state_dict"]
        state_dict = torch.load("checkpoints/exp-schp-201908301523-atr.pth")["state_dict"]
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()
        self.model.eval()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])]
        )
        self.upsample = torch.nn.Upsample(size=self.input_size, mode="bilinear", align_corners=True)

        print("Init model")

    def preprocess(self, image):
        image = np.array(image)
        aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
        h, w, _ = image.shape

        # Get person center and scale
        def _box2cs(box):
            x, y, w, h = box[:4]
            return _xywh2cs(x, y, w, h)

        def _xywh2cs(x, y, w, h):
            center = np.zeros((2), dtype=np.float32)
            center[0] = x + w * 0.5
            center[1] = y + h * 0.5
            if w > aspect_ratio * h:
                h = w * 1.0 / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            scale = np.array([w, h], dtype=np.float32)
            return center, scale

        person_center, s = _box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)

        scaled_img = cv2.warpAffine(
            image,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        tensor = self.transform(scaled_img)
        meta = {"center": person_center, "height": h, "width": w, "scale": s, "rotation": r}
        return tensor, meta

    def postprocess(self, output, meta):
        c = meta["center"]
        s = meta["scale"]
        w = meta["width"]
        h = meta["height"]
        upsample_output = self.upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
        parsing_result = np.argmax(logits_result, axis=2)
        output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        output_img.putpalette(self.palette)
        return output_img

    def predict(self, image):
        tensor, meta = self.preprocess(image)
        output = self.model(tensor.unsqueeze(0).cuda())
        output_image = self.postprocess(output, meta)
        return output_image
