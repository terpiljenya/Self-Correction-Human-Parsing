import glob
import os
import sys
import uuid
from glob import glob
from random import gammavariate

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from clothes_extractor import Extractor


def imporve_mask(item_mask, dilate=True):
    gray = np.uint8(item_mask)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # finding contours
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea)
    # create a blank mask
    mask = np.zeros((item_mask.shape[0], item_mask.shape[1]), np.uint8)
    # draw selected ROI to form the mask
    mask = cv2.drawContours(mask, [max_contour], 0, 255, -1)
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.dilate(mask, kernel, iterations=4)
    return mask / 255


def crop_item(
    image,
    mask,
    item_id,
    keep_original_back=False,
    improve_mask=True,
    dilate_mask=False,
    path_background="/home/jt/Self-Correction-Human-Parsing/2021-09-11 11.13.18.jpg",
):
    mask = np.asarray(mask)
    unique, counts = np.unique(mask, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    item_mask = np.where(np.isin(mask, [item_id]), 1, 0)

    if np.argwhere(item_mask == 1).sum() == 0:
        return None
    # print(mask)

    if improve_mask:
        item_mask = imporve_mask(item_mask, dilate_mask)

    x_min = np.argwhere(item_mask == 1).min(axis=0)[1]
    y_min = np.argwhere(item_mask == 1).min(axis=0)[0]
    x_max = np.argwhere(item_mask == 1).max(axis=0)[1]
    y_max = np.argwhere(item_mask == 1).max(axis=0)[0]

    if keep_original_back:
        return image.crop((x_min, y_min, x_max, y_max))

    matte = np.repeat(np.asarray(item_mask)[:, :, None], 3, axis=2)
    if path_background:
        back = Image.open(path_background)
        back = back.resize(image.size)
    else:
        back = Image.new("L", size=image.size, color=(255)).convert("RGB")
    foreground = image * matte + back * (1 - matte)
    return Image.fromarray(np.uint8(foreground)).crop((x_min, y_min, x_max, y_max))


lip_classes = [
    "Background",
    "Hat",
    "Hair",
    "Glove",
    "Sunglasses",
    "Upper-clothes",
    "Dress",
    "Coat",
    "Socks",
    "Pants",
    "Jumpsuits",
    "Scarf",
    "Skirt",
    "Face",
    "Left-arm",
    "Right-arm",
    "Left-leg",
    "Right-leg",
    "Left-shoe",
    "Right-shoe",
]


extractor = Extractor()

# input_folder = "../ai_stylist/my_looks/"
input_folder = "../ai_stylist/test_data/"
output_folder = "test_data"

results_folder = os.path.join(output_folder, "results")

category_to_garment = {
    "tops": ["Upper-clothes", "Coat"],
    "dress": ["Dress"],
    "pants": ["Pants"],
    "skirt": ["Skirt"],
    "jumpsuits": ["Jumpsuits"],
    "footwear": ["Left-shoe", "Right-shoe"],
}

garment_to_category = {}
for category, garments in category_to_garment.items():
    for garment in garments:
        garment_to_category[garment] = category


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

garment_category_folder = {}

for category in category_to_garment.keys():
    garment_category = os.path.join(output_folder, "garment", category)
    garment_category_folder[category] = garment_category
    if not os.path.exists(garment_category):
        os.makedirs(garment_category)


for path in glob(os.path.join(input_folder, "*.jpg")):
    if not os.path.exists(path):
        print("Img does not exists..")
        continue
    img = Image.open(path)
    mask = extractor.predict(img)
    mask.save(os.path.join(results_folder, os.path.basename(path).replace(".jpg", ".png")))

    for category, garments in category_to_garment.items():

        extracted_garment = crop_item(img, mask, [lip_classes.index(garment) for garment in garments])
        if extracted_garment:
            print("Extracted", category)
            img_id = os.path.splitext(os.path.basename(path))[0]
            extracted_garment.save(os.path.join(garment_category_folder[category], f"{img_id}.jpg"))

    print("Processed - ", path)
