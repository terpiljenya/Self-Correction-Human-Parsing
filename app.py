import os
import pickle
import subprocess

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import ExifTags, Image

from clothes_extractor import Extractor


@st.cache()
def init_extractor():
    return Extractor()


def run_command(args):
    """Run command, transfer stdout/stderr back into Streamlit and manage error"""
    st.info(f"Doing magic...")
    result = subprocess.run(args, capture_output=True, text=True)
    try:
        result.check_returncode()
        st.info(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(result.stderr)
        raise e


def open_image(img_path):
    img = Image.open(img_path)
    print("Size oriignal", img.size)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = img._getexif()

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
    except Exception as ex:
        print(ex)
        pass
    return img.convert("RGB")


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
    dilate_mask=True,
    path_background="/home/jt/Self-Correction-Human-Parsing/2021-09-11 11.13.18.jpg",
):
    print("Size croping", image.size)
    mask = np.asarray(mask)
    unique, counts = np.unique(mask, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    item_mask = np.where(np.isin(mask, [item_id]), 1, 0)
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


extractor = init_extractor()

st.sidebar.header("Clothes Magic Extractor")

uploaded_file = st.sidebar.file_uploader("Choose an image...")

cloth_mapping = {
    "top": [
        5,  # upper cloth
        6,  # dress
        7,  # coat
        14,  # left arm
        15,  # right arm
    ],
    "bottom": [
        8,  # sock
        9,  # pant
        12,  # skirt
        16,  # left leg
        17,  # right leg
    ],
}

# cloth = st.sidebar.selectbox("cloth", cloth_mapping.keys())

if uploaded_file is not None:
    image = open_image(uploaded_file)
    img_name = uploaded_file.name.replace(".JPG", ".jpg")
    # image.save(f"inputs/{img_name}")
    mask = extractor.predict(image)
    with open("mask.pickle", "wb") as handle:
        pickle.dump(mask, handle)
    # mask.save(f"mask.jpg")
    # mask_name = img_name.replace(".jpg", ".png")
    # if not os.path.exists(f"outputs/{mask_name}"):
    #     run_command(["python3", "simple_extractor.py"])

    st.sidebar.image(image)
    st.sidebar.image(
        mask,
        caption="Result",
        use_column_width=True,
    )

    left_column, right_column = st.columns(2)

    top = crop_item(image, mask, cloth_mapping["top"])
    left_column.image(top)
    bottom = crop_item(image, mask, cloth_mapping["bottom"])
    left_column.image(bottom)
