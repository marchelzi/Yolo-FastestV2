import os
import cv2
import time
import argparse
import numpy as np

import torch
import model.detector
import utils.utils
import streamlit as st
from PIL import Image
cfg = utils.utils.load_datafile("data/coco.data")
device = torch.device("cpu")
model = model.detector.Detector(
    cfg["classes"], cfg["anchor_num"], True).to(device)
model.load_state_dict(torch.load(
    "modelzoo/coco2017-0.241078ap-model.pth", map_location=device))

model.eval()


def run_inference(image):
    ori_img = image
    res_img = cv2.resize(
        ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    preds = model(img)

    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(
        output, conf_thres=0.3, iou_thres=0.4)

    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]
    for box in output_boxes[0]:
        x1, y1, x2, y2 = box[:4]
        x1, y1, x2, y2 = int(x1 * scale_w), int(y1 * scale_h), int(
            x2 * scale_w), int(y2 * scale_h)
        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(ori_img, LABEL_NAMES[int(box[5])], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return ori_img


def main():
    with st.sidebar:
        st.title("YOLOv3")
        st.subheader("Object Detection")
        st.write("This is a demo of YOLOv3 Object Detection")
        st.write("Please upload an image to test")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(
            uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        image_result = run_inference(image)
        st.image(image_result, caption='Classified Image.',
                 use_column_width=True)


if __name__ == "__main__":
    main()
