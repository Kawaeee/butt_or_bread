import streamlit as st
from streamlit.logger import get_logger

import time
import os
import requests

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms

st_logger = get_logger(__name__)

st.set_option("deprecation.showfileUploaderEncoding", False)

st.set_page_config(
    layout="centered",
    page_title="Corgi butt or loaf of bread?",
    page_icon="icon/corgi-icon.png",
)

# Markdown
repo = "[![GitHub Star](https://img.shields.io/github/stars/Kawaeee/butt_or_bread)](https://github.com/Kawaeee/butt_or_bread)"
version = "[![GitHub Release](https://img.shields.io/github/v/release/Kawaeee/butt_or_bread)](https://github.com/Kawaeee/butt_or_bread/releases/tag/v1.0)"
follower = "[![GitHub Follow](https://img.shields.io/github/followers/Kawaeee?style=social)](https://github.com/Kawaeee)"
visitor = "![Visitor Badge](https://visitor-badge.glitch.me/badge?page_id=Kawaeee.butt_or_bread.visitor-badge)"

model_url_path = "https://github.com/Kawaeee/butt_or_bread/releases/download/v1.0/buttbread_resnet152_3.h5"

# Test images
test_images_path = "test_images"
labels = ["Corgi butt 🐕", "Loaf of bread 🍞"]

corgi_images_file = [
    "corgi_1.jpg",
    "corgi_2.jpg",
    "corgi_3.jpg",
    "corgi_4.jpg",
    "corgi_5.jpg",
]

corgi_images_name = [
    "A loaf of corgi",
    "Corgi butt pressed against window",
    "Corgi butt wearing a glasses",
    "Thicc corgi butt post",
    "Cute corgi butt walking outdoor",
]
corgi_images_dict = {
    name: os.path.join(test_images_path, c_file)
    for name, c_file in zip(corgi_images_name, corgi_images_file)
}

bread_images_file = [
    "bread_1.jpg",
    "bread_2.jpg",
    "bread_3.jpg",
    "bread_4.jpg",
    "bread_5.jpg",
]

bread_images_name = [
    "A close up of a corgi butt bread",
    "A loaf of bread on the wooden table",
    "Big loaf of bread",
    "Burnt version of corgi butt bread",
    "Corgi butt bun",
]

bread_images_dict = {
    name: os.path.join(test_images_path, b_file)
    for name, b_file in zip(bread_images_name, bread_images_file)
}

# Model configuration
processing_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_normalizer = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

img_transformer = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        img_normalizer,
    ]
)


@st.cache(allow_output_mutation=True, max_entries=5, ttl=3600)
def initialize_model(device=processing_device):
    """Retrieves the butt_bread trained model and maps it to the CPU by default, can also specify GPU here."""
    model = models.resnet152(pretrained=False).to(device)
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2),
    ).to(device)

    model.load_state_dict(torch.load("buttbread_resnet152_3.h5", map_location=device))
    model.eval()

    return model

def predict(img, model):
    """Make a prediction on a single image"""
    input_img = img_transformer(img).float()
    input_img = input_img.unsqueeze(0)

    pred_logits_tensor = model(input_img)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

    bread_prob = pred_probs[0][0]
    butt_prob = pred_probs[0][1]

    json_output = {
        "name": img.filename,
        "format": img.format,
        "mode": img.mode,
        "width": img.width,
        "height": img.height,
        "prediction": {
            "labels": {
                "Corgi butt 🐕": "{:.3%}".format(float(butt_prob)),
                "Loaf of bread 🍞": "{:.3%}".format(float(bread_prob)),
            }
        },
    }

    return json_output

def download_model():
    """Download model weight, if model does not exist in Streamlit server."""
    if os.path.isfile("buttbread_resnet152_3.h5") == False:
        print("Downloading butt_bread model !!")
        req = requests.get(model_url_path, allow_redirects=True)
        open("buttbread_resnet152_3.h5", "wb").write(req.content)
        return True

    return True


if __name__ == "__main__":
    img_file = None
    img = None
    prediction = None

    download_model()
    model = initialize_model()

    st_logger.info("[INFO] Initialize %s model successfully", "buttbread_resnet152_3.h5", exc_info=0)

    st.title("Corgi butt or loaf of bread? 🐕🍞")
    st.markdown(version + " " + repo + " " + visitor + " " + follower, unsafe_allow_html=True)

    processing_mode = st.radio("", ("Upload an image", "Select pre-configured image"))

    if processing_mode == "Upload an image":
        img_file = st.file_uploader("Upload an image", accept_multiple_files=False)
    elif processing_mode == "Select pre-configured image":
        img_labels = st.selectbox("Pick a labels:", labels)

        if img_labels == labels[0]:
            corgi_list = st.selectbox("Pick your favorite corgi butt image 🐕:", corgi_images_name)
            img_file = corgi_images_dict[corgi_list]

        elif img_labels == labels[1]:
            bread_list = st.selectbox("Pick your favorite loaf of bread image 🍞:", bread_images_name)
            img_file = bread_images_dict[bread_list]

    if img_file:
        try:
            img = Image.open(img_file)

            if img.mode != "RGB":
                tmp_format = img.format
                img = img.convert("RGB")
                img.format = tmp_format
            if processing_mode == "Upload an image":
                img.filename = img_file.name
            elif processing_mode == "Select pre-configured image":
                img.filename = os.path.basename(img_file)

            prediction = predict(img, model)

            st_logger.info("[INFO] Predict %s image successfully", img.filename, exc_info=0)

        except Exception as e:
            st.error("ERROR: Unable to predict {} ({}) !!!".format(img_file.name, img_file.type))
            st_logger.error("[ERROR] Unable to predict %s (%s) !!!", img_file.name, img_file.type, exc_info=0)
            img_file = None
            img = None
            prediction = None

    if img != None or prediction != None:
        st.header("Here is the image you've chosen")
        resized_image = img.resize((400, 400))
        st.image(resized_image)
        st.write("Prediction:")
        st.json(prediction)
