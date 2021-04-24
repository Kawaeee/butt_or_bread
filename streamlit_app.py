import streamlit as st

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

# Markdown
repo = "[![GitHub Star](https://img.shields.io/github/stars/Kawaeee/butt_or_bread)](https://github.com/Kawaeee/butt_or_bread)"
version = "[![GitHub Release](https://img.shields.io/github/v/release/Kawaeee/butt_or_bread)](https://github.com/Kawaeee/butt_or_bread/releases/tag/v1.0)"
follower = "[![GitHub Follow](https://img.shields.io/github/followers/Kawaeee?style=social)](https://github.com/Kawaeee)"
visitor = "![Visitor Badge](https://visitor-badge.glitch.me/badge?page_id=Kawaeee.butt_or_bread.visitor-badge)"

model_url_path = "https://github.com/Kawaeee/butt_or_bread/releases/download/v1.0/buttbread_resnet152_3.h5"

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


@st.cache()
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


@st.cache()
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
                "Corgi butt": "{:.3%}".format(float(butt_prob)),
                "Loaf of bread": "{:.3%}".format(float(bread_prob)),
            }
        },
    }

    return json_output


@st.cache(suppress_st_warning=True)
def download_model():
    """Download model weight, if model does not exist in Streamlit server."""
    if os.path.isfile("buttbread_resnet152_3.h5") == False:
        print("Downloading butt_bread model !!")
        req = requests.get(model_url_path, allow_redirects=True)
        open("buttbread_resnet152_3.h5", "wb").write(req.content)
        st.balloons()
    return True


if __name__ == "__main__":
    img = None
    prediction = None

    download_model()
    model = initialize_model()
    st.title("Corgi butt or loaf of bread?")
    st.markdown(version + " " + repo + " " + visitor + " " + follower, unsafe_allow_html=True)

    file = st.file_uploader("Upload An Image", accept_multiple_files=False)

    if file:
        try:
            img = Image.open(file)

            if img.mode != "RGB":
                tmp_format = img.format
                img = img.convert("RGB")
                img.format = tmp_format

            img.filename = file.name

            prediction = predict(
                img,
                model,
            )
        except Exception as e:
            img = None
            prediction = None
            st.error("ERROR: Unable to predict {} ({}) !!!".format(file.name, file.type))
    if img != None or prediction != None:
        st.header("Here is the image you've chosen")
        resized_image = img.resize((400, 400))
        st.image(resized_image)
        st.write("Prediction:")
        st.json(prediction)
