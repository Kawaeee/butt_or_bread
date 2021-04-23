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
    input_img = img_transformer(img).float()
    input_img = input_img.unsqueeze(0)

    pred_logits_tensor = model(input_img)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

    return pred_probs


@st.cache(suppress_st_warning=True)
def download_model():
    model_url_path = "https://github.com/Kawaeee/butt_or_bread/releases/download/v1.0/buttbread_resnet152_3.h5"

    if os.path.isfile("buttbread_resnet152_3.h5") == False:
        print("butt_bread model not found")
        req = requests.get(model_url_path, allow_redirects=True)
        open("buttbread_resnet152_3.h5", "wb").write(req.content)
        st.balloons()

    return True


if __name__ == "__main__":
    img = None
    prediction = None

    download_model()
    model = initialize_model()
    st.write("# Corgi butt or loaf of bread?")

    file = st.file_uploader("Upload An Image")

    if file:
        img = Image.open(file)
        prediction = predict(
            img,
            model,
        )
        st.title("Here is the image you've selected")

    if img != None or prediction != None:
        resized_image = img.resize((336, 336))
        st.image(resized_image)
        st.write(prediction)