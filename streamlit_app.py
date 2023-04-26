import json
import os

import streamlit as st
from streamlit.logger import get_logger
from PIL import Image

from butt_or_bread.core import ButtBreadClassifier
from butt_or_bread.utils import health_check

# Create Streamlit logger
st_logger = get_logger(__name__)
st.set_option("deprecation.showfileUploaderEncoding", False)

# Load Streamlit configuration file
with open("streamlit_app.json") as cfg_file:
    st_app_cfg = json.load(cfg_file)

ui_cfg = st_app_cfg["ui"]
model_cfg = st_app_cfg["model"]
image_cfg = st_app_cfg["image"]

st.set_page_config(
    layout="centered",
    page_title=ui_cfg["title"],
    page_icon=ui_cfg["icon"],
)


@st.cache_resource
def get_classifier():
    """
    Retrieves a cached instance of a ButtBreadClassifier model, or creates a new instance if none exists.

    Returns:
        A ButtBreadClassifier object that has been downloaded and initialized.

    Usage:
        To retrieve a cached classifier, call this function without any arguments. The cached classifier will be returned
        if it exists, or a new one will be created and cached. 

        Example:
        >>> classifier = get_classifier()

        To force the creation of a new instance and bypass the cache, pass a new `model_url` parameter as a keyword
        argument.

        Example:
        >>> new_classifier = get_classifier(model_url='https://new-model-url.com')

    Raises:
        Any exceptions raised during the initialization of the ButtBreadClassifier object, such as if the download
        or initialization fails.

    Note:
        This function makes use of the `@st.cache_resource` decorator, which allows the resulting classifier object to be
        cached and reused across different sessions of the Streamlit app. This can greatly improve performance, but also
        means that changes to the underlying model will not be reflected until the cache is cleared.

    """
    classifier = ButtBreadClassifier(model_url=model_cfg["url"])
    classifier.download()
    classifier.initialize()

    return classifier


if __name__ == "__main__":
    image_file, image, prediction = None, None, None

    classifier = get_classifier()
    st_logger.info("[INFO] Initialize %s model successfully", "buttbread_resnet152_3.h5", exc_info=0)
    st_logger.info("[DEBUG] %s", health_check(), exc_info=0)

    st.title(body=ui_cfg["title"])
    st.markdown(
        body=f'{ui_cfg["markdown"]["release"]} {ui_cfg["markdown"]["star"]} {ui_cfg["markdown"]["visitor"]}',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        label="options?",
        options=[ui_cfg["mode"]["upload"]["main_label"], ui_cfg["mode"]["select"]["main_label"]],
        label_visibility="hidden",
    )

    if mode == ui_cfg["mode"]["upload"]["main_label"]:
        image_file = st.file_uploader(label=mode, accept_multiple_files=False)
    elif mode == ui_cfg["mode"]["select"]["main_label"]:
        class_label = st.selectbox(label=ui_cfg["mode"]["select"]["class_label"], options=model_cfg["label"].values())

        if class_label == model_cfg["label"]["corgi"]:
            image_label = st.selectbox(label=ui_cfg["mode"]["select"]["corgi_label"], options=[*image_cfg["corgi"]])
            image_file = os.path.join(image_cfg["base_path"], image_cfg["corgi"][image_label])
        elif class_label == model_cfg["label"]["bread"]:
            image_label = st.selectbox(label=ui_cfg["mode"]["select"]["bread_label"], options=[*image_cfg["bread"]])
            image_file = os.path.join(image_cfg["base_path"], image_cfg["bread"][image_label])

    if image_file:
        try:
            image = Image.open(image_file)

            if image.mode != "RGB":
                temporary_format = image.format
                image = image.convert("RGB")
                image.format = temporary_format

            if mode == ui_cfg["mode"]["upload"]["main_label"]:
                image.filename = image_file.name
            elif mode == ui_cfg["mode"]["select"]["main_label"]:
                image.filename = os.path.basename(image_file)

            prediction = classifier.predict(image)

            st_logger.info("[DEBUG] %s", health_check(), exc_info=0)
            st_logger.info("[INFO] Predict %s image successfully", image.filename, exc_info=0)

        except Exception as ex:
            st.error("ERROR: Unable to predict {} ({}) !!!".format(image_file.name, image_file.type))
            st_logger.error("[ERROR] Unable to predict %s (%s) !!!", image_file.name, image_file.type, exc_info=0)
            image_file, image, prediction = None, None, None

    if image is not None or prediction is not None:
        st.header("Here is the image you've chosen")
        resized_image = image.resize((400, 400))
        st.image(resized_image)
        st.write("Prediction:")
        st.json(prediction)
