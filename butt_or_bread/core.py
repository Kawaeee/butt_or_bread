import logging
import os
import requests

import torch
from torchvision import models, transforms


class ButtBreadClassifier:
    """
    A classifier for predicting whether an image is of a corgi butt or a loaf of bread.

    Attributes:
        model_url (str): The URL where the trained model weights can be downloaded.
        model_name (str): The filename of the trained model weights.
        device (torch.device): The device (CPU or GPU) where the model will be loaded.
        model (torchvision.models.ResNet): The ResNet152 model with a custom fully connected layer for classifying images.
        preprocessor (torchvision.transforms.Compose): The image preprocessing pipeline used to normalize and resize images.
        logger (logging.Logger): The logger object used for logging messages during model download.

    Methods:
        initialize(): Initializes the model by loading the pretrained ResNet152 model, adding a custom fully connected layer with two output classes, loading the model weights, and putting the model on the CPU or GPU depending on availability. Returns the initialized model.
        download(): Downloads the model weights from the specified URL if the weights are not already present in the server. If the download is successful, returns True, otherwise raises an error.
        predict(image): Takes an image as input, preprocesses it using standard image transformations, runs the preprocessed image through the loaded model, and returns a JSON object containing the image metadata, the predicted labels (corgi butt or loaf of bread), and the corresponding probabilities. Returns the JSON object.
    """

    def __init__(self, model_url):
        self.model_url = model_url
        self.model_name = "buttbread_resnet152_3.h5"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        """
        Initializes the model by loading the pretrained ResNet152 model, adding a custom fully connected layer with two output classes, loading the model weights, and putting the model on the CPU or GPU depending on availability. Returns the initialized model.
        """
        self.model = models.resnet152(weights=None).to(self.device)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 2),
        ).to(self.device)

        self.model.load_state_dict(torch.load(self.model_name, map_location=self.device, weights_only=True))
        self.model.eval()

        return self.model

    def download(self):
        """
        Downloads the model weights from the specified URL if the weights are not already present in the server. If the download is successful, returns True, otherwise raises an error.
        """
        try:
            if not os.path.isfile(self.model_name):
                self.logger.info("Downloading butt_or_bread model !!")
                model_weights_request = requests.get(self.model_url, allow_redirects=True)
                with open(self.model_name, "wb") as f:
                    f.write(model_weights_request.content)
                return True

        except Exception as e:
            self.logger.error(f"Failed to download model. Error: {e}")
            raise

        return False

    def predict(self, image):
        """
        Takes an image as input, preprocesses it using standard image transformations, runs the preprocessed image through the loaded model, and returns a JSON object containing the image metadata, the predicted labels (corgi butt or loaf of bread), and the corresponding probabilities. Returns the JSON object.

        Args:
            image (PIL.Image): The input image to make a prediction on.

        Returns:
            dict: A JSON object containing the image metadata, the predicted labels (corgi butt or loaf of bread), and the corresponding probabilities.
        """
        input_image = self.preprocessor(image)
        input_image = input_image.unsqueeze(0)

        prediction_logits_tensor = self.model(input_image)
        prediction_probabilities = torch.nn.functional.softmax(prediction_logits_tensor, dim=1).cpu().data.numpy()

        bread_probability = prediction_probabilities[0][0]
        corgi_probability = prediction_probabilities[0][1]

        json_output = {
            "name": image.filename,
            "format": image.format_description,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "prediction": {
                "labels": {
                    "Corgi butt 🐕": "{:.3%}".format(float(corgi_probability)),
                    "Loaf of bread 🍞": "{:.3%}".format(float(bread_probability)),
                }
            },
        }

        return json_output
