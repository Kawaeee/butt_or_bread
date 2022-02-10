import os
import requests

import torch
from torchvision import models, transforms


class ButtBreadClassifier:
    def __init__(self, model_url):
        self.model_url = model_url
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

    def initialize(self):
        """Retrieves the butt_bread trained model and maps it to the CPU by default, can also specify GPU here."""

        self.model = models.resnet152(pretrained=False).to(self.device)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 2),
        ).to(self.device)

        self.model.load_state_dict(torch.load("buttbread_resnet152_3.h5", map_location=self.device))
        self.model.eval()

        return self.model

    def download(self):
        """Download model weight, if model does not exist in Streamlit server."""

        if os.path.isfile("buttbread_resnet152_3.h5") is False:
            print("Downloading butt_bread model !!")
            req = requests.get(self.model_url, allow_redirects=True)
            open("buttbread_resnet152_3.h5", "wb").write(req.content)
            req = None
            return True

        return False

    def predict(self, image):
        """Make a prediction on a single image"""

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
