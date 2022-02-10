import argparse
import os
import time

from tqdm import tqdm

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


class ButtBreadModel:
    """Corgi butt or loaf of bread? model"""

    def __init__(self, device):
        self.model = None
        self.device = device
        self.criterion = None
        self.optimizer = None

    def initialize(self):
        """Transfer Learning by using ResNet-152 as pre-trained weight"""
        self.model = models.resnet152(pretrained=True).to(self.device)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 2),
        ).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.fc.parameters())

    def train(self, image_dataloaders, image_datasets, epochs=1):
        for epoch in range(epochs):
            time_start = time.monotonic()
            print(f"Epoch {epoch + 1}/{epochs}")

            # Phase check
            for phase in ["train", "valid"]:

                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate and try to predict input and check with output -> generate loss and correct label
                for inputs, labels in tqdm(image_dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.detach() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_accuracy = running_corrects.float() / len(image_datasets[phase])

                print(f"{phase} loss: {epoch_loss.item():.4f}, acc: {epoch_accuracy.item():.4f}")

            print("Runtime: (", "{0:.2f}".format(time.monotonic() - time_start), " seconds)", sep="")

        return self.model

    def test(self, image_dataloaders):
        """Test with test set"""
        test_accuracy_count = 0

        for k, (test_images, test_labels) in tqdm(enumerate(image_dataloaders["test"])):
            test_outputs = self.model(test_images.to(self.device))
            _, prediction = torch.max(test_outputs.data, 1)
            test_accuracy_count += torch.sum(prediction == test_labels.to(self.device).data).item()

        test_accuracy = test_accuracy_count / len(image_dataloaders["test"])

        return test_accuracy

    def save(self, model_path):
        """Saving model weight"""
        return torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        """Loading model weight"""
        return self.model.load_state_dict(torch.load(model_path, map_location=self.device)).eval()


def get_dataset(dataset_path: str):
    """
    Data transformation steps
    Train set :: Resize -> Random affine -> Random horizontal flip -> To Tensor -> Normalize
    Valid/test set :: Resize -> To Tensor -> Normalize
    """
    data_transformers = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    }

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(dataset_path, "train"), data_transformers["train"]),
        "valid": datasets.ImageFolder(os.path.join(dataset_path, "valid"), data_transformers["valid"]),
        "test": datasets.ImageFolder(os.path.join(dataset_path, "test"), data_transformers["test"]),
    }

    image_dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=32, shuffle=True, num_workers=2),
        "valid": DataLoader(image_datasets["valid"], batch_size=32, shuffle=False, num_workers=2),
        "test": DataLoader(image_datasets["test"], batch_size=1, shuffle=False, num_workers=2),
    }

    return image_datasets, image_dataloaders


def main(opt):
    dataset_path, model_path, epochs = opt.dataset_path, opt.model_path, opt.epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, image_dataloaders = get_dataset(dataset_path)

    butt_bread_obj = ButtBreadModel(device=device)
    butt_bread_obj.initialize()

    butt_bread_obj.train(
        image_dataloaders=image_dataloaders,
        image_datasets=image_datasets,
        epochs=epochs,
    )

    test_accuracy = butt_bread_obj.test(image_dataloaders=image_dataloaders)
    print(f"Test accuracy: {test_accuracy}")

    butt_bread_obj.save(model_path=model_path)
    print(f"Saved model at {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="datasets/", help="Dataset path")
    parser.add_argument("--model-path", type=str, default="buttbread_resnet152_3.h5", help="Output model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")

    args = parser.parse_args()
    main(args)
