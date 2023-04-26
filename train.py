"""
This script trains a convolutional neural network (CNN) to distinguish between images of corgi butts and loaf of bread. ]
It uses transfer learning with the ResNet-152 model pre-trained on the ImageNet dataset. 
The script loads the dataset, preprocesses the images, initializes the model, trains the model, and saves the trained weights to a specified file path.

Usage:
python train_model.py --dataset_path [path to dataset] --model_path [path to save model] --epochs [number of epochs to train for]

Args:
- dataset_path (str): The path to the directory containing the dataset. The dataset should be organized into three subdirectories: 'train', 'valid', and 'test', each containing subdirectories for the two classes ('butt' and 'bread').
- model_path (str): The path to save the trained model's weights.
- epochs (int): The number of epochs to train the model for.

Returns:
The trained CNN model saved to the specified file path.

Example usage:
python train_model.py --dataset_path ./data --model_path ./models/butt_bread_model.pt --epochs 10
"""

import argparse
import os
import time

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm


class ButtBreadModel:
    """
    A PyTorch model that predicts whether an image contains a corgi's butt or a loaf of bread.

    Attributes:
        model (torch.nn.Module): The PyTorch model.
        device (torch.device): The device (CPU or GPU) on which to run the model.
        criterion (torch.nn.Module): The loss function used to train the model.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.

    Methods:
        initialize(): Initializes the model's architecture by loading a pre-trained ResNet-152 model and replacing the
            fully connected layer with a new one that outputs two classes (corgi butt or loaf of bread).
        train(image_dataloaders, image_datasets, epochs=1): Trains the model on the given image datasets for the given
            number of epochs. Returns the trained model.
        test(image_dataloaders): Evaluate the model on the test set and return the accuracy.
        save(model_path): Save the model weight to a file.
        load(model_path): Load the model weight to a file
    """

    def __init__(self, device):
        """Initializes the ButtBreadModel with the given device (CPU or GPU)."""
        self.model = None
        self.device = device
        self.criterion = None
        self.optimizer = None

    def initialize(self):
        """
        Initializes the model's architecture by loading a pre-trained ResNet-152 model
        and replacing the fully connected layer with a new one
        that outputs two classes (corgi butt or loaf of bread).
        """
        self.model = models.resnet152(weights="IMAGENET1K_V1").to(self.device)

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
        """
        Trains the model on the given image datasets for the given number of epochs.

        Args:
            image_dataloaders (dict): A dictionary containing PyTorch DataLoader objects for the training and validation
                datasets.
            image_datasets (dict): A dictionary containing PyTorch Dataset objects for the training and validation
                datasets.
            epochs (int, optional): The number of epochs to train the model. Defaults to 1.

        Returns:
            The trained PyTorch model.
        """
        for epoch in range(epochs):
            time_start = time.monotonic()
            print(f"Epoch {epoch + 1}/{epochs}")

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
        """
        Evaluate the model on the test set and return the accuracy.

        Args:
            image_dataloaders (dict): A dictionary containing PyTorch DataLoader objects for the train, validation, and test sets.

        Returns:
            float: The accuracy of the model on the test set.
        """
        test_accuracy_count = 0

        for k, (test_images, test_labels) in tqdm(enumerate(image_dataloaders["test"])):
            test_outputs = self.model(test_images.to(self.device))
            _, prediction = torch.max(test_outputs.data, 1)
            test_accuracy_count += torch.sum(prediction == test_labels.to(self.device).data).item()

        test_accuracy = test_accuracy_count / len(image_dataloaders["test"])

        return test_accuracy

    def save(self, model_path):
        """
        Save the model weights to a file.

        Args:
            model_path (str): The path to the file where the model weights should be saved.
        """
        return torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        """
        Load the model weights from a file.

        Args:
            model_path (str): The path to the file where the model weights are stored.

        Returns:
            The loaded model with the saved weights.
        """
        return self.model.load_state_dict(torch.load(model_path, map_location=self.device)).eval()


def get_dataset(dataset_path: str):
    """
    This function takes in a dataset path and returns two dictionaries
    containing the image datasets and dataloaders for training, validation, and testing.
    The function applies different data transformations to each dataset depending on
    whether it's the train, validation, or test dataset.

    The train dataset is transformed with resize, random affine, random horizontal flip, to tensor, and normalization.
    The validation and test datasets are transformed with resize, to tensor, and normalization.

    Args:
        dataset_path (str): The path to the dataset directory.

    Returns:
        image_datasets (dict): A dictionary containing three image datasets: "train", "valid", and "test". Each dataset is an instance of ImageFolder class from torchvision.datasets, and is associated with its own set of data transformations defined by data_transformers dictionary.
        image_dataloaders (dict): A dictionary containing three dataloaders: "train", "valid", and "test". Each dataloader is associated with its own dataset in image_datasets and is responsible for loading the dataset with a given batch size and shuffling the data randomly for the train set. The test dataloader has a batch size of 1 since it is only used for evaluating the model. The num_workers parameter specifies how many subprocesses to use for data loading.
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
    """
    Train and test the ButtBreadModel on the specified dataset, and save the trained model.

    Args:
        opt (argparse.Namespace): The command-line arguments.

    Returns:
        None
    """
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
    parser.add_argument("--model-path", type=str, default="buttbread_resnet152_1.h5", help="Output model name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")

    args = parser.parse_args()
    main(args)
