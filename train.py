import time 
from tqdm import tqdm

from PIL import Image

import torch
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optimizer

dataset_path = '/data'

# https://pytorch.org/docs/stable/torchvision/models.html

#data transformation = image augmentation  -> to tensor -> normalize
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Train with random affine and horizontal flip
# Valid/test no augmentation

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'valid':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
    'test': 
        transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder(dataset_path+'/train', data_transforms['train']),
    'valid': datasets.ImageFolder(dataset_path+'/valid', data_transforms['valid']),
    'test': datasets.ImageFolder(dataset_path+'test', data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=2),
    'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False, num_workers=2),
    'test': DataLoader(image_datasets['test'], batch_size=1, shuffle=False, num_workers=2)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet152(pretrained=True).to(device)
    
for param in model.parameters():
    param.requires_grad = False  
 
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)

criterion = nn.CrossEntropyLoss()
adam = optimizer.Adam(model.fc.parameters())

# Train model function
def train_model(model, criterion, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        time_start = time.monotonic()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        #phase check
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #iterate and try to predict input and check with output => generate loss and correct label
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss.item(), epoch_acc.item()))
            
        print('Runtime: (', '{0:.2f}'.format(time.monotonic()-time_start), ' seconds)', sep='')

    return model

# Training
model_training = train_model(model, criterion, adam, num_epochs=3).to(device)

# Test with test set
test_acc_count = 0

for k, (test_images, test_labels) in tqdm(enumerate(dataloaders['test'])):
    test_outputs = model_training(test_images.to(device))
    _, prediction = torch.max(test_outputs.data, 1)
    test_acc_count += torch.sum(prediction == test_labels.to(device).data).item()

test_accuracy = test_acc_count / len(dataloaders['test'])
print('Test acc: ',test_accuracy)

# Save model weight
torch.save(model_training.state_dict(), '/model/buttbread_resnet152_3.h5')
