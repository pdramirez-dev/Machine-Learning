import os
import torchvision
from torchvision import transforms


# ruta de Archivos de entrenamineto
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


train_data_path = os.path('/Datasets/Images/train')
validation_data_path = os.path('/Datasets/Images/validation')
transforms = transforms.Compose(
    [transforms.Resize(64),
     transforms.ToTensor(),
     transforms.Normalize(mean=[
         0.485, 0.456, 0.406
     ], std=[.229, .224, .225])
     ])
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transforms)
validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=transforms)


def data_loader(batch_sice):
    pass
