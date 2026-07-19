import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

#import matplotlib.pyplot as plt

#Image transformtion/resizing augmented transform for training only
train_transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])
    


validation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

#Load the base dataset
base_dataset = datasets.ImageFolder(root="data/raw")
classes = base_dataset.classes
dataset_size = len(base_dataset)

#Creating datasets is respective transforms for training and validation
train_dataset = datasets.ImageFolder( root="data/raw", transform=train_transform)
validation_dataset = datasets.ImageFolder( root="data/raw", transform=validation_transform)

#Split dataset into training and validation sets (80%/20%)
train_size = int(0.8 * dataset_size)
validation_size = dataset_size - train_size

#Fixed seed for reprodicable results
generator = torch.Generator().manual_seed(42)

train_subset, validation_subset = random_split(range(dataset_size), [train_size, validation_size], generator=generator)

#apply the split indices
train_dataset = Subset(train_dataset, train_subset.indices)
validation_dataset = Subset(validation_dataset, validation_subset.indices)

#create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


#images, labels = next(iter(train_loader))

#print(f"Number of images: {len(dataset)}")
#print(f"Number of classes: {len(dataset.classes)}")
#print(dataset.classes[:10])

#print(images.shape)
#print(labels.shape)
#print(labels[0])
#print(dataset.classes[labels[0]])

#image = images[0]
#print(image.shape)

#image = image.permute(1, 2, 0)

#plt.imshow(image)
#plt.show()