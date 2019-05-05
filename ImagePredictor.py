# Fruit image source: https://github.com/Horea94/Fruit-Images-Dataset
# Code based on Heartbeat's PyTorch Guide
# URL: https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864

import torch
from NeuralNetModels import *
import torch.nn as nn
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt


# given an image file path, predicts the fruit in that image.
def predict_single_image(image_path):
    checkpoint = torch.load("./fruitimagemodel_79.model")  # Best model given in repository for testing
    model = BasicModel(num_classes=11)

    model.load_state_dict(checkpoint)
    model.eval()
    image = Image.open(image_path)
    resize_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = resize_transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    classes = ImageFolder(
        root="./Test/",
        transform=resize_transform).classes

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    image_input = Variable(image_tensor)

    # Predict the class of the image
    output = model(image_input)

    _, predicted = torch.max(output, 1)

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(1, len(image_input), 1)
    sub.set_title(str(classes[predicted]))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def predict_multiple_images():
    checkpoint = torch.load("./fruitimagemodel_79.model")  # Best model given in repository for testing
    model = BasicModel(num_classes=11)

    model.load_state_dict(checkpoint)
    model.eval()

    resize_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(
        root="./TestFruitsCustom/SingleClass/",
        transform=resize_transform)
    classes = ImageFolder(
        root="./TestFruitsCustom/SingleClass/",
        transform=resize_transform).classes
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        shuffle=False
    )
    # Predict fruits given a pretrained model
    correct_labels = list(0.0 for i in range(11))
    total_labels = list(0.0 for i in range(11))
    for i, (images, labels) in enumerate(data_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        pred_answer = (predicted == labels).squeeze()
        for j in range(4):
            label = labels[j]
            correct_labels[label] += pred_answer[j].item()
            total_labels[label] += 1

    for i in range(11):
        if total_labels[i] > 0:
            print(f'Accuracy of {classes[i]} : {100 * correct_labels[i] / total_labels[i]}')


# outputs the accuracy for each fruit image in the Test set.
def output_label_accuracy():
    checkpoint = torch.load("./fruitimagemodel_79.model")  # Best model given in repository for testing
    model = BasicModel(num_classes=11)

    model.load_state_dict(checkpoint)
    model.eval()

    resize_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(
        root="./Test/",
        transform=resize_transform)
    classes = ImageFolder(
        root="./Test/",
        transform=resize_transform).classes
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        shuffle=False
    )
    # Predict fruits given a pretrained model
    correct_labels = list(0.0 for i in range(11))
    total_labels = list(0.0 for i in range(11))
    for i, (images, labels) in enumerate(data_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        pred_answer = (predicted == labels).squeeze()
        for j in range(4):
            label = labels[j]
            correct_labels[label] += pred_answer[j].item()
            total_labels[label] += 1

    for i in range(11):
        if total_labels[i] > 0:
            print(f'Accuracy of {classes[i]} : {100 * correct_labels[i] / total_labels[i]}')


if __name__ == "__main__":
    output_label_accuracy()
    print("-----------------")
    predict_multiple_images()
