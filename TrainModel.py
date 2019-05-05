# Fruit image source: https://github.com/Horea94/Fruit-Images-Dataset
# Code based on Heartbeat's PyTorch Guide
# URL: https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864

import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


class TrainModel:
    def __init__(self, p_model, p_optimizer, p_loss_func, p_num_epochs, p_train_path, p_test_path):
        self.model = p_model
        self.optimizer = p_optimizer
        self.num_epochs = p_num_epochs
        self.loss_fn = p_loss_func

        # load in the training and test data, as well as a list of images to know how many images there are.
        self.train_loader, self.train_images = self.load_files(p_train_path)
        self.test_loader, self.test_images = self.load_files(p_test_path)

        self.cuda_avail = torch.cuda.is_available()

        self.train_acc_list = []
        self.train_loss_list = []
        self.test_acc_list = []

    def load_files(self, path):
        resize_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = ImageFolder(
            root=path,
            transform=resize_transform)
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            shuffle=False
        )
        return data_loader, dataset

    # Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
    def adjust_learning_rate(self, epoch):
        lr = 0.001

        if epoch > 180:
            lr = lr / 1000000
        elif epoch > 150:
            lr = lr / 100000
        elif epoch > 120:
            lr = lr / 10000
        elif epoch > 90:
            lr = lr / 1000
        elif epoch > 60:
            lr = lr / 100
        elif epoch > 30:
            lr = lr / 10

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save_models(self, epoch):
        torch.save(self.model.state_dict(), "fruitimagemodel_{}.model".format(epoch))
        print("Checkpoint saved")

    # plot accuracy results
    def graph_results(self):
        plt.plot(self.train_acc_list, label='train accuracy')
        plt.plot(self.test_acc_list, label='test accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def test(self):
        self.model.eval()
        test_acc = 0.0
        for i, (images, labels) in enumerate(self.test_loader):
            if self.cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)
            # Predict classes using images from the test set
            outputs = self.model(images)
            _, prediction = torch.max(outputs.data, 1)

            test_acc += torch.sum(prediction == labels.data)

        # Compute the average acc and loss over all 10000 test images
        test_acc = float(test_acc) / float(len(self.test_images))

        return test_acc

    def train(self):
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()
            train_acc = 0.0
            train_loss = 0.0
            total = 0

            for i, (images, labels) in enumerate(self.train_loader):
                # Move images and labels to gpu if available
                if self.cuda_avail:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images)
                    labels = Variable(labels)

                # Clear all accumulated gradients
                self.optimizer.zero_grad()
                # Predict classes using images from the test set
                outputs = self.model(images)
                # Compute the loss based on the predictions and actual labels
                loss = self.loss_fn(outputs, labels)
                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                self.optimizer.step()

                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_acc += torch.sum(prediction == labels.data)

            # Call the learning rate adjustment function
            self.adjust_learning_rate(epoch)

            # Compute the average acc and loss over all 50000 training images
            train_acc = float(train_acc) / float(len(self.train_images))
            train_loss = train_loss / float(len(self.train_images))

            # save the train accuracy and loss to graph later
            self.train_acc_list.append(train_acc)
            self.train_loss_list.append(train_loss)

            # Evaluate on the test set
            test_acc = self.test()

            # Save the model if the test acc is greater than our current best
            if test_acc > best_acc:
                self.save_models(epoch)
                best_acc = test_acc

            # save the test accuracy and loss to graph later
            self.test_acc_list.append(test_acc)

            # Print the metrics
            print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss, test_acc))
