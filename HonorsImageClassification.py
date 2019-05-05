# Fruit image source: https://github.com/Horea94/Fruit-Images-Dataset
# Code based on Heartbeat's PyTorch Guide
# URL: https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864

from TrainModel import TrainModel
from NeuralNetModels import *
from torch.optim import Adam


def main():
    # create model
    model = BasicModel(num_classes=11)

    # Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    image_classifier = TrainModel(p_model=model, p_optimizer=optimizer, p_loss_func=loss_fn, p_num_epochs=100,
                                  p_train_path="./Training", p_test_path="./Test")
    image_classifier.train()
    image_classifier.graph_results()


if __name__ == "__main__":
    main()