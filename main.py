import numpy as np
import matplotlib.pyplot as plt

def download_mnist(path="./mnist"):
    pass
def relu(x):
    pass
def relu_backward(gradient, x):
    pass
def softmax(x):
    pass
def cross_loss(resultatss, y_true):
    pass
def train(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64):
    """

    :param model:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param epochs:
    :param batch_size:
    :return:
    """
    pass
class Conv:
    def __init__(self, n_filters, filter_size, n_channels, lr=0.01):
        """

        :param n_filters:
        :param filter_size:
        :param n_channels:
        :param lr:
        """
        pass
    def forward(self, entree):
        """

        :param entree:
        :return:
        """
        pass
    def backward(self, gradient):
        pass

class Pooling:
    def __init__(self, pool_size=2):
        pass

    def forward(self, entree):
        pass
    def backward(self, gradient):
        pass
class Dense:
    def __init__(self, nentree, nsortie, lr=0.01):
        pass
    def forward(self, X):
        pass
    def backward(self, gradient):
        pass

class CNN:
    def __init__(self, lr=0.001):
        pass
    def forward(self, X):
        pass
    def backward(self, gradientscore):
        """

        :param gradientscore:
        :return:
        """
        pass
    def predict(self, X):
        pass