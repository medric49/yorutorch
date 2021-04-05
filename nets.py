from torch import nn
import torch


class GenericNet(nn.Module):
    """Defines a generic neural network model. This class contains a required attribute 'network'.
    This attribute is the neural network which will be used to make forwarding processes.
    """

    def __init__(self, network):
        super(GenericNet, self).__init__()
        self.network = network

    def forward(self, x):
        if self.network is not None:
            x = self.network(x)
        return x

    def load(self, load_file):
        """Loads the parameters' state of the model from the file 'load_file'"""
        self.load_state_dict(torch.load(load_file))

    def save(self, save_file):
        """Saves the parameters' state of the model into the file 'save_file'"""
        torch.save(self.state_dict(), save_file)


class ClassifierNet(GenericNet):

    def __init__(self, network, classifier):
        super(ClassifierNet, self).__init__(network=network)
        self.classifier = classifier
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = super(ClassifierNet, self).forward(x)
        if self.classifier is not None:
            x = self.flatten(x)
            x = self.classifier(x)
        return x


