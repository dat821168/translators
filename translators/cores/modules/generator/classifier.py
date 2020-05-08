import torch.nn as nn


class Classifier(nn.Module):
    """
    Fully-connected classifier
    """
    def __init__(self, in_features, out_features, init_weight=0.1):
        """
        Constructor for the Classifier.
        :param in_features: number of input features
        :param out_features: number of output features (size of vocabulary)
        :param init_weight: range for the uniform initializer
        """
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_features, out_features)
        nn.init.uniform_(self.classifier.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.classifier.bias.data, -init_weight, init_weight)

    def forward(self, x):
        """
        Execute the classifier.
        :param x: output from decoder
        """
        out = self.classifier(x)
        return out