import torch
import torch.nn as nn
import torch.optim as optim

class SharedLayerNetwork(nn.Module):
    def __init__(self, text_space, internal_space, image_space):
        super(SharedLayerNetwork, self).__init__()
        self.text_layers = nn.Sequential(
            nn.Linear(text_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, internal_space),
        )
        self.image_layers = nn.Sequential(
            nn.Linear(image_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, internal_space),
        )
        self.shared_layers = nn.Sequential(
            nn.Linear(internal_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, image_space),
        )
        self.input_mode = None
    def forward(self, x, input_mode=None):
        if input_mode is None: input_mode = self.input_mode
        if input_mode is None: return
        elif input_mode == "text":
            return self.shared_layers(self.text_layers(x))
        else:
            return self.shared_layers(self.image_layers(x))


