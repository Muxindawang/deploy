import torch



class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * 2
        x.add_(0)
        x = x.view(-1)
        if x[0] > 1:
            return x[0]
        else:
            return x[1]

