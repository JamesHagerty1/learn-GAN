import torch.nn as nn

# randomly init model weights from a Normal distribution where mean=0, stdev=0.02
# used by both the Discriminator and the Generator
# reinits all convolutional, convolutional-transpose, and batch normalization layers
def weights_init(m):
    """
    m: initial model
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
