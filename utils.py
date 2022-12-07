import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from hyperparameters import dataroot, image_size, batch_size, dataset_size, workers, ngpu
from generator_model import Generator
from discriminator_model import Discriminator

#
def dataloader_init(reduce=False):
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    print("Dataset ready")

    if reduce:
        dataset.samples = dataset.samples[:dataset_size]
        print("Reduced dataset samples")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    print("Dataloader ready")

    return dataloader

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

#
def generator_init(device):
    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)
    # print(netG)
    print("Generator ready")
    return netG

#
def discriminator_init(device):
    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)
    # print(netD)
    print("Discriminator ready")
    return netD
