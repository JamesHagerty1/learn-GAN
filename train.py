import torch
import torch.nn as nn

from hyperparameters import ngpu
from generator_model import Generator
from discriminator_model import Discriminator
from weights import weights_init

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
# print(netG)
print("Generator ready")


netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
# print(netD)
print("Discriminator ready")
