import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from hyperparameters import ngpu, dataroot, image_size, batch_size, dataset_size, workers, lr, beta1, nz, num_epochs
from generator_model import Generator
from discriminator_model import Discriminator
from weights import weights_init

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
print("Dataset ready")

dataset.samples = dataset.samples[:dataset_size]
print("Reduced dataset samples")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
print("Dataloader ready")

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

criterion = nn.BCELoss()
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Progress checking
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
img_list = []
iters = 0

print("Starting training")
for epoch in range(num_epochs):
    print("Epoch ", epoch)
    for i, data in enumerate(dataloader, 0):
        
        # D training
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        
        optimizerD.step()

        # G training
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        # Sanity check, currently per epoch
        if (i == len(dataloader)-1):
            print("Generating progress images grid...")
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img = vutils.make_grid(fake, padding=2, normalize=True) # can be appended to img_list faster
            disk_img = np.transpose(img, (1,2,0)).numpy()
            plt.imsave(f"generated_data/gen{epoch}.jpg", disk_img)

        iters += 1
print("Finished training")
