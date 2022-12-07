import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from hyperparameters import ngpu, batch_size, lr, beta1, nz, num_epochs
from generator_model import Generator
from discriminator_model import Discriminator
from utils import dataloader_init, generator_init, discriminator_init

def main():
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    dataloader = dataloader_init()
    netG = generator_init(device)
    netD = discriminator_init(device)

    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, nz, 1, 1, device=device) # Used for progress checking generations
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
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                print("Generating progress images...")
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img = vutils.make_grid(fake, padding=2, normalize=True) # can be appended to img_list faster
                disk_img = np.transpose(img, (1,2,0)).numpy()
                plt.imsave(f"generated_data/gen{epoch}-{iters}.jpg", disk_img)

            iters += 1
    print("Finished training")

if __name__ == "__main__":
    main()
