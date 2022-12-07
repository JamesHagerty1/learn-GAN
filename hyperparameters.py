dataroot = "data/celeba/" 
workers = 2 # num dataloader workers
batch_size = 128 
dataset_size = batch_size * 100 # train on data[:dataset_size] images (if in utils I choose to reduce dataset size)           
image_size = 64
nc = 3 # num channels of training images
nz = 100 # z latent vector size (generator input)
ngf = 64 # generator feature maps size
ndf = 64 # discriminator feature maps size
num_epochs = 5
lr = 0.0002
beta1 = 0.5 # for Adam optimizers
ngpu = 1 # num of GPUs to use (if available)
