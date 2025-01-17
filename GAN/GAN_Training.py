import os
import cv2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image

'''def extract_validation_set(input_dir, output_dir, validation_split=0.2):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the input directory
    images = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Split the data into training and validation
    train_images, val_images = train_test_split(images, test_size=validation_split, random_state=42)

    # Move validation images to the output directory
    for image in val_images:
        shutil.move(os.path.join(input_dir, image), os.path.join(output_dir, image))

    print(f"Moved {len(val_images)} images to the validation directory: {output_dir}")
    print(f"Remaining images in the training directory: {len(train_images)}")

# Example usage
input_directory = 'C:\\IITI\\Notebooks\\animefacedataset\\images'
output_directory = 'C:\\IITI\\AIML\\Task2_Dataset\\images'
extract_validation_set(input_directory, output_directory, validation_split=0.2)'''

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

num_epochs = 15
lr = 0.001
batch_size = 128
image_size = 64
latent_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
train_dir = '/kaggle/input/animefacedataset'
train_ds = ImageFolder(train_dir, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)
    ]))
train_dl=DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
   
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
    
train_dl = DeviceDataLoader(train_dl, device)

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model= nn.Sequential(
            # in: 3 x 64 x 64

            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1
            
            nn.Flatten(),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_size=latent_size
        self.model = nn.Sequential(
             # in: latent_size x 1 x 1
             
             nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(512),
             nn.ReLU(True),
             # out: 512 x 4 x 4

             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(256),
             nn.ReLU(True),
             # out: 256 x 8 x 8

             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(128),
             nn.ReLU(True),
             # out: 128 x 16 x 16

             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
             nn.BatchNorm2d(64),
             nn.ReLU(True),
             # out: 64 x 32 x 32

             nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
             nn.Tanh()
             # out: 3 x 64 x 64
)

    def forward(self, z):
        return self.model(z)
    
generator = Generator(latent_size)
generator = to_device(generator, device)
discriminator = Discriminator()
discriminator = to_device(discriminator, device)

def train_gan(real_images, opt_d, opt_g):
    # Train the Discriminator
    opt_d.zero_grad()   # Clear discriminator gradients

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    d_loss = real_loss + fake_loss
    d_loss.backward()
    opt_d.step()

    # Train the Generator

    opt_g.zero_grad()    # Clear generator gradients

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)  # Generator wants discriminator to output 1 for fakes
    g_loss = F.binary_cross_entropy(preds, targets)

    # Update generator weights
    g_loss.backward()
    opt_g.step()
    return d_loss.item(), g_loss.item(), real_score, fake_score

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images*stats[1][0] + stats[0][0], os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

def train_model(num_epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for real_images,_ in tqdm(train_dl):
            real_images = real_images.to(device)
            d_loss, g_loss, real_score, fake_score = train_gan(real_images, opt_d, opt_g)

        losses_g.append(g_loss)
        losses_d.append(d_loss)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        print("Epoch [{}/{}] || g_loss: {:.4f}, d_loss: {:.4f} || real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, num_epochs, g_loss, d_loss, real_score, fake_score))
    
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

run_gan=train_model(num_epochs,lr)
losses_g, losses_d, real_scores, fake_scores = run_gan

# Create a figure with two subplots side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Plot the losses on the first subplot
ax[0].plot(losses_d, '-', label='Discriminator')
ax[0].plot(losses_g, '-', label='Generator')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend()
ax[0].set_title('Losses')

# Plot the scores on the second subplot
ax[1].plot(real_scores, '-', label='Real')
ax[1].plot(fake_scores, '-', label='Fake')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('score')
ax[1].legend()
ax[1].set_title('Scores')


plt.tight_layout()
plt.show()
