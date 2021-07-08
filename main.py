import os.path
from PIL import Image
import torch
from glob import glob
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
from dataset import PlantDataset
from model2 import W_Net
from training import Trainer

#Distance Regression Version

train_rgbs_path = glob('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_training/A1/*_rgb.png')
val_rgbs_path = glob('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_training/A1/plant079_rgb.png')
train_rgbs_path.remove(val_rgbs_path[0])

distmaps_dir = '/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/dismap/'
labels_dir = '/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_training/'
neighbors_dir = './neighbors'

save_model_path = '/home/lab/ssd1/Plant/PEIS/models/Model2_Bcnorm2_IntraSplitInterTogether_InputImageNormWithCombinedChannel_A1'
if os.path.isdir(save_model_path) is False:
    os.mkdir(save_model_path)

middle_imgs_path = os.path.join('./middle_imgs', os.path.basename(save_model_path))
if os.path.isdir(middle_imgs_path) is False:
    os.mkdir(middle_imgs_path)

lamda1 = 1
lamda2 = 1
input_dim=3
Dfeature_channels = 32
E_channels = 8
image_size = (512, 512)
epochs=501
dismap_mode = '255'

lr = 1e-4
device='cuda:1'
#dataset
train_ds = PlantDataset(train_rgbs_path, distmaps_dir, labels_dir, neighbors_dir, image_size, dismap_mode)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)
val_ds = PlantDataset(val_rgbs_path, distmaps_dir, labels_dir, neighbors_dir, image_size, dismap_mode)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True)

#model
wnet = W_Net(input_dim, Dfeature_channels, E_channels)


optimizer = torch.optim.Adam(wnet.parameters(), lr=lr)
scheduler = MultiStepLR(optimizer, milestones=[100,150,200,250, 300, 400, 500],gamma= 0.9)

#validation_rgb
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

validation_rgb = Image.open('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_test_data/A1/plant160/plant160_rgb.png')
validation_rgb = transform(validation_rgb).unsqueeze(0)

#trainer
trainer = Trainer(train_loader, val_loader, wnet, optimizer,scheduler, device, save_model_path, dismap_mode,
                  middle_imgs_path, E_channels)
trainer.train(epochs, lamda1, lamda2)


