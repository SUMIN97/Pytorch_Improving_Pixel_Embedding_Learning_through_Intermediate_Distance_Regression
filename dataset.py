import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, rgbs_path, image_size):
        super(TestDataset, self).__init__()

        self.rgbs_path = rgbs_path
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        name = os.path.basename(self.rgbs_path[idx]).split('_')[0]
        image = Image.open(self.rgbs_path[idx])
        image = self.transform(image)

        # get original data type
        orig_dtype = image.dtype
        image_mean = torch.mean(image, dim=(-1, -2, -3))
        stddev = torch.std(image, axis=(-1, -2, -3))
        num_pixels = torch.tensor(torch.numel(image), dtype=torch.float32)
        min_stddev = torch.rsqrt(num_pixels)
        adjusted_stddev = torch.max(stddev, min_stddev)
        # normalize image
        image -= image_mean
        image = torch.div(image, adjusted_stddev)
        # make sure that image output dtype  == input dtype
        assert image.dtype == orig_dtype

        return (image, name)

    def __len__(self):
        return len(self.rgbs_path)

class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, rgbs_path, dismaps_dir, labels_dir, neighbors_dir, image_size, dismap_mode):
        super(PlantDataset, self).__init__()

        self.rgbs_path = rgbs_path
        self.dismaps_dir = dismaps_dir
        self.labels_dir = labels_dir
        self.neighbors_dir = neighbors_dir
        self.image_size = image_size

        #transforms.Resize 는 bilinear 시 antialising으로 different 할수 있
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        self.totensor = transforms.ToTensor()
        self.dismap_mode = dismap_mode


    def __getitem__(self, idx):
        image = Image.open(self.rgbs_path[idx])
        image = self.transform(image)
        image = torch.div(image - torch.mean(image),  torch.std(image))
        """
        # get original data type
        orig_dtype = image.dtype
        image_mean = torch.mean(image, dim=(-1, -2, -3))
        stddev = torch.std(image, axis=(-1, -2, -3))
        num_pixels = torch.tensor(torch.numel(image), dtype=torch.float32)
        min_stddev = torch.rsqrt(num_pixels)
        adjusted_stddev = torch.max(stddev, min_stddev)
        #normalize image
        image -= image_mean
        image = torch.div(image, adjusted_stddev)
        # make sure that image output dtype  == input dtype
        assert image.dtype == orig_dtype
        """


        name = os.path.basename(self.rgbs_path[idx]).split('_')[0]
        A_folder = self.rgbs_path[idx].split('/')[-2]
        folder = self.rgbs_path[idx].split('/')[-2]

        distmap = Image.open(os.path.join(self.dismaps_dir, A_folder,  name + '.png'))
        distmap= self.transform(distmap) * 255.0
        # distmap = (distmap - distmap.min()) / (distmap.max() - distmap.min()) * 255.0


        labelmap = Image.open(os.path.join(self.labels_dir, A_folder, name + '_label.png'))
        #labelmap 이 uint이기 때문에 transform을 거치면서 [0,1]로 normalize. 이를 255를 곱하고 int로 만들어 준다
        labelmap = (self.transform(labelmap) * 255).int()


        neighbor = np.load(os.path.join(self.neighbors_dir, folder+'_' + name + '.npy'))

        return (image, distmap, labelmap, neighbor)

    def __len__(self):
        return len(self.rgbs_path)

if __name__ == '__main__':
    rgbs_path = glob( '/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_training/A1/*_rgb.png')
    distmaps_dir = '/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/dismap/'
    labels_dir = '/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_training/'
    neighbors_dir = './neighbors'
    save_model_path = '/home/lab/ssd1/Plant/PEIS/models/v1.pth'

    trans2pil = transforms.ToPILImage()
    image_size = (512, 512)

    ds = PlantDataset(rgbs_path, distmaps_dir, labels_dir, neighbors_dir, image_size, dismap_mode='255')
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    loader = iter(loader)

    rgb, distmap, labelmap, neighbor = loader.next()

    print(torch.unique(labelmap))
    print(distmap.max())
    np_distmap = distmap[0,0].numpy()
    pil_distmap = Image.fromarray(np_distmap)
    pil_distmap.show()
    # plt.figure()
    # plt.imshow(pil_distmap)
    # plt.show()
    # print(one_hot_neighbor)

    # pil_rgb = trans2pil(rgb[0])
    # pil_rgb.show()
    # pil_dismap = trans2pil(dismap[0])
    # pil_dismap.show()
    #
    # pil_labelmap = trans2pil((labelmap[0] == 1)).type(torch.int)
    # print(labelmap[0])
    # pil_labelmap.show()



