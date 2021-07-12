from model2 import W_Net as Wnet

from postprocessing import *
from glob import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
from dataset import TestDataset
from skimage.filters import gaussian
import cv2


save_model_path = '/home/lab/ssd1/Plant/PEIS/models/IntraTogetherInterSplit_A1234/epoch_490_1194.7046.pth'
save_dir = os.path.join('/home/lab/ssd1/Plant/PEIS/final_imgs', save_model_path.split('/')[-2])
if os.path.isdir(save_dir) is False:
    os.mkdir(save_dir)
save_folder = os.path.join(save_dir, save_model_path.split('/')[-1].split('.')[0])
if os.path.isdir(save_folder) is False:
    os.mkdir(save_folder)


input_dim=3
Dfeature_channels = 32
E_channels = 8
image_size = (512, 512)
distmap_mode='255'
device='cuda:0'
seeds_thres = 0.3
seeds_min_dis=3
similarity_thres = 0.7

w_net = Wnet(input_dim, Dfeature_channels, E_channels)
checkpoint = torch.load(save_model_path)
w_net.load_state_dict(checkpoint['W_Net'])

#deactivate dropout
w_net.to(device)
w_net.eval()

rgbs_path = glob('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_test_data/A1/*/*_rgb.png') \
            + glob('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_test_data/A2/*/*_rgb.png') \
            + glob('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_test_data/A3/*/*_rgb.png') \
            + glob('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_test_data/A4/*/*_rgb.png')


for i in range(5):
    A_folder = os.path.join(save_folder, 'A' + str(i+1))
    if os.path.isdir(A_folder) is False:
        os.mkdir(A_folder)


test_ds = TestDataset(rgbs_path, image_size)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=False)


for idx, data in enumerate(test_loader):
    rgbs, As, names = data

    pd_distmaps, norm_embeddings = w_net(rgbs.to(device))

    bs = rgbs.shape[0]

    for i in range(bs):
        name = names[i]
        print(name)
        A = As[i]
        plant_folder = os.path.join(save_folder,A, name)
        if os.path.isdir(plant_folder) is False:
            os.mkdir(plant_folder)

        #save rgb
        rgb_path = os.path.join('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_test_data',
                                         A, name, name+'_rgb.png')
        shutil.copy(src = rgb_path,dst=plant_folder)
        img_width, img_height, _ = cv2.imread(rgb_path).shape

        pd_distmap = pd_distmaps[i].squeeze(0).cpu().detach().numpy()
        pd_distmap[pd_distmap > 255] = 255


        #gaussian
        pd_distmap = gaussian(pd_distmap, sigma=3)

        #(c,h,w) to (h,w)
        # pd_distmap = np.squeeze(pd_distmap)

        #get seeds
        seeds = get_seeds(pd_distmap)

        embedding = norm_embeddings[i].permute((1, 2, 0))  # (h, ,w c)

        embedding = smooth_emb(embedding, radius=3)

        #segment from seeds

        seg = mask_from_seeds(embedding, seeds, similarity_thres)

        #remove noise
        seg = remove_noise(seg, pd_distmap, min_size=10, min_intensity=0.1)

        #use save_indexed_png
        resize_seg = cv2.resize(seg, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        save_indexed_png(os.path.join(plant_folder, '{}_indexed_seg.png'.format(name)), resize_seg)

        """
        #save seg
        pil_seg = Image.fromarray(cv2.resize(seg, (img_width, img_height), interpolation=cv2.INTER_NEAREST))
        pil_seg.save(os.path.join(plant_folder, '{}_seg.png'.format(name)))


        #distmap 저장
        pd_distmap = pd_distmap.astype(np.uint8)
        Image.fromarray(pd_distmap).save(os.path.join(plant_folder, '{}_distmap.png'.format(name)))

        #seeds 저장
        seeds = (seeds * 255).astype(np.uint8)
        Image.fromarray(seeds).save(os.path.join(plant_folder, '{}_seeds.png'.format(name)))

        n_clus = np.max(seg)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_clus + 1)]

        # save color argmax
        argmax_color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        embedding_argmax = embedding.argmax(dim=2).cpu().detach().numpy()
        for i in range(0, n_clus + 1):
            argmax_color[embedding_argmax == i] = (np.array(colors[i][:3]) * 255).astype('int')
        argmax_color = Image.fromarray(argmax_color).resize((img_width, img_height))
        argmax_color.save(os.path.join(plant_folder, name + '_argmax_color.png'))
        
        # color segmentation 저장

        seg_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        seg = np.array(pil_seg)
        for i in range(1, n_clus + 1):
            seg_color[seg == i] = (np.array(colors[i][:3]) * 255).astype('int')

        seg_color_pil = Image.fromarray(seg_color)
        seg_color_pil.save(os.path.join(plant_folder, '{}_seg_color.png'.format(name)))
        """






