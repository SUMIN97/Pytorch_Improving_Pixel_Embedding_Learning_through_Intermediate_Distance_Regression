import numpy as np
import torch
from skimage.measure import regionprops, label
from skimage.feature import peak_local_max
from skimage.morphology import dilation as im_dilation
from skimage.morphology import square as mor_square
from PIL import Image
import math
import torch.nn.functional as F

def get_seeds(dist_map, seeds_thres=0.7, seeds_min_dis=5):
    dist_map = np.squeeze(dist_map)  # (h,w)
    mask = peak_local_max(dist_map, min_distance=seeds_min_dis, threshold_abs=seeds_thres * dist_map.max(), indices=False)
    return mask

def smooth_emb(emb, radius):
    from scipy import ndimage
    from skimage.morphology import disk
    emb = emb.cpu().detach().numpy()
    w = disk(radius) / np.sum(disk(radius))
    for i in range(emb.shape[-1]):
        emb[:, :, i] = ndimage.convolve(emb[:, :, i], w, mode='reflect')
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    emb = torch.tensor(emb)
    return emb

def mask_from_seeds(norm_1_embedding, seeds, similarity_thres):
    """
    :param norm_1_embedding: (h, w, c)
    :param seeds:
    :param similarity_thres:
    :return:
    """
    seeds = label(seeds).astype('uint8')
    props = regionprops(seeds)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]

        emb_mean = torch.mean(norm_1_embedding[row, col], dim=0)
        emb_mean = F.normalize(emb_mean, p=2, dim=0)
        mean[p.label] = emb_mean


    while True:
        dilated = im_dilation(seeds, mor_square(3))
        front_r, front_c = np.nonzero(seeds != dilated)
        #numpy
        similarity = [torch.dot(norm_1_embedding[r, c, :], mean[dilated[r, c]]) for r, c in zip(front_r, front_c)]
        add_ind = torch.stack([s > similarity_thres for s in similarity], dim=0).cpu().detach().numpy()


        # print(add_ind.sum())
        if np.sum(add_ind) == 0: break

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seeds

def remove_noise(l_map, d_map, min_size=10, min_intensity=0.1):
    max_instensity = d_map.max()
    props = regionprops(l_map, intensity_image=d_map)
    for p in props:
        if p.area < min_size:
            l_map[l_map==p.label] = 0
        if p.mean_intensity/max_instensity < min_intensity:
            l_map[l_map==p.label] = 0
    return label(l_map)

P = [252, 233, 79, 114, 159, 207, 239, 41, 41, 173, 127, 168, 138, 226, 52,
     233, 185, 110, 252, 175, 62, 211, 215, 207, 196, 160, 0, 32, 74, 135, 164, 0, 0,
     92, 53, 102, 78, 154, 6, 143, 89, 2, 206, 92, 0, 136, 138, 133, 237, 212, 0, 52,
     101, 164, 204, 0, 0, 117, 80, 123, 115, 210, 22, 193, 125, 17, 245, 121, 0, 186,
     189, 182, 85, 87, 83, 46, 52, 54, 238, 238, 236, 0, 0, 10, 252, 233, 89, 114, 159,
     217, 239, 41, 51, 173, 127, 178, 138, 226, 62, 233, 185, 120, 252, 175, 72, 211, 215,
     217, 196, 160, 10, 32, 74, 145, 164, 0, 10, 92, 53, 112, 78, 154, 16, 143, 89, 12,
     206, 92, 10, 136, 138, 143, 237, 212, 10, 52, 101, 174, 204, 0, 10, 117, 80, 133, 115,
     210, 32, 193, 125, 27, 245, 121, 10, 186, 189, 192, 85, 87, 93, 46, 52, 64, 238, 238, 246]

P = P * math.floor(255*3/len(P))
l = int(255 - len(P)/3)
P = P + P[3:(l+1)*3]
P = [0,0,0] + P

def save_indexed_png(fname, label_map, palette=P):
    label_map = np.squeeze(label_map.astype(np.uint8))
    im = Image.fromarray(label_map, 'P')
    im.putpalette(palette)
    im.save(fname, 'PNG')



if __name__ == '__main__':
    distmap = np.array(Image.open('/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/dismap/A1/plant001.png'))
    seeds = get_seeds(distmap)
    # pos = np.transpose(np.nonzero(seeds))
    pil_seeds = Image.fromarray(seeds)
    pil_seeds.show()

