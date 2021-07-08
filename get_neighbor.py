import os
import numpy as np
import cv2
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

def get_neighbor_by_distance(label_map, distance=10, max_neighbor=32):

    label_map = label_map.copy()

    def _adjust_size(x):
        if len(x) >= max_neighbor:
            return x[0:max_neighbor]
        else:
            return np.pad(x, (0, max_neighbor-len(x)), 'constant',  constant_values=(0, 0))

    unique = np.unique(label_map)
    assert unique[0] == 0
    # only one object
    if len(unique) <= 2:
        return None

    # neighbor_indice = np.zeros((len(unique)-1, max_neighbor))
    neighbor_indice = np.zeros((max_neighbor, max_neighbor))
    label_flat = label_map.reshape((-1))

    #타원모양
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance * 2 + 1, distance * 2 + 1))
    kernel = np.ones((distance * 2 + 1, distance * 2 + 1))

    for i, label in enumerate(unique[1:]):
        assert i+1 == label
        mask = label_map == label
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).reshape((-1))
        neighbor_pixel_ind = np.logical_and(dilated_mask > 0, label_flat != 0)
        neighbor_pixel_ind = np.logical_and(neighbor_pixel_ind, label_flat != label)
        neighbors = np.unique(label_flat[neighbor_pixel_ind])
        neighbor_indice[i,:] = _adjust_size(neighbors)

    return neighbor_indice.astype(np.int32)

dir = '/home/lab/ssd1/Plant/Improving_Pixel_Embedding_Learning/data/raw/CVPPP2017_LSC_training/'
folder = 'A2'
label_maps_paths = glob(os.path.join(dir, folder, '*_label.png'))
distance = 10
max_neighbor = 32

final_mapping_max = 0
for label_path in label_maps_paths:
    name = os.path.basename(label_path).split('_')[0]

    #neighbor npy 저장
    label_map = np.array(Image.open(label_path))
    neighbor = get_neighbor_by_distance(label_map, distance, max_neighbor)
    # neighbor.save('./neighbors/{}_{}.npy'.format(folder, name))


    #neighbor map 저장
    n_ins = neighbor.max()
    mapping = {}
    possible_color = np.ones(n_ins)

    for main_label in range(1, 32):
        if main_label in mapping.keys():continue

        possible_color = np.ones(n_ins)

        for nei_label in neighbor[main_label - 1]:
            if nei_label == 0: break

            if nei_label in mapping.keys():
                possible_color[mapping[nei_label]] = False

                for nei_nei_label in neighbor[nei_label]:
                    if nei_nei_label in mapping.keys():
                        possible_color[mapping[nei_label]] = False

            for i in range(len(possible_color)):
                if possible_color[i] == True:
                    mapping[main_label] = i
                    break

    nmap = np.zeros(label_map.shape)
    for i in range(1, n_ins + 1):
        nmap[label_map == i] = mapping[i] + 1

    pil_nmap = Image.fromarray(nmap.astype(np.uint8))
    save_path =  './png_neighbors/' + (folder + '_' + name + '.png')
    # pil_nmap.save(save_path)

    #최대로 필요한 neighbor 수 계산
    mapping_max = np.array(list(mapping.values())).max()
    print(name, mapping_max)
    if final_mapping_max < mapping_max:
        final_mapping_max = mapping_max

    #neighbor map_color 저장
    h, w = label_map.shape
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, mapping_max + 2)]
    nmap_color = np.zeros((h, w, 3),  dtype=np.uint8)

    for i in range(0, mapping_max + 2):
        nmap_color[nmap == i] = (np.array(colors[i][:3]) * 255).astype('int')

    nmap_color_pil = Image.fromarray(nmap_color)
    nmap_color_pil.save('./png_neighbors/' + (folder + '_' + name + '_color.png'))




print(final_mapping_max)


