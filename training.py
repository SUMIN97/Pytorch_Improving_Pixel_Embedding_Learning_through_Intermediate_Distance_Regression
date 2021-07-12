from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from postprocessing import *
import os
import torch.nn as nn


class Trainer():
    def __init__(self, train_loader, val_loader, wnet, wnet_optimizer, scheduler, device, save_model_path, dismap_mode,
                 middle_imgs_path, E_channels):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wnet = wnet

        self.wnet_optimizer = wnet_optimizer
        self.scheduler = scheduler

        self.device = device
        self.save_model_path = save_model_path
        self.dismap_mode = dismap_mode
        self.middle_imgs_path = middle_imgs_path
        self.E_channels = E_channels

        self.DLoss = nn.MSELoss(reduction='mean')
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        self.cos_sim_dim0 = torch.nn.CosineSimilarity(dim=0)
        self.criterion = nn.CrossEntropyLoss()

        self.wnet.to(self.device)

        self.topilimage = ToPILImage()

        self.seeds_thres = 0.7
        self.seeds_min_dis = 10

        self.relu = torch.nn.ReLU()

    def train(self, epochs, lamda1, lamda2):
        epoch_min_loss = 1e100000
        print(epochs)
        for epoch in range(epochs):
            print('Epoch: ', epoch)
            epoch_sum_loss = 0.0
            for idx, data in enumerate(self.train_loader):
                rgb, distmap, labelmap, neighbor = data

                bs = rgb.size()[0]
                rgb = rgb.to(self.device)
                distmap = distmap.to(self.device)

                self.wnet.zero_grad()
                self.wnet_optimizer.zero_grad()

                pd_distmap, norm_embedding = self.wnet(rgb)

                d_loss = torch.square(pd_distmap - distmap)
                d_loss = torch.mean(d_loss)

                b_inter, b_intra = 0.0, 0.0

                for i in range(bs):
                    means, intra = self.get_means_split_intra(labelmap[i], norm_embedding[i])
                    inter = self.get_inter(means, neighbor[i], self.E_channels)

                    b_intra += intra
                    b_inter += inter


                if epoch < 3:
                    e_loss = b_inter / bs
                else:
                    e_loss = (b_intra + b_inter) / bs


                # loss = d_loss
                loss = d_loss * lamda1 + e_loss * lamda2
                epoch_sum_loss += loss
                loss.backward()
                self.wnet_optimizer.step()

                if idx % 10 == 0:
                    print(torch.argmax(means, dim=1))
                    print('DLoss : ', d_loss.item())
                    print('B Inter : ', b_inter.item())
                    print('B Intra :', b_intra.item())


            # self.scheduler.step()
            self.validation(epoch)

            if epoch_sum_loss < epoch_min_loss:
                epoch_min_loss = epoch_sum_loss

                torch.save({
                    'W_Net': self.wnet.state_dict(),
                    'optimizer': self.wnet_optimizer.state_dict()
                }, os.path.join(self.save_model_path, 'epoch_{}_{:.4f}.pth'.format(epoch, epoch_min_loss)))

    def get_means_and_together_intra(self, labelmap, embedding):
        max_label = int(labelmap.unique().max().item())
        means = []
        labels_intra_sum = 0.0

        embedding_flat = embedding.permute(1, 2, 0).view(-1, self.E_channels)

        #label 이 0 인 bg 경우도 포함
        for label in range(0, max_label +1):
            mask = (labelmap == label).flatten()
            count = mask.sum()

            mask_embedding = embedding_flat[mask, :]
            mean = torch.sum(mask_embedding, dim=0) / count
            means.append(mean)

        means = torch.stack(means, dim=0)
        means = F.normalize(means, p=2, dim=1)
        labelmap_flat = labelmap.flatten().long()
        means_expand = means[labelmap_flat]
        #bio
        loss_inner = torch.mean(1 - self.cos_sim(means_expand, embedding_flat))

        return means, loss_inner

    def get_means_split_intra(self, labelmap, embedding):
        max_label = int(labelmap.unique().max().item())
        norm_means = []
        labels_intra_sum = 0.0

        embedding_flat = embedding.view(self.E_channels, -1)

        # label 이 0 인 bg 경우도 포함
        for label in range(0, max_label + 1):
            mask = (labelmap == label).flatten()
            count = mask.sum()

            mask_embedding = embedding_flat[:, mask]
            mean = torch.sum(mask_embedding, dim=1) / count

            # l2_norm
            norm_mean = F.normalize(mean, p=2, dim=0)
            norm_means.append(norm_mean)

            norm_mean_repeat = norm_mean.unsqueeze(dim=1).expand(self.E_channels, count)
            intra = 1 - self.cos_sim_dim0(norm_mean_repeat, mask_embedding)
            labels_intra_sum += intra.sum() / count

        #distance regression 방식
        intra_mean = labels_intra_sum / (1 + max_label)
        norm_means = torch.stack(norm_means, dim=0)

        return norm_means, intra_mean

    def get_inter(self, means, neighbor, n_emb):
        bg_include_n_labels = len(means)

        main_means = means.unsqueeze(1).expand(bg_include_n_labels, bg_include_n_labels, n_emb)
        neighbor_means = main_means.clone().permute(1, 0, 2)
        main_means = main_means.reshape(-1, n_emb)
        neighbor_means = neighbor_means.reshape(-1, n_emb)

        inter = self.cos_sim(neighbor_means, main_means).view(bg_include_n_labels, bg_include_n_labels).abs()

        #local neighbor
        inter_mask = torch.zeros(bg_include_n_labels, bg_include_n_labels, dtype=torch.float)

        for main_label in range(1, bg_include_n_labels):
            for nei_label in neighbor[main_label -1]:
                if nei_label == 0: break

                inter_mask[main_label][nei_label] = 1.0

        inter_mask[0] = torch.ones(bg_include_n_labels)
        inter_mask[:, 0] = torch.ones(bg_include_n_labels)
        inter_mask[0][0] = 0.0

        inter_mask = inter_mask.to(self.device)

        inter_mean = 0.0
        inter_mean = torch.sum(inter * inter_mask) / torch.sum(inter_mask)
        # for i in range(bg_include_n_labels):
        #     inter_mean += torch.sum(inter[i] * inter_mask[i]) / torch.sum(inter_mask[i])
        # inter_mean /= bg_include_n_labels
        return inter_mean

    def validation(self, epoch):
        for idx, data in enumerate(self.val_loader):
            rgb, distmap, labelmap, neighbor = data
            neighbor = neighbor.numpy()
            bs = rgb.size()[0]

            rgb = rgb.to(self.device)
            distmap = distmap.to(self.device)

            self.wnet.zero_grad()
            self.wnet_optimizer.zero_grad()

            pd_distmap, norm_embedding = self.wnet(rgb)

            """
            d_loss = self.build_dist_loss(pd_distmap, distmap)
            e_loss = self.build_embedding_loss(norm_embedding, labelmap, neighbor, include_bg=True)
            print('DLoss : ', d_loss.item())
            print('ELoss : ', e_loss.item())

            """

            d_loss = torch.mean(torch.square(pd_distmap - distmap))

            b_inter, b_intra = 0.0, 0.0

            for i in range(bs):
                means, mean_intra = self.get_means_split_intra(labelmap[i], norm_embedding[i])
                mean_inter = self.get_inter(means, neighbor[i], self.E_channels)

                b_intra += mean_intra
                b_inter += mean_inter

            b_intra /= bs
            b_inter /= bs

            print('Val DLoss : ', d_loss.item())
            print('Val Inter : ', b_inter.item(), 'Val Intra: ', b_intra.item())


            if epoch % 10 == 0:
                # dismap 저장

                pd_distmap = pd_distmap.cpu().detach().numpy()[0, 0]
                pd_distmap[pd_distmap > 255.0] = 255.0
                pd_distmap = pd_distmap.astype(np.uint8)

                pil_pd_distmap = Image.fromarray(pd_distmap)
                pil_pd_distmap.save(os.path.join(self.middle_imgs_path, 'distmap_{}.png'.format(epoch)))

                if epoch < 10: continue
                seeds = get_seeds(pd_distmap)

                Image.fromarray(seeds).save(os.path.join(self.middle_imgs_path, 'seeds_{}.png'.format(epoch)))

                bc_1_norm_embedding = norm_embedding[0].permute(1, 2, 0)
                seg = mask_from_seeds(bc_1_norm_embedding, seeds, 0.7).astype(np.uint8)
                Image.fromarray(seg).save(os.path.join(self.middle_imgs_path, 'seg_{}.png'.format(epoch)))

                # color segmentation 저장
                n_clus = np.max(seg)
                colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_clus + 1)]
                seg_color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

                for i in range(1, n_clus + 1):
                    seg_color[seg == i] = (np.array(colors[i][:3]) * 255).astype('int')

                seg_color_pil = Image.fromarray(seg_color)
                seg_color_pil.save(os.path.join(self.middle_imgs_path, 'seg_color_{}.png'.format(epoch)))






