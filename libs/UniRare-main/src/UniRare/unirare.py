
import math
import torch
from torch import nn
import cv2
import numpy as np
import torch.nn.functional as F



class RarityNetwork(nn.Module):
    def __init__(self, threshold= None):
        super(RarityNetwork, self).__init__()
        self.threshold= threshold

    @staticmethod
    def add_rarity(tensor1, tensor2):
        
        # check num of dimnesions
        assert tensor1.ndim == 3, "tensor1 must have 3 dimensions C , W , H"
        assert tensor2.ndim == 3, "tensor2 must have 3 dimensions C , W , H"

        C , W, H = tensor1.shape

        # resize to same size
        tensor2 = F.interpolate(tensor2.unsqueeze(0), size=(tensor1.shape[1], tensor1.shape[2]), mode='bilinear', align_corners=False).squeeze(0)

        tensor2 = tensor2.view(tensor2.shape[0], -1)
        tensor1 = tensor1.view(tensor1.shape[0], -1)

        # normalise both tensor
        tensor_min = tensor1.min(dim=1, keepdim=True)[0]
        tensor_max = tensor1.max(dim=1, keepdim=True)[0]
        tensor1 = (tensor1 - tensor_min) / (tensor_max - tensor_min + 1e-8)

        tensor_min = tensor2.min(dim=1, keepdim=True)[0]
        tensor_max = tensor2.max(dim=1, keepdim=True)[0]
        tensor2 = (tensor2 - tensor_min) / (tensor_max - tensor_min + 1e-8)

        # sum both tensor
        tensor = tensor1 + tensor2

        # normalise 0 and 1 
        tensor_min = tensor.min(dim=1, keepdim=True)[0]
        tensor_max = tensor.max(dim=1, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        return tensor.view(C,W,H)
    
    @staticmethod
    def prod_rarity(tensor1, tensor2):
        
        # check num of dimnesions
        assert tensor1.ndim == 3, "tensor1 must have 3 dimensions C , W , H"
        assert tensor2.ndim == 3, "tensor2 must have 3 dimensions C , W , H"

        C , W, H = tensor1.shape

        # resize to same size
        tensor2 = F.interpolate(tensor2.unsqueeze(0), size=(tensor1.shape[1], tensor1.shape[2]), mode='bilinear', align_corners=False).squeeze(0)

        tensor2 = tensor2.view(tensor2.shape[0], -1)
        tensor1 = tensor1.view(tensor1.shape[0], -1)

        # normalise both tensor
        tensor_min = tensor1.min(dim=1, keepdim=True)[0]
        tensor_max = tensor1.max(dim=1, keepdim=True)[0]
        tensor1 = (tensor1 - tensor_min) / (tensor_max - tensor_min + 1e-8)

        tensor_min = tensor2.min(dim=1, keepdim=True)[0]
        tensor_max = tensor2.max(dim=1, keepdim=True)[0]
        tensor2 = (tensor2 - tensor_min) / (tensor_max - tensor_min + 1e-8)

        # sum both tensor
        tensor = tensor1 * tensor2

        # normalise 0 and 1 
        tensor_min = tensor.min(dim=1, keepdim=True)[0]
        tensor_max = tensor.max(dim=1, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        return tensor.view(C,W,H)


    @staticmethod
    def sub_rarity(tensor1, tensor2):
        # check num of dimnesions
        assert tensor1.ndim == 3, "tensor1 must have 3 dimensions C , W , H"
        assert tensor2.ndim == 3, "tensor2 must have 3 dimensions C , W , H"

        C , W, H = tensor1.shape

        # resize to same size
        tensor2 = F.interpolate(tensor2.unsqueeze(0), size=(tensor1.shape[1], tensor1.shape[2]), mode='bilinear', align_corners=False).squeeze(0)

        tensor2 = tensor2.view(tensor2.shape[0], -1)
        tensor1 = tensor1.view(tensor1.shape[0], -1)

        # normalise both tensor
        tensor_min = tensor1.min(dim=1, keepdim=True)[0]
        tensor_max = tensor1.max(dim=1, keepdim=True)[0]
        tensor1 = (tensor1 - tensor_min) / (tensor_max - tensor_min + 1e-8)

        tensor_min = tensor2.min(dim=1, keepdim=True)[0]
        tensor_max = tensor2.max(dim=1, keepdim=True)[0]
        tensor2 = (tensor2 - tensor_min) / (tensor_max - tensor_min + 1e-8)

        # sum both tensor
        tensor = torch.abs(tensor1 - tensor2)

        # normalise 0 and 1 
        tensor_min = tensor.min(dim=1, keepdim=True)[0]
        tensor_max = tensor.max(dim=1, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        return tensor.view(C,W,H)
    
    
    @staticmethod
    def fuse_rarity(tensor1, tensor2):
        # check num of dimnesions
        assert tensor1.ndim == 3, "tensor1 must have 3 dimensions C , W , H"
        assert tensor2.ndim == 3, "tensor2 must have 3 dimensions C , W , H"

        C , W, H = tensor1.shape

        # resize to same size
        tensor2 = F.interpolate(tensor2.unsqueeze(0), size=(tensor1.shape[1], tensor1.shape[2]), mode='bilinear', align_corners=False).squeeze(0)

        tensor2 = tensor2.view(tensor2.shape[0], -1)
        tensor1 = tensor1.view(tensor1.shape[0], -1)

        # normalise both tensor
        tensor_min = tensor1.min(dim=1, keepdim=True)[0]
        tensor_max = tensor1.max(dim=1, keepdim=True)[0]
        tensor1 = (tensor1 - tensor_min) / (tensor_max - tensor_min + 1e-8)

        tensor_min = tensor2.min(dim=1, keepdim=True)[0]
        tensor_max = tensor2.max(dim=1, keepdim=True)[0]
        tensor2 = (tensor2 - tensor_min) / (tensor_max - tensor_min + 1e-8)

        # compute weights
        tensor_max = tensor1.max(dim=1, keepdim=True)[0]
        tensor_mean = tensor1.mean(dim=1, keepdim=True)
        w1 = torch.square(tensor_max - tensor_mean)
        tensor1 = w1 * tensor1

        tensor_max = tensor2.max(dim=1, keepdim=True)[0]
        tensor_mean = tensor2.mean(dim=1, keepdim=True)
        w2 = torch.square(tensor_max - tensor_mean)
        tensor2 = w2 * tensor2

        # sum both tensor
        tensor = tensor1 + tensor2

        # normalise 0 and 1 
        tensor_min = tensor.min(dim=1, keepdim=True)[0]
        tensor_max = tensor.max(dim=1, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        return tensor.view(C,W,H)

    def rarity_tensor(self, channel):
        B,C , a , b = channel.shape
        if a > 16:  # manage margins for high-level features
            channel[:,:,0:1, :] = 0
            channel[:,:,:, a - 1:a] = 0
            channel[:,:,:, 0:1] = 0
            channel[:,:,b - 1:b, :] = 0
        elif a > 28:  # manage margins for mid-level features
            channel[:,:,0:2, :] = 0
            channel[:,:,:, a - 2:a] = 0
            channel[:,:,:, 0:2] = 0
            channel[:,:,b - 2:b, :] = 0
        elif a > 50:  # manage margins for low-level features
            channel[:,:,0:3, :] = 0
            channel[:,:,:, a - 3:a] = 0
            channel[:,:,:, 0:3] = 0
            channel[:,:,b - 3:b, :] = 0
        
        channel = channel.view(B,C, -1)
        tensor_min = channel.min(dim=2, keepdim=True)[0]
        tensor_max = channel.max(dim=2, keepdim=True)[0]
        channel = (channel - tensor_min) / (tensor_max - tensor_min + 1e-8)
        channel = channel * 256

        bins = 8

        min_val, max_val = channel.min(), channel.max()
        bin_edges = torch.linspace(min_val, max_val, steps=bins + 1, device=channel.device)

        bin_indices = torch.bucketize(channel, bin_edges, right=True) - 1
        bin_indices = bin_indices.clamp(0, bins - 1)

        histograms = torch.zeros((B,C, bins), device=channel.device)
        histograms.scatter_add_(2, bin_indices, torch.ones_like(bin_indices, dtype=torch.float, device=channel.device))    

        histograms = histograms / histograms.sum(dim=2, keepdim=True)
        histograms = -torch.log(histograms + 1e-4)
        hists_idx = ((channel/256.) * (bins - 1)).long().clamp(0, bins - 1)

        dst = histograms.gather(2, hists_idx)

        tensor_min = dst.min(dim=2, keepdim=True)[0]
        tensor_max = dst.max(dim=2, keepdim=True)[0]
        dst = (dst - tensor_min) / (tensor_max - tensor_min + 1e-8)

        if self.threshold is not None:
            dst[dst < self.threshold] = 0

        map_max = dst.max(dim=2, keepdim=True)[0]
        map_mean = dst.mean(dim=2, keepdim=True)
        map_weight = (map_max - map_mean) ** 2  # Itti-like weight
        dst = dst * map_weight

        dst = torch.pow(dst, 2)
        ma = dst.max(dim=2, keepdim=True)[0]
        me = dst.mean(dim=2, keepdim=True)
        w = (ma - me) * (ma - me)
        dst = w * dst

        tensor_min = dst.min(dim=2, keepdim=True)[0]
        tensor_max = dst.max(dim=2, keepdim=True)[0]
        dst = (dst - tensor_min) / (tensor_max - tensor_min + 1e-8)
        dst = dst.view(B,C, a,b)

        if a > 18:
            dst[:,:,0:1, :] = 0
            dst[:,:,:, a - 1:a] = 0
            dst[:,:,:, 0:1] = 0
            dst[:,:,b - 1:b, :] = 0

        elif a > 28:
            dst[:,:,0:2, :] = 0
            dst[:,:,:, a - 2:a] = 0
            dst[:,:,:, 0:2] = 0
            dst[:,:,b - 2:b, :] = 0
        elif a > 50:
            dst[:,:,0:3, :] = 0
            dst[:,:,:, a - 3:a] = 0
            dst[:,:,:, 0:3] = 0
            dst[:,:,b - 3:b, :] = 0

        dst = dst.view(B,C, -1)
        dst = torch.pow(dst, 2)
        ma = dst.max(dim=2, keepdim=True)[0]
        me = dst.mean(dim=2, keepdim=True)
        w = (ma - me) * (ma - me)
        dst = w * dst

        return dst.view(B,C, a,b)

    def apply_rarity(self, layer_output, layer_ind):
        features_processed = self.rarity_tensor(layer_output[layer_ind - 1].clone())

        features_processed= features_processed.sum(dim=1)

        features_processed= F.interpolate(features_processed.clone().unsqueeze(0), (240,240), mode='bilinear', align_corners=False)

        # features_processed= F.interpolate(features_processed.clone(), (240,240), mode='bilinear', align_corners=False)

        return features_processed

    def fuse_itti_tensor(self, tensor):
        # Itti-like fusion between two maps
        B, C , W , H = tensor.shape
        tensor= tensor.view(B, C, -1)

        # Normalize 0 1
        tensor_min = tensor.min(dim=2, keepdim=True)[0]
        tensor_max = tensor.max(dim=2, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        tensor_min = tensor.min(dim=2, keepdim=True)[0]
        tensor_max = tensor.max(dim=2, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        # Compute Weights
        tensor_max = tensor.max(dim=2, keepdim=True)[0]
        tensor_mean = tensor.mean(dim=2, keepdim=True)
        w1 = torch.square(tensor_max - tensor_mean)
        tensor = w1 * tensor
        tensor= tensor.sum(dim=1)

        # normalize 0 255
        tensor_min = tensor.min(dim=1, keepdim=True)[0]
        tensor_max = tensor.max(dim=1, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
        tensor *= 255

        return tensor.view(B,W,H)
    
    def forward(
            self,
            layers_input,
            layers_index=[
                [4,5],
                [7,8],
                [10,11],
                [13,14],
                [16,17]
            ]
        ):

        assert len(layers_input) > 0, "layer_output should not be empty"
        B, C, W, H = layers_input[0].shape
        for layer in layers_input:
            assert layer.ndim == 4, "Each layer in layer_output must have the same dimensions B, C, W, H"

        # for i,layer in enumerate(layers_input):
            # print(f"Layer {i+1}: ",layer.shape)

        groups = []
        for layer_index in layers_index:
            tempo = []
            for index in layer_index:
                tempo.append(self.apply_rarity(layers_input, index))
            tempo = torch.cat(tempo, dim=1)
            # tempo = torch.stack(tempo, dim=1)
            groups.append(self.fuse_itti_tensor(tempo))
        groups= torch.stack(groups, dim=1)

        SAL = groups.sum(dim= 1)

        B,W,H = SAL.shape
        SAL = SAL.view(B , -1)

        SAL_min = SAL.min(dim=1, keepdim=True)[0]
        SAL_max = SAL.max(dim=1, keepdim=True)[0]
        SAL = (SAL - SAL_min) / (SAL_max - SAL_min + 1e-8)

        SAL = torch.exp(SAL)

        SAL_min = SAL.min(dim=1, keepdim=True)[0]
        SAL_max = SAL.max(dim=1, keepdim=True)[0]
        SAL = (SAL - SAL_min) / (SAL_max - SAL_min + 1e-8)

        SAL= SAL.view(B,W,H)

        # # Convert SAL to numpy array
        # sal_np = SAL.detach().cpu().numpy()

        # # App                                                                                                                                                                                                                                                                                                                                                                                                                                                   