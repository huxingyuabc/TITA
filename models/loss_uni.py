import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from models.loss_ssim import ssim


class fusion_loss_general(nn.Module):
    def __init__(self):
        super(fusion_loss_general, self).__init__()
        self.vgg = VGGFeatureExtractor()
        self.L_MSE = L_MSE()
        self.L_SSIM = L_SSIM()

    def features_grad(self, features):
        kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8],
                               [1 / 8, -1, 1 / 8],
                               [1 / 8, 1 / 8, 1 / 8]], dtype=torch.float32).to(features.device)
        n, c, h, w = features.shape
        kernel = kernel.repeat(c, 1, 1, 1)
        # kernel = kernel.unsqueeze(1)

        fgs = F.conv2d(features, kernel, padding=1, groups=c)

        return fgs

    def get_weights(self, image_A, image_B, c=3e3):
        S1_FEAS = self.vgg(torch.cat([image_A, image_A, image_A], dim=1))
        S2_FEAS = self.vgg(torch.cat([image_B, image_B, image_B], dim=1))

        ws1_list = []
        ws2_list = []

        for i in range(len(S1_FEAS)):
            m1 = torch.mean(self.features_grad(S1_FEAS[i]) ** 2, dim=[1, 2, 3])
            m2 = torch.mean(self.features_grad(S2_FEAS[i]) ** 2, dim=[1, 2, 3])

            ws1_list.append(m1.unsqueeze(-1))
            ws2_list.append(m2.unsqueeze(-1))

        ws1 = torch.cat(ws1_list, dim=-1)
        ws2 = torch.cat(ws2_list, dim=-1)

        s1 = torch.mean(ws1, dim=-1) / c
        s2 = torch.mean(ws2, dim=-1) / c
        weights = torch.softmax(torch.cat([s1.unsqueeze(-1), s2.unsqueeze(-1)], dim=-1), dim=-1)
        return weights[:, 0], weights[:, 1]

    def forward(self, image_A, image_B, image_fused, weight_A, weight_B):
        loss_l1 = 20 * self.L_MSE(image_A, image_B, image_fused, weight_A, weight_B)
        loss_SSIM = self.L_SSIM(image_A, image_B, image_fused, weight_A, weight_B)
        fusion_loss = loss_l1 + loss_SSIM
        return fusion_loss, loss_l1, loss_SSIM


def Fro_LOSS(batchimg):
    n, c, h, w = batchimg.shape
    fro_norm = (torch.norm(batchimg, dim=[2, 3], p='fro')) ** 2 / (h * w)
    return torch.mean(fro_norm, dim=1)


class L_MSE(nn.Module):
    def __init__(self):
        super(L_MSE, self).__init__()

    def forward(self, image_A, image_B, image_fused, weight_A, weight_B):
        mse_A = Fro_LOSS(image_A - image_fused)
        mse_B = Fro_LOSS(image_B - image_fused)
        Loss_MSE = weight_A * mse_A + weight_B * mse_B
        return torch.mean(Loss_MSE)


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()

    def forward(self, image_A, image_B, image_fused, weight_A, weight_B):
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return 1. - torch.mean(Loss_SSIM)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2, 7, 16, 25, 34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
		use_input_norm: If True, x: [0, 1] --> (x - mean) / std
		use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
		'''
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer) - 1):
                self.features.add_module('child' + str(i), nn.Sequential(
                    *list(model.features.children())[(feature_layer[i] + 1):(feature_layer[i + 1] + 1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)
