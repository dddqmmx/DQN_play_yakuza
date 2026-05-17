import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from config import NETWORK_CONFIG


def cfg_get(config, key, default=None):
    return (config or NETWORK_CONFIG).get(key, default)

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration (Factorized Gaussian Noise)"""
    def __init__(self, in_features, out_features, std_init=NETWORK_CONFIG.get('noisy_std', 0.5)):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class SpatialAttention(nn.Module):
    """空间注意力模块，增强3D位置感知"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (通道注意力)"""
    def __init__(self, channel, reduction=NETWORK_CONFIG.get('se_reduction', 8)):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, config=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=cfg_get(config, 'se_reduction', 8))
        self.sa = SpatialAttention()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.sa(self.se(out)) # 双重注意力
        out += self.shortcut(x)
        return F.relu(out)

class OptimizedNet(nn.Module):
    """增强版架构：Spatio-Temporal Conv3D + Dual Attention + Dueling"""
    def __init__(self, num_actions, input_channels=4):
        super().__init__()
        self.use_recurrent = NETWORK_CONFIG.get('use_recurrent', True)
        self.use_noisy = NETWORK_CONFIG.get('use_noisy', True)

        # 1. 3D 卷积层：提取 4 帧之间的时空演变特征 (3D空间感)
        # input shape: (batch, 1, 4, 160, 160) -> (batch, channel, time, h, w)
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # 2. 2D Backbone (处理融合后的特征)
        self.backbone = nn.Sequential(
            ResidualBlock(32 * 4, 64, stride=2), # 4是时间维度的融合
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.health_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU(inplace=True))
        fusion_dim = 256 + 32
        
        hidden_size = NETWORK_CONFIG.get('dueling_hidden_size', 256)
        
        if self.use_recurrent:
            self.recurrent = nn.GRU(fusion_dim, NETWORK_CONFIG['recurrent_hidden_size'], batch_first=True)
            head_input_dim = NETWORK_CONFIG['recurrent_hidden_size']
        else:
            self.feature_layer = nn.Sequential(
                nn.Linear(fusion_dim, hidden_size),
                nn.ReLU(inplace=True)
            )
            head_input_dim = hidden_size

        linear_layer = NoisyLinear if self.use_noisy else nn.Linear
        self.value_head = linear_layer(head_input_dim, 1)
        self.advantage_head = linear_layer(head_input_dim, num_actions)

    def forward(self, x, boss_health, self_health, hidden=None):
        # x shape: (batch, 4, 160, 160)
        batch_size = x.size(0)
        
        # 转换为 3D 卷积输入: (batch, 1, time=4, h, w)
        x_3d = x.unsqueeze(1)
        feat_3d = self.conv3d(x_3d) # (batch, 32, 4, 40, 40)
        
        # 融合时间维度到通道维度
        feat_2d = feat_3d.view(batch_size, 32 * 4, 40, 40)
        
        visual_feat = self.backbone(feat_2d).view(batch_size, -1)
        
        health = torch.cat([boss_health, self_health], dim=1)
        health_feat = self.health_fc(health)
        
        combined = torch.cat([visual_feat, health_feat], dim=1)
        
        if self.use_recurrent:
            out, next_hidden = self.recurrent(combined.unsqueeze(1), hidden)
            feat = out.squeeze(1)
        else:
            feat = self.feature_layer(combined)
            next_hidden = None
            
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        if self.use_recurrent:
            return q_values, next_hidden
        return q_values

    def reset_noise(self):
        if self.use_noisy:
            self.value_head.reset_noise()
            self.advantage_head.reset_noise()

class ProNet(nn.Module):
    """Pro版架构：FPN + Distributional RL (QR-DQN) + Aux Tasks"""
    def __init__(self, num_actions, input_channels=4, config=None):
        super().__init__()
        config = config or NETWORK_CONFIG
        self.num_actions = num_actions
        self.use_recurrent = config.get('use_recurrent', True)
        self.use_noisy = config.get('use_noisy', True)
        self.num_quantiles = config.get('num_quantiles', 51)
        self.use_fpn = config.get('use_fpn', True)

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(32 * 4, 64, stride=2, config=config)
        self.layer2 = ResidualBlock(64, 128, stride=2, config=config)
        self.layer3 = ResidualBlock(128, 256, stride=2, config=config)

        if self.use_fpn:
            self.latlayer1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
            self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.health_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU(inplace=True))
        fusion_dim = 256 + 32
        
        hidden_size = config.get('dueling_hidden_size', 1024)
        
        if self.use_recurrent:
            recurrent_hidden_size = config['recurrent_hidden_size']
            self.recurrent = nn.GRU(fusion_dim, recurrent_hidden_size, batch_first=True)
            self.feature_layer = nn.Sequential(
                nn.Linear(recurrent_hidden_size, hidden_size),
                nn.ReLU(inplace=True)
            )
        else:
            self.feature_layer = nn.Sequential(
                nn.Linear(fusion_dim, hidden_size),
                nn.ReLU(inplace=True)
            )

        def linear_layer(in_features, out_features):
            if self.use_noisy:
                return NoisyLinear(in_features, out_features, std_init=config.get('noisy_std', 0.5))
            return nn.Linear(in_features, out_features)
        self.value_head = linear_layer(hidden_size, self.num_quantiles)
        self.advantage_head = linear_layer(hidden_size, num_actions * self.num_quantiles)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x, boss_health, self_health, hidden=None):
        batch_size = x.size(0)
        x_3d = x.unsqueeze(1)
        feat_3d = self.conv3d(x_3d)
        feat_2d = feat_3d.view(batch_size, 32 * 4, feat_3d.size(3), feat_3d.size(4))
        
        c1 = self.layer1(feat_2d)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        if self.use_fpn:
            p3 = c3
            p2 = self._upsample_add(p3, self.latlayer1(c2))
            p2 = self.smooth1(p2)
            p1 = self._upsample_add(p2, self.latlayer2(c1))
            p1 = self.smooth2(p1)
            visual_feat = self.pool(p1).view(batch_size, -1)
        else:
            visual_feat = self.pool(c3).view(batch_size, -1)
            
        health = torch.cat([boss_health, self_health], dim=1)
        health_feat = self.health_fc(health)
        combined = torch.cat([visual_feat, health_feat], dim=1)
        
        if self.use_recurrent:
            out, next_hidden = self.recurrent(combined.unsqueeze(1), hidden)
            feat = out.squeeze(1)
            feat = self.feature_layer(feat)
        else:
            feat = self.feature_layer(combined)
            next_hidden = None
            
        value = self.value_head(feat).view(batch_size, 1, self.num_quantiles)
        advantage = self.advantage_head(feat).view(batch_size, self.num_actions, self.num_quantiles)
        
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        if self.use_recurrent:
            return q_dist, next_hidden
        return q_dist

    def reset_noise(self):
        if self.use_noisy:
            self.value_head.reset_noise()
            self.advantage_head.reset_noise()

