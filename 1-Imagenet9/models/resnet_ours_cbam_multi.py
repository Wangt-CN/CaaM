import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from .cbam import *
from .bam import *

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, split=False, bi_path_bn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.split = split

        if bi_path_bn:
            self.bn3 = nn.BatchNorm2d(planes)
            self.bn4 = nn.BatchNorm2d(planes)

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None


    def forward(self, x):
        if isinstance(x, list):
            x_causal, x_spurious = x[0], x[1]
            residual_causal = x_causal
            residual_spurious = x_spurious

            out_causal_ = self.conv1(x_causal)
            out_causal_= self.bn1(out_causal_)
            out_causal_ = self.relu(out_causal_)
            out_causal_ = self.conv2(out_causal_)
            out_causal_ = self.bn2(out_causal_)

            out_spurious_ = self.conv1(x_spurious)
            out_spurious_= self.bn3(out_spurious_)
            out_spurious_ = self.relu(out_spurious_)
            out_spurious_ = self.conv2(out_spurious_)
            out_spurious_ = self.bn4(out_spurious_)

            if self.downsample is not None:
                residual_causal = self.downsample(x_causal)
                residual_spurious = self.downsample(x_spurious)

            out = out_causal_ + out_spurious_ # combine the 2 feature
            assert not self.cbam is None
            out_causal = self.cbam(out)
            out_spurious = out - out_causal

            out_causal += residual_causal
            out_spurious += residual_spurious

            out_mix = out_causal + out_spurious
            out_causal = self.relu(out_causal)
            out_spurious = self.relu(out_spurious)
            out_mix = self.relu(out_mix)
            return [out_causal, out_spurious, out_mix]

        else:
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            assert not self.cbam is None
            out_causal = self.cbam(out)
            out_spurious = out - out_causal
            out_causal += residual
            out_causal = self.relu(out_causal)

            if self.split:
                # out_spurious += residual
                out_mix = out.clone()
                out_mix += residual
                out_spurious = self.relu(out_spurious)
                out_mix = self.relu(out_mix)
                return [out_causal, out_spurious, out_mix]

            else:
                return out_causal



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, att_type=None, split_layer=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        if split_layer == 1:
            self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type, split_num=2)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type, bi_path_bn=True)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type, bi_path_bn=True)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type, bi_path_bn=True)
        elif split_layer == 2:
            self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type, split_num=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type, bi_path_bn=True)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type, bi_path_bn=True)
        elif split_layer == 3:
            self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type, split_num=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type, bi_path_bn=True)
        elif split_layer == 4:
            self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type, split_num=2)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None, split_num=0, bi_path_bn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM', bi_path_bn=bi_path_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == split_num-1:
                layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM', split=True))
            else:
                layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM', bi_path_bn=bi_path_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            if isinstance(x, list):
                new_x = []
                for idx, x_ in enumerate(x):
                    x_ = self.avgpool(x_)
                    new_x.append(x_.view(x_.size(0), -1))
                return new_x
            else:
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
        else:
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class ResNet_Classifier(nn.Module):

    def __init__(self, block, num_classes=1000, bias=True):
        super(ResNet_Classifier, self).__init__()
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=bias)
        init.kaiming_normal_(self.fc.weight)
    def forward(self, x):
        x = self.fc(x)
        return x


def ResidualNet(network_type, depth, num_classes, att_type, split_layer=4):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type, split_layer=split_layer)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model


def classifier(pretrained=False, **kwargs):
    classifier_model = ResNet_Classifier(BasicBlock, **kwargs)
    return classifier_model