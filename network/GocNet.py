'''
Created by: Zhiqing Guo
Institutions: Hunan University
Email: guozhiqing@hnu.edu.cn
Copyright (c) 2020
'''
import torch
import torch.nn as nn
import network.resnet_mta as resnet_mta
import network.resnet_plain as resnet_plain
import torch.nn.functional as F

# Load model
plain_stream = resnet_plain.resnet18_plain(pretrained=False)
mta_stream = resnet_mta.resnet18_mta(pretrained=False)

# Remove the last layer of ResNet (i.e. FC layer)
plain_extract_feature = nn.Sequential(*list(plain_stream.children())[:-1])
mta_extract_feature = nn.Sequential(*list(mta_stream.children())[:-1])

# TP module
class TP_module(nn.Module):
    def __init__(self):
        super(TP_module, self).__init__()

        kernel_2 = [[0, 1, 1],
                  [-1, 0, 1],
                  [-1, -1, 0]]

        kernel = torch.FloatTensor(kernel_2).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class GocNet(nn.Module):
    def __init__(self, num_classes=None):
        print('num_classes:',num_classes)
        super(GocNet, self).__init__()
        self.num_classes = num_classes
        
        self.plain_extract_feature = plain_extract_feature
        self.mta_extract_feature = mta_extract_feature
        self.fc = nn.Linear(8192, num_classes)
        
        self.tp = TP_module()
            
    def forward(self, rgb_data):
        # TP module
        x = self.tp(rgb_data)
        
        # extract features
        output_1 = self.plain_extract_feature(x)        # TP stream
        output_2 = self.mta_extract_feature(rgb_data)   # MTA stream
        
        # feature fusion
        output = output_1 + output_2
        
        # decision
        final_out = torch.flatten(output, 1)
        final_out = self.fc(final_out)
        
        return final_out

