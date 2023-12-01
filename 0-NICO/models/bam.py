import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """
    Flatten operation for the input tensor.

    Parameters:
    - x (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Flattened tensor.
    """
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelGate(nn.Module):
    """
    Bottleneck Attention Module (BAM) - Channel Gate.

    Parameters:
    - gate_channel (int): Number of input channels for the gate.
    - reduction_ratio (int, optional): Reduction ratio for the intermediate channels. Default is 16.
    - num_layers (int, optional): Number of linear layers in the gate. Default is 1.
    """
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()

        # Define the components of the channel gate
        self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )

        # Define the number of channels for each layer in the gate
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        # Create linear layers with batch normalization and ReLU activation
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )

        # Final linear layer without activation
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        # Apply global average pooling to the input tensor
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )

        # Apply the channel gate to the averaged features and expand the dimensions
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    """
    Bottleneck Attention Module (BAM) - Spatial Gate.

    Parameters:
    - gate_channel (int): Number of input channels for the gate.
    - reduction_ratio (int, optional): Reduction ratio for the intermediate channels. Default is 16.
    - dilation_conv_num (int, optional): Number of dilated convolutions in the gate. Default is 2.
    - dilation_val (int, optional): Dilation value for the dilated convolutions. Default is 4.
    """
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()

        # Define the components of the spatial gate
        self.gate_s = nn.Sequential()

        # Reduce dimensionality using 1x1 convolution
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )

        # Apply dilated convolutions
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )

        # Final 1x1 convolution to produce spatial attention map
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )

    def forward(self, in_tensor):
        # Apply the spatial gate to the input tensor and expand the dimensions
        return self.gate_s( in_tensor ).expand_as(in_tensor)
    
class BAM(nn.Module):
    """
    Bottleneck Attention Module (BAM).

    Parameters:
    - gate_channel (int): Number of input channels for the gate.
    """
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        # Channel-wise attention module
        self.channel_att = ChannelGate(gate_channel)

        # Spatial-wise attention module
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        """
        Forward pass of the BAM module.

        Parameters:
        - in_tensor (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying BAM.
        """
        # Calculate attention weights using channel and spatial attention modules
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )

        # Multiply the input tensor by the attention weights
        return att * in_tensor