import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleExtension(nn.Module):
    """
    Creates a decoder which takes the output of the base model and generates multi-scale predictions.
    Assumes that the output of the base model contains <input_channels> channels.
    """
    def __init__(self, input_channels=16, output_channels=1, scales=[1]):
        super(MultiScaleExtension, self).__init__()
        self.scales = scales
        
        # Downsample layers for each scale
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding="same")
            for _ in self.scales
        ])
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        multi_scale_outputs = []
        
        for scale, downsample_layer in zip(self.scales, self.downsample_layers):
            if scale > 1:
                downsampled_x = F.interpolate(x, scale_factor=1/scale, mode='nearest')
            else:
                downsampled_x = x
                
            out = self.sigmoid(downsample_layer(downsampled_x))
            multi_scale_outputs.append(out)
        
        return multi_scale_outputs
    
class MultiScaleExtensionPyramid(nn.Module):
    def __init__(self, input_channels=16, output_channels=1, scales=[1], original_dim=(65, 65)):
        super(MultiScaleExtensionPyramid, self).__init__()
        self.scales = scales
        self.original_dim = original_dim
        
        # Initial convolution layer for scale 1
        self.initial_conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
        
        # List to hold downsample and convolution layers for subsequent scales
        self.pyramid_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(self.original_dim[0]//scale, self.original_dim[1]//scale), mode='nearest'), # Downsampling to the correct dimension
                nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1) # Convolution layer
            )
            for scale in self.scales[1:] # We already have a layer for scale 1
        ])
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        multi_scale_outputs = []
        
        # Passing through the initial convolution layer (scale 1)
        out = self.sigmoid(self.initial_conv(x))
        multi_scale_outputs.append(out)
        
        # Passing through the pyramid structure for subsequent scales
        for pyramid_layer in self.pyramid_layers:
            out = pyramid_layer(out) # The output of one layer is input to the next
            multi_scale_outputs.append(self.sigmoid(out))
        
        return multi_scale_outputs
    
class MultiScaleExtensionPyramid_scaleConsistency(nn.Module):
    def __init__(self, input_channels=16, output_channels=1, scales=[1], original_dim=(65, 65), pooling_type='max'):
        super(MultiScaleExtensionPyramid_scaleConsistency, self).__init__()
        self.scales = scales
        self.original_dim = original_dim
        self.pooling_type = pooling_type
        
        # Single convolution layer to process output from the base model
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        multi_scale_outputs = []
        
        # Passing through the convolution layer
        out = self.sigmoid(self.conv(x))
        
        # Downsampling for each scale in the pyramid structure using pooling
        for scale in self.scales:
            if scale == 1:
                # No pooling required for scale 1
                multi_scale_outputs.append(out)
            else:
                if self.pooling_type == 'max':
                    pool_layer = nn.MaxPool2d(kernel_size=scale, stride=scale)
                elif self.pooling_type == 'avg':
                    pool_layer = nn.AvgPool2d(kernel_size=scale, stride=scale)
                elif self.pooling_type == 'min':
                    pool_layer = lambda x: -nn.MaxPool2d(kernel_size=scale, stride=scale)(-x)
                else:
                    raise ValueError("Invalid pooling type")
                
                downsampled_output = pool_layer(out)
                multi_scale_outputs.append(downsampled_output)
        
        return multi_scale_outputs
