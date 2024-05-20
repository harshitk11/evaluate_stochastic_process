import torch
import torch.nn as nn

def get_downsample_padding(kernel_size):
    lut = {3:1, 5:2, 7:3}
    return lut[kernel_size]

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, config=None):
        """
        If downsample is set to True, the ResBlock will use MaxPool2d(2) as the downsampling layer.
        Else, no downsampling.
        """
        super(ResBlock, self).__init__()
        if config is not None:
            KERNEL_SIZE = config.model.segmentation_model.conv_kernel_size
        else:
            KERNEL_SIZE = 3
        
        PADDING = KERNEL_SIZE // 2
        DOWNSAMPLE_PADDING = get_downsample_padding(KERNEL_SIZE)
            
        
        self.downsample_flag = downsample
        
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        
        if self.downsample_flag:
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=2, padding=DOWNSAMPLE_PADDING)
        else:
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        
        
        if self.downsample_flag:
            self.downsample = nn.MaxPool2d(2)
        else:
            self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)

    def forward(self, x):
        x_res = self.conv_res(x)
        x_res = self.dropout(x_res)
        
        out = self.leaky_relu(x)
        out = self.dropout(out)
        out = self.downsample(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.conv_out(out)
        out = self.dropout(out) + x_res
        
        return out

class SegmentationModel(nn.Module):
    def __init__(self, use_bottleneck=True, config=None):
        super(SegmentationModel, self).__init__()
        if config is not None:
            KERNEL_SIZE = config.model.segmentation_model.conv_kernel_size
            BOTTLENECK_CHANNEL = config.model.segmentation_model.bottleneck_channel
        else:
            KERNEL_SIZE = 3
            BOTTLENECK_CHANNEL = 16
            
        PADDING = KERNEL_SIZE // 2
        
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            print("*"*10, "Using bottleneck", "*"*10)
        else:
            print("*"*10, "Not using bottleneck", "*"*10)
            
        # Initial layers
        # self.conv1 = nn.Conv2d(12, 16, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        self.conv1 = nn.Conv2d(12, BOTTLENECK_CHANNEL, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        self.conv2 = nn.Conv2d(BOTTLENECK_CHANNEL, BOTTLENECK_CHANNEL, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        
        # self.conv_res = nn.Conv2d(12, 16, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        self.conv_res = nn.Conv2d(12, BOTTLENECK_CHANNEL, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        
        # Encoder ResBlocks
        # self.resblock1 = ResBlock(16,32, downsample=self.use_bottleneck)
        self.resblock1 = ResBlock(BOTTLENECK_CHANNEL,2*BOTTLENECK_CHANNEL, downsample=self.use_bottleneck)
        # self.resblock2 = ResBlock(32,32, downsample=self.use_bottleneck)
        self.resblock2 = ResBlock(2*BOTTLENECK_CHANNEL,2*BOTTLENECK_CHANNEL, downsample=self.use_bottleneck)
        
        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.resblock3 = ResBlock(32, 16, downsample=False)
        self.resblock3 = ResBlock(2*BOTTLENECK_CHANNEL, BOTTLENECK_CHANNEL, downsample=False)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.resblock4 = ResBlock(16, 16, downsample=False)
        self.resblock4 = ResBlock(BOTTLENECK_CHANNEL, BOTTLENECK_CHANNEL, downsample=False)
        
        # Final layer
        # self.conv_final = nn.Conv2d(16, 1, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        self.conv_final = nn.Conv2d(BOTTLENECK_CHANNEL, 1, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Convert from (B, T, C, H, W) to (B, C, H, W)
        x = x.squeeze(1)
        
        # Residual connection
        x_res = self.conv_res(x)
        x_res = self.dropout(x_res)
        
        # Initial layers
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x) + x_res
        
        # Using bottleneck if the flag is set to True
        x = self.resblock1(x)
        x = self.resblock2(x)
    
        # Decoder
        if self.use_bottleneck:
            x = self.upsample1(x)
        x = self.resblock3(x)
        
        if self.use_bottleneck:
            x = self.upsample2(x)
        x = self.resblock4(x)
        
        # Final layer
        x = self.conv_final(x)
        x = self.sigmoid(x)
        
        
        # For compatibility with the trainer
        x = x.unsqueeze(1)
        return [x] 

def main():
    # To use the model with bottleneck:
    model_with_bottleneck = SegmentationModel(use_bottleneck=True)

    # To use the model without bottleneck:
    model_without_bottleneck = SegmentationModel(use_bottleneck=False)


if __name__ == "__main__":
    main()    
