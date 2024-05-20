import torch
import torch.nn as nn
from .base_models.depth_convlstm import depth_convlstm
from .base_models.depth_convlstm_multilayer import depth_convlstm_multiscale
from multiscale_extension import MultiScaleExtension,MultiScaleExtensionPyramid,MultiScaleExtensionPyramid_scaleConsistency

class BaseModel_WithMultiScale(nn.Module):
    def __init__(self, config, base_model, writer=None):
        self.writer = writer
        super(BaseModel_WithMultiScale, self).__init__()
        self.scales = config.experiment_setting.train.multiscale.scales
        self.input_channels_multiscale = config.experiment_setting.train.multiscale.input_channels
        self.output_channels_multiscale = config.experiment_setting.train.multiscale.output_channels
        self.H = config.dataloader.image_spatial_size[0]
        self.W = config.dataloader.image_spatial_size[1]
        self.model_prediction_pooling = config.experiment_setting.train.multiscale.model_prediction_pooling
        
        if base_model == "d_convlstm":
            self.base_model = depth_convlstm(config)
        elif base_model == "depth_convlstm_multiscale":
            self.base_model = depth_convlstm_multiscale(config)
        # self.multiscale_extension = MultiScaleExtension(input_channels=self.input_channels_multiscale, 
        #                                                 output_channels=self.output_channels_multiscale,
        #                                                 scales=self.scales)  
        self.multiscale_extension = MultiScaleExtensionPyramid_scaleConsistency(input_channels=self.input_channels_multiscale, 
                                                        output_channels=self.output_channels_multiscale,
                                                        scales=self.scales,
                                                        original_dim=(self.H,self.W),
                                                        pooling_type=self.model_prediction_pooling)  
       
    def forward(self, x):
        """
        Args:
            - x: (B, T_obs, C, H, W) -> Batch of observed chunks
        """
        # Reshape from BTCHW to TBCHW
        if len(x.shape) == 5:
            x = x.permute(1,0,2,3,4)
        else:
            raise ValueError("Input shape must be (B, T_obs, C, H, W)")
        
        x_pred = self.base_model(x) 
        T,B,C,H,W = x_pred.shape # (T_obs+T_pred-1, B, C, H, W)
        x_pred = x_pred.view(-1, C, H, W) # (T*B, C, H, W)
        x_ms = self.multiscale_extension(x_pred) # [(T*B, C, H1, W1), (T*B, C, H2, W2), ...]
        HW_ms = [(y.shape[-2],y.shape[-1]) for y in x_ms] # [(H1,W1), (H2,W2), ...]
        x_ms_t = [y.view(T, B, self.output_channels_multiscale, H, W) for y,(H,W) in zip(x_ms, HW_ms)] # [(T, B, C, H1, W1), (T, B, C, H2, W2), ...] 
        
        # Reshape from TBCHW to BTCHW
        x_ms_t = [y.permute(1, 0, 2, 3, 4) for y in x_ms_t]
        
        return x_ms_t
