import torch
import torch.nn as nn
import math

########################################################### Encoder ###########################################################
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        INPUT_CHANNELS = args.model.d_convLSTM.Encoder.in_channels
        OUTPUT_CHANNELS = args.model.d_convLSTM.Encoder.out_channels
        KERNEL_SIZE = args.model.d_convLSTM.Encoder.kernel_size

        # Encoder	
        self.conv1 = nn.Conv2d(in_channels=INPUT_CHANNELS, 
                                 out_channels=OUTPUT_CHANNELS, 
                                   kernel_size=KERNEL_SIZE, 
                                 padding="same")
        self.bn1 = nn.BatchNorm2d(OUTPUT_CHANNELS)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

class DownsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample_stride, padding):
        super(DownsampleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=downsample_stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EncoderBottleneck(nn.Module):
    def __init__(self, args, bottleneck_factor):
        super().__init__()
        
        INPUT_CHANNELS = args.model.d_convLSTM.EncoderBottleneck.in_channels
        OUTPUT_CHANNELS = args.model.d_convLSTM.EncoderBottleneck.out_channels
        KERNEL_SIZE = args.model.d_convLSTM.EncoderBottleneck.kernel_size
        BOTTLENECK_FACTOR = bottleneck_factor  
        
        # Calculate the number of downsampling stages
        num_stages = int(math.log2(BOTTLENECK_FACTOR))
        assert BOTTLENECK_FACTOR == 2**num_stages, "Bottleneck factor must be a power of 2."
        
        # Create a sequence of downsampling and convolution blocks
        layers = []
        current_channels = INPUT_CHANNELS
        for i in range(num_stages):
            layers.append(DownsampleConvBlock(current_channels, 
                                              OUTPUT_CHANNELS, 
                                              KERNEL_SIZE, 
                                              downsample_stride=2, 
                                              padding=KERNEL_SIZE // 2))
            current_channels = OUTPUT_CHANNELS
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
########################################################### Decoder ###########################################################

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
          
        INPUT_CHANNELS = args.model.d_convLSTM.Decoder.in_channels
        OUTPUT_CHANNELS = args.model.d_convLSTM.Decoder.out_channels
        KERNEL_SIZE = args.model.d_convLSTM.Decoder.kernel_size
          
        # Decoder
        self.conv1 = nn.Conv2d(in_channels=INPUT_CHANNELS, 
                                 out_channels=OUTPUT_CHANNELS, 
                                   kernel_size=KERNEL_SIZE, 
                                 padding="same")
        self.relu = nn.ReLU()

    def forward(self, context):
        x = self.relu(self.conv1(context))
        return x
    
class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample_mode='bilinear'):
        super(UpsampleConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True if upsample_mode == 'bilinear' else None)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class DecoderBottleneck(nn.Module):
    def __init__(self, args, bottleneck_factor):
        super().__init__()
        
        INPUT_CHANNELS = args.model.d_convLSTM.DecoderBottleneck.in_channels
        OUTPUT_CHANNELS = args.model.d_convLSTM.DecoderBottleneck.out_channels
        KERNEL_SIZE = args.model.d_convLSTM.DecoderBottleneck.kernel_size
        BOTTLENECK_FACTOR = bottleneck_factor  
        UPSAMPLE_MODE = 'bilinear'
        
        # Calculate the number of upsampling stages
        num_stages = int(math.log2(BOTTLENECK_FACTOR))
        assert BOTTLENECK_FACTOR == 2**num_stages, "Bottleneck factor must be a power of 2."
        
        # Create a sequence of upsampling and convolution blocks
        layers = []
        for i in range(num_stages):
            layers.append(UpsampleConvBlock(INPUT_CHANNELS if i == 0 else OUTPUT_CHANNELS, 
                                            OUTPUT_CHANNELS, 
                                            KERNEL_SIZE, 
                                            upsample_mode=UPSAMPLE_MODE))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward pass of the ConvLSTM cell.

        Parameters
        ----------
        input_tensor: torch.Tensor
            4D tensor of shape (batch_size, input_channels, height, width).
        cur_state: tuple
            Tuple containing the current hidden state and cell state, each of shape (batch_size, hidden_dim, height, width).

        Returns
        -------
        h_next: torch.Tensor
            Next hidden state, of shape (batch_size, hidden_dim, height, width).
        c_next: torch.Tensor
            Next cell state, of shape (batch_size, hidden_dim, height, width).
        """

        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden and cell states.

        Parameters
        ----------
        batch_size: int
            Size of the batch.
        image_size: tuple
            Spatial dimensions of the input image, formatted as (height, width).

        Returns
        -------
        tuple
            Zero initialized hidden and cell states, each of shape (batch_size, hidden_dim, height, width).
        """

        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))        
        
        
# This creates a multi-layered LSTM. Each layer is defined by ConvLSTMCell class that is defined above.
class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels  (If using multiple layers, then list of hidden_dim for each layer)
        kernel_size: Size of kernel in convolutions (If using multiple layers, then list of kernel_size for each layer. Each kernel_size is a tuple)
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, args):
        super(ConvLSTM, self).__init__()
        self.input_dim = args.model.d_convLSTM.ConvLSTM.in_channels
        self.hidden_dim = args.model.d_convLSTM.ConvLSTM.hidden_channels
        self.output_dim = args.model.d_convLSTM.ConvLSTM.output_channels
        self.kernel_size = (3,3)
        self.num_layers = args.model.d_convLSTM.ConvLSTM.num_layers
        self.batch_first = False
        self.bias = True
        self.return_all_layers = False

        # To check that kernel size is a tuple or a list of tuples
        self._check_kernel_size_consistency(self.kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        self.kernel_size = self._extend_for_multilayer(self.kernel_size, self.num_layers)
        self.hidden_dim = self._extend_for_multilayer(self.hidden_dim, self.num_layers)
        if not len(self.kernel_size) == len(self.hidden_dim) == self.num_layers: # Checks to make sure that we have hidden_dim and kernel_size for every layer
            raise ValueError('Inconsistent list length.')

        self.n_observation = args.experiment_setting.chunk_params.num_observed
        self.n_prediction = args.experiment_setting.chunk_params.num_predicted
  
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1] # If first layer, then input is the original input. Else input is the the output of the previous layer

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list) # Holds submodules in a list: List of ConvLSTMCell. Each ConvLSTM cell represents one layer.

        # For the final output convolution
        out_conv_kernel = self.kernel_size[-1] # Using the same kernel size as the last output. Can change this.
        out_conv_padding = out_conv_kernel[0] // 2, out_conv_kernel[1] // 2  # Padding to keep the input and output size same
        self.out_conv = nn.Conv2d(in_channels=self.hidden_dim[-1], 
                                    out_channels=self.output_dim, 
                                      kernel_size=out_conv_kernel, 
                                    padding=out_conv_padding, 
                                     bias=self.bias)

    def forward(self, input_tensor):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        output: Shape (t, b, c, h, w)
        """
        if self.batch_first: # If tensor is of shape (b, t, c, h, w) then convert it to (t, b, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        _, b, _, h, w = input_tensor.size() # (t,b,c,h,w)
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w)) # [(h1, c1)] -> shape : b,c,h,w

        cur_layer_input = input_tensor # Input to the first layer is the input_tensor
        for layer_idx in range(self.num_layers): # Iterating over all the layers

            h, c = hidden_state[layer_idx] # Initial hidden state of layer layer_idx
            predicted_voxels = []

            for t in range(self.n_observation):
                # h,c shape : b,c_out,h,w
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[t, :, :, :, :], cur_state=[h, c])
                predicted_voxels.append(self.out_conv(h))

            for t in range(self.n_prediction): 
                # h,c shape : b,c,h,w
                h, c = self.cell_list[layer_idx](input_tensor=predicted_voxels[-1], cur_state=[h, c])
                if t < self.n_prediction - 1: # ignore last prediction
                    predicted_voxels.append(self.out_conv(h))

            output = torch.stack(predicted_voxels, dim=0)
            
        return output

    def _init_hidden(self, batch_size, image_size): 
        """ To initialize the hidden state for each layer. """
        init_states = [] 
        for i in range(self.num_layers): # Need to initialize hidden state for each layer
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size)) 
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list): # If not a list, then extend to a list
            param = [param] * num_layers
        return param


class depth_convlstm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model.name
        try:
            self.BOTTLENECK_FACTOR = args.model.d_convLSTM.bottleneck_factor  
        except:
            print("Error: No bottleneck factor specified. Defaulting to 1.")
            self.BOTTLENECK_FACTOR = 1
        
        # Encoder
        if self.BOTTLENECK_FACTOR > 1:
            print(f"******* Botlenecked ConvLSTM : {self.BOTTLENECK_FACTOR} *******")
            self.encoder = EncoderBottleneck(args, self.BOTTLENECK_FACTOR)
        else:
            self.encoder = Encoder(args)

        # ConvLSTM
        self.convlstm = ConvLSTM(args)
        
        # Decoder
        if self.BOTTLENECK_FACTOR > 1:
            self.decoder = DecoderBottleneck(args, self.BOTTLENECK_FACTOR)
        # else:
        #     self.decoder = Decoder(args)
        
        

    def forward(self, image):
        """
        Args:	
             image: (num_frame, num_batch, num_channel, H, W)
        Returns:
            pred_context: (num_frame, num_batch, num_channel, H, W)
        """
        # Encoder		
        num_frame, num_batch, num_channel, image_size = image.shape[:-1]
        image = image.reshape(num_frame * num_batch, num_channel, image_size, image_size)
        context = self.encoder(image)
        _, C, H, W = context.shape
        context = context.reshape(num_frame, num_batch, C, H, W) # T, B, C, H, W
  
        # ConvLSTM | input: (Tobs, B, C, H, W) | output: (Tobs+Tpred-1, B, C, H, W)
        pred_context  = self.convlstm(context)  
        
        if self.BOTTLENECK_FACTOR > 1:
            # Decoder
            pred_context = pred_context.reshape((self.convlstm.n_observation + self.convlstm.n_prediction - 1) * num_batch, pred_context.shape[2], pred_context.shape[3], pred_context.shape[4])
            image_pred = self.decoder(pred_context).squeeze()
            assert image_size == image_pred.shape[-1], f"image_size: {image_size}, image_pred.shape[-1]: {image_pred.shape[-1]}"
            image_pred = image_pred.reshape((self.convlstm.n_observation + self.convlstm.n_prediction - 1), num_batch, -1,image_size, image_size)
            pred_context = image_pred
            
        return pred_context


if __name__ == '__main__':
    print("here")