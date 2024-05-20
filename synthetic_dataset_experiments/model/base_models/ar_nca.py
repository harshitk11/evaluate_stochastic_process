import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# This script defines the architecture of the ConvLSTM.
# Reference to the Github repository : https://github.com/ndrplz/ConvLSTM_pytorch

# This is a single LSTM cell
# Dimensions of input tensor (which is an image) : Number of channels x Height of image x Width of image
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
        # In ConvLSTM, the inputs, cell states, hidden states, and the gates are all 3D tensors whose last two dimensions are spatial dimensions (rows and columns)

        self.input_dim = input_dim # Number of channels (not dimension) of the input tensor
        self.hidden_dim = hidden_dim # Number of channels (not dimension) of the hidden state tensor    

        self.kernel_size = kernel_size # Kernel size of the Convolution. (int, int) 2-D kernel
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # // : Floor division - Rounds the result down to the nearest whole number. Padding = (Filter Size - 1)/2 to keep the output shape same as the input shape after convolution
        self.bias = bias

        # Initializing a 2D convolutional layer
        # We are performing two convolutions : Conv(x_t;W_x) [Convolution over the input tensor] and Conv(h_t-1;W_h) [convolution over the hidden state tensor]
        # So we perform the two convolutions together [Parrallelization]
        # On the output side, we need to perform the two convolutions for each of the input gate (i_t), forget gate (f_t), output gate (o_t), and intermediate cell state (g_t)
        # Therefore, for every channel in the input, we create 4 channels of output, in our Conv2d. We will separate the 4 channels in the output later.

        # Reference for Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # NOTE: arguments are CHANNELS and not features
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    # Defining the forward pass
    def forward(self, input_tensor, cur_state):
        # Shape of the tensor : Number of channels * H * W  (Last two dimensions are spatial dimensions)
        h_cur, c_cur = cur_state    # current hidden state, current cell state [They are both 3D tensor]

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis (axis0 [batch] * axis1 [channel] * axis2 [height] * axis3 [width])

        combined_conv = self.conv(combined) # Convolve

        # Remember we had 4 output channels for the convolutions for it, ft, ot, gt
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) # Split along the channel axis
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    # Use this to initialize the initial hidden state tensors in your code
    def init_hidden(self, batch_size, image_size):
        # Returns a zero initialized tensor for the cell state and the hidden state
        height, width = image_size # Spatial dimensions of your image (Must be a tuple)
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

		self.input_dim = 16
		self.hidden_dim = 16
		self.kernel_size = (3,3)
		self.num_layers = 1
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

		self.n_batch = 0 # will be changed in the training code
		self.n_observation = 10
		self.n_prediction = 50

		cell_list = []
		for i in range(0, self.num_layers):
			cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1] # If first layer, then input is the original input. Else input is the the output of the previous layer

			cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
											hidden_dim=self.hidden_dim[i],
											kernel_size=self.kernel_size[i],
											bias=self.bias))

		self.cell_list = nn.ModuleList(cell_list) # Holds submodules in a list: List of ConvLSTMCell. Each ConvLSTM cell represents one layer.

		# Convolution layer for converting hidden states over all the time steps (B x T x Ch x H x W) to output (B x T x Ci x H x W)
		# Since conv2d takes input of the form [B' x Ch x H x W] we will reshape the tensor such that B'= B x T 
		# Applying conv2d to the hidden states of the last layer only. Output will have same number of channels as that of the input i.e. Ci 

		# For the final output convolution
		self.out_conv_kernel = self.kernel_size[-1] # Using the same kernel size as the last output. Can change this.
		self.out_conv_padding = self.out_conv_kernel[0] // 2, self.out_conv_kernel[1] // 2  # Padding to keep the input and output size same
		self.out_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=self.out_conv_kernel, padding=self.out_conv_padding, bias=self.bias)

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
		last_state_list, layer_output
		"""
		if self.batch_first: # If batch first is not selected then reshape the input tensor
			# (t, b, c, h, w) -> (b, t, c, h, w)
			input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

		_, b, _, h, w = input_tensor.size() # (t,b,c,h,w)
		hidden_state = self._init_hidden(batch_size=b, image_size=(h, w)) # [(h1, c1)]

		final_output_tensor_list = []
		layer_output_list = [] # Output over all the time steps for all the layers. Each element contains output of all the time steps for one layer.
		last_state_list = [] # Hidden state after the last time step for all the layers. Each element contains [h,c] after the last time step for one layer.

		cur_layer_input = input_tensor # Input to the first layer is the input_tensor

		for layer_idx in range(self.num_layers): # Iterating over all the layers

			h, c = hidden_state[layer_idx] # Initial hidden state of layer layer_idx
			
			output_inner = [] # Stores the output of a given layer for all the time steps
			final_output = [] # Stores the final output (with num channels same as the channels in the input image)
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

			output = predicted_voxels
			output = torch.stack(output, dim=0)

		return output

		"""
			layer_output = torch.stack(output_inner, dim=1) # Shape :B x T x Ch x H x W | Output (hidden states h) over all the time steps for this layer.
			final_output_tensor = torch.stack(final_output, dim=1)

			cur_layer_input = layer_output # Output of this layer will be the input to the next layer

			layer_output_list.append(layer_output) # Append the hidden states of this layer to a list
			last_state_list.append([h, c]) 
			final_output_tensor_list.append(final_output_tensor) # Each element contains the output tensor for each layer

		if not self.return_all_layers:
			layer_output_list = layer_output_list[-1:] # return output of the last layer only. Includes all the time steps of the last layer.
			last_state_list = last_state_list[-1:] # return the hidden state of the last layer only.

		# Need output tensor in the shape B x T x C x H x W
		last_hidden_state = layer_output_list[-1] # b,t,Ch,h,w
		last_layer_final_output = final_output_tensor_list[-1]

		B,_,Ch,H,W = last_hidden_state.size()
		last_layer_final_output = torch.sigmoid(last_layer_final_output)
		
		return layer_output_list, last_state_list, last_layer_final_output
		"""

	# To initialize the hidden state for each layer.
	def _init_hidden(self, batch_size, image_size): 
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


class HebbianBasicCNN(nn.Module):
	def __init__(self, args):

		super().__init__()
		self.args = args

		if self.args.mode == "attention_lstm": # output vector: (1, 64)
			self.fc1 = nn.Linear(3, 32) # (12, 32) for wildfire
			self.fc2 = nn.Linear(32, 32) # (32, 32) for default
			self.ln1 = nn.LayerNorm(32)
			self.ln2 = nn.LayerNorm(32)

		elif self.args.mode == "attention_rnn": # output vector: (1, 64)
			self.fc1 = nn.Linear(3, 32) # (12, 32) for wildfire
			self.fc2 = nn.Linear(32, 32) # (32, 32) for default
			self.ln1 = nn.LayerNorm(32)
			self.ln2 = nn.LayerNorm(32)		

		elif self.args.mode == "attention":
			self.fc1 = nn.Linear(3, 32)
			self.fc2 = nn.Linear(32, 32)
			self.ln1 = nn.LayerNorm(32)
			self.ln2 = nn.LayerNorm(32)

		elif self.args.mode == "convlstm":
			self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=2)
			self.bn1 = nn.BatchNorm2d(16)
			self.pool = nn.MaxPool2d(kernel_size=3, stride=1)

		elif self.args.mode == "d_convlstm":
			self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding="same")
			self.bn1 = nn.BatchNorm2d(16)
			self.pool = nn.MaxPool2d(kernel_size=3, stride=1)

		elif self.args.mode == "cnn_lstm": # output vector: (1, 512)
			self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2)
			self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
			self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)

			self.bn1 = nn.BatchNorm2d(32)
			self.bn2 = nn.BatchNorm2d(64)
			self.bn3 = nn.BatchNorm2d(128)
			self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x, location):
		if self.args.mode == "attention_lstm": # patch
			#x = self.relu(self.fc1(x))
			#x = self.relu(self.fc2(x))
			x = self.relu(self.ln1(self.fc1(x))) # 2023/08/28
			x = self.relu(self.ln2(self.fc2(x))) # 2023/08/28

		elif self.args.mode == "attention_rnn": # patch
			x = self.relu(self.ln1(self.fc1(x))) # 2023/08/28
			x = self.relu(self.ln2(self.fc2(x))) # 2023/08/28

		elif self.args.mode == "attention":
			x = self.relu(self.ln1(self.fc1(x))) # 2023/08/28
			x = self.relu(self.ln2(self.fc2(x))) # 2023/08/28

		elif self.args.mode == "convlstm":
			x = self.relu(self.bn1(self.conv1(x)))
			x = self.pool(x)

		elif self.args.mode == "d_convlstm":
			x = self.relu(self.bn1(self.conv1(x)))

		elif self.args.mode == "cnn_lstm":
			x = self.relu(self.bn1(self.conv1(x)))
			x = self.pool(x)

		return x


class Decoder(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		if self.args.mode == "attention_lstm":
			self.fc4 = nn.Linear(32, 64) # (32, 64) for default
			self.fc5 = nn.Linear(64, 1)
			self.ln4 = nn.LayerNorm(64) # 230829
			#self.num_patch = 64*64

		elif self.args.mode == "attention_rnn":
			self.fc4 = nn.Linear(32, 64) # (32, 64) for default
			self.fc5 = nn.Linear(64, 1)
			self.ln4 = nn.LayerNorm(64) # 230829

		elif self.args.mode == "attention":
			self.fc4 = nn.Linear(32, 64)
			self.fc5 = nn.Linear(64, 1)
			self.ln4 = nn.LayerNorm(64) # 230829

		elif self.args.mode == "convlstm":
			self.upsample_2x2 = nn.Upsample(scale_factor=2, mode='nearest')
			self.conv1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=2)
			self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=2)
			self.bn1 = nn.BatchNorm2d(8)

		elif self.args.mode == "d_convlstm":
			self.conv1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding="same")

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


	def forward(self, context, location):
		if self.args.mode == "attention_lstm":
			num_frame = context.shape[0]
			num_patch = 64*64
			num_batch = int(context.shape[1] / num_patch)

			y = self.relu(self.ln4(self.fc4(context.reshape(num_frame, num_batch, num_patch, -1)))) # 230829
			y = self.sigmoid(self.fc5(y)) # for pburn prediction
			#y = self.fc5(y) # for rgb prediction
			return None, y

		if self.args.mode == "attention_rnn":
			num_frame = context.shape[0]
			num_patch = 64*64
			num_batch = int(context.shape[1] / num_patch)

			y = self.relu(self.ln4(self.fc4(context.reshape(num_frame, num_batch, num_patch, -1)))) # 230829
			y = self.sigmoid(self.fc5(y)) # for pburn prediction
			#y = self.fc5(y) # for rgb prediction
			return None, y

		elif self.args.mode == "attention":
			num_frame = context.shape[0]
			num_patch = 512*512
			num_batch = int(context.shape[1] / num_patch)

			y = self.relu(self.ln4(self.fc4(context.reshape(num_frame, num_batch, num_patch, -1)))) # 230829
			y = self.sigmoid(self.fc5(y))
			return None, y

		elif self.args.mode == "convlstm":
			x = self.relu(self.bn1(self.conv1(context))) # 16,31,31
			x = self.upsample_2x2(x) # 8,62,62
			x = self.sigmoid((self.conv2(x))) # 1,64,64
			return x

		elif self.args.mode == "d_convlstm":
			x = self.sigmoid(self.conv1(context)) # 16,31,31
			return x


class GraphAttention(nn.Module):
	def __init__(self, num_node, args):
		super().__init__()
		self.args = args

		neighbor_indices_list = [[-1,-1], [-1,0], [-1,1],
								[0,-1], [0,0], [0,1],
								[1,-1], [1,0], [1,1]]

		self.adjacency = torch.zeros(num_node, num_node).to(self.args.device)
		for node_idx in range(self.adjacency.shape[0]):
			node_row = int(node_idx / int(self.adjacency.shape[0] ** 0.5))
			node_col = int(node_idx % int(self.adjacency.shape[0] ** 0.5))

			for neighbor_indice_idx in range(9):
				neighbor_index = neighbor_indices_list[neighbor_indice_idx]
				if (node_row + neighbor_index[0]) >= 0 and (node_row + neighbor_index[0]) < 5 and \
					(node_col + neighbor_index[1]) >= 0 and (node_col + neighbor_index[1]) < 5:
					neighbor_idx = int(self.adjacency.shape[0] ** 0.5) * (node_row + neighbor_index[0]) + (node_col + neighbor_index[1])
					self.adjacency[node_idx][neighbor_idx] = 1

		self.fc = nn.Linear(66, 64)
		self.fc_at = nn.Linear(128, 1)
		self.leaky_relu = nn.LeakyReLU()

		# Generate the indices for the 625 elements in the matrix
		# Calculate the row and column indices for each element
		indices = torch.arange(num_node)
		rows = indices // int(num_node ** 0.5)
		cols = indices % int(num_node ** 0.5)

		# Generate the indices for the 9 neighboring elements of each element
		neighbor_indices = torch.tensor([
			[-1, -1], [-1, 0], [-1, 1],
			[0, -1], [0, 0], [0, 1],
			[1, -1], [1, 0], [1, 1]
		])

		# Calculate the row and column indices for the neighboring elements
		neighbor_rows = rows.view(-1, 1) + neighbor_indices[:, 0]
		neighbor_cols = cols.view(-1, 1) + neighbor_indices[:, 1]

		# Shift the out-of-bound indices
		neighbor_rows = neighbor_rows.reshape(25,25,-1)
		neighbor_rows[0,:,:] += 1
		neighbor_rows[-1,:,:] -= 1
		neighbor_rows = neighbor_rows.reshape(625,-1)

		neighbor_cols = neighbor_cols.reshape(25,25,-1)
		neighbor_cols[:,0,:] += 1
		neighbor_cols[:,-1,:] -= 1
		neighbor_cols = neighbor_cols.reshape(625,-1)
		
		self.neighbor_rows, self.neighbor_cols = neighbor_rows, neighbor_cols

	def forward(self, x):
		# input is context vector of patch
		""" 1) using adjacent matrix
		x = self.fc(x)

		x1 = x.unsqueeze(-2).repeat(1,1,x.shape[-2],1)
		x2 = x.unsqueeze(-3).repeat(1,x.shape[-2],1,1)

		z = torch.cat([x1,x2],-1)
		z = self.leaky_relu(self.fc_at(z)).squeeze()

		# softmax with adjacent matrix
		z = torch.exp(z) * self.adjacency
		attn_weight = z / torch.sum(z, dim=-1, keepdim=True)
		attn_vector = torch.sum(x.unsqueeze(-2) * attn_weight.unsqueeze(-1), dim=-2)
		"""

		#""" 2) using neighbor rows and cols
		x = self.fc(x)

		x1 = x.unsqueeze(-2).repeat(1,1,9,1)
		x2 = x.reshape(-1,25,25,64)[:,self.neighbor_rows,self.neighbor_cols,:]

		z = torch.cat([x1,x2], -1)
		z = self.leaky_relu(self.fc_at(z)).squeeze()

		# softmax with adjacent matrix
		z = torch.exp(z)
		attn_weight = z / torch.sum(z, dim=-1, keepdim=True)
		attn_vector = torch.sum(x.unsqueeze(-2) * attn_weight.unsqueeze(-1), dim=-2)
		#"""

		#print("self.adjacency, z: ", self.adjacency.shape, z.shape)
		#print("attn_vector, attn_weight: ", attn_vector.shape, attn_weight.shape)
		#input("here")
		return attn_vector, attn_weight

class LocalAttention(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

		# parameters
		self.n_input = 32
		self.n_hidden = 32
		self.n_latent = 32
		self.n_batch = 0 # will be changed in the training code
		self.n_patch = 64*64 # will be changed in the training code

		self.n_observation = 10
		self.n_prediction = 50
	
		self.fc2 = nn.Linear(32, 32)
		self.ln2 = nn.LayerNorm([32])

		self.fc_sa = nn.Linear(32+2, 32)
		self.mhsa = nn.MultiheadAttention(embed_dim=32+2, num_heads=1)

		# Calculate the row and column indices for each element
		num_node = self.n_patch
		indices = torch.arange(num_node)
		rows = indices // int(num_node ** 0.5)
		cols = indices % int(num_node ** 0.5)

		# Generate the indices for the 9 neighboring elements of each element
		neighbor_indices = torch.tensor([
			[-1, -1], [-1, 0], [-1, 1],
			[ 0, -1], [ 0, 0], [ 0, 1],
			[ 1, -1], [ 1, 0], [ 1, 1]
		])

		# Calculate the row and column indices for the neighboring elements
		neighbor_rows = rows.view(-1, 1) + neighbor_indices[:, 0]
		neighbor_cols = cols.view(-1, 1) + neighbor_indices[:, 1]

		# Shift the out-of-bound indices
		neighbor_rows = neighbor_rows.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
		neighbor_rows[0,:,:] += 1
		neighbor_rows[-1,:,:] -= 1
		neighbor_rows = neighbor_rows.reshape(self.n_patch,-1)

		neighbor_cols = neighbor_cols.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
		neighbor_cols[:,0,:] += 1
		neighbor_cols[:,-1,:] -= 1
		neighbor_cols = neighbor_cols.reshape(self.n_patch, -1)

		self.neighbor_rows, self.neighbor_cols = neighbor_rows, neighbor_cols

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


	def init_states(self):
		num_node = self.n_patch
		# Calculate the row and column indices for each element
		indices = torch.arange(num_node)
		rows = indices // int(num_node ** 0.5)
		cols = indices % int(num_node ** 0.5)

		if self.neighbor_rows.shape[-1] == 9:
			# Generate the indices for the 9 neighboring elements of each element
			neighbor_indices = torch.tensor([
				[-1, -1], [-1, 0], [-1, 1],
				[0, -1], [0, 0], [0, 1],
				[1, -1], [1, 0], [1, 1]
			])

		# Calculate the row and column indices for the neighboring elements
		neighbor_rows = rows.view(-1, 1) + neighbor_indices[:, 0]
		neighbor_cols = cols.view(-1, 1) + neighbor_indices[:, 1]
		neighbor_rows = neighbor_rows.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
		neighbor_cols = neighbor_cols.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)

		if neighbor_indices.shape[0] == 9: # 3x3 perception
			# Shift the out-of-bound indices
			neighbor_rows[0,:,:] += 1
			neighbor_rows[-1,:,:] -= 1

			neighbor_cols[:,0,:] += 1
			neighbor_cols[:,-1,:] -= 1

		neighbor_rows = neighbor_rows.reshape(self.n_patch,-1)
		neighbor_cols = neighbor_cols.reshape(self.n_patch, -1)		
		self.neighbor_rows, self.neighbor_cols = neighbor_rows, neighbor_cols

	def forward(self, x):
		self.init_states()
		pred_encoded_vectors = torch.zeros(self.n_batch * self.n_patch, self.n_hidden).to(self.args.device)

		encoded_vectors = x
		predicted_vectors = []
		
		# patch position as patch label: [x,y] indicates patch is at x-th row and y-th col
		patch_label_row = torch.arange(start=0, end=1, step=1 / int(self.n_patch ** 0.5)).to(self.args.device)
		patch_label_row = patch_label_row.unsqueeze(1).repeat(1, int(self.n_patch ** 0.5))
		patch_label_col = torch.arange(start=0, end=1, step=1 / int(self.n_patch ** 0.5)).to(self.args.device)
		patch_label_col = patch_label_col.unsqueeze(0).repeat(int(self.n_patch ** 0.5), 1)
		patch_label = torch.cat((patch_label_row.unsqueeze(0), patch_label_col.unsqueeze(0)), dim=0)
		patch_label = patch_label.reshape(2, -1).permute(1, 0)

		if x.shape[1] > self.n_patch:
			patch_label = patch_label.unsqueeze(1).repeat(1, self.n_batch, 1)
		elif x.shape[1] == self.n_patch:
			patch_label = patch_label.unsqueeze(1).repeat(1, self.n_batch, 1)

		attn_weights_list = []

		# observation
		for time_idx in range(self.n_observation):				
			# multi-head attention
			if x.shape[1] > self.n_patch: # for training data
				"""
				query, key, value = encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				#self.h:  torch.Size([16384, 32])
				#encoded_vectors[time_idx]:  torch.Size([16384, 32])
				#self.n_batch, self.n_patch, self.n_hidden:  4 4096 32

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				"""

				query = encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query = torch.cat((query, patch_label), dim=-1)

				#self.h:  torch.Size([16384, 32])
				#encoded_vectors[time_idx]:  torch.Size([16384, 32])
				#self.n_batch, self.n_patch, self.n_hidden:  4 4096 32

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)

				#neighbor_patch_label = torch.tensor([[0.33,0.33], [0.33,0.66], [0.33,0.99], [0.66,0.33], [0.66,0.66], [0.66,0.99], [0.99,0.33], [0.99,0.33], [0.99,0.99]], requires_grad=False).to(self.args.device)
				#neighbor_patch_label = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [ 0, 0], [ 0, 1], [1, -1], [ 1, 0], [ 1, 1]], requires_grad=False).to(self.args.device)
				#neighbor_patch_label = torch.tensor([[1, 1], [0, 0], [1, 1], [0, 0], [0.5, 0.5], [0, 0], [0, 0], [0, 0], [0, 0]], requires_grad=False).to(self.args.device)
				#new_query[:,:,-2:] = neighbor_patch_label.unsqueeze(1)

				#attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
				attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
				new_encoded_vectors = encoded_vectors[time_idx] + self.fc_sa(attn_vectors[4,:,:].squeeze())
				#new_encoded_vectors = self.fc_sa(attn_vectors[4,:,:].squeeze())

			elif x.shape[1] == self.n_patch: # for test data
				"""
				query, key, value = encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				"""

				query = encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query = torch.cat((query, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)

				#neighbor_patch_label = torch.tensor([[0.33,0.33], [0.33,0.66], [0.33,0.99], [0.66,0.33], [0.66,0.66], [0.66,0.99], [0.99,0.33], [0.99,0.33], [0.99,0.99]], requires_grad=False).to(self.args.device)
				#neighbor_patch_label = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [ 0, 0], [ 0, 1], [1, -1], [ 1, 0], [ 1, 1]], requires_grad=False).to(self.args.device)
				#neighbor_patch_label = torch.tensor([[1, 1], [0, 0], [1, 1], [0, 0], [0.5, 0.5], [0, 0], [0, 0], [0, 0], [0, 0]], requires_grad=False).to(self.args.device)
				#new_query[:,:,-2:] = neighbor_patch_label.unsqueeze(1)

				#attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
				attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
				new_encoded_vectors = encoded_vectors[time_idx] + self.fc_sa(attn_vectors[4,:,:].squeeze())
				#new_encoded_vectors = self.fc_sa(attn_vectors[4,:,:].squeeze())

			# save predicted latent vectors
			predicted_vectors.append(self.ln2(self.fc2(new_encoded_vectors)))
			attn_weights_list.append(attn_weights.squeeze())

		# prediction
		for time_idx in range(self.n_prediction):
			# multi-head attention
			if x.shape[1] > self.n_patch:
				"""
				query, key, value = predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				"""

				query = predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query = torch.cat((query, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)

				#neighbor_patch_label = torch.tensor([[0.33,0.33], [0.33,0.66], [0.33,0.99], [0.66,0.33], [0.66,0.66], [0.66,0.99], [0.99,0.33], [0.99,0.33], [0.99,0.99]], requires_grad=False).to(self.args.device)
				#neighbor_patch_label = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [ 0, 0], [ 0, 1], [1, -1], [ 1, 0], [ 1, 1]], requires_grad=False).to(self.args.device)
				#neighbor_patch_label = torch.tensor([[1, 1], [0, 0], [1, 1], [0, 0], [0.5, 0.5], [0, 0], [0, 0], [0, 0], [0, 0]], requires_grad=False).to(self.args.device)
				#new_query[:,:,-2:] = neighbor_patch_label.unsqueeze(1)

				attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
				new_encoded_vectors = predicted_vectors[-1] + self.fc_sa(attn_vectors[4,:,:].squeeze())
				#new_encoded_vectors = self.fc_sa(attn_vectors[4,:,:].squeeze())

			elif x.shape[1] == self.n_patch:
				"""
				query, key, value = predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				"""

				query = predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query = torch.cat((query, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)

				#neighbor_patch_label = torch.tensor([[0.33,0.33], [0.33,0.66], [0.33,0.99], [0.66,0.33], [0.66,0.66], [0.66,0.99], [0.99,0.33], [0.99,0.33], [0.99,0.99]], requires_grad=False).to(self.args.device)
				#neighbor_patch_label = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [ 0, 0], [ 0, 1], [1, -1], [ 1, 0], [ 1, 1]], requires_grad=False).to(self.args.device)
				#neighbor_patch_label = torch.tensor([[1, 1], [0, 0], [1, 1], [0, 0], [0.5, 0.5], [0, 0], [0, 0], [0, 0], [0, 0]], requires_grad=False).to(self.args.device)
				#new_query[:,:,-2:] = neighbor_patch_label.unsqueeze(1)

				#attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
				attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
				new_encoded_vectors = predicted_vectors[-1] + self.fc_sa(attn_vectors[4,:,:].squeeze())
				#new_encoded_vectors = self.fc_sa(attn_vectors[4,:,:].squeeze())

			# save predicted latent vectors
			if time_idx < self.n_prediction - 1: # ignore last prediction
				predicted_vectors.append(self.ln2(self.fc2(new_encoded_vectors)))
				attn_weights_list.append(attn_weights.squeeze())

		output = predicted_vectors
		output = torch.stack(output, dim=0)
		
		return output


class LocalAttention_backup(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

		# parameters
		self.n_input = 48
		self.n_hidden = 48
		self.n_latent = 48
		self.n_batch = 0 # will be changed in the training code
		self.n_patch = 64*64 # will be changed in the training code

		self.n_observation = 10
		self.n_prediction = 50
	
		self.fc2 = nn.Linear(48, 48)
		self.ln2 = nn.LayerNorm([48])

		self.fc_sa = nn.Linear(48+2, 48)
		self.mhsa = nn.MultiheadAttention(embed_dim=48+2, num_heads=1)

		# Calculate the row and column indices for each element
		num_node = self.n_patch
		indices = torch.arange(num_node)
		rows = indices // int(num_node ** 0.5)
		cols = indices % int(num_node ** 0.5)

		# Generate the indices for the 9 neighboring elements of each element
		neighbor_indices = torch.tensor([
			[-1, -1], [-1, 0], [-1, 1],
			[ 0, -1], [ 0, 0], [ 0, 1],
			[ 1, -1], [ 1, 0], [ 1, 1]
		])

		# Calculate the row and column indices for the neighboring elements
		neighbor_rows = rows.view(-1, 1) + neighbor_indices[:, 0]
		neighbor_cols = cols.view(-1, 1) + neighbor_indices[:, 1]

		# Shift the out-of-bound indices
		neighbor_rows = neighbor_rows.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
		neighbor_rows[0,:,:] += 1
		neighbor_rows[-1,:,:] -= 1
		neighbor_rows = neighbor_rows.reshape(self.n_patch,-1)

		neighbor_cols = neighbor_cols.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
		neighbor_cols[:,0,:] += 1
		neighbor_cols[:,-1,:] -= 1
		neighbor_cols = neighbor_cols.reshape(self.n_patch, -1)

		self.neighbor_rows, self.neighbor_cols = neighbor_rows, neighbor_cols

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		pred_encoded_vectors = torch.zeros(self.n_batch * self.n_patch, self.n_hidden).to(self.args.device)

		encoded_vectors = x
		predicted_vectors = []
		
		# patch position as patch label: [x,y] indicates patch is at x-th row and y-th col
		patch_label_row = torch.arange(start=0, end=1, step=1 / int(self.n_patch ** 0.5)).to(self.args.device)
		patch_label_row = patch_label_row.unsqueeze(1).repeat(1, int(self.n_patch ** 0.5))
		patch_label_col = torch.arange(start=0, end=1, step=1 / int(self.n_patch ** 0.5)).to(self.args.device)
		patch_label_col = patch_label_col.unsqueeze(0).repeat(int(self.n_patch ** 0.5), 1)
		patch_label = torch.cat((patch_label_row.unsqueeze(0), patch_label_col.unsqueeze(0)), dim=0)
		patch_label = patch_label.reshape(2, -1).permute(1, 0)

		if x.shape[1] > self.n_patch:
			patch_label = patch_label.unsqueeze(1).repeat(1, self.n_batch, 1)
		elif x.shape[1] == self.n_patch:
			patch_label = patch_label.unsqueeze(1).repeat(1, self.n_batch, 1)

		attn_weights_list = []

		# observation
		for time_idx in range(self.n_observation):				
			# multi-head attention
			if x.shape[1] > self.n_patch: # for training data
				query, key, value = encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				#self.h:  torch.Size([16384, 32])
				#encoded_vectors[time_idx]:  torch.Size([16384, 32])
				#self.n_batch, self.n_patch, self.n_hidden:  4 4096 32

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
				new_encoded_vectors = encoded_vectors[time_idx] + self.fc_sa(attn_vectors[4,:,:].squeeze())

			elif x.shape[1] == self.n_patch: # for test data
				query, key, value = encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), encoded_vectors[time_idx].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
				new_encoded_vectors = encoded_vectors[time_idx] + self.fc_sa(attn_vectors[4,:,:].squeeze())

			# save predicted latent vectors
			predicted_vectors.append(self.ln2(self.fc2(new_encoded_vectors)))
			attn_weights_list.append(attn_weights.squeeze())

		# prediction
		for time_idx in range(self.n_prediction):
			# multi-head attention
			if x.shape[1] > self.n_patch:
				query, key, value = predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
				new_encoded_vectors = predicted_vectors[-1] + self.fc_sa(attn_vectors[4,:,:].squeeze())

			elif x.shape[1] == self.n_patch:
				query, key, value = predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), predicted_vectors[-1].reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				# local attention
				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
				attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
				new_encoded_vectors = predicted_vectors[-1] + self.fc_sa(attn_vectors[4,:,:].squeeze())

			# save predicted latent vectors
			if time_idx < self.n_prediction - 1: # ignore last prediction
				predicted_vectors.append(self.ln2(self.fc2(new_encoded_vectors)))
				attn_weights_list.append(attn_weights.squeeze())

		output = predicted_vectors
		output = torch.stack(output, dim=0)
		
		return output


class LSTM(nn.Module):
	"""
	batch : number of games to be trained together
	timestep : number of timestep for training each game
	input : latent vector (timestep, batch, 64)
	output : binary latent vector (timestep, batch, 64)
	"""
	def __init__(self, args):
		super().__init__()
		self.args = args

		# parameters
		self.n_input = 32 # 32 for default, 2023-11-13
		self.n_hidden = 32 # 32 for default, 2023-11-13
		self.n_latent = 32 # 32 for default, 2023-11-13
		self.n_layer = 1
		self.n_batch = 0 # will be changed in the training code
		self.n_patch = 64*64 # will be changed in the training code

		self.n_observation = 10
		self.n_prediction = 50
		
		# layers
		if self.args.mode == "cnn_lstm":
			self.fc1 = nn.Linear(512, 64) # nn.Linear(1152, 64) #for same_density2 101 x 101
			self.fc2 = nn.Linear(64, 512) # nn.Linear(64, 1152) # for same_density2 101 x 101
			self.ln1 = nn.LayerNorm([64])
			self.ln2 = nn.LayerNorm([64])

		if self.args.mode == "attention_lstm":
			self.fc2 = nn.Linear(32, 32) # (32, 32) for default
			self.ln2 = nn.LayerNorm([32])

			self.fc_sa = nn.Linear(32+2, 32) # (32+2, 32) for patch_label
			self.mhsa = nn.MultiheadAttention(embed_dim=32+2, num_heads=1) # embed_dim=32+2 for patch_label

			#self.gsa = GraphAttention(num_node=self.n_patch, args=self.args)
			num_node = self.n_patch
			# Calculate the row and column indices for each element
			indices = torch.arange(num_node)
			rows = indices // int(num_node ** 0.5)
			cols = indices % int(num_node ** 0.5)

			"""
			# Generate the indices for the 25 (5x5) neighboring elements of each element
			neighbor_indices = torch.tensor([
				[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
				[-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
				[0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
				[1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
				[2, -2], [2, -1], [2, 0], [2, 1], [2, 2],
			])

			# Generate the indices for the 16 (4x4) neighboring elements of each element - V1
			neighbor_indices = torch.tensor([
				[-1, -1], [-1, 0], [-1, 1], [-1, 2],
				[0, -1], [0, 0], [0, 1], [0, 2],
				[1, -1], [1, 0], [1, 1], [1, 2],
				[2, -1], [2, 0], [2, 1], [2, 2],
			])

			# Generate the indices for the 16 (4x4) neighboring elements of each element - V2
			neighbor_indices = torch.tensor([
				[-2, -2], [-2, -1], [-2, 0], [-2, 1],
				[-1, -2], [-1, -1], [-1, 0], [-1, 1],
				[0, -2], [0, -1], [0, 0], [0, 1],
				[1, -2], [1, -1], [1, 0], [1, 1],
			])

			# Generate the indices for the 9 (3x3) neighboring elements of each element
			neighbor_indices = torch.tensor([
				[-1, -1], [-1, 0], [-1, 1],
				[0, -1], [0, 0], [0, 1],
				[1, -1], [1, 0], [1, 1]
			])
			"""

			# Generate the indices for the 9 (3x3) neighboring elements of each element
			neighbor_indices = torch.tensor([
				[-1, -1], [-1, 0], [-1, 1],
				[0, -1], [0, 0], [0, 1],
				[1, -1], [1, 0], [1, 1]
			])

			# Calculate the row and column indices for the neighboring elements
			neighbor_rows = rows.view(-1, 1) + neighbor_indices[:, 0]
			neighbor_cols = cols.view(-1, 1) + neighbor_indices[:, 1]
			neighbor_rows = neighbor_rows.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
			neighbor_cols = neighbor_cols.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)

			if neighbor_indices.shape[0] == 9: # 3x3 perception
				# Shift the out-of-bound indices
				neighbor_rows[0,:,:] += 1
				neighbor_rows[-1,:,:] -= 1

				neighbor_cols[:,0,:] += 1
				neighbor_cols[:,-1,:] -= 1

			elif neighbor_indices.shape[0] == 16: # 4x4 perception
				# Shift the out-of-bound indices
				#""" V1
				neighbor_rows[0,:,:] += 1
				neighbor_rows[-1,:,:] -= 2
				neighbor_rows[-2,:,:] -= 1

				neighbor_cols[:,0,:] += 1
				neighbor_cols[:,-1,:] -= 2
				neighbor_cols[:,-2,:] -= 1
				#"""

				""" V2
				neighbor_rows[0,:,:] += 2
				neighbor_rows[1,:,:] += 1
				neighbor_rows[-1,:,:] -= 1

				neighbor_cols[:,0,:] += 2
				neighbor_cols[:,1,:] += 1
				neighbor_cols[:,-1,:] -= 1
				"""

			elif neighbor_indices.shape[0] == 25: # 5x5 perception
				# Shift the out-of-bound indices
				neighbor_rows[0,:,:] += 2
				neighbor_rows[1,:,:] += 1
				neighbor_rows[-1,:,:] -= 2
				neighbor_rows[-2,:,:] -= 1

				neighbor_cols[:,0,:] += 2
				neighbor_cols[:,1,:] += 1
				neighbor_cols[:,-1,:] -= 2
				neighbor_cols[:,-2,:] -= 1

			neighbor_rows = neighbor_rows.reshape(self.n_patch,-1)
			neighbor_cols = neighbor_cols.reshape(self.n_patch, -1)
			self.neighbor_rows, self.neighbor_cols = neighbor_rows, neighbor_cols

			#print("self.neighbor_rows: ", self.neighbor_rows, self.neighbor_rows.shape)
			#print("self.neighbor_cols: ", self.neighbor_cols, self.neighbor_cols.shape)
			#input("here")

		self.lstm = nn.LSTMCell(input_size=self.n_input, hidden_size=self.n_hidden)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


	def init_states(self):
		if self.n_patch == 0:
			self.h = torch.zeros(self.n_batch, self.n_hidden).to(self.args.device)
			self.c = torch.zeros(self.n_batch, self.n_hidden).to(self.args.device)
		else:
			self.h = torch.zeros(self.n_batch * self.n_patch, self.n_hidden).to(self.args.device)
			self.c = torch.zeros(self.n_batch * self.n_patch, self.n_hidden).to(self.args.device)

		num_node = self.n_patch
		# Calculate the row and column indices for each element
		indices = torch.arange(num_node)
		rows = indices // int(num_node ** 0.5)
		cols = indices % int(num_node ** 0.5)

		if self.neighbor_rows.shape[-1] == 9:
			# Generate the indices for the 9 neighboring elements of each element
			neighbor_indices = torch.tensor([
				[-1, -1], [-1, 0], [-1, 1],
				[0, -1], [0, 0], [0, 1],
				[1, -1], [1, 0], [1, 1]
			])

		elif self.neighbor_rows.shape[-1] == 16:
			#"""
			# Generate the indices for the 16 (4x4) neighboring elements of each element - V1
			neighbor_indices = torch.tensor([
				[-1, -1], [-1, 0], [-1, 1], [-1, 2],
				[0, -1], [0, 0], [0, 1], [0, 2],
				[1, -1], [1, 0], [1, 1], [1, 2],
				[2, -1], [2, 0], [2, 1], [2, 2],
			])
			#"""

			"""
			# Generate the indices for the 16 (4x4) neighboring elements of each element - V2
			neighbor_indices = torch.tensor([
				[-2, -2], [-2, -1], [-2, 0], [-2, 1],
				[-1, -2], [-1, -1], [-1, 0], [-1, 1],
				[0, -2], [0, -1], [0, 0], [0, 1],
				[1, -2], [1, -1], [1, 0], [1, 1],
			])
			"""

		elif self.neighbor_rows.shape[-1] == 25:
			# Generate the indices for the 25 (5x5) neighboring elements of each element
			neighbor_indices = torch.tensor([
				[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
				[-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
				[0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
				[1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
				[2, -2], [2, -1], [2, 0], [2, 1], [2, 2],
			])

		# Calculate the row and column indices for the neighboring elements
		neighbor_rows = rows.view(-1, 1) + neighbor_indices[:, 0]
		neighbor_cols = cols.view(-1, 1) + neighbor_indices[:, 1]
		neighbor_rows = neighbor_rows.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
		neighbor_cols = neighbor_cols.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)

		if neighbor_indices.shape[0] == 9: # 3x3 perception
			# Shift the out-of-bound indices
			neighbor_rows[0,:,:] += 1
			neighbor_rows[-1,:,:] -= 1

			neighbor_cols[:,0,:] += 1
			neighbor_cols[:,-1,:] -= 1

		elif neighbor_indices.shape[0] == 16: # 5x5 perception
			# Shift the out-of-bound indices
			#""" V1
			neighbor_rows[0,:,:] += 1
			neighbor_rows[-1,:,:] -= 2
			neighbor_rows[-2,:,:] -= 1

			neighbor_cols[:,0,:] += 1
			neighbor_cols[:,-1,:] -= 2
			neighbor_cols[:,-2,:] -= 1
			#"""
			
			""" V2
			neighbor_rows[0,:,:] += 2
			neighbor_rows[1,:,:] += 1
			neighbor_rows[-1,:,:] -= 1

			neighbor_cols[:,0,:] += 2
			neighbor_cols[:,1,:] += 1
			neighbor_cols[:,-1,:] -= 1	
			"""

		elif neighbor_indices.shape[0] == 25: # 5x5 perception
			# Shift the out-of-bound indices
			neighbor_rows[0,:,:] += 2
			neighbor_rows[1,:,:] += 1
			neighbor_rows[-1,:,:] -= 2
			neighbor_rows[-2,:,:] -= 1

			neighbor_cols[:,0,:] += 2
			neighbor_cols[:,1,:] += 1
			neighbor_cols[:,-1,:] -= 2
			neighbor_cols[:,-2,:] -= 1

		neighbor_rows = neighbor_rows.reshape(self.n_patch,-1)
		neighbor_cols = neighbor_cols.reshape(self.n_patch, -1)		
		self.neighbor_rows, self.neighbor_cols = neighbor_rows, neighbor_cols


	def forward(self, x, batch_size):
		self.n_batch = batch_size

		if self.args.mode == "attention_lstm":
			self.init_states()

			encoded_vectors = x
			predicted_vectors = []
			
			# patch position as patch label: [x,y] indicates patch is at x-th row and y-th col
			patch_label_row = torch.arange(start=0, end=1, step=1 / int(self.n_patch ** 0.5)).to(self.args.device)
			patch_label_row = patch_label_row.unsqueeze(1).repeat(1, int(self.n_patch ** 0.5))
			patch_label_col = torch.arange(start=0, end=1, step=1 / int(self.n_patch ** 0.5)).to(self.args.device)
			patch_label_col = patch_label_col.unsqueeze(0).repeat(int(self.n_patch ** 0.5), 1)
			patch_label = torch.cat((patch_label_row.unsqueeze(0), patch_label_col.unsqueeze(0)), dim=0)
			patch_label = patch_label.reshape(2, -1).permute(1, 0)

			if x.shape[1] > self.n_patch:
				patch_label = patch_label.unsqueeze(1).repeat(1, self.n_batch, 1)
			elif x.shape[1] == self.n_patch:
				patch_label = patch_label.unsqueeze(1).repeat(1, self.n_batch, 1)

			attn_weights_list = []

			# observation
			for time_idx in range(self.n_observation):				
				# multi-head attention
				if x.shape[1] > self.n_patch:
					self.h, self.c = self.lstm(encoded_vectors[time_idx], (self.h, self.c))
					query, key, value = self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
					query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

					# all-to-all attention
					#attn_vectors, attn_weights = self.mhsa(query, key, value)
					#self.h = self.h + self.fc_sa(attn_vectors.permute(1,0,2).reshape(-1, 66))

					# local attention
					#"""
					new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					new_query = new_query.permute(1,2,0,3).reshape(self.neighbor_rows.shape[-1], self.n_batch * self.n_patch, -1)
					#new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					#new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
					#new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					#new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)

					attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
					#attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
					if self.neighbor_rows.shape[-1] == 9:
						self.h = self.h + self.fc_sa(attn_vectors[4,:,:].squeeze())
					elif self.neighbor_rows.shape[-1] == 16:
						self.h = self.h + self.fc_sa(attn_vectors[5,:,:].squeeze()) # V1
						#self.h = self.h + self.fc_sa(attn_vectors[10,:,:].squeeze()) # V2
					elif self.neighbor_rows.shape[-1] == 25:
						self.h = self.h + self.fc_sa(attn_vectors[12,:,:].squeeze())
					#"""

				elif x.shape[1] == self.n_patch:
					self.h, self.c = self.lstm(encoded_vectors[time_idx], (self.h, self.c))
					query, key, value = self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
					query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

					# all-to-all attention
					#attn_vectors, attn_weights = self.mhsa(query, key, value)
					#self.h = self.h + self.fc_sa(attn_vectors.permute(1,0,2).reshape(-1, 66))

					# local attention
					#"""
					new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					new_query = new_query.permute(1,2,0,3).reshape(self.neighbor_rows.shape[-1], self.n_batch * self.n_patch, -1)
					#new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					#new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
					#new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					#new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)

					attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
					#attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
					if self.neighbor_rows.shape[-1] == 9:
						self.h = self.h + self.fc_sa(attn_vectors[4,:,:].squeeze())
					elif self.neighbor_rows.shape[-1] == 16:
						self.h = self.h + self.fc_sa(attn_vectors[5,:,:].squeeze()) # V1
						#self.h = self.h + self.fc_sa(attn_vectors[10,:,:].squeeze()) # V2
					elif self.neighbor_rows.shape[-1] == 25:
						self.h = self.h + self.fc_sa(attn_vectors[12,:,:].squeeze())

					#"""

				#"""

				# save predicted latent vectors
				predicted_vectors.append(self.ln2(self.fc2(self.h)))
				attn_weights_list.append(attn_weights.squeeze())

			# prediction
			for time_idx in range(self.n_prediction):
				# multi-head attention
				if x.shape[1] > self.n_patch:
					self.h, self.c = self.lstm(predicted_vectors[-1], (self.h, self.c))
					query, key, value = self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
					query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

					# all-to-all attention
					#attn_vectors, attn_weights = self.mhsa(query, key, value)
					#self.h = self.h + self.fc_sa(attn_vectors.permute(1,0,2).reshape(-1, 66))

					# local attention
					#"""
					new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					new_query = new_query.permute(1,2,0,3).reshape(self.neighbor_rows.shape[-1], self.n_batch * self.n_patch, -1)
					#new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					#new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
					#new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					#new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)

					attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
					#attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
					if self.neighbor_rows.shape[-1] == 9:
						self.h = self.h + self.fc_sa(attn_vectors[4,:,:].squeeze())
					elif self.neighbor_rows.shape[-1] == 16:
						self.h = self.h + self.fc_sa(attn_vectors[5,:,:].squeeze()) # V1
						#self.h = self.h + self.fc_sa(attn_vectors[10,:,:].squeeze()) # V2
					elif self.neighbor_rows.shape[-1] == 25:
						self.h = self.h + self.fc_sa(attn_vectors[12,:,:].squeeze())

					#"""

				elif x.shape[1] == self.n_patch:
					self.h, self.c = self.lstm(predicted_vectors[-1], (self.h, self.c))
					query, key, value = self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
					query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

					# all-to-all atention
					#attn_vectors, attn_weights = self.mhsa(query, key, value)
					#self.h = self.h + self.fc_sa(attn_vectors.permute(1,0,2).reshape(-1, 66))

					# local attention
					#"""
					new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					new_query = new_query.permute(1,2,0,3).reshape(self.neighbor_rows.shape[-1], self.n_batch * self.n_patch, -1)
					#new_key = key.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					#new_key = new_key.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)
					#new_value = value.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
					#new_value = new_value.permute(1,2,0,3).reshape(9, self.n_batch * self.n_patch, -1)

					#neighbor_patch_label = torch.tensor([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]], requires_grad=False).to(self.args.device)
					#new_query[:,:,-2:] = neighbor_patch_label.unsqueeze(1)
					#new_key[:,:,-2:] = neighbor_patch_label.unsqueeze(1)
					#new_value[:,:,-2:] = neighbor_patch_label.unsqueeze(1)

					attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
					#attn_vectors, attn_weights = self.mhsa(new_query, new_key, new_value)
					if self.neighbor_rows.shape[-1] == 9:
						self.h = self.h + self.fc_sa(attn_vectors[4,:,:].squeeze())
					elif self.neighbor_rows.shape[-1] == 16:
						self.h = self.h + self.fc_sa(attn_vectors[5,:,:].squeeze()) # V1
						#self.h = self.h + self.fc_sa(attn_vectors[10,:,:].squeeze()) # V2
					elif self.neighbor_rows.shape[-1] == 25:
						self.h = self.h + self.fc_sa(attn_vectors[12,:,:].squeeze())
					#"""

				#"""

				# save predicted latent vectors
				if time_idx < self.n_prediction - 1: # ignore last prediction
					predicted_vectors.append(self.ln2(self.fc2(self.h)))
					#attn_weights_list.append(attn_weights.squeeze()) # to save gpu memory

			output = predicted_vectors
			output = torch.stack(output, dim=0)
			
			""" plot attention score
			f, ax = plt.subplots(1, 7, figsize=(28, 4))
			for i in range(7):
				if i == 6:
					ax[i].imshow(attn_weights_list[10*i-2].detach().cpu())
				else:
					ax[i].imshow(attn_weights_list[10*i].detach().cpu())
			#plt.colorbar()
			plt.show()
			plt.savefig("attn_weights.png")
			plt.close()
			input("attention plot done at attn_weights.png")
			"""

			return output

class RNN(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args

		# parameters
		self.n_input = 32 # 32 for default, 2023-11-13
		self.n_hidden = 32 # 32 for default, 2023-11-13
		self.n_latent = 32 # 32 for default, 2023-11-13
		self.n_layer = 1
		self.n_batch = 0 # will be changed in the training code
		self.n_patch = 64*64 # will be changed in the training code

		self.n_observation = 10
		self.n_prediction = 50
		
		self.x2h = nn.Linear(32, 32)
		self.h2h = nn.Linear(32, 32)

		self.fc2 = nn.Linear(32, 32) # (32, 32) for default
		self.ln2 = nn.LayerNorm([32])

		self.fc_sa = nn.Linear(32+2, 32) # (32+2, 32) for patch_label
		self.mhsa = nn.MultiheadAttention(embed_dim=32+2, num_heads=1) # embed_dim=32+2 for patch_label

		#self.gsa = GraphAttention(num_node=self.n_patch, args=self.args)
		num_node = self.n_patch
		# Calculate the row and column indices for each element
		indices = torch.arange(num_node)
		rows = indices // int(num_node ** 0.5)
		cols = indices % int(num_node ** 0.5)

		# Generate the indices for the 9 (3x3) neighboring elements of each element
		neighbor_indices = torch.tensor([
			[-1, -1], [-1, 0], [-1, 1],
			[0, -1], [0, 0], [0, 1],
			[1, -1], [1, 0], [1, 1]
		])

		# Calculate the row and column indices for the neighboring elements
		neighbor_rows = rows.view(-1, 1) + neighbor_indices[:, 0]
		neighbor_cols = cols.view(-1, 1) + neighbor_indices[:, 1]
		neighbor_rows = neighbor_rows.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
		neighbor_cols = neighbor_cols.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)

		if neighbor_indices.shape[0] == 9: # 3x3 perception
			# Shift the out-of-bound indices
			neighbor_rows[0,:,:] += 1
			neighbor_rows[-1,:,:] -= 1

			neighbor_cols[:,0,:] += 1
			neighbor_cols[:,-1,:] -= 1

		neighbor_rows = neighbor_rows.reshape(self.n_patch,-1)
		neighbor_cols = neighbor_cols.reshape(self.n_patch, -1)
		self.neighbor_rows, self.neighbor_cols = neighbor_rows, neighbor_cols

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


	def init_states(self):
		if self.n_patch == 0:
			self.h = torch.zeros(self.n_batch, self.n_hidden).to(self.args.device)
		else:
			self.h = torch.zeros(self.n_batch * self.n_patch, self.n_hidden).to(self.args.device)

		num_node = self.n_patch
		# Calculate the row and column indices for each element
		indices = torch.arange(num_node)
		rows = indices // int(num_node ** 0.5)
		cols = indices % int(num_node ** 0.5)

		if self.neighbor_rows.shape[-1] == 9:
			# Generate the indices for the 9 neighboring elements of each element
			neighbor_indices = torch.tensor([
				[-1, -1], [-1, 0], [-1, 1],
				[0, -1], [0, 0], [0, 1],
				[1, -1], [1, 0], [1, 1]
			])

		# Calculate the row and column indices for the neighboring elements
		neighbor_rows = rows.view(-1, 1) + neighbor_indices[:, 0]
		neighbor_cols = cols.view(-1, 1) + neighbor_indices[:, 1]
		neighbor_rows = neighbor_rows.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)
		neighbor_cols = neighbor_cols.reshape(int(self.n_patch ** 0.5), int(self.n_patch ** 0.5),-1)

		if neighbor_indices.shape[0] == 9: # 3x3 perception
			# Shift the out-of-bound indices
			neighbor_rows[0,:,:] += 1
			neighbor_rows[-1,:,:] -= 1

			neighbor_cols[:,0,:] += 1
			neighbor_cols[:,-1,:] -= 1

		neighbor_rows = neighbor_rows.reshape(self.n_patch,-1)
		neighbor_cols = neighbor_cols.reshape(self.n_patch, -1)		
		self.neighbor_rows, self.neighbor_cols = neighbor_rows, neighbor_cols

	def forward(self, x):
		self.init_states()

		encoded_vectors = x
		predicted_vectors = []
		
		# patch position as patch label: [x,y] indicates patch is at x-th row and y-th col
		patch_label_row = torch.arange(start=0, end=1, step=1 / int(self.n_patch ** 0.5)).to(self.args.device)
		patch_label_row = patch_label_row.unsqueeze(1).repeat(1, int(self.n_patch ** 0.5))
		patch_label_col = torch.arange(start=0, end=1, step=1 / int(self.n_patch ** 0.5)).to(self.args.device)
		patch_label_col = patch_label_col.unsqueeze(0).repeat(int(self.n_patch ** 0.5), 1)
		patch_label = torch.cat((patch_label_row.unsqueeze(0), patch_label_col.unsqueeze(0)), dim=0)
		patch_label = patch_label.reshape(2, -1).permute(1, 0)

		if x.shape[1] > self.n_patch:
			patch_label = patch_label.unsqueeze(1).repeat(1, self.n_batch, 1)
		elif x.shape[1] == self.n_patch:
			patch_label = patch_label.unsqueeze(1).repeat(1, self.n_batch, 1)

		attn_weights_list = []

		# observation
		for time_idx in range(self.n_observation):				
			# multi-head attention
			if x.shape[1] > self.n_patch:
				self.h = torch.tanh(self.x2h(encoded_vectors[time_idx]) + self.h2h(self.h))
				query, key, value = self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(self.neighbor_rows.shape[-1], self.n_batch * self.n_patch, -1)

				attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
				if self.neighbor_rows.shape[-1] == 9:
					self.h = self.h + self.fc_sa(attn_vectors[4,:,:].squeeze())

			elif x.shape[1] == self.n_patch:
				self.h = torch.tanh(self.x2h(encoded_vectors[time_idx]) + self.h2h(self.h))
				query, key, value = self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(self.neighbor_rows.shape[-1], self.n_batch * self.n_patch, -1)

				attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
				if self.neighbor_rows.shape[-1] == 9:
					self.h = self.h + self.fc_sa(attn_vectors[4,:,:].squeeze())

			# save predicted latent vectors
			predicted_vectors.append(self.ln2(self.fc2(self.h)))
			attn_weights_list.append(attn_weights.squeeze())

		# prediction
		for time_idx in range(self.n_prediction):
			# multi-head attention
			if x.shape[1] > self.n_patch:
				self.h = torch.tanh(self.x2h(predicted_vectors[-1]) + self.h2h(self.h))
				query, key, value = self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(self.neighbor_rows.shape[-1], self.n_batch * self.n_patch, -1)

				attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
				if self.neighbor_rows.shape[-1] == 9:
					self.h = self.h + self.fc_sa(attn_vectors[4,:,:].squeeze())

			elif x.shape[1] == self.n_patch:
				self.h = torch.tanh(self.x2h(predicted_vectors[-1]) + self.h2h(self.h))
				query, key, value = self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2), self.h.reshape(self.n_batch, self.n_patch, self.n_hidden).permute(1,0,2)
				query, key, value = torch.cat((query, patch_label), dim=-1), torch.cat((key, patch_label), dim=-1), torch.cat((value, patch_label), dim=-1)

				new_query = query.reshape(int(self.n_patch ** 0.5),int(self.n_patch ** 0.5),self.n_batch,-1)[self.neighbor_rows, self.neighbor_cols, :, :]
				new_query = new_query.permute(1,2,0,3).reshape(self.neighbor_rows.shape[-1], self.n_batch * self.n_patch, -1)

				attn_vectors, attn_weights = self.mhsa(new_query, new_query, new_query)
				if self.neighbor_rows.shape[-1] == 9:
					self.h = self.h + self.fc_sa(attn_vectors[4,:,:].squeeze())

			# save predicted latent vectors
			if time_idx < self.n_prediction - 1: # ignore last prediction
				predicted_vectors.append(self.ln2(self.fc2(self.h)))

		output = predicted_vectors
		output = torch.stack(output, dim=0)
		
		return output


class ForestFirePredictor_ARNCA(nn.Module):
	def __init__(self, args):
		super().__init__()
		if args.mode == "attention_lstm":
			self.args = args
			self.encoder = HebbianBasicCNN(args)
			self.lstm = LSTM(args)
			self.decoder = Decoder(args)


	def forward(self, image, location):
		if "attention" not in self.args.mode:
			num_frame, num_batch, num_channel, image_size = image.shape[:-1]
			image = image.reshape(num_frame * num_batch, num_channel, image_size, image_size)
		else:
			num_frame, num_batch, num_channel, image_size = image.shape[:-1]
			image = image.reshape(num_frame, num_batch, num_channel, image_size*image_size).permute(0,1,3,2) # move rgb dimension to the last

		if self.args.mode == "attention_lstm":
			context = self.encoder(image, location)
			context = context.reshape(num_frame, num_batch*image_size*image_size, -1) # (num_frame=60, num_batch=4, num_patch=25, 64)
			pred_context = self.lstm(context, num_batch)
			N_burn, image_pred = self.decoder(pred_context, location) # prediction

		if self.args.mode in ["burning_prob", "hybrid", "attention_lstm"]:
			if len(image_pred.shape) == 1:
				image_pred = image_pred[:, None]

		if self.args.mode == "attention_lstm" or self.args.mode == "attention_rnn":
			return image_pred


if __name__ == '__main__':
	print("here")
	#model = HebbianLSTM()