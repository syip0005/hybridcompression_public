from unicodedata import bidirectional
import torch
from torch import nn
from torch.nn import functional as F
import logging, sys
from tqdm import tqdm
import numpy
from .gdn import GDN

scale_factor = 1

#######################################################################################################
### SUPPLEMENTARY CNN CALCULATORS
########################################################################################################

import numpy

def pool_calculator(input_height, input_width, kernel_height, kernel_width, stride_height,
                    stride_width, input_filters):

    """Calculates resulting shape of pooling layer
    """

    output_height_ = int(numpy.floor((input_height - kernel_height) / stride_height) + 1)
    output_width_ = int(numpy.floor((input_width - kernel_width) / stride_width) + 1)

    return (input_filters, output_height_, output_width_)

def conv_shape_calculator(input_height, input_width, kernel_shape, stride, padding, filters,
                         kernel_height = None, kernel_width = None):

    """Calculates resulting shape of convolution
    """
    
    if kernel_shape is not None:
        kernel_height = kernel_shape
        kernel_width = kernel_shape
        

    output_height_ = int(numpy.floor((input_height + 2*padding - kernel_height) / stride) + 1)
    output_width_ = int(numpy.floor((input_width + 2*padding - kernel_width) / stride) + 1)

    return (filters, output_height_, output_width_)

def inverse_conv_shape_calculator(output_height, output_width, kernel_shape, stride, padding, in_filters,
                                 kernel_height = None, kernel_width = None):
    
    """Calculates inverse of a convolution
    """
        
    if kernel_shape is not None:
        kernel_height = kernel_shape
        kernel_width = kernel_shape
    
    input_height = int(kernel_height - 2 * padding + (output_height -1 ) * stride)
    input_width = int(kernel_width - 2 * padding + (output_width -1 ) * stride)
    
    return (in_filters, input_height, input_width)
    

def tconv_shape_calculator(input_height, input_width, kernel_shape, stride, padding, out_padding, filters,
                          kernel_height = None, kernel_width = None):

    """Calculates resulting shape of transposed convolution
    """
    # TODO: update for height and widths
        
    if kernel_shape is not None:
        kernel_height = kernel_shape
        kernel_width = kernel_shape

    output_height_ = int((input_height - 1) * stride - 2 * padding +(kernel_height - 1) + out_padding + 1)
    output_width_ = int((input_width - 1) * stride - 2 * padding +(kernel_width - 1) + out_padding + 1)

    return (filters, output_height_, output_width_)


def inverse_tconv_shape_calculator(output_height, output_width, kernel_shape, 
                                   stride, padding, out_padding, filters,
                                  kernel_height = None, kernel_width = None,
                                  stride_height = None, stride_width = None,
                                  padding_height = None, padding_width = None,
                                  out_padding_height = None, out_padding_width = None):

    """Calculates resulting shape of transposed convolution
    """
        
    if kernel_shape is not None:
        kernel_height = kernel_shape
        kernel_width = kernel_shape
    if stride is not None:
        stride_height = stride
        stride_width = stride
    if padding is not None:
        padding_height = padding
        padding_width = padding
    if out_padding is not None:
        out_padding_height = out_padding
        out_padding_width = out_padding
    
    input_height_ = int((output_height - 1 - out_padding_height - (kernel_height - 1) + 2*padding_height) / stride_height + 1)
    input_width_ = int((output_width - 1 - out_padding_width - (kernel_width - 1) + 2*padding_width) / stride_width + 1)

    return (filters, input_height_, input_width_)

########################################################################################################
### MODELS
########################################################################################################

class SeparableConv2d(nn.Module):

    """
    Depth-wise Separable Convolution in PyTorch

    Basic implementation taken from: https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride, bias: bool = True, padding: int = 1):

        """
        Parameters
        ----------
        in_channels : int
            Input channels (filters)
        out_channels : int
            Number of filters for convolutional operation
        kernel_size : int
            Size of kernel, same as nn.Conv2d
        stride : int
            Stride size, same as nn.Conv2d
        """

        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                                   groups=in_channels, bias=bias, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):

        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class AE_Encoder(nn.Module):

    """
    Encoder for AE
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, input_days: int = 12, wide_freq: int = 48):

        super().__init__()

        input_n = input_days * wide_freq

        # Encoder
        self.fc1 = nn.Linear(in_features = input_n, out_features = 96 * scale_factor) 
        self.fc2 = nn.Linear(in_features = 96 * scale_factor, out_features = 48 * scale_factor) 
        self.fc3 = nn.Linear(in_features = 48 * scale_factor, out_features = 24 * scale_factor) 
        self.fc4 = nn.Linear(in_features = 24 * scale_factor, out_features = latent_n)

        # General
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        # x input: (N, input_n)
        
        logging.debug('Input Shape: %s', x.shape)
        
        x = F.relu(self.fc1(x)) # (N, 96)
        x = F.relu(self.fc2(x)) # (N, 48)
        x = F.relu(self.fc3(x)) # (N, 24)        
        x = F.relu(self.fc4(x)) # (N, latent_n)
        
        return x

class AE_Decoder(nn.Module):

    """
    Decoder for AE
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, input_days: int = 12, wide_freq: int = 48):

        super().__init__()

        input_n = input_days * wide_freq

         # Decoder
        self.fc5 = nn.Linear(in_features = latent_n, out_features = 24 * scale_factor)
        self.fc6 = nn.Linear(in_features = 24 * scale_factor, out_features = 48 * scale_factor)
        self.fc7 = nn.Linear(in_features = 48 * scale_factor, out_features = 96 * scale_factor)
        self.fc8 = nn.Linear(in_features = 96 * scale_factor, out_features = input_n)
        
        # General
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # x input: (N, latent_n)
        
        x = self.dropout(x)
        x = F.relu(self.fc5(x)) # (N, 24)
        x = F.relu(self.fc6(x)) # (N, 48)
        x = F.relu(self.fc7(x)) # (N, 96)        
        x = self.sigmoid(self.fc8(x)) # (N, in_shape)
        
        return x


class AE(nn.Module):

    """
    Basic fully-connected Auto-Encoder architecture with 4 encoder and 4 decoder layers.
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, input_days: int = 12, wide_freq: int = 48):

        """
        Parameters
        ----------
        dropout : float
            Dropout probability to beu sed
        input_days : int
            Number of input days - i.e., each example contains 48 x input_days (has to be same as CERN dataset)
        latent_n : int
            Number of neurons in latent layer
        wide_freq: int
            Number of features per row, generally the daily frequency (i.e., 48 for CER, 96 for UMASS)
        """

        super().__init__()

        self.encoder = AE_Encoder(dropout, latent_n, input_days, wide_freq)
        self.decoder = AE_Decoder(dropout, latent_n, input_days, wide_freq)
       
    def forward(self, x):

        latent = self.encoder(x)
        x = self.decoder(latent)

        return x, latent


class SCSAE_Encoder(nn.Module):

    """
    SCSAE Encoder
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2):

        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=2, padding=1) # padding to make work
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = SeparableConv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Calculate shape
        shape = (1, input_days * reshape_factor, wide_freq / reshape_factor)
        shape = conv_shape_calculator(shape[1], shape[2], kernel_shape = 3, stride = 2, padding = 1, filters = 16)
        shape = conv_shape_calculator(shape[1], shape[2], kernel_shape = 3, stride = 2, padding = 1, filters = 32)
        shape = conv_shape_calculator(shape[1], shape[2], kernel_shape = 3, stride = 1, padding = 1, filters = 64)
        shape = pool_calculator(shape[1], shape[2], 2, 2, 2, 2, shape[0])

        # Linear layer
        self.fc1 = nn.Linear(shape[0] * shape[1] * shape[2], latent_n)
        
        # Batch normalize layers
        if batchnormalize:
            self.batchnorm1 = nn.BatchNorm2d(16)
            self.batchnorm2 = nn.BatchNorm2d(32)
            self.batchnorm3 = nn.BatchNorm2d(64)
        else:
            self.batchnorm1, self.batchnorm2, self.batchnorm3 = nn.Identity(), nn.Identity(), nn.Identity()

    def forward(self, x):
        
        # x input: (N, 1, 24, 24)
        
        x = self.conv1(x)                       # (N, 16, 12, 12)
        x = self.batchnorm1(x)
        x = F.relu(x)
        
        x = self.conv2(x)                       # (N, 32, 6, 6)
        x = self.batchnorm2(x)
        x = F.relu(x)

        x = self.conv3(x)                       # (N, 64, 6, 6)
        x = self.batchnorm3(x)
        x = F.relu(x)
        
        x = self.pool1(x)                       # (N, 64, 3, 3)
        x = torch.flatten(x, start_dim = 1)     # (N, 64, 3, 3)
        x = self.fc1(x)                         # (N, latent_n)
        
        return x

class SCSAE_Decoder(nn.Module):

    """
    SCSAE Decoder
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2):

        super().__init__()

        # Required shape
        shape = (1, input_days * reshape_factor, wide_freq / reshape_factor)
        shape = inverse_conv_shape_calculator(shape[1], shape[2],
                                                    kernel_shape = 1, stride = 1, padding = 0, in_filters = 32)
        shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                    kernel_shape = 3, stride = 2, padding = 1, out_padding = 1, filters = 64)
        shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                    kernel_shape = 3, stride = 2, padding = 1, out_padding = 1, filters = 64)
        self.shape = shape

        # Decoder
        self.fc2 = nn.Linear(latent_n, shape[0] * shape[1] * shape[2])
        self.tconv1 = nn.ConvTranspose2d(in_channels=64, out_channels = 64, 
                                         kernel_size = 3, stride=2, padding=1, output_padding=1, bias=False) # pad and outpad to make it work
        self.tconv2 = nn.ConvTranspose2d(in_channels=64, out_channels = 32, 
                                         kernel_size = 3, stride=2, padding=1, output_padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels = 1, kernel_size = (1,1))
        
        # General
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # x input: (N, latent_n)
        
        x = self.fc2(x) # (N, 64*6*6)
        x = x.reshape(-1, self.shape[0], self.shape[1], self.shape[2]) # (N, 64, 6, 6)
        x = F.relu(self.tconv1(x)) # (N, 64, 12, 12)
        x = F.relu(self.tconv2(x)) # (N, 32, 24, 24)
        x = self.sigmoid(self.conv4(x)) # (N, 1, 24, 24)
        
        return x

class SCSAE(nn.Module):

    """
    SCSAE as per Wang et al. (2020) paper, my implementation.
    
    nb: Not all implementation details were included in the paper (such as padding) and I have incorporated details where
        required to ensure that model runs.
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2):

        """
        Parameters
        ----------
        dropout : float
            Dropout probability to be used
        latent_n : int
            Number of neurons in latent layer
        batchnormalize : bool
            Whether to apply batch normalization to CNN layers
        input_days : int
            Input days rows (needs to be same as dataset)
        wide_freq : bool
            Features per row (should be daily frequency (i.e, CER 48, UMASS 96))
        reshape_factor : int
            Same as dataset, to reshape into square by original authors
        """

        super().__init__()

        self.encoder = SCSAE_Encoder(dropout, latent_n, batchnormalize, input_days, wide_freq, reshape_factor)
        self.decoder = SCSAE_Decoder(dropout, latent_n, batchnormalize, input_days, wide_freq, reshape_factor)

    def forward(self, x):

        latent = self.encoder(x)
        x = self.decoder(latent)

        return x, latent

class HAE_Encoder(nn.Module):

    """
    HAE Encoder
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2):

        super().__init__()

        # Encoder
        self.lstm1 = nn.LSTM(input_size = wide_freq, hidden_size = 24, num_layers = 1, batch_first = True) # padding to make work
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = SeparableConv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride = (1,2))

        # Calculate shape
        shape = (1, input_days * reshape_factor, wide_freq / reshape_factor) # INPUT
        shape = (1, input_days * reshape_factor, 24 * 1) # LSTM1
        shape = conv_shape_calculator(shape[1], shape[2], kernel_shape = 3, stride = 2, padding = 1, filters = 32) # CONV2 
        shape = conv_shape_calculator(shape[1], shape[2], kernel_shape = 3, stride = 2, padding = 1, filters = 64) # CONV3
        shape = pool_calculator(shape[1], shape[2], 1, 2, 1, 2, shape[0]) # POOL

        # Linear layer
        self.fc1 = nn.Linear(shape[0] * shape[1] * shape[2], latent_n)

        # Batch normalize
        if batchnormalize:
            self.batchnorm1 = nn.BatchNorm2d(16)
            self.batchnorm2 = nn.BatchNorm2d(32)
            self.batchnorm3 = nn.BatchNorm2d(64)
        else:
            self.batchnorm1, self.batchnorm2, self.batchnorm3 = nn.Identity(), nn.Identity(), nn.Identity()

    def forward(self, x):

        # x input: (N, 1, 12, 48)
        
        logging.debug('Input Shape: %s', x.shape)

        x = x.squeeze(1)                        # for LSTM input
        
        x, _ = self.lstm1(x)                    # (N, 12, 24)
        x = x.unsqueeze(1)                     # (N, 1, 12, 24)

        x = self.conv2(x)                       # (N, 32, 6, 12)
        x = self.batchnorm2(x)
        x = F.relu(x)
        
        x = self.conv3(x)                       # (N, 64, 3, 6)
        x = self.batchnorm3(x)
        x = F.relu(x)

        x = self.pool1(x)
        
        x = x.flatten(start_dim = 1)     # (N, 64 * 3 * 6)

        x = self.fc1(x)                         # (N, latent_n)
        
        return x

class HAE_Decoder(nn.Module):

    """
    HAE Decoder
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2):

        super().__init__()

        # Calculate shape
        shape = (1, input_days * reshape_factor, wide_freq / reshape_factor)
        shape = inverse_conv_shape_calculator(shape[1], shape[2],
                                                    kernel_shape = 1, stride = 1, padding = 0, in_filters = 32) # conv4
        shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                    kernel_shape = 3, stride = 2, padding = 1, out_padding = 1, filters = 64) # tconv2
        shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                    kernel_shape = 3, stride = 2, padding = 1, out_padding = 1, filters = 64) # tconv1
        self.shape = shape

        # Decoder
        self.fc2 = nn.Linear(latent_n, shape[0] * shape[1] * shape[2])
        self.tconv1 = nn.ConvTranspose2d(in_channels=64, out_channels = 64, 
                                         kernel_size = 3, stride=2, padding=1, output_padding=1, bias=False) # pad and outpad to make it work
        self.tconv2 = nn.ConvTranspose2d(in_channels=64, out_channels = 32, 
                                         kernel_size = 3, stride=2, padding=1, output_padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels = 1, kernel_size = (1,1))
        
        # General
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # x input: (N, latent_n)
        
        x = self.fc2(x) # (N, 64*3*12)
        x = x.reshape(-1, self.shape[0], self.shape[1], self.shape[2]) # (N, 64, 3, 12)
        x = F.relu(self.tconv1(x)) # (N, 64, 6, 24)
        x = F.relu(self.tconv2(x)) # (N, 32, 12, 48)
        x = self.sigmoid(self.conv4(x)) # (N, 1, 12, 48)

        return x

class HAE(nn.Module):
    
    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 1):

        """
        Parameters
        ----------
        dropout : float
            Dropout probability to be used
        latent_n : int
            Number of neurons in latent layer
        """

        super().__init__()

        self.encoder = HAE_Encoder(dropout, latent_n, batchnormalize, input_days, wide_freq, reshape_factor)
        self.decoder = HAE_Decoder(dropout, latent_n, batchnormalize, input_days, wide_freq, reshape_factor)

    def forward(self, x):

        latent = self.encoder(x)
        x = self.decoder(latent)
        clipped_x = x.clamp(0., 1.)

        return clipped_x, latent
        

class HAE_Encoder_V2(nn.Module):

    """
    HAE Encoder
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2, device = 'cuda'):

        super().__init__()

        # Calculate LSTM shape
        LSTM_HIDDEN = 24

        # Encoder
        self.lstm1 = nn.LSTM(input_size = wide_freq, hidden_size = LSTM_HIDDEN, num_layers = 2, batch_first = True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = SeparableConv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride = (1,2))

        # Calculate shape
        shape = (1, input_days * reshape_factor, wide_freq / reshape_factor) # INPUT
        shape = (shape[0], shape[1], LSTM_HIDDEN) # LSTM1 
        shape = conv_shape_calculator(shape[1], shape[2], 3, 2, 1, 32) # CONV2
        gdn2_channels = shape[0]
        shape = conv_shape_calculator(shape[1], shape[2], 3, 2, 1, 64) # CONV2
        shape = pool_calculator(shape[1], shape[2], 1, 2, 1, 2, shape[0]) # POOL1
        gdn3_channels = shape[0]

        # Linear layer
        self.fc1 = nn.Linear(shape[0] * shape[1] * shape[2], latent_n)

        # Batch normalize
        if batchnormalize:
            # self.gdn1 = GDN(gdn1_channels, device)
            self.gdn2 = GDN(gdn2_channels, device)
            self.gdn3 = GDN(gdn3_channels, device)
            # self.gdn1 = nn.PReLU(gdn1_channels)
            # self.gdn2 = nn.PReLU(gdn2_channels)
            # self.gdn3 = nn.PReLU(gdn3_channels)            
        else:
            self.gdn1, self.gdn2, self.gdn3 = nn.Identity(), nn.Identity(), nn.Identity()

    def forward(self, x):

        # x input: (N, 1, 12, 48)
        
        logging.debug('Input Shape: %s', x.shape)

        x = x.squeeze(1)                        # for LSTM input
        x, _ = self.lstm1(x)                    # tanh non linearity
        x = x.unsqueeze(1)                    

        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.gdn3(x)
        
        x = x.flatten(start_dim = 1)     # (N, 64 * 3 * 6)

        x = self.fc1(x)                         # (N, latent_n)
        
        return x

class HAE_Decoder_V2(nn.Module):

    """
    HAE Decoder
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2, device = 'cuda'):

        super().__init__()

        if (input_days * reshape_factor) % 4 == 3:

            # Calculate shape
            shape = (1, input_days * reshape_factor, wide_freq / reshape_factor)
            shape = inverse_conv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 1, stride = 1, padding = 0, in_filters = 32) # conv4
            gdn2_channels = shape[0]
            shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 2, stride = 2, padding = 1, out_padding = None, 
                                                        filters = 64, out_padding_height = 1, out_padding_width = 0) # tconv2
            gdn1_channels = shape[0]
            shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 5, stride = 2, padding = 1, out_padding = None, 
                                                        filters = 64, out_padding_height=1, out_padding_width=0) # tconv1
            self.shape = shape

            # Decoder
            self.fc2 = nn.Linear(latent_n, shape[0] * shape[1] * shape[2])
            self.tconv1 = nn.ConvTranspose2d(in_channels=64, out_channels = 64, 
                                            kernel_size = (5,5), stride=2, padding=(1,1), output_padding=(1,0), bias=False) # pad and outpad to make it work
            self.tconv2 = nn.ConvTranspose2d(in_channels=64, out_channels = 32, 
                                            kernel_size = 2, stride=2, padding=1, output_padding=(1,0), bias=False)
            self.conv4 = nn.Conv2d(in_channels=32, out_channels = 1, kernel_size = (1,1))

        elif (input_days * reshape_factor) % 4 == 0:

            shape = (1, input_days * reshape_factor, wide_freq / reshape_factor)
            shape = inverse_conv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 1, stride = 1, padding = 0, in_filters = 32) # conv4
            gdn2_channels = shape[0]
            shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 3, stride = 2, padding = 1, out_padding = 1, 
                                                        filters = 64) # tconv2
            gdn1_channels = shape[0]
            shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 3, stride = 2, padding = 1, out_padding = 1, 
                                                        filters = 64) # tconv1
            self.shape = shape

            # Decoder
            self.fc2 = nn.Linear(latent_n, shape[0] * shape[1] * shape[2])
            self.tconv1 = nn.ConvTranspose2d(in_channels=64, out_channels = 64, 
                                            kernel_size = 3, stride=2, padding=1, output_padding=1, bias=False) # pad and outpad to make it work
            self.tconv2 = nn.ConvTranspose2d(in_channels=64, out_channels = 32, 
                                            kernel_size = 3, stride=2, padding=1, output_padding=1, bias=False)
            self.conv4 = nn.Conv2d(in_channels=32, out_channels = 1, kernel_size = (1,1))

        else:
            raise NotImplementedError('please reshape')
        
        # General
        self.sigmoid = nn.Sigmoid()

        if batchnormalize:
            self.gdn1 = GDN(gdn1_channels, device)
            self.gdn2 = GDN(gdn2_channels, device)
            # self.gdn1 = nn.PReLU(gdn1_channels)
            # self.gdn2 = nn.PReLU(gdn2_channels)
        else:
            self.gdn1, self.gdn2 = nn.Identity(), nn.Identity()

    def forward(self, x):
        
        # x input: (N, latent_n)
        
        x = self.fc2(x) # (N, 64*3*12)
        x = x.reshape(-1, self.shape[0], self.shape[1], self.shape[2]) # (N, 64, 3, 12)
        x = self.tconv1(x) # (N, 64, 6, 24)
        x = self.gdn1(x)
        x = self.tconv2(x) # (N, 32, 12, 48)
        x = self.gdn2(x)
        x = self.sigmoid(self.conv4(x)) # (N, 1, 12, 48)

        return x

class HAE_V2(nn.Module):

    """
    HAE V2
    Allows 7 days shape, and uses GDN.
    """
    
    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 1, device = 'cuda'):

        """
        Parameters
        ----------
        dropout : float
            Dropout probability to be used
        latent_n : int
            Number of neurons in latent layer
        """

        super().__init__()

        self.encoder = HAE_Encoder_V2(dropout, latent_n, batchnormalize, input_days, wide_freq, reshape_factor, device)
        self.decoder = HAE_Decoder_V2(dropout, latent_n, batchnormalize, input_days, wide_freq, reshape_factor, device)

    def forward(self, x):

        latent = self.encoder(x)
        x = self.decoder(latent)
        clipped_x = x.clamp(0., 1.)

        return clipped_x, latent


class HAE_Encoder_V3(nn.Module):

    """
    HAE Encoder
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2, device = 'cuda'):

        super().__init__()

        # Calculate LSTM shape
        LSTM_HIDDEN = 48

        # Encoder
        self.lstm1 = nn.LSTM(input_size = wide_freq, hidden_size = LSTM_HIDDEN, num_layers = 3, batch_first = True) # 1 x 12 x 48
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0) # 32 x 10 x 46 
        gdn2_channels = 32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0) # 64 x 10 x 46
        self.pool1 = nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)) # 64 x 10 x 23           
        gdn3_channels = 64
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=2, stride=2, padding=(0,1)) # 96 x 5 x 12
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=1, stride=1, padding=0) # 128 x 5 x 12
        self.pool2 = nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)) # 128 x 5 x 6 
        gdn4_channels = 128
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=1, stride=1, padding=0) # 256 x 5 x 6
        gdn5_channels = 192
        # self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0) # 256 x 5 x 6
        # Linear layer
        self.fc1 = nn.Linear(192*29*6, latent_n)

        # Batch normalize
        if batchnormalize:
            self.gdn2 = GDN(gdn2_channels, device)
            self.gdn3 = GDN(gdn3_channels, device)
            self.gdn4 = GDN(gdn4_channels, device)
            self.gdn5 = GDN(gdn5_channels, device)
       
        else:
            self.gdn1, self.gdn2, self.gdn3, self.gdn4, self.gdn5 = nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()

    def forward(self, x):

        # x input: (N, 1, 12, 48)
        
        logging.debug('Input Shape: %s', x.shape)

        x = x.squeeze(1)                        # for LSTM input
        x, _ = self.lstm1(x)                    # tanh non linearity
        x = x.unsqueeze(1)                    

        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.gdn4(x)
        x = self.conv6(x)
        x = self.gdn5(x)
        # x = self.conv7(x)
        x = x.flatten(start_dim = 1)     # (N, 64 * 3 * 6)
        x = self.fc1(x)                         # (N, latent_n)
        
        return x

class HAE_Decoder_V3(nn.Module):

    """
    HAE Decoder
    """

    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 2, device = 'cuda'):

        super().__init__()

        if (input_days * reshape_factor) % 4 == 3:

            # Calculate shape
            shape = (1, input_days * reshape_factor, wide_freq / reshape_factor)
            shape = inverse_conv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 1, stride = 1, padding = 0, in_filters = 32) # conv4
            gdn2_channels = shape[0]
            shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 2, stride = 2, padding = 1, out_padding = None, 
                                                        filters = 64, out_padding_height = 1, out_padding_width = 0) # tconv2
            gdn1_channels = shape[0]
            shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 5, stride = 2, padding = 1, out_padding = None, 
                                                        filters = 64, out_padding_height=1, out_padding_width=0) # tconv1
            self.shape = shape

            # Decoder
            self.fc2 = nn.Linear(latent_n, shape[0] * shape[1] * shape[2])
            self.tconv1 = nn.ConvTranspose2d(in_channels=64, out_channels = 64, 
                                            kernel_size = (5,5), stride=2, padding=(1,1), output_padding=(1,0), bias=False) # pad and outpad to make it work
            self.tconv2 = nn.ConvTranspose2d(in_channels=64, out_channels = 32, 
                                            kernel_size = 2, stride=2, padding=1, output_padding=(1,0), bias=False)
            self.conv4 = nn.Conv2d(in_channels=32, out_channels = 1, kernel_size = (1,1))

        elif (input_days * reshape_factor) % 4 == 0:

            shape = (1, input_days * reshape_factor, wide_freq / reshape_factor)
            shape = inverse_conv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 1, stride = 1, padding = 0, in_filters = 32) # conv4
            gdn2_channels = shape[0]
            shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 3, stride = 2, padding = 1, out_padding = 1, 
                                                        filters = 64) # tconv2
            gdn1_channels = shape[0]
            shape = inverse_tconv_shape_calculator(shape[1], shape[2],
                                                        kernel_shape = 3, stride = 2, padding = 1, out_padding = 1, 
                                                        filters = 64) # tconv1
            self.shape = shape

            # Decoder
            self.fc2 = nn.Linear(latent_n, shape[0] * shape[1] * shape[2])
            self.tconv1 = nn.ConvTranspose2d(in_channels=64, out_channels = 64, 
                                            kernel_size = 3, stride=2, padding=1, output_padding=1, bias=False) # pad and outpad to make it work
            self.tconv2 = nn.ConvTranspose2d(in_channels=64, out_channels = 32, 
                                            kernel_size = 3, stride=2, padding=1, output_padding=1, bias=False)
            self.conv4 = nn.Conv2d(in_channels=32, out_channels = 1, kernel_size = (1,1))

        else:
            raise NotImplementedError('please reshape')
        
        # General
        self.sigmoid = nn.Sigmoid()

        if batchnormalize:
            self.gdn1 = GDN(gdn1_channels, device)
            self.gdn2 = GDN(gdn2_channels, device)
            # self.gdn1 = nn.PReLU(gdn1_channels)
            # self.gdn2 = nn.PReLU(gdn2_channels)
        else:
            self.gdn1, self.gdn2 = nn.Identity(), nn.Identity()

    def forward(self, x):
        
        # x input: (N, latent_n)
        
        x = self.fc2(x) # (N, 64*3*12)
        x = x.reshape(-1, self.shape[0], self.shape[1], self.shape[2]) # (N, 64, 3, 12)
        x = self.tconv1(x) # (N, 64, 6, 24)
        x = self.gdn1(x)
        x = self.tconv2(x) # (N, 32, 12, 48)
        x = self.gdn2(x)
        x = self.sigmoid(self.conv4(x)) # (N, 1, 12, 48)

        return x

class HAE_V3(nn.Module):

    """
    HAE V2
    Allows 7 days shape, and uses GDN.
    Used to test theory about expanding filter sizes.
    """
    
    def __init__(self, dropout: float = 0.1, latent_n: int = 48, batchnormalize: bool = True,
    input_days: int = 12, wide_freq: int = 48, reshape_factor: int = 1, device = 'cuda'):

        """
        Parameters
        ----------
        dropout : float
            Dropout probability to be used
        latent_n : int
            Number of neurons in latent layer
        """

        super().__init__()

        self.encoder = HAE_Encoder_V3(dropout, latent_n, batchnormalize, input_days, wide_freq, reshape_factor, device)
        self.decoder = HAE_Decoder_V3(dropout, latent_n, batchnormalize, input_days, wide_freq, reshape_factor, device)

    def forward(self, x):

        latent = self.encoder(x)
        x = self.decoder(latent)
        clipped_x = x.clamp(0., 1.)

        return clipped_x, latent


