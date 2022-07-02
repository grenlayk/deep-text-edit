# source: https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/master/model.py
# 
# Copyright (c) 2019 Siskon

import torch
import torch.nn as nn
import math

# Constraints
# Input: [batch_size, in_channels, height, width]

# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''
    def __init__(self, name):
        self.name = name
    
    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        
        return weight * math.sqrt(2 / fan_in)
    
    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)
    
    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)

# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module

# Uniformly set the hyperparameters of Linears
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
# 5/13: Apply scaled weights
class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        
        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)

# Uniformly set the hyperparameters of Conv2d
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
# 5/13: Apply scaled weights
class SConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        
        self.conv = quick_scale(conv)

    def forward(self, x):
        return self.conv(x)

# Normalization on every element of input vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

# "learned affine transform" A
class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style
    
# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''
    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)
        
    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias  
        return result

# "learned per-channel scaling factors" B
# 5/13: Debug - tensor -> nn.Parameter
class Scale_B(nn.Module):
    '''
    Learned per-channel scale factor, used to scale the noise
    '''
    def __init__(self, n_channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))
    
    def forward(self, noise):
        result = noise * self.weight
        return result 

# Early convolutional block
class Early_StyleConv_Block(nn.Module):
    '''
    This is the very first block of generator that get the constant value as input
    '''
    def __init__ (self, n_channel, dim_latent):
        super().__init__()
        # Style generators
        self.style1   = FC_A(dim_latent, n_channel)
        self.style2   = FC_A(dim_latent, n_channel)
        # Noise processing modules
        self.noise1   = quick_scale(Scale_B(n_channel))
        self.noise2   = quick_scale(Scale_B(n_channel))
        # AdaIn
        self.adain    = AdaIn(n_channel)
        self.lrelu    = nn.LeakyReLU(0.2)
        # Convolutional layer
        self.conv     = SConv2d(n_channel, n_channel, 3, padding=1)


    def forward(self, x, latent_w):
        noise1 = torch.normal(mean=0,std=torch.ones(x.shape)).cuda()
        result = x + self.noise1(noise1)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv(result)
        noise2 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        result = result + self.noise2(noise2)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)
        
        return result
    
# General convolutional blocks
class StyleConv_Block(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''
    def __init__ (self, in_channel, out_channel, dim_latent):
        super().__init__()
        # Style generators
        self.style1   = FC_A(dim_latent, out_channel)
        self.style2   = FC_A(dim_latent, out_channel)
        # Noise processing modules
        self.noise1   = quick_scale(Scale_B(out_channel))
        self.noise2   = quick_scale(Scale_B(out_channel))
        # AdaIn
        self.adain    = AdaIn(out_channel)
        self.lrelu    = nn.LeakyReLU(0.2)
        # Convolutional layers
        self.conv1    = SConv2d(in_channel, out_channel, 3, padding=1)
        self.conv2    = SConv2d(out_channel, out_channel, 3, padding=1)
    
    def forward(self, previous_result, latent_w):
        # Conv 3*3
        result = self.conv1(previous_result)
        #noise1 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        #noise2 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        # Conv & Norm
        #result = result + self.noise1(noise1)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv2(result)
        #result = result + self.noise2(noise2)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)
        
        return result    
        
# Main components
class Intermediate_Generator(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''
    def __init__(self, dim_latent):
        super().__init__()
        layers = [PixelNorm()]
        layers.append(SLinear(dim_latent, dim_latent))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(SLinear(dim_latent, dim_latent))
        layers.append(nn.LeakyReLU(0.2))
            
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, latent_z):
        latent_w = self.mapping(latent_z.squeeze())
        return latent_w    

# Generator
class StyleBased_Generator(nn.Module):
    '''
    Main Module
    '''
    def __init__(self, dim_latent):
        super().__init__()
        # Waiting to adjust the size
        self.fcs    = Intermediate_Generator(dim_latent)
        self.first_conv = nn.Identity()
        self.convs  = nn.ModuleList([
            nn.Identity(),# Early_StyleConv_Block(512, dim_latent),
            StyleConv_Block(512, 512, dim_latent),
            StyleConv_Block(512, 512, dim_latent),
            StyleConv_Block(512, 512, dim_latent),
            StyleConv_Block(512, 256, dim_latent),
            StyleConv_Block(256, 128, dim_latent),
            StyleConv_Block(128, 64, dim_latent),
            StyleConv_Block(64, 32, dim_latent),
            StyleConv_Block(32, 16, dim_latent)
        ])
        self.to_rgbs = nn.ModuleList([
            SConv2d(512, 3, 1),
            SConv2d(512, 3, 1),
            SConv2d(512, 3, 1),
            SConv2d(512, 3, 1),
            SConv2d(256, 3, 1),
            SConv2d(128, 3, 1),
            SConv2d(64, 3, 1),
            SConv2d(32, 3, 1),
            SConv2d(16, 3, 1)
        ])
    def forward(self, 
                content_embed, 
                style_embed, 
                step = 5,       # Step means how many layers (count from 4 x 4) are used to train
                alpha=-1):      # Alpha is the parameter of smooth conversion of resolution):

        latent_w = self.fcs(style_embed)
        result = 0

        first_content = self.first_conv(content_embed)

        for i, conv in enumerate(self.convs):
                
            # Not the first layer, need to upsample
            if i > 0 and step > 0:
                result_upsample = nn.functional.interpolate(result, scale_factor=2, mode='bilinear',
                                                            align_corners=False)
                result = conv(result_upsample, latent_w)
            else:
                result = first_content
            
            # Final layer, output rgb image
            if i == step:
                result = self.to_rgbs[i](result)
                
                if i > 0 and 0 <= alpha < 1:
                    result_prev = self.to_rgbs[i - 1](result_upsample)
                    result = alpha * result + (1 - alpha) * result_prev
                    
                # Finish and break
                break
        
        return result