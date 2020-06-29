# -*- coding: utf-8 -*-

import torch
import numpy as np
from skimage.transform import resize

class RandomCropTensor(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, random_state=np.random):
        self.size = size
        self.random_state = random_state

    def __call__(self, img):
        h, w = img.shape[1:3]
        th, tw = self.size
        if w == tw and h == th:
            return img
        x1 = self.random_state.randint(0, w - tw)
        y1 = self.random_state.randint(0, h - th)
        return img[:, y1: y1 + th, x1:x1 + tw]
    

class RandomFlipTensor(object):
    """
    Horizontal flip a given matrix randomly with a given probability
    """

    def __init__(self, p):
        self.p = p
        
    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be flipped.
        Returns:
            Numpy array: Randomly flipped matrix.
        """
        if np.random.random() < self.p:
            return data[:, :, ::-1] - np.zeros_like(data)
        return data
    
class ResizeTensor(object):
    """
    Rescale a matrix to a given size.

    Args:
        output_size (int or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of matrix edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[1:3]
        new_h, new_w = self.output_size
        out_data = np.zeros((data.shape[0], new_h, new_w))
        for c in range(data.shape[0]):
            out_data[c] = resize(data[c], (new_h, new_w), mode='constant', anti_aliasing=True)
        return out_data
    
class PadTensor(object):
    """
    Pad the given matrix on all sides with the given "pad" value.
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.
        Returns:
            PIL Image: Padded image.
        """
        p = self.padding
        return np.pad(img, ((0,0), (p[1], p[3]), (p[0], p[2])), self.padding_mode, constant_values=self.fill)

class NormalizeTensor(object):
    """
    Normalize an tensor with given mean (M1,...,Mn) and std (S1,..,Sn) 
    for n channels. This transform will normalize each channel of the input 
    input[channel] = (input[channel] - mean[channel]) / std[channel]
    
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Data of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized data
        """
        for c in range(data.shape[0]):
            data[c] = (data[c] - self.mean[c]) / self.std[c]
        return data   

class NoiseGaussian(object):
    """
    Adds gaussian noise to a given matrix.
    
    Args:
        factor (int): scale of the Gaussian noise. default: 1e-5
    """

    def __init__(self, factor=1e-5):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Noisy tensor with additive Gaussian noise
        """
        data = data + (np.random.randn(data.shape[0], data.shape[1]) * self.factor)
        return data;
    
class OutliersZeroRandom(object):
    """
    Randomly add zeroed-out outliers (without structure)
    
    Args:
        factor (int): Percentage of outliers to add. default: .25
    """

    def __init__(self, factor=.25):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Tensor with randomly zeroed-out outliers
        """
        dataSize = data.size
        tmpData = data.copy();
        # Add random outliers (here similar to dropout mask)
        tmpIDs = np.floor(np.random.rand(int(np.floor(dataSize * self.factor))) * dataSize)
        for i in range(tmpIDs.shape[0]):
            if (tmpIDs[i] < data.size):
                tmpData.ravel()[int(tmpIDs[i])] = 0
        return tmpData
    
class FilterMeanRandom(object):
    """
    Perform randomized abundance filtering (under mean)
    
    Args:
        factor (int): Percentage of outliers to add. default: .25
    """

    def __init__(self, factor=.25):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            tensor (Tensor): Tensor of data
        Returns:
            Tensor: Noisy tensor with additive Gaussian noise
        """
        data = data.copy()
        meanVal = np.mean(data[data > 0])
        cutThresh = np.random.rand(1) * (meanVal / 4)
        for iS in range(data.size):
            if (data.ravel()[iS] < cutThresh[0]):
                data.ravel()[iS] = 0;
        return data

class Binarize(object):
    """
    Binarize a given matrix
    """

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data[data > 0] = 1
        return data

class MaskRows(object):
    """
    Put random rows to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data = data.copy()
        tmpIDs = np.floor(np.random.rand(int(np.floor(data.shape[0] * self.factor))) * (data.shape[0]))
        for i in range(tmpIDs.shape[0]):
            if tmpIDs[i] < data.shape[0]:
                data[int(tmpIDs[i]), :] = 0
        return data

class MaskColumns(object):
    """
    Put random columns to zeros
    
    Args:
        factor (int): Percentage to be put to zero. default: .2
    """

    def __init__(self, factor=.2):
        self.factor = factor

    def __call__(self, data):
        """
        Args:
            data (Numpy array): Matrix to be masked
        Returns:
            Numpy array: Masked matrix.
        """
        data = data.copy()
        tmpIDs = np.floor(np.random.rand(int(np.floor(data.shape[1] * self.factor))) * (data.shape[1]))
        for i in range(tmpIDs.shape[0]):
            if tmpIDs[i] < data.shape[1]:
                data[:, int(tmpIDs[i])] = 0
        return data