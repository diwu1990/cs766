from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, img_as_float, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython import display
# Ignore warnings
import warnings
import csv
import copy
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class HE_SHG_Dataset(Dataset):

    def __init__(self, csv_file, transform=None):
     
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        he_name = os.path.join(self.files_list.iloc[idx, 0])
        shg_name = os.path.join(self.files_list.iloc[idx, 1])

        he_image = io.imread(he_name)
        he_image = img_as_float(he_image)
        shg_image= io.imread(shg_name)
        shg_image= img_as_float(shg_image)
        
        sample = {'input': he_image, 'output': shg_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Compose(object):


    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class Rescale(object):


    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        input, output = sample['input'], sample['output']

        h, w = input.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        he_img = transform.resize(input, (new_h, new_w))
        shg_img = transform.resize(output, (new_h, new_w))

  

        return {'input': he_img, 'output': shg_img}


class RandomCrop(object):
    

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        input, output = sample['input'], sample['output']

        h, w = input.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input = input[top: top + new_h,
                      left: left + new_w]

        output = output[top: top + new_h,
                      left: left + new_w]

        return {'input': input, 'output': output}


class ToTensor(object):

    def __call__(self, sample):
        input, output = sample['input'], sample['output']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input = input.transpose((2, 0, 1))
        output = output.transpose((0, 1))

        return {'input': torch.from_numpy(input),
                'output': torch.from_numpy(output)}
    
class Normalize(object):
    

#     def __init__(self, mean, std):
#          self.mean = mean
#          self.std = std

    def __call__(self, sample):
        
        #nparray
        input, output = sample['input'], sample['output']
        
        # get HE RGB mean
        mean = np.mean(input, axis=(0, 1))
#         r_mean = np.mean(input, axis=(2, 1))
#         g_mean = np.mean(input, axis=(2, 2))
#         b_mean = np.mean(input, axis=(2, 3))
        # get HE RGB std
        std = np.std(input, axis=(0, 1))
#         r_std = np.std(input, axis=(2, 1))
#         g_std = np.std(input, axis=(2, 2))
#         b_std = np.std(input, axis=(2, 3))
        # normalize
        input = (input-mean)/std
#        print(str(mean) + " " + str(std))
#         input[:, :, 1] = (input[:, :, 0]-r_mean)/r_std
#         input[:, :, 2] = (input[:, :, 1]-g_mean)/g_std
#         input[:, :, 3] = (input[:, :, 2]-b_mean)/b_std
        
#         gray = color.rgb2gray(input)
        
#         gray_mean=np.mean(gray)
#         gray_std=np.std(gray)
        
        SHG_mean = np.mean(output)
        SHG_std = np.std(output)
        output = output/1 # not unit variance!
#         output = (output+gray_mean) * gray_std
        

        
        # HE
#         for t, m, s in zip(input, self.mean, self.std):
#             t.sub_(m).div_(s)
        sample['input'] = input
        sample['output'] = output
        
        return sample

    
def get_default_image_length():
    return 100
def get_default_input_channels():
    return 3
def get_default_batch_size():
    return 32
def get_default_num_workers():
    return 1

def get_csv_path():
    path = os.path.join(os.getcwd(), 'Data', 'image_files.csv')
    return path

def generate_csv():
    cwd = os.getcwd()
    csvFilePath = os.path.join(cwd, 'Data', 'image_files.csv')
    hePath = os.path.join(cwd, 'Data', 'HE_JPEG')
    shgPath = os.path.join(cwd, 'Data', 'SHG_JPEG')
    numOfPatch = len([name for name in os.listdir(hePath) if 
                      os.path.isfile(os.path.join(hePath, name))])
    with open(csvFilePath, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL)
        for i in range(numOfPatch):
            he = os.path.join(hePath, str(i+1) + '.jpeg')
            shg = os.path.join(shgPath, str(i+1) + '.jpeg')
            filewriter.writerow([he, shg])


def show_patch(dataloader, index = 3):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['input'].size(), 
              sample_batched['output'].size())

        # observe 4th batch and stop.
        if i_batch == index:
            plt.figure()
            input_batch, label_batch = sample_batched['input'], sample_batched['output']
            batch_size = len(input_batch)
            im_size = input_batch.size(2)
            label_batch=label_batch.reshape([batch_size,1,im_size,im_size])
            print(label_batch.size())
#             input_batch = unnormalize_img(input_batch, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             label_batch = unnormalize_img(label_batch, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


            grid = utils.make_grid(input_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.figure()

            grid = utils.make_grid(label_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
        
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

def tensormeanstd(img):
    mean = torch.tensor([0.0, 0.0, 0.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    for i, channel in enumerate(img):
        mean[i] = torch.mean(img[i, :, :])
        std[i] = torch.std(img[i, :, :])
    return mean, std
            
def unnormalize_img(batch, mean, std):
    for img, imgmean, imgstd in zip(batch, mean, std):
        for t, m, s in zip(img, imgmean, imgstd):
            t.mul_(s).add_(m)
    return batch

def normalize_img(batch, mean, std):
    for img, imgmean, imgstd in zip(batch, mean, std):
        for t, m, s in zip(img, imgmean, imgstd):
            t.sub_(m).div_(s)
    return batch

def get_one_batch(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['input'].size(), 
              sample_batched['output'].size())
        if i_batch == 0:  
            break
    return sample_batched

def batch_mean_std(input_batch):
    batch_size = len(input_batch)
    batch_mean = torch.zeros([3, batch_size])
    batch_std = torch.zeros([3, batch_size])
    for i, img in enumerate(input_batch):
        batch_mean[:, i], batch_std[:, i] = tensormeanstd(img)    
    return batch_mean.permute(1, 0), batch_std.permute(1, 0)

def show_one_batch(sample, meanHE, stdHE, meanSHG, stdSHG):
    input_batch, label_batch = sample['input'], sample['output']
    batch_size = len(input_batch)
    im_size = input_batch.size(2)
    label_batch=label_batch.reshape([batch_size,1,im_size,im_size])
    print(label_batch.size())
    
    input_batch = unnormalize_img(input_batch, meanHE, stdHE)
    stdSHG = torch.ones(3, batch_size)
#     label_batch = unnormalize_img(label_batch, meanSHG, stdSHG)
    
    grid = utils.make_grid(input_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.figure()

    grid = utils.make_grid(label_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
        
    plt.axis('off')
    plt.ioff()
    plt.show()   
    return

def normalizebatch(sample):
    input_batch, label_batch = sample['input'], sample['output']
    batch_size = len(input_batch)
    im_size = input_batch.size(2)
    label_batch=label_batch.reshape([batch_size,1,im_size,im_size])
    print(label_batch.size())
    
    meanHE, stdHE = batch_mean_std(input_batch)
    meanSHG, stdSHG = batch_mean_std(label_batch)
    input_batch = normalize_img(input_batch, meanHE, stdHE)
    stdSHG = torch.ones(3, batch_size)
#     label_batch = normalize_img(label_batch, meanSHG, stdSHG)
    
    grid = utils.make_grid(input_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.figure()

    grid = utils.make_grid(label_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
        
    plt.axis('off')
    plt.ioff()
    plt.show() 
    sample['input'] = input_batch
    sample['output'] = label_batch
    
    return(sample, meanHE, stdHE, meanSHG, stdSHG)