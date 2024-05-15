import torch
import torch.nn as nn 
import numpy
from numpy import random
from PIL import Image
import math
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import gaussian_random_fields as gr
from torchvision.utils import save_image
import time

class ProRandomConv():
  def __init__(self, img, size=3):
    self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    self.image = img.to(self.device)
    self.color = 3
    batch_size = img.shape[0]

    epsilon = 0.0001 #paper says small value
    b_delta = 0.5
    sigma_weight = 1 / math.sqrt(size*size*self.color)
    sigma_delta = torch.rand(1) * (epsilon - b_delta) + b_delta
    sigma_g = random.uniform(low=epsilon,high=1.0)

    self.gamma = torch.tensor([random.normal(loc=0.0, scale=0.5)], device=self.device)
    self.beta = torch.tensor([random.normal(loc=0.0, scale=0.5)], device=self.device)
    self.epsilon = epsilon
    
    #initialize weight of filter
    self.filters = torch.zeros(3,3,size,size).to(self.device)
    nn.init.kaiming_normal_(self.filters, nonlinearity='conv2d')
    for i in range(-1,2):
      for j in range(-1,2):
        g = math.exp(-(i*i+j*j)/(2*sigma_g*sigma_g))
        self.filters[:,:,i+1,j+1] = self.filters[:,:,i+1,j+1] * g

    #initialize offsets 論文には二個やり方書いてある
    self.offsets = torch.zeros(batch_size, size * size * 2, img.size()[2], img.size()[3], device=self.device)
    self.offsets[:,:] = torch.from_numpy(gr.gaussian_random_field(alpha=10,size=img.size()[2])).to(self.device) * sigma_delta.to(self.device)
    #self.offsets = torch.normal(mean=torch.zeros(batch_size,size*size*2, img.size()[2], img.size()[3]),std=sigma_delta).to(self.device)

  def affine_transform(self,image):
    mean = image.mean(dim=[2,3],keepdim=True)
    std = image.std(dim=[2,3],keepdim=True) + self.epsilon
    output = self.gamma * (image - mean) / std + self.beta
    return output

  def tanh(self,image):
    m = nn.Tanh()
    output = m(image)
    return output
  
  def forward(self):
    output = torchvision.ops.deform_conv2d(self.image, self.offsets, self.filters, padding=1)
    output = self.affine_transform(output)
    output = self.tanh(output)
    self.image = output
    return self

# 画像にどの様な変形を加えるか
transform = transforms.Compose([
  #transforms.Resize(size=(224,224)),# Digit 32, other 224 
  #transforms.ToTensor() ## このぎょうあとでコメントアウト
])

def pro_rand(image, n_loop=1, r=0.2):
  with torch.no_grad():
    original_image = image
    img = ProRandomConv(img=image)
    for _ in range(n_loop):
      img = img.forward()
    img.image = img.image*r + original_image*(1-r)
    return img.image


'''
transform = transforms.Compose([
  transforms.Resize(size=(224,224)),# Digit 32, other 224 
  transforms.ToTensor() ## このぎょうあとでコメントアウト
])
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
im = Image.open("input.jpg")
img = transform(im)
img = torch.unsqueeze(img,0)
img = pro_rand(img)
save_image(img[0], "test.png")
'''