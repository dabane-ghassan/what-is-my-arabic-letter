import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import ImageGrab, Image
import PIL.ImageOps 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage import rotate

class ImageDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, self.data.columns != 'label'].values.astype(np.uint8).reshape(32, 32)
        label = self.data.iloc[index, -1] - 1
        
        if self.transform is not None:
            image = self.transform(image)
            
        img = torch.Tensor(rotate(torch.flip(image, (0, 1)), -90, axes=(1,2)))
        
        img = transforms.functional.resize(img, (128,128))
        
        return img, label

def view_data(data: torch.Tensor, label: torch.Tensor, n: int) -> plt.Figure:

    arab_labs = ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط' , 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى']

    fig, axs = plt.subplots(1, n, figsize=(21, 5))
    for i_ax, ax in enumerate(axs):
        ax.imshow(data[i_ax, 0, :, :], cmap=plt.gray())
        ax.set_title("Label = %s" % (arab_labs[int(label[i_ax].item())]))
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def view_data_rand(loader: torch.utils.data.DataLoader, n: int = 10) -> plt.Figure:

    rand_data, rand_label = next(iter(loader))

    return view_data(rand_data, rand_label, n)

def predict(model, device, tensor):

    arab_labs = ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط' , 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى']
    model.to(device)
    with torch.no_grad():

        output = model(torch.unsqueeze(tensor, dim=0).to(device))
        output = nn.functional.softmax(output, dim=1) * 100
        confidence, predicted = torch.max(output.data, 1)
        return confidence.item(), arab_labs[int(predicted.detach())]

def img_to_tensor(image):

    TO_TENSOR = transforms.ToTensor()
    TO_NORM= transforms.Normalize(mean=(0.5,), std=(0.5,))
    TO_GRAY = transforms.Grayscale(num_output_channels=1)

    image = np.array(image)
    image = cv2.resize(image,(128,128),interpolation = cv2.INTER_AREA)
    image = Image.fromarray(image)
    image = PIL.ImageOps.invert(image)

    return TO_NORM(TO_GRAY(TO_TENSOR(image)))