import numpy as np
import cv2
import torch
from PIL import ImageGrab, Image
import PIL.ImageOps 
import torch.nn as nn
from torchvision import transforms

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