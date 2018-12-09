import os
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasetsfrom torchvision.datasets import ImageFolder

class MyCub2011(Dataset):

	def __init__(self, root, formattedData, transform=None):
		self.root = root
		self.transform = transform
		#self.data = formattedData # Training set or test set

		# Split up data
		self.filenames = [d['filename'] for d in formattedData]
		self.labels = [d['class']['label'] for d in formattedData]
		self.bboxes = [d['object']['bbox'] for d in formattedData]

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		img_name = self.filenames[idx]

		image = Image.open(img_name)
		image = image.convert('RGB')
		left, top, right, bottom = self.bboxes[idx]['xmin'], self.bboxes[idx]['ymin'], self.bboxes[idx]['xmax'], self.bboxes[idx]['ymax']
		image = image.crop((left, top, right, bottom))

		if self.transform is not None:
			image = self.transform(image)

		label = self.labels[idx]

		return image, label
		

