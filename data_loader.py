# augmentation: https://hoya012.github.io/blog/albumentation_tutorial/

import os
import random
from random import shuffle
import numpy as np
import torch
import torchvision
import PIL
import cv2
import time
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
import albumentations.pytorch
from matplotlib import pyplot as plt


from torch.utils import data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import KFold
from tqdm import tqdm



class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=512,mode='train',augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
		self.GT_paths = root[:-1]+'_GT/'
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		filename = image_path.split('_')[-1][:-len(".jpg")]
		GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'

		image = Image.open(image_path)
		GT = Image.open(GT_path)

		aspect_ratio = image.size[1]/image.size[0]

		Transform = []

		ResizeRange = random.randint(300,320)
		Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
		p_transform = random.random()

		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			RotationDegree = random.randint(0,3)
			RotationDegree = self.RotationDegree[RotationDegree]
			if (RotationDegree == 90) or (RotationDegree == 270):
				aspect_ratio = 1/aspect_ratio

			Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
						
			RotationRange = random.randint(-10,10)
			Transform.append(T.RandomRotation((RotationRange,RotationRange)))
			CropRange = random.randint(250,270)
			Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
			Transform = T.Compose(Transform)
			
			image = Transform(image)
			GT = Transform(GT)

			ShiftRange_left = random.randint(0,20)
			ShiftRange_upper = random.randint(0,20)
			ShiftRange_right = image.size[0] - random.randint(0,20)
			ShiftRange_lower = image.size[1] - random.randint(0,20)
			image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
			GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

			if random.random() < 0.5:
				image = F.hflip(image)
				GT = F.hflip(GT)

			if random.random() < 0.5:
				image = F.vflip(image)
				GT = F.vflip(GT)

			Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

			image = Transform(image)

			Transform =[]


		Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)
		
		image = Transform(image)
		GT = Transform(GT)

		Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		image = Norm_(image)

		return image, GT


class CustomTensorDataset(Dataset):
	"""TensorDataset with support of transforms."""
	def __init__(self, tensors, transform=None):

		self.tensors = tensors
		if not all(len(tensors[0]) == len(tensors[1])):
			raise ValueError('not all lists have same length!')
		self.transform = transform

	def __len__(self):
		print('__len__: '.format(len(self.tensors)))
		return len(self.tensors)


	def __getitem__(self, index):
		x = self.tensors[0][index]
		y = self.tensors[1][index]
		print('size X: {}'.format(np.shape(x)))
		print('size Y: {}'.format(np.shape(y)))

		start_t = time.time()
		if self.transform:
			print('transform start !')
			x = self.transform(x)
			y = self.transform(y)
		total_time = (time.time()-start_t)
		print('Calculation time: {}'.format(total_time))
		return x, y


class AlbumentationsDataset(Dataset):
    """TensorDataset with support of transforms."""
    #def __init__(self, tensors, batchSize=4, shuffle=False, transform=None):
    def __init__(self, tensors, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform


    def __len__(self):
        return len(self.tensors[0])


    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        #start_t = time.time()
        #print('transform start !')
        sample = {"image": x, "mask": y}

        if self.transform:
            augmented = self.transform(**sample)
            x = augmented['image']
            y = augmented['mask']

        #total_time = (time.time()-start_t)
        #print('Calculation time: {}'.format(total_time))
        return x, y

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader

def get_loader_mat(image_path, batch_size, num_workers=2, kfold=1, currentCVNum=1):
	# wjcheon

	dbLocs = image_path
	dbInput = os.path.join(dbLocs, 'Input')
	dbGT = os.path.join(dbLocs, 'GT')
	filesLists_dbInput = [os.path.join(dbInput, f) for f in os.listdir(dbInput) if
						  os.path.isfile(os.path.join(dbInput, f))]
	filesLists_dbGT = [os.path.join(dbGT, f) for f in os.listdir(dbGT) if os.path.isfile(os.path.join(dbGT, f))]
	filesLists_dbInput.sort()
	filesLists_dbGT.sort()

	filesLists_dbGT_length = filesLists_dbInput.__len__()
	rn = range(0, filesLists_dbGT_length)
	kf5 = KFold(n_splits=kfold, shuffle=False)
	training_indexes = {}
	test_indexes = {}
	counter = 1
	for train_index, test_index in kf5.split(rn):
		training_indexes["trainIndex_CV{0}".format(counter)] = train_index
		test_indexes["testIndex_CV{0}".format(counter)] = test_index
		counter = counter + 1
		#print("trainIndex:{}".format(train_index))
		#print("testIndex:{}".format(test_index))

	trainingDicKey_list = list(training_indexes.keys())
	trainingIndex_CV_F = training_indexes.get(trainingDicKey_list[currentCVNum-1])

	testDicKey_list = list(test_indexes.keys())
	testIndex_CV_F = test_indexes.get(testDicKey_list[currentCVNum-1])

	input_x = []
	output_gt = []

	for iter1 in tqdm(trainingIndex_CV_F):
		# print(iter1)
		tempFilenameInput = filesLists_dbInput[iter1]
		tempFilenameGT = filesLists_dbGT[iter1]

		trainXtemp = loadmat(tempFilenameInput)
		try:
			trainXtemp = trainXtemp["ctImagesData_rot_zeroNorm_HalfNHalf"]
			trainXtemp = np.swapaxes(trainXtemp, axis1=0, axis2=2)
			trainXtemp = np.expand_dims(trainXtemp, axis=3)
		except:
			trainXtemp = trainXtemp["ctImagesData_rot_zeroNorm_3channel"]
		#trainXtemp = trainXtemp["ctImagesData_rot_zeroNorm"]


		trainYtemp = loadmat(tempFilenameGT)
		trainYtemp = trainYtemp["ntotalGrayMap_Oncologist_rot_HalfNHalf"]
		#trainYtemp = trainYtemp["ntotalGrayMap_Oncologist_rot"]
		trainYtemp = np.swapaxes(trainYtemp, axis1=0, axis2=2)
		trainYtemp = np.expand_dims(trainYtemp, axis=3)


		input_x.extend(trainXtemp)
		output_gt.extend(trainYtemp)

	del trainXtemp
	del trainYtemp

	# Data (input, output) type was maintain as numpy.
	#input_x = torch.tensor(input_x)
	#output_gt = torch.tensor(output_gt)
	# use FloatTensor
	#input_x = torch.FloatTensor(input_x)
	#output_gt = torch.FloatTensor(output_gt)

	input_x_k = []
	output_gt_k = []
	for iter1 in tqdm(testIndex_CV_F):
		# print(iter1)
		tempFilenameInput = filesLists_dbInput[iter1]
		tempFilenameGT = filesLists_dbGT[iter1]

		testXtemp = loadmat(tempFilenameInput)

		try:
			testXtemp = testXtemp["ctImagesData_rot_zeroNorm_HalfNHalf"]
			#testXtemp = testXtemp["ctImagesData_rot_zeroNorm"]
			testXtemp = np.swapaxes(testXtemp, axis1=0, axis2=2)
			testXtemp = np.expand_dims(testXtemp, axis=3)
		except:
			testXtemp = testXtemp["ctImagesData_rot_zeroNorm_3channel"]

		testYtemp = loadmat(tempFilenameGT)
		testYtemp = testYtemp["ntotalGrayMap_Oncologist_rot_HalfNHalf"]
		#testYtemp = testYtemp["ntotalGrayMap_Oncologist_rot"]
		testYtemp = np.swapaxes(testYtemp, axis1=0, axis2=2)
		testYtemp = np.expand_dims(testYtemp, axis=3)

		input_x_k.extend(testXtemp)
		output_gt_k.extend(testYtemp)

	del testXtemp
	del testYtemp

	print("Data is successfully loaded !!")
	print("Train input:{}, Train gt:{}".format(np.shape(input_x), np.shape(output_gt)))
	print("Test input:{}, Test gt:{}".format(np.shape(input_x_k), np.shape(output_gt_k)))


	# Data (input, output) type was maintain as numpy.
	# input_x_k = torch.FloatTensor(input_x_k)
	# output_gt_k = torch.FloatTensor(output_gt_k)

	# torchvision_transform = torchvision.transforms.Compose([
	# 	transforms.RandomHorizontalFlip(),
	# 	transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
	# 	transforms.ToTensor()
	#
	# 	# transforms.Resize((256, 256)),
	# 	# transforms.RandomCrop(224),
	# 	# transforms.RandomHorizontalFlip(),
	# 	# transforms.ToTensor(),
	# ])

	# # visual debug for input and outpu
	# sampleimg = np.squeeze(input_x[50])
	# sampleimg = sampleimg[:,:,0]
	# plt.figure()
	# plt.imshow(sampleimg)
	#
	# sampleimg_gt = np.squeeze(output_gt[50])
	# plt.figure()
	# plt.imshow(sampleimg_gt)


	albumentations_transform = albumentations.Compose([
		# albumentations.Resize(256, 256),
		# albumentations.RandomCrop(224, 224),
		albumentations.HorizontalFlip(),  # Same with transforms.RandomHorizontalFlip()
		albumentations.Rotate(),
		albumentations.pytorch.transforms.ToTensor()
	])

	albumentations_transform_testSet = albumentations.Compose([
		# albumentations.Resize(256, 256),
		# albumentations.RandomCrop(224, 224),
		albumentations.pytorch.transforms.ToTensor()
	])

	train_dataset_transform = AlbumentationsDataset(tensors=(input_x, output_gt), transform=albumentations_transform)
	train_loader = torch.utils.data.DataLoader(train_dataset_transform, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_dataset_transform = AlbumentationsDataset(tensors=(input_x_k, output_gt_k), transform=albumentations_transform_testSet)
	test_loader = torch.utils.data.DataLoader(test_dataset_transform, batch_size=1, shuffle=False, num_workers=num_workers)

	return train_loader, test_loader




