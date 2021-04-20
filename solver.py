import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from torch.utils.tensorboard import SummaryWriter
import csv
import glob
import tensorboardEye
import logging
import tkinter as tk
from tkinter import filedialog





timestr = time.strftime("%Y%m%d-%H%M%S")
filenameLog = os.path.join('./Logs','TrainingLog-'+timestr+'.txt')
filenameWriter = os.path.join('./runs','Writer-'+timestr) # This is directory name

logging.basicConfig(filename=filenameLog, filemode='a',level=logging.DEBUG, format='%(asctime)s %(msecs)d- %(process)d-%(levelname)s - %(message)s')


class Solver(object):
	def __init__(self, config, train_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		#self.criterion = torch.nn.BCELoss()
		self.criterion = torch.nn.MSELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step
		self.validation_period = config.validation_period

		self.mode = config.mode
		self.cuda_id = config.cuda_idx
		#self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.device = torch.device(f'cuda:{self.cuda_id}' if torch.cuda.is_available() else 'cpu')
		if self.device.type == 'cuda':
			print("GPU: {}".format(torch.cuda.get_device_name(self.cuda_id)), "/  Index: {}".format(self.cuda_id))
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()


		# Path
		self.model_path = config.model_path
		self.writer_path = filenameWriter
		self.val_imgpath = os.path.join(config.val_imgpath,
										'model_{0}_CV{1}_AugProb_{2:.3f}-'.format(self.model_type, self.num_epochs,
																				 self.augmentation_prob) \
										+ timestr)
		if not os.path.exists(self.val_imgpath):
			os.makedirs(self.val_imgpath)
			print(self.val_imgpath)

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=1,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=1,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=1,output_ch=1,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		print('Starting the training process !!!')
		writer = SummaryWriter(self.writer_path)
		# Check and Load U-Net for Train
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		if os.path.isfile(unet_path):
		# 	# Load the pretrained Encoder
		# 	#self.unet.load_state_dict(torch.load(unet_path))
		# 	#print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		 	print('hi')
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			totalEpoch =0
			
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				totalEpoch = totalEpoch+1
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				MSE = 0.
				length = 0
				#i, (image, GT) = self.train_loader.__iter__()
				for i, (images, GT) in enumerate(self.train_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = self.unet(images)
					# GT : Ground Truth
					# SR : Segmentation Result

					#SR_probs = F.sigmoid(SR)
					SR_probs = torch.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)
					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()

					# Visualization
					writer.add_figure('predictions vs. actuals (train)',
									  tensorboardEye.imshow_on_tensorboard(images,GT,SR),
									  global_step=totalEpoch)

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR_flat,GT_flat)
					SP += get_specificity(SR_flat,GT_flat)
					PC += get_precision(SR_flat,GT_flat)
					F1 += get_F1(SR_flat,GT_flat)
					JS += get_JS(SR_flat,GT_flat)
					DC += get_DC(SR_flat,GT_flat)
					MSE += get_MSE(SR_flat,GT_flat)
					length += images.size(0)


				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				MSE = MSE/length

				trainingACC = acc
				trainingMSE = MSE

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, MSE: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,SE,SP,PC,F1,JS,DC, MSE))


				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
				
			#===================================== Validation ====================================#
				logPrintPeriod = self.validation_period
				#print("totalEpoch: {}".format(totalEpoch))
				#print("logPrintPeriod: {}".format(logPrintPeriod))
				#print(np.mod(totalEpoch, logPrintPeriod))
				if np.mod(totalEpoch, logPrintPeriod) == 0:
					with torch.no_grad():
						#self.unet.train(False)
						self.unet.eval()

						acc = 0.	# Accuracy
						SE = 0.		# Sensitivity (Recall)
						SP = 0.		# Specificity
						PC = 0. 	# Precision
						F1 = 0.		# F1 Score
						JS = 0.		# Jaccard Similarity
						DC = 0.		# Dice Coefficient
						MSE = 0.    # MSE

						length=0
						for i, (images, GT) in enumerate(self.test_loader):

							images = images.to(self.device)
							GT = GT.to(self.device)
							#SR = F.sigmoid(self.unet(images))
							SR = torch.sigmoid(self.unet(images))
							SR_probs = torch.sigmoid(SR)
							SR_flat = SR_probs.view(SR_probs.size(0), -1)
							GT_flat = GT.view(GT.size(0), -1)

							acc += get_accuracy(SR, GT)
							SE += get_sensitivity(SR_flat, GT_flat)
							SP += get_specificity(SR_flat, GT_flat)
							PC += get_precision(SR_flat, GT_flat)
							F1 += get_F1(SR_flat, GT_flat)
							JS += get_JS(SR_flat, GT_flat)
							DC += get_DC(SR_flat, GT_flat)
							MSE += get_MSE(SR_flat,GT_flat)

							length += images.size(0)

							writer.add_figure('predictions vs. actuals (test)',
											  tensorboardEye.imshow_on_tensorboard(images, GT, SR),
											  global_step=totalEpoch)
							tensorboardEye.save_on_local(images, GT, SR, self.val_imgpath, i)

						acc = acc/length
						SE = SE/length
						SP = SP/length
						PC = PC/length
						F1 = F1/length
						JS = JS/length
						DC = DC/length
						MSE = MSE/length

						# Score for records and save model parameters!
						#unet_score = JS + DC
						unet_score = 1.-MSE


						print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, MSE: %.4f'%(acc,SE,SP,PC,F1,JS,DC,MSE))


						# torchvision.utils.save_image(images.data.cpu(),
						# 							os.path.join(self.result_path,
						# 										'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
						# torchvision.utils.save_image(SR.data.cpu(),
						# 							os.path.join(self.result_path,
						# 										'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
						# torchvision.utils.save_image(GT.data.cpu(),
						# 							os.path.join(self.result_path,
						# 										'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

						# Save Best U-Net model
						if unet_score > best_unet_score:
							best_unet_score = unet_score
							best_epoch = epoch
							#modelSave_path = os.path.join('./models','model_MSE:{0:.3f}_DC:{1:.3f}_epoch:{2}.pth'.format(MSE, DC, totalEpoch))


							# Save model
							filename_save_model = 'model_{0}_CV{1}_AugProb_{2:.3f}-'.format(self.model_type,self.num_epochs,self.augmentation_prob)\
												  + timestr +'.pth'
							modelSave_path = os.path.join('./models',filename_save_model)
							torch.save({
								'epoch': best_epoch,
								'model_state_dict': self.unet.state_dict(),
								'optimizer_state_dict': self.optimizer.state_dict(),
								'loss': epoch_loss,},
							modelSave_path)
							print('Best %s model score : %.4f' % (self.model_type, best_unet_score))

							# Write log
							logging.info("Epoch: {0} --- Training ACC:{1:.3f}, Test ACC:{2:.3f}, Training MSE:{3:.3f}, Test MSE:{4:.3f}".format(
								best_epoch, trainingACC, acc, trainingMSE, MSE))

							with torch.no_grad():
								# self.unet.train(False)
								self.unet.eval()

								acc = 0.  # Accuracy
								SE = 0.  # Sensitivity (Recall)
								SP = 0.  # Specificity
								PC = 0.  # Precision
								F1 = 0.  # F1 Score
								JS = 0.  # Jaccard Similarity
								DC = 0.  # Dice Coefficient
								MSE = 0.  # MSE

								length = 0
								for i, (images, GT) in enumerate(self.test_loader):
									images = images.to(self.device)
									GT = GT.to(self.device)
									# SR = F.sigmoid(self.unet(images))
									SR = torch.sigmoid(self.unet(images))
									SR_probs = torch.sigmoid(SR)
									SR_flat = SR_probs.view(SR_probs.size(0), -1)
									GT_flat = GT.view(GT.size(0), -1)

									acc += get_accuracy(SR, GT)
									SE += get_sensitivity(SR_flat, GT_flat)
									SP += get_specificity(SR_flat, GT_flat)
									PC += get_precision(SR_flat, GT_flat)
									F1 += get_F1(SR_flat, GT_flat)
									JS += get_JS(SR_flat, GT_flat)
									DC += get_DC(SR_flat, GT_flat)
									MSE += get_MSE(SR_flat, GT_flat)

									length += images.size(0)
									tensorboardEye.save_on_local(images, GT, SR, self.val_imgpath, i)
						#===================================== Validation ====================================#

						# del self.unet
						# del best_unet
						# self.build_model()
						# self.unet.load_state_dict(torch.load(unet_path))
						#
						# self.unet.train(False)
						# self.unet.eval()
						#
						# acc = 0.	# Accuracy
						# SE = 0.		# Sensitivity (Recall)
						# SP = 0.		# Specificity
						# PC = 0. 	# Precision
						# F1 = 0.		# F1 Score
						# JS = 0.		# Jaccard Similarity
						# DC = 0.		# Dice Coefficient
						# length=0
						# for i, (images, GT) in enumerate(self.valid_loader):
						#
						# 	images = images.to(self.device)
						# 	GT = GT.to(self.device)
						# 	#SR = F.sigmoid(self.unet(images))
						# 	SR = torch.sigmoid(self.unet(images))
						# 	acc += get_accuracy(SR,GT)
						# 	SE += get_sensitivity(SR,GT)
						# 	SP += get_specificity(SR,GT)
						# 	PC += get_precision(SR,GT)
						# 	F1 += get_F1(SR,GT)
						# 	JS += get_JS(SR,GT)
						# 	DC += get_DC(SR,GT)
						#
						# 	length += images.size(0)
						#
						# acc = acc/length
						# SE = SE/length
						# SP = SP/length
						# PC = PC/length
						# F1 = F1/length
						# JS = JS/length
						# DC = DC/length
						# unet_score = JS + DC
						#
						#
						# f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
						# wr = csv.writer(f)
						# wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
						# f.close()

	def test(self):
		# wjcheon (212018)
		#unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		root = tk.Tk()
		root.withdraw()

		model_path = filedialog.askopenfilename()
		self.model_path = model_path
		unet_path = self.model_path

		pathname_selected = os.path.split(self.val_imgpath)[0]
		filename_selected = 'test-'+os.path.split(self.val_imgpath)[-1]
		independent_inference_locs = os.path.join(pathname_selected, filename_selected)


		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.build_model()

			checkpoint = torch.load(unet_path)
			self.unet.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			trained_loss =  checkpoint['loss']
			print('%s is Successfully Loaded from %s !! loss: %1.4f'%(self.model_type,unet_path, trained_loss))

		with torch.no_grad():
			self.unet.eval()

			acc = 0.  # Accuracy
			SE = 0.  # Sensitivity (Recall)
			SP = 0.  # Specificity
			PC = 0.  # Precision
			F1 = 0.  # F1 Score
			JS = 0.  # Jaccard Similarity
			DC = 0.  # Dice Coefficient
			MSE = 0.
			length = 0

			for i, (images, GT) in enumerate(self.test_loader):
				images = images.to(self.device)
				GT = GT.to(self.device)
				# SR = F.sigmoid(self.unet(images))
				SR = torch.sigmoid(self.unet(images))
				SR_probs = torch.sigmoid(SR)
				SR_flat = SR_probs.view(SR_probs.size(0), -1)
				GT_flat = GT.view(GT.size(0), -1)

				acc += get_accuracy(SR, GT)
				SE += get_sensitivity(SR_flat, GT_flat)
				SP += get_specificity(SR_flat, GT_flat)
				PC += get_precision(SR_flat, GT_flat)
				F1 += get_F1(SR_flat, GT_flat)
				JS += get_JS(SR_flat, GT_flat)
				DC += get_DC(SR_flat, GT_flat)
				MSE += get_MSE(SR_flat, GT_flat)

				length += images.size(0)




				tensorboardEye.save_on_local(images, GT, SR, independent_inference_locs, i)

		# f = open(os.path.join(self.result_path, 'result_testdata.csv'), 'a', encoding='utf-8', newline='')
		# wr = csv.writer(f)
		# # wr.writerow(
		# # 	[self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, best_epoch, self.num_epochs, self.num_epochs_decay,
		# # 	 self.augmentation_prob])
		# wr.writerow(
		# 	[self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, self.num_epochs, self.num_epochs_decay,
		# 	 self.augmentation_prob])
		# f.close()


