from sklearn.model_selection import KFold
import argparse
import os
from solver import Solver
from data_loader import get_loader, get_loader_mat, get_loader_mat_dropout
from torch.backends import cudnn
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy  as np


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    # path for saving model
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    # path for saving image of Validation set
    if not os.path.exists(config.val_imgpath):
        os.makedirs(config.val_imgpath)

    # learning rate
    lr = random.random()*0.0005 + 0.0000005
    config.lr = lr
    # Decaying ratio for learning rate
    decay_ratio = 0.3
    decay_epoch = int(config.num_epochs*decay_ratio)
    config.num_epochs_decay = decay_epoch

    datapath_splited = str.split(config.data_path,'_')
    if (datapath_splited[-1] == '2.5D'):
        #config.img_ch = 3
        config.img_ch = 5
        print("The number channel of input: {}".format(config.img_ch))


    print(config)
    # k fold validation
    # train_loader, test_loader = get_loader_mat(image_path=config.data_path,
    #                                            batch_size=config.batch_size,
    #                                            num_workers=config.num_workers,
    #                                            kfold=config.kfold,
    #                                            currentCVNum=config.currentCVNum)

    # # Dropout instead of k fold validation
    train_loader, test_loader = get_loader_mat_dropout(image_path_train=config.data_path,
                                                image_path_test= config.data_path_test,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers)

    solver = Solver(config, train_loader, test_loader)




    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=5)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--validation_period', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--val_imgpath', type=str, default='ResultValidationAsImage')

    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--currentCVNum', type=int, default=4) # start from 1

    # misc
    #parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')   # path for saving model
    #parser.add_argument('--writer_path',type=str,default='runs/grapMap-1011-AdamMaXAug-SmothL1-Fullbatch-BS10-Trail3-beta0_5_CV{}_Writer'.format(1))

    #DB
    #parser.add_argument('--data_path', type=str, default='/home/shared/DB/NSCLC/DB_NSCLC_HalfNHalf')
    #parser.add_argument('--data_path', type=str, default='/home/shared/DB/NSCLC/DB_NSCLC_HalFNHalF_5slices_GTV1_corrected_2.5D')
    parser.add_argument('--data_path', type=str, default='/home/shared/DB/NSCLC/DB_NSCLC_HalFNHalF_5slices_GTV1_corrected_2.5D-Train-CV2')
    parser.add_argument('--data_path_test', type=str, default='/home/shared/DB/NSCLC/DB_NSCLC_HalFNHalF_5slices_GTV1_corrected_2.5D-Test-CV2')

    #parser.add_argument('--data_path', type=str, default='/home/shared/DB/NSCLC/DB_NSCLC_HalfNHalf_SLIM')
    #


    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
