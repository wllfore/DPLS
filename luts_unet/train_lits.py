import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.autograd import Variable

import albumentations as albu

from segmentation_models_pytorch import utils as smp_utils
import time
from lits import LiverDataSet, get_preprocessing, get_training_augmentation

import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from luts_func import do_tumorseg_one_volume, generate_msl_atsw
import SimpleITK as sitk

import shutil

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def model_valid_by_volume(model, data_dir):
    print ('welcome to model_valid_by_volume ...')
    model.eval()
    
    pos = data_dir.rfind('/')
    fold_dir = data_dir[:pos]    
    valid_vol_txt = os.path.join(fold_dir, 'valid_vol.txt')
    volume_dir = os.path.join(fold_dir, 'volumes/')
    segmentation_dir = os.path.join(fold_dir, 'labels/')
    
    # print (train_vol_txt)
    fp = open(valid_vol_txt, 'r')
    volume_ids = fp.readlines()
    fp.close()
    volume_ids = [x.replace('\n', '') for x in volume_ids]
    
    # print (len(volume_ids), volume_ids[0])
    
    pos_cnt = 0
    sum_w = 0
    mean_valid_dice = 0
    mean_valid_iou = 0
    
    file_num = len(volume_ids)
    count = 0
    for volume_id in volume_ids:
        count += 1
        print ('\r', 'process: {}/{}'.format(count, file_num), end='', flush=True)
        
        volume_path = os.path.join(volume_dir, volume_id + '.nii.gz')
        segmentation_path = os.path.join(segmentation_dir, volume_id.replace('volume', 'segmentation') + '.nii.gz')
            
        dice_score, iou_score, hd95_score, pred_prob = do_tumorseg_one_volume(volume_path, segmentation_path, model, eval_metrics=['dice'])
        
        mean_valid_dice += dice_score
            
    mean_valid_dice /= count
    
    print ('\nmodel_valid_by_volume end ...') 
    
    return mean_valid_dice              

def parse_args():
    parser = argparse.ArgumentParser(description='Train a liver segmentation model')
    parser.add_argument('--tumor_dir', default=None, help='the path to train images and masks')
    parser.add_argument('--frame', default='unet', help='unet or unet++')
    parser.add_argument('--encoder', default='mit_b0', help='encoder net')
    parser.add_argument("--max_epochs", default=20, type=int, help="max number of training epochs")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dsl_mode', default=0, type=int)      ### 0 close / 1 open
    parser.add_argument('--msl_mode', default=0, type=int)      ### 0 close / 1 open
    parser.add_argument('--atsw_mode', default=0, type=int)     ### 0 close / 1 open
    parser.add_argument('--loss_mode', default=0, type=int)     ### 0->origin label / 1->DSL / 2->origin+DSL / 3->origin+MSL / 4->origin+DSL+MSL 
    parser.add_argument('--job_name', default='dataset_network_description')
    parser.add_argument('--dataset', default='LiTS')            ### LiTS
    
    args = parser.parse_args()
    
    return args

def train():
    
    args = parse_args()  
    print(args)
    
    ### make dirs according to job name
    job_dir = './work_dir/{}'.format(args.job_name)
    make_dir(job_dir)
    
    model_dir = os.path.join(job_dir, 'model/')
    msl_mask_dir = os.path.join(job_dir, 'MSL/')
    
    make_dir(model_dir)
    make_dir(msl_mask_dir)
    
    train_image_dir = os.path.join(args.tumor_dir, 'image/train')
    train_origin_mask_dir = os.path.join(args.tumor_dir, 'mask/train')
    train_msl_mask_dir = msl_mask_dir
           
    train_ids_txt = os.path.join(args.tumor_dir, 'train.txt')

    pos = args.tumor_dir.rfind('/')
    upper_data_dir = args.tumor_dir[:pos] 
    
    ENCODER = args.encoder
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['lesion']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    # ACTIVATION = 'softmax2d'
    INITIAL_LEARNING_RATE = 0.0001
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print ('DEVICE = ', DEVICE)    

    # create segmentation model with pretrained encoder
    model = None
    if args.frame == 'unet':
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif args.frame == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)    

    train_dataset = LiverDataSet(
        train_image_dir, 
        train_origin_mask_dir,
        None,
        train_msl_mask_dir,
        train_ids_txt,
        3, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES
    )
  
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    #### step 3. set train configs
    dice_loss = smp_utils.losses.DiceLoss()
    metrics = [
        smp_utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=INITIAL_LEARNING_RATE),
    ])

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp_utils.train.TrainEpoch(
        model, 
        loss=dice_loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    loss_mode = args.loss_mode 
    if args.loss_mode == 3:
        loss_mode = 0
    if args.loss_mode == 4:
        loss_mode = 2
             
    #### adaptive sample weight
    train_sample_weight = None
    do_sample_reweight  = False
    if args.atsw_mode == 1:
        do_sample_reweight = True

    #### step 4. train model for 40 epochs
    for i in range(args.max_epochs):
        print ('\n')

        if args.dsl_mode > 0:                       
            input_dir = os.path.join(upper_data_dir, 'DSL/epoch_{}'.format(i))       ### 1. data-driven soft label       
            train_dataset.set_dsl_mask_path(input_dir)
            
        if i >= args.max_epochs/4 and args.msl_mode == 1:
            loss_mode = args.loss_mode
            train_sample_weight = generate_msl_atsw(args.tumor_dir, msl_mask_dir, model, do_sample_reweight=do_sample_reweight)      ### 2. model-driven soft label + adaptive sample weight
        
        ### change learning rate
        if i == args.max_epochs/2:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        
        print('Epoch: {}, loss mode: {}'.format(i, loss_mode))
        
        ### training
        train_logs = train_epoch.run(train_loader, loss_mode, train_sample_weight)

        ### validation
        valid_dice_score = model_valid_by_volume(model, args.tumor_dir)
        print ('valid dice = ', valid_dice_score)
        
        ### save models
        if i>= args.max_epochs/2:
            save_model_path = os.path.join(model_dir, 'epoch_{}_dice_{:.3f}.pth'.format(i, valid_dice_score))
            torch.save(model, save_model_path)
            print ('model saved in epoch ', i)

if __name__ == '__main__':
    print ('let us start segmentation model training on LiTS now ...')
    
    train()
    
    print ('finish model training ...')    
