import os
from os.path import join, exists

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import numpy as np
import SimpleITK as sitk
import scipy

import time
from medpy import metric
import math

input_img_size = 480

def calculate_metric_percase(pred, gt, spacing, metric_list):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = 0
        if 'dice' in metric_list:
            dice = metric.binary.dc(pred, gt)
        
        iou = 0
        if 'iou' in metric_list:
            iou = metric.binary.jc(pred, gt)
        
        hd95 = 0
        if 'hd95' in metric_list:
            hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        
        return dice, iou, hd95
    elif gt.sum()==0:
        return 1, 1, 0
    else:
        return 0, 0, 0


def pred_one_image(ori_img, seg_model):
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rez_img = cv2.resize(ori_img, (input_img_size, input_img_size), interpolation = cv2.INTER_LINEAR)
    image = cv2.cvtColor(rez_img, cv2.COLOR_BGR2RGB)
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    max_pixel_value = 255.0
    r, g, b = cv2.split(image)
    r = (r - mean[0]*max_pixel_value) / (std[0]*max_pixel_value)
    g = (g - mean[1]*max_pixel_value) / (std[1]*max_pixel_value)
    b = (b - mean[2]*max_pixel_value) / (std[2]*max_pixel_value)
    
    input_image = cv2.merge((r, g, b))
    
    input_image = input_image.transpose(2, 0, 1).astype('float32')
    
    x_tensor = torch.from_numpy(input_image).to(DEVICE).unsqueeze(0)
    pr_map = seg_model.predict(x_tensor)
    pr_map = pr_map.squeeze().cpu().numpy()
    pr_mask = pr_map.round().astype('uint8')
    
    # print (ori_img.shape[0], ori_img.shape[1])
    pred_res_prob = cv2.resize(pr_map, (ori_img.shape[1], ori_img.shape[0]))
    pred_res_mask = cv2.resize(pr_mask, (ori_img.shape[1], ori_img.shape[0]), interpolation = cv2.INTER_NEAREST)  
    
    return pred_res_mask, pred_res_prob

def do_tumorseg_one_slice(slice_image, liver_mask, tumor_seg_model):
    
    pred_tumor_slice_mask = np.zeros((slice_image.shape[0], slice_image.shape[1]), dtype='int')
    pred_tumor_slice_prob = np.zeros((slice_image.shape[0], slice_image.shape[1]), dtype='float')
    
    if np.sum(liver_mask) > 30:    
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        liver_mask = cv2.dilate(liver_mask, kernel)

        slice_image = cv2.add(slice_image.astype('uint8'), np.zeros(np.shape(slice_image), dtype=np.uint8), mask=liver_mask.astype('uint8'))

        pred_tumor_slice_mask, pred_tumor_slice_prob = pred_one_image(slice_image, tumor_seg_model)
        
    return pred_tumor_slice_mask, pred_tumor_slice_prob

def do_tumorseg_one_volume(volume_path, segmentation_path, tumor_seg_model, eval_metrics=[]):
    
    tumor_dice_score = 0
    tumor_iou_score = 0
    
    volume = sitk.ReadImage(volume_path)
    volume_data = sitk.GetArrayFromImage(volume)
    
    segmentation = sitk.ReadImage(segmentation_path)
    gt_seg_map = sitk.GetArrayFromImage(segmentation)

    spacing = volume.GetSpacing()
        
    # print('shape info: ', volume_data.shape, gt_seg_map.shape)
    # print ('spacing info: ', spacing)
    zoom_flag = False
    norm_spacing_z = 2.5
    zoom_scale = spacing[2] / norm_spacing_z
    zoom_volume_data = None
    if spacing[2] > norm_spacing_z:
        zoom_flag = True
        zoom_volume_data = scipy.ndimage.zoom(volume_data, [zoom_scale, 1, 1])
    

    min_HU = -100
    max_HU = 300
    volume_data = 255.0 * (volume_data - min_HU) / (max_HU - min_HU)
    volume_data[volume_data <= 0 ] = 0
    volume_data[volume_data >= 255 ] = 255
    
    if zoom_flag:
        zoom_volume_data = 255.0 * (zoom_volume_data - min_HU) / (max_HU - min_HU)
        zoom_volume_data[zoom_volume_data <= 0 ] = 0
        zoom_volume_data[zoom_volume_data >= 255 ] = 255
    
    ## organ+tumor -> organ
    gt_organ = gt_seg_map.copy()
    gt_organ[gt_organ >= 1] = 1
    
    gt_tumor = gt_seg_map.copy()
    gt_tumor[gt_tumor == 1] = 0
    gt_tumor[gt_tumor == 2] = 1
    
    pred_tumor_seg_map = np.zeros_like(gt_tumor)
    pred_tumor_seg_prob = np.zeros(gt_tumor.shape, dtype='float')
    
    for i in range(volume_data.shape[0]):
        # print ('process slice ', i)
        
        current_slice_data = volume_data[i,:,:]
        
        former_slice_data = None
        next_slice_data = None
        
        if not zoom_flag:
            former_slice_index = i-1
            next_slice_index = i+1
            # former_slice_index = round(i - 1.0 / zoom_scale)
            # next_slice_index = round(i - 1.0 / zoom_scale)

            if former_slice_index < 0:
                former_slice_index = 0
            if next_slice_index > volume_data.shape[0] - 1:
                next_slice_index = volume_data.shape[0] - 1
                
            former_slice_data = volume_data[former_slice_index,:,:]
            next_slice_data = volume_data[next_slice_index,:,:]
        else:
            former_slice_index = round(i*zoom_scale - 1)
            next_slice_index = round(i*zoom_scale + 1)
    
            if former_slice_index < 0:
                former_slice_index = 0
            if next_slice_index > zoom_volume_data.shape[0] - 1:
                next_slice_index = zoom_volume_data.shape[0] - 1
                
            former_slice_data = zoom_volume_data[former_slice_index,:,:]
            next_slice_data = zoom_volume_data[next_slice_index,:,:]
            
        img_3c = cv2.merge((former_slice_data, current_slice_data, next_slice_data))

        # print (img_3c.shape)
        
        #### predict tumor mask via tumor seg model on the basis of organ mask (gt or pred)
        organ_slice_mask = gt_organ[i,:,:]
        pred_tumor_slice_mask, pred_tumor_slice_prob = do_tumorseg_one_slice(img_3c, organ_slice_mask, tumor_seg_model)

        # print (pred_tumor_slice_mask.shape)

        pred_tumor_seg_map[i,:,:] = pred_tumor_slice_mask
        pred_tumor_seg_prob[i,:,:] = pred_tumor_slice_prob  

    tumor_dice_score, tumor_iou_score, tumor_hd95_score = calculate_metric_percase(pred_tumor_seg_map.astype('uint8'), gt_tumor.astype('uint8'), spacing, eval_metrics)
    
    return tumor_dice_score, tumor_iou_score, tumor_hd95_score, pred_tumor_seg_prob

def compute_sample_weight(pred, gt, slice_num):
    
    inter = np.sum(gt * pred)
    sets_sum = np.sum(gt.reshape(-1)) + np.sum(pred.reshape(-1))
    epsilon = 1e-6
    if np.sum(gt) == 0:
        epsilon = slice_num
    d = (2 * inter + epsilon) / (sets_sum + epsilon)    
    dice_loss = 1 - d    
    w = 1 / (1 + math.exp(-1 * dice_loss))

    return w

def generate_msl_atsw(data_dir, msl_mask_save_dir, latest_model, do_sample_reweight=False):
    
    print ('welcome to generate_msl_atsw ...')
    
    pos = data_dir.rfind('/')
    fold_dir = data_dir[:pos] 

    spatial_smooth_dir = os.path.join(fold_dir, 'SSL/')
    train_vol_txt = os.path.join(fold_dir, 'train_vol.txt')
    volume_dir = os.path.join(fold_dir, 'volumes/')
    segmentation_dir = os.path.join(fold_dir, 'labels/')
    
    fp = open(train_vol_txt, 'r')
    volume_ids = fp.readlines()
    fp.close()
    volume_ids = [x.replace('\n', '') for x in volume_ids]

    train_sample_weights = dict() 

    belta = 0.7
    
    file_num = len(volume_ids)
    count = 0
    for volume_id in volume_ids:
        count += 1 
        print ('\r', 'process: {}/{}'.format(count, file_num), end='', flush=True)

        segmentation_file = volume_id.replace('volume', 'segmentation') + '.nii.gz'
        
        spatial_smooth_mask = sitk.ReadImage(os.path.join(spatial_smooth_dir, segmentation_file))
        spatial_smooth_mask_data = sitk.GetArrayFromImage(spatial_smooth_mask)

        segmentation_path = os.path.join(segmentation_dir, segmentation_file)
        segmentation = sitk.ReadImage(segmentation_path)
        segmentation_data = sitk.GetArrayFromImage(segmentation)
        
        volume_path = os.path.join(volume_dir, volume_id + '.nii.gz')
        dice_score, iou_score, hd95_score, pred_prob = do_tumorseg_one_volume(volume_path, segmentation_path, latest_model, eval_metrics=['dice'])
        ### 1. model distillation based soft label
        MDSL = spatial_smooth_mask_data * (belta + (1.0 - belta) * pred_prob)
        
        tumor_gt = segmentation_data.copy()
        tumor_gt[tumor_gt == 1] = 0
        tumor_gt[tumor_gt == 2] = 1

        ### compute volume weight
        volume_w = 1
        if do_sample_reweight:
            slice_num = 0
            for i in range(segmentation_data.shape[0]):
                slice_mask = segmentation_data[i, :, :]
                if np.sum(slice_mask) >= 10:
                    slice_num += 1

            volume_w = compute_sample_weight(pred_prob, tumor_gt, slice_num)
            # print ('volume_w = ', volume_w, ', dice_score = ', dice_score, ', slice_num = ', slice_num)

        
        for i in range(segmentation_data.shape[0]):
            slice_mask = segmentation_data[i, :, :]

            ####没有liver的slice跳过 有liver无tumor的slice间隔采样
            if np.sum(slice_mask) == 0:
                continue
            elif np.sum(slice_mask == 2) == 0 and i % 2 == 0:
                continue
            
            slice_name = volume_id + '_slice_' + str(i)
            slice_modelmask = MDSL[i, :, :]
            save_modelmask_path = os.path.join(msl_mask_save_dir, slice_name + '.png')            
            cv2.imwrite(save_modelmask_path, slice_modelmask.astype(np.uint8)) 

            if do_sample_reweight:
                slice_pred = pred_prob[i, :, :]
                slice_tumor_gt = tumor_gt[i, :, :]
                slice_w = 1
                slice_w = compute_sample_weight(slice_pred, slice_tumor_gt, 1)
                # slice_w = math.sqrt(slice_w * volume_w)
                train_sample_weights[slice_name] = slice_w

    if do_sample_reweight:
        sum_w = 0
        for k, v in train_sample_weights.items():
            sum_w += v
        
        norm_scale = len(train_sample_weights) / sum_w
        for k, v in train_sample_weights.items():
            train_sample_weights[k] = v * norm_scale

        # print ('nCount = ', len(train_sample_weights), ', sum_w = ', sum_w, ', ratio = ', norm_scale)
            
    print ('\ngenerate_msl_atsw end ...')

    return train_sample_weights
