import math
import shutil

import numpy as np
import os
import SimpleITK as sitk
import cv2
import pandas as pd
import random
import scipy
from data_smooth import guass_smooth_3d

import argparse

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def overlap_liver_region(slice_image, slice_mask):
    liver_mask = slice_mask.copy()
    liver_mask[liver_mask >= 1] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    liver_mask = cv2.dilate(liver_mask, kernel)
    liver_mask = liver_mask.astype(np.uint8)

    liver_image = cv2.add(slice_image.astype(np.uint8), np.zeros(np.shape(slice_image), dtype=np.uint8), mask=liver_mask)

    tumor_mask = slice_mask.copy()
    tumor_image = liver_image.copy()

    tumor_mask[tumor_mask == 1] = 0
    tumor_mask[tumor_mask == 2] = 255

    return tumor_image, tumor_mask

#### convert the LiTS dataset to slice images and masks for tumor segmentation model training
def convert_tumor_data(volumes_dir, segmentations_dir, volume_list, save_dir, mode='train'):
    nCount = 0

    images_3c_dir = os.path.join(save_dir, 'tumor/image/' + mode)
    masks_dir = os.path.join(save_dir, 'tumor/mask/' + mode)

    print(images_3c_dir, masks_dir)

    make_dir(images_3c_dir)
    make_dir(masks_dir)

    image_ids_txt = os.path.join(save_dir, 'tumor/' + mode + '.txt')
    print(image_ids_txt)

    fp = open(image_ids_txt, 'w')
    for volume_id in volume_list:
        print('start process ', volume_id)

        volume_path = os.path.join(volumes_dir, volume_id + '.nii.gz')
        segmentation_path = os.path.join(segmentations_dir, volume_id.replace('volume', 'segmentation') + '.nii.gz')

        volume = sitk.ReadImage(volume_path)
        volume_data = sitk.GetArrayFromImage(volume)

        segmentation = sitk.ReadImage(segmentation_path)
        segmentation_data = sitk.GetArrayFromImage(segmentation)

        print('shape info: ', volume_data.shape, segmentation_data.shape)

        spacing = volume.GetSpacing()
        NORM_SPACING_Z = 2.5
        zoom_flag = False
        zoom_scale = spacing[2] / NORM_SPACING_Z

        zoom_volume_data = None
        if spacing[2] > NORM_SPACING_Z:
            zoom_flag = True
            zoom_volume_data = scipy.ndimage.zoom(volume_data, [zoom_scale, 1, 1])

        ## 腹部软组织窗HU值变换
        min_HU = -100.0
        max_HU = 300.0
        volume_data = 255.0 * (volume_data - min_HU) / (max_HU - min_HU)
        volume_data[volume_data <= 0] = 0
        volume_data[volume_data >= 255] = 255

        if zoom_flag:
            zoom_volume_data = 255.0 * (zoom_volume_data - min_HU) / (max_HU - min_HU)
            zoom_volume_data[zoom_volume_data <= 0] = 0
            zoom_volume_data[zoom_volume_data >= 255] = 255

        for i in range(volume_data.shape[0]):
            slice_mask = segmentation_data[i, :, :]

            ####没有liver的slice跳过; 有liver无tumor的slice间隔采样
            if np.sum(slice_mask) == 0:
                continue
            elif np.sum(slice_mask == 2) == 0 and i % 2 == 0:
                continue

            ### save image contain current + former + next slices ###
            current_slice_data = volume_data[i, :, :]
            former_slice_data = None
            next_slice_data = None

            if not zoom_flag:
                former_slice_index = i - 1
                next_slice_index = i + 1

                if former_slice_index < 0:
                    former_slice_index = 0
                if next_slice_index > volume_data.shape[0] - 1:
                    next_slice_index = volume_data.shape[0] - 1

                former_slice_data = volume_data[former_slice_index, :, :]
                next_slice_data = volume_data[next_slice_index, :, :]
            else:
                former_slice_index = round(i * zoom_scale - 1)
                next_slice_index = round(i * zoom_scale + 1)

                if former_slice_index < 0:
                    former_slice_index = 0
                if next_slice_index > zoom_volume_data.shape[0] - 1:
                    next_slice_index = zoom_volume_data.shape[0] - 1

                former_slice_data = zoom_volume_data[former_slice_index, :, :]
                next_slice_data = zoom_volume_data[next_slice_index, :, :]


            img_3c = cv2.merge((former_slice_data, current_slice_data, next_slice_data))

            tumor_image, tumor_mask = overlap_liver_region(img_3c, slice_mask)

            slice_name = volume_id + '_slice_' + str(i)
            appdix = '.png'
            print(slice_name)
            fp.write(slice_name + '\n')

            save_image_path = os.path.join(images_3c_dir, slice_name + appdix)
            save_mask_path = os.path.join(masks_dir, slice_name + appdix)

            cv2.imwrite(save_mask_path, tumor_mask.astype(np.uint8))
            cv2.imwrite(save_image_path, tumor_image.astype(np.uint8))

    fp.close()

### generate SSL, ISM and CDF for train volumes
def prepare_SSL_ISM_CDF(volumes_dir, segmentations_dir, save_dir, volume_list):
    
    SSL_dir = os.path.join(save_dir, 'SSL')
    ISM_dir = os.path.join(save_dir, 'ISM')
    CDF_dir = os.path.join(save_dir, 'CDF')
    make_dir(SSL_dir)
    make_dir(ISM_dir)
    make_dir(CDF_dir)

    for volume_id in volume_list:
        print ('start process ', volume_id)

        volume_path = os.path.join(volumes_dir, volume_id + '.nii.gz')
        segmentation_path = os.path.join(segmentations_dir, volume_id.replace('volume', 'segmentation') + '.nii.gz')

        volume = sitk.ReadImage(volume_path)
        volume_data = sitk.GetArrayFromImage(volume)

        mask = sitk.ReadImage(segmentation_path)
        mask_data = sitk.GetArrayFromImage(mask)
        spacing = mask.GetSpacing()

        mask_data[mask_data == 1] = 0
        mask_data[mask_data == 2] = 255

        ### 1. compute and save SSL
        smooth_mask_data = guass_smooth_3d(mask_data, kernel_sizeXY=5, kernel_sizeZ=3, mask=None, do_3d=True)       
        smooth_mask = sitk.GetImageFromArray(smooth_mask_data)
        smooth_mask.SetSpacing(mask.GetSpacing())
        smooth_mask.SetOrigin(mask.GetOrigin())
        smooth_mask.SetDirection(mask.GetDirection())
        spatial_smooth_path = os.path.join(SSL_dir, volume_id.replace('volume', 'segmentation')+'.nii.gz')
        sitk.WriteImage(smooth_mask, spatial_smooth_path)

        intensity_similarity_mask = np.zeros(mask_data.shape, dtype='float')
        distance_field_mask = np.zeros(mask_data.shape, dtype='float')

        input_mask = sitk.GetImageFromArray(mask_data.astype('uint8'))
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)
        output_mask = cc_filter.Execute(input_mask)

        num_connected_regions = cc_filter.GetObjectCount()
        if num_connected_regions >= 1:
            lss_filter = sitk.LabelShapeStatisticsImageFilter()
            lss_filter.Execute(output_mask)

            np_output_mask = sitk.GetArrayFromImage(output_mask)
            for label in range(1, num_connected_regions+1):
                area = lss_filter.GetNumberOfPixels(label)
                # print ('label = ', label, ', area = ', area)
                lesion_mask = np.zeros_like(np_output_mask)
                lesion_mask[np_output_mask == label] = 1
                lesion_size = np.sum(lesion_mask)

                lesion_mean_HU = np.sum(volume_data * lesion_mask) / lesion_size
                # print ('lesion_size = ', lesion_size, ', lesion_mean_HU = ', lesion_mean_HU)

                dilated_lesion_mask = np.zeros_like(lesion_mask)
                for i in range(lesion_mask.shape[0]):
                    lesion_slice_mask = lesion_mask[i, :, :]
                    if np.sum(lesion_slice_mask) > 0:
                        ### dilate slice mask
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
                        lesion_slice_mask = cv2.dilate(lesion_slice_mask.astype('uint8'), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                        dilated_lesion_mask[i,:,:] = lesion_slice_mask

                #### distance constraint
                #### find center point of a 3d lesion and max radius
                lesion_3d_pts = np.nonzero(dilated_lesion_mask)
                # print ('xxxx', lesion_3d_pts)
                lesion_center_z = np.mean(lesion_3d_pts[0])
                lesion_center_y = np.mean(lesion_3d_pts[1])
                lesion_center_x = np.mean(lesion_3d_pts[2])
                # print('lesion center: ', lesion_center_x, lesion_center_y, lesion_center_z)

                max_d = 0
                lesion_distance_field = np.zeros_like(lesion_mask)
                for k in range(lesion_3d_pts[2].shape[0]):
                    pt_x = lesion_3d_pts[2][k]
                    pt_y = lesion_3d_pts[1][k]
                    pt_z = lesion_3d_pts[0][k]
                    dx = (pt_x - lesion_center_x) * spacing[0]
                    dy = (pt_y - lesion_center_y) * spacing[1]
                    dz = (pt_z - lesion_center_z) * spacing[2]
                    d = math.sqrt(dx*dx + dy*dy + dz*dz)
                    lesion_distance_field[pt_z, pt_y, pt_x] = d
                    if max_d < d:
                        max_d = d

                # print('max_d = ', max_d)
                for k in range(lesion_3d_pts[2].shape[0]):
                    pt_x = lesion_3d_pts[2][k]
                    pt_y = lesion_3d_pts[1][k]
                    pt_z = lesion_3d_pts[0][k]
                    d = lesion_distance_field[pt_z, pt_y, pt_x]
                    wd = 1 - 0.2 * math.pow(d/max_d, 0.5)                   ### weight of distance

                    # pt_HU = smooth_volume_data[pt_z, pt_y, pt_x]
                    pt_HU = 0
                    cnt = 0
                    for z in range(pt_z - 1, pt_z + 2):
                        for y in range(pt_y - 2, pt_y + 3):
                            for x in range(pt_x - 2, pt_x + 3):
                                pt_HU += volume_data[z, y, x]
                                cnt += 1
                    pt_HU /= cnt
                    wi = math.exp(abs(pt_HU - lesion_mean_HU) * 0.004 * -1)     ### weight of intensity

                    intensity_similarity_mask[pt_z, pt_y, pt_x] = 255 * wi
                    distance_field_mask[pt_z, pt_y, pt_x] = 255 * wd

        distance_field = sitk.GetImageFromArray(distance_field_mask)
        distance_field.SetSpacing(mask.GetSpacing())
        distance_field.SetOrigin(mask.GetOrigin())
        distance_field.SetDirection(mask.GetDirection())

        intensity_similarity = sitk.GetImageFromArray(intensity_similarity_mask)
        intensity_similarity.SetSpacing(mask.GetSpacing())
        intensity_similarity.SetOrigin(mask.GetOrigin())
        intensity_similarity.SetDirection(mask.GetDirection())

        distance_field_path = os.path.join(CDF_dir, volume_id.replace('volume', 'segmentation')+'.nii.gz')
        sitk.WriteImage(distance_field, distance_field_path)

        intensity_similarity_path = os.path.join(ISM_dir, volume_id.replace('volume', 'segmentation')+'.nii.gz')
        sitk.WriteImage(intensity_similarity, intensity_similarity_path)

def generate_DSL_epoch(data_dir, alpha, epoch=0):
    
    segmentation_dir = os.path.join(data_dir, 'labels/')
    
    spatial_smooth_dir = os.path.join(data_dir, 'SSL/')
    intensity_similarity_dir = os.path.join(data_dir, 'ISM/')
    distance_field_dir = os.path.join(data_dir, 'CDF/')
    
    DSL_epoch_dir = os.path.join(data_dir, 'DSL/epoch_{}'.format(epoch))
    make_dir(DSL_epoch_dir)
 
    print (DSL_epoch_dir)

    train_vol_txt = os.path.join(data_dir, 'train_vol.txt')
    fp = open(train_vol_txt, 'r')
    volume_ids = fp.readlines()
    fp.close()
    volume_ids = [x.replace('\n', '') for x in volume_ids]
    
    file_num = len(volume_ids)
    count = 0
    for volume_id in volume_ids:
        # print ('start process ', volume_id)

        count += 1
        print ('\r', 'process: {}/{}'.format(count, file_num), end='', flush=True)
        
        ### 1. generate soft mask based on prior knowledge
        segmentation_file = volume_id.replace('volume', 'segmentation') + '.nii.gz'

        segmentation_path = os.path.join(segmentation_dir, segmentation_file)
        segmentation = sitk.ReadImage(segmentation_path)
        segmentation_data = sitk.GetArrayFromImage(segmentation)
        
        spatial_smooth_mask = sitk.ReadImage(os.path.join(spatial_smooth_dir, segmentation_file))
        spatial_smooth_mask_data = sitk.GetArrayFromImage(spatial_smooth_mask)

        intensity_similarity_mask = sitk.ReadImage(os.path.join(intensity_similarity_dir, segmentation_file))
        intensity_similarity_mask_data = sitk.GetArrayFromImage(intensity_similarity_mask)

        distance_field_mask = sitk.ReadImage(os.path.join(distance_field_dir, segmentation_file))
        distance_field_mask_data = sitk.GetArrayFromImage(distance_field_mask)

        intensity_similarity_mask_data = intensity_similarity_mask_data / 255.0
        distance_field_mask_data = distance_field_mask_data / 255.0
        
        soft_mask_data = spatial_smooth_mask_data * np.power(intensity_similarity_mask_data * distance_field_mask_data, alpha)      ### merge result

        ### save slice prior mask
        for i in range(segmentation_data.shape[0]):
            slice_mask = segmentation_data[i, :, :]

            ####没有liver的slice跳过 有liver无tumor的slice间隔采样
            if np.sum(slice_mask) == 0:
                continue
            elif np.sum(slice_mask == 2) == 0 and i % 2 == 0:
                continue
            
            #### save prior mask of slice
            slice_soft_mask = soft_mask_data[i, :, :]
            slice_name = volume_id + '_slice_' + str(i)
            save_soft_mask_path = os.path.join(DSL_epoch_dir, slice_name + '.png')            
            cv2.imwrite(save_soft_mask_path, slice_soft_mask.astype(np.uint8))

def generate_DSL(data_dir, max_epochs):
    for epoch in range(max_epochs):
        alpha = pow(0.90, epoch)
        print ('dynamic alpha value: ', epoch, alpha)    
        generate_DSL_epoch(data_dir, alpha, epoch=epoch)
        print ('\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for LiTS')
    parser.add_argument('--data_dir', default=None, help='the path to data set')
    parser.add_argument('--run_mode', default=0, type=int)    ### 0 convert_tumor_data / 1 prepare_SSL_ISM_CDF / 2 generate_DSL
    
    args = parser.parse_args()
    
    return args

def lits_data_process():
    
    args = parse_args()  
    print(args)
    
    lits_data_dir = args.data_dir
    run_mode = args.run_mode

    volumes_dir = os.path.join(lits_data_dir, 'volumes/')       ### path to place ct volume files
    labels_dir = os.path.join(lits_data_dir, 'labels/')         ### path to place label files

    fp = open(os.path.join(lits_data_dir, 'train_vol.txt'), 'r')
    vol_ids = fp.readlines()
    fp.close()
    train_volume_list = [id.replace('\n', '') for id in vol_ids] 

    print('train volumes num = ', len(train_volume_list))

    if run_mode == 0:
        convert_tumor_data(volumes_dir, labels_dir, train_volume_list, lits_data_dir, 'train')
    elif run_mode == 1:
        prepare_SSL_ISM_CDF(volumes_dir, labels_dir, lits_data_dir, train_volume_list)
    else:
        generate_DSL(lits_data_dir, 20)

if __name__ == '__main__':
    
    print ('let us start data process ...')
    lits_data_process()

