import cv2
import numpy as np


def double2uint8(I, ratio=1.0):
    return np.clip(np.round(I*ratio), 0, 255).astype(np.uint8)

def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))
    return kernel/kernel.sum()

def NLmeansfilter(I, h_=10, templateWindowSize=5,  searchWindowSize=11):
    f = int(templateWindowSize / 2)
    t = int(searchWindowSize / 2)
    height, width = I.shape[:2]
    padLength = t+f
    I2 = np.pad(I, padLength, 'symmetric')
    kernel = make_kernel(f)
    h = (h_**2)
    I_ = I2[padLength-f:padLength+f+height, padLength-f:padLength+f+width]

    average = np.zeros(I.shape)
    sweight = np.zeros(I.shape)
    wmax =  np.zeros(I.shape)
    for i in range(-t, t+1):
        for j in range(-t, t+1):
            if i==0 and j==0:
                continue
            I2_ = I2[padLength+i-f:padLength+i+f+height, padLength+j-f:padLength+j+f+width]
            w = np.exp(-cv2.filter2D((I2_ - I_)**2, -1, kernel)/h)[f:f+height, f:f+width]
            sweight += w
            wmax = np.maximum(wmax, w)
            average += (w*I2_[f:f+height, f:f+width])
    return (average+wmax*I)/(sweight+wmax)

def nonlocal_smooth_3d(input_data):
    output_data = np.zeros_like(input_data)
    
    for i in range(input_data.shape[0]):
        slice_data = input_data[i, :, :]
        
        if np.sum(slice_data) >= 1:
            smooth_slice_data = NLmeansfilter(slice_data, 20.0, 5, 11)
            output_data[i, :, :] = smooth_slice_data
    
    return output_data

### 先对每个XY横截面做GaussBlur，然后z方向对相邻上中下3层做平滑处理
def guass_smooth_3d(input_data, kernel_sizeXY=5, kernel_sizeZ=3, mask=None, do_3d=True):

    if kernel_sizeZ != 3:
        raise Exception('kernel_sizeZ must equal to 1.')

    if mask is None:
        mask = np.ones(input_data.shape, dtype='int')

    smooth_data_2d = np.zeros(input_data.shape, dtype='float')
    for i in range(input_data.shape[0]):
        slice_data = input_data[i, :, :]
        slice_mask = mask[i, :, :]
        if np.sum(slice_mask) > 0:
            smooth_slice_data = cv2.GaussianBlur(slice_data.astype('float'), (kernel_sizeXY, kernel_sizeXY), 0)
            smooth_data_2d[i, :, :] = smooth_slice_data

    if not do_3d:
        # print ('just do 2d smooth ...')
        return smooth_data_2d

    output_data = np.zeros(input_data.shape, dtype='float')
    w_z = [0.20, 0.60, 0.20]
    for i in range(input_data.shape[0]):
        slice_mask = np.zeros(slice_data.shape, dtype='int')
        slice_mask = mask[i, :, :]
        if np.sum(slice_mask) > 0:
            current_slice_data = smooth_data_2d[i, :, :]
            former_slice_index = i - 1
            next_slice_index = i + 1
            if former_slice_index < 0:
                former_slice_index = 0
            if next_slice_index > smooth_data_2d.shape[0] - 1:
                next_slice_index = smooth_data_2d.shape[0] - 1

            former_slice_data = smooth_data_2d[former_slice_index, :, :]
            next_slice_data = smooth_data_2d[next_slice_index, :, :]

            new_slice_data = w_z[0] * former_slice_data + w_z[1] * current_slice_data + w_z[2] * next_slice_data
            output_data[i, :, :] = new_slice_data

    # print ('finsh 3d smooth ...')
    return output_data        
        

if __name__ == '__main__':

    I = cv2.imread('D:/download/lena.jpeg', 0)

    sigma = 10.0
    smoothed = double2uint8(NLmeansfilter(I.astype('float'), sigma, 5, 11))

    # 显示结果图像
    cv2.imshow('Original Image', I)
    cv2.imshow('Smoothed Image', smoothed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()