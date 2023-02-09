# from segmentation_models import Unet, Nestnet, Xnet
import cv2
import numpy as np
from helper_functions import *       ###########it's only for A2C segmentaion model loading, Soumya may not need this
import collections
import math
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d ##Nov 1st, 2022
import tensorflow as tf

IMG_WIDTH_SEG = 224
IMG_WIDTH_ROI = 256
IMG_WIDTH_EF = 128
IMG_WIDTH_CLS = 224
IMG_WIDTH_SPK = 224  ##Nov 1st, 2022


ROImodel = tf.keras.models.load_model("./Models/model-Echo-ROI-AW-256-256-MobileNetV2_Jan_2022.h5", compile = True)
print('ROI model loaded')

clsmodel = tf.keras.models.load_model('./Models/model-comprehensive-view-classification-EfficientNetB4-one_Dense-24-classes-add-weights-add-vertical-flip.h5', compile = True)
pd_cls_model = tf.keras.models.load_model('./Models/model-binary-pediatric-classification-EfficientNetB0.h5', compile = True)
doppler_clsmodel = tf.keras.models.load_model('./Models/model-binary-doppler-view-classification-EfficientNetB0.h5', compile = True)
print('Classification model loaded')
segmodel = tf.keras.models.load_model('./Models/model-A2C-A4C-224-224-Unetpp-with-Augmentation-inceptionresnetv2-March-2022-12-0.0897.h5', compile = False)
segmodel.compile(optimizer="Adam", 
                 loss=bce_dice_loss,     
                 metrics=["binary_crossentropy", dice_coef])
segmodel_A2C = tf.keras.models.load_model('./Models/model-A2C-new-224-224-Unetpp-with-ROI-inceptionresnetv2-finetune-March-2022-16-0.0391.h5', compile = False)
segmodel_A2C.compile(optimizer="Adam", 
                 loss=bce_dice_loss,
                 metrics=["binary_crossentropy", dice_coef])
print('Segmentation model loaded')
EF_pred_model = tf.keras.models.load_model("./Models/model-Stanford-EF-EDArea-ESArea-regression-128-128-EfficientNetB6-Multi_loss_0.h5", compile = True)
print('EF Prediction model loaded')
GLS_pred_model = tf.keras.models.load_model('./Models/model-Stanford-GLS-regression-128-128-InceptionResNetV2_MSE_one_dense-Esad.h5', compile = True)
speckle_init_model = tf.keras.models.load_model('./Models/model-Echo-speckle-init-points-EfficientNetB4.h5', compile = True) ##Nov 1st, 2022
print('All Models are successfully loaded')
pi = 3.1415926535

def get_init_from_model(data):   ##Nov 1st, 2022
    test_data = cv2.resize(data, (224, 224), interpolation = cv2.INTER_LINEAR)
    ini = speckle_init_model.predict(test_data[np.newaxis, :, :, :])
    x1, y1, x2, y2, x3, y3 = int(ini[0][0][0]), int(ini[1][0][0]), \
    int(ini[2][0][0]), int(ini[3][0][0]),\
    int(ini[4][0][0]), int(ini[5][0][0])
    return x1, y1, x2, y2, x3, y3

def get_modified_points(ori_pts, model_pt, num_points): ##Nov 1st, 2022
    '''
    combine the control points from speckle_init_model with the segmentation results to get the speckles
    '''      
    dist_list = []
    for i in range(ori_pts.shape[0]):
        dist_list.append(np.sqrt(np.power(ori_pts[i, 0] - model_pt[0], 2) + np.power(ori_pts[i, 1] - model_pt[1], 2)))
    mod_pts = ori_pts[0:np.argmin(dist_list) + 1]
    mod_pts[-1, :] = model_pt
    
    #######remove duplicates on x-axis
    mod_pts= mod_pts[mod_pts[:, 1].argsort()]
    _, counts = np.unique(mod_pts[:, 1], return_counts=True)
    cum_count = (np.cumsum(counts) - 1).tolist()
    del_list = list(set(range(max(cum_count) + 1)) - set(cum_count))
    mod_pts = np.delete(mod_pts, del_list, 0)
    ##################################
    f = interp1d(mod_pts[:, 1], mod_pts[:, 0], kind='cubic')
    ynew = np.array(np.linspace(mod_pts[0, 1], mod_pts[-1, 1], num=num_points, endpoint=True).astype(np.int16))
    xnew = np.array(f(ynew).astype(np.int16))
    new_points = np.stack((xnew, ynew), axis = 0).T
    return new_points


def get_empty_array_from_video(cv2cap):
    frameCount = int(cv2cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cv2cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cv2cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount + 1, frameHeight, frameWidth, 3), np.dtype('uint8'))   #####cv2 will capture the next frame of the last
    
    return buf

def ROI_frame(frame_array):
    h_resize_rate = frame_array.shape[0] / 256
    w_resize_rate = frame_array.shape[1] / 256
    frame_array = cv2.resize(frame_array, (IMG_WIDTH_ROI, IMG_WIDTH_ROI))
    ROI = ROImodel.predict(frame_array[np.newaxis, :, :, :])
    xx, yy, ww, hh = int(ROI[0][0][0] * w_resize_rate), int(ROI[1][0][0] * h_resize_rate), int(ROI[2][0][0] * w_resize_rate), int(ROI[3][0][0] * h_resize_rate)
    return xx, yy, ww, hh

def cls_frame(frame_array):  
    try:
        frame_array = cv2.resize(frame_array, (IMG_WIDTH_CLS, IMG_WIDTH_CLS))
        cls_result = np.argmax(clsmodel.predict(frame_array[np.newaxis, :, :, :]).ravel())
        if cls_result == 0 or cls_result == 2:
                cls_doppler = doppler_clsmodel.predict(frame_array[np.newaxis, :, :, :])[0][0]
                if cls_doppler > 0.5:
                    cls_result = 21
    except:
        cls_result = 21
    return cls_result

def cls_pd(frame_array):
    frame_array = cv2.resize(frame_array, (IMG_WIDTH_CLS, IMG_WIDTH_CLS))
    pd = np.argmin([pd_cls_model.predict(frame_array[np.newaxis, :, :, :])[0][0], 0.5])
    return pd
        
def pred_EF(ED_frame, ES_frame, mid_frame):

    img_rgb = (np.dstack((ED_frame, mid_frame, ES_frame))).astype(np.uint8)
    img_rgb = cv2.resize(img_rgb, (IMG_WIDTH_EF, IMG_WIDTH_EF), interpolation = cv2.INTER_NEAREST)
    cv2.normalize(img_rgb, img_rgb, 0, 255, cv2.NORM_MINMAX)
    img_rgb = np.expand_dims(img_rgb, axis = 0)
    EF_pred = EF_pred_model.predict(img_rgb)[0][0]                     #Call the EF prediction model
    # print(EF_pred)
    return EF_pred  

def pred_GLS(ED_frame, ES_frame, mid_frame):
    img_rgb = (np.dstack((ED_frame, mid_frame, ES_frame))).astype(np.uint8)
    img_rgb = cv2.resize(img_rgb, (IMG_WIDTH_EF, IMG_WIDTH_EF), interpolation = cv2.INTER_NEAREST)
    # cv2.normalize(img_rgb, img_rgb, 0, 255, cv2.NORM_MINMAX)
    img_rgb = np.expand_dims(img_rgb, axis = 0)
    GLS_pred = GLS_pred_model.predict(img_rgb)[0]
    print(GLS_pred)
    #Call the GLS prediction model
    # print(EF_pred)
    return GLS_pred 

def pred_area(frame_batch):
    img_rgb = (np.dstack((frame_batch, frame_batch, frame_batch))).astype(np.uint8)
    img_rgb = cv2.resize(img_rgb, (IMG_WIDTH_EF, IMG_WIDTH_EF), interpolation = cv2.INTER_NEAREST)
    cv2.normalize(img_rgb, img_rgb, 0, 255, cv2.NORM_MINMAX)
    img_rgb = np.expand_dims(img_rgb, axis = 0)
    area_pred = EF_pred_model.predict(img_rgb)[1][0][0]                     #Call the EF prediction model
    return area_pred 


def get_mask(echo_frame):
    prediction_val = segmodel.predict(echo_frame)                         #call the EF prediction model
    # prediction_proba = segmodel.predict_proba(echo_frame)
    mask = np.argmax(prediction_val, axis = -1)
    maskbin = mask * 16
    return mask, maskbin, prediction_val

def get_mask_A2C(echo_frame):
    prediction_val = segmodel_A2C.predict(echo_frame)                         #call the EF prediction model
    # prediction_proba = segmodel.predict_proba(echo_frame)
    mask = np.argmax(prediction_val, axis = -1)
    maskbin = mask * 16
    return mask, maskbin, prediction_val

def get_empty_array_from_video(cv2cap):
    frameCount = int(cv2cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cv2cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cv2cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount + 1, frameHeight, frameWidth, 3), np.dtype('uint8'))   #####cv2 will capture the next frame of the last
    
    return buf

def get_top_bottom_intersect_points(mask, val_L, val_R):    ####Modified Oct 2021
    mask_L = np.zeros(mask.shape)
    mask_R = np.zeros(mask.shape)
    mask_L_R = np.zeros(mask.shape)
    mask_L[np.where(mask == val_L)] = 10
    mask_R[np.where(mask == val_R)] = 10
    mask_L_new = np.zeros((mask.shape[0], mask.shape[1] + 1))
    mask_L_new[:, 1::] = mask_L
    mask_L = mask_L_new[:, 0:-1]
    mask_L = mask_L.astype(np.uint8)
    mask_R = mask_R.astype(np.uint8)
    
    ret,thresh_L = cv2.threshold(mask_L,9,2,cv2.THRESH_BINARY)
    contours_L, hierarchy = cv2.findContours(thresh_L, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour_L_len = 0
    for i in range(len(contours_L)):
        if contours_L[i].shape[0] > contour_L_len:
            contour_L_len = contours_L[i].shape[0]
            contour_L_ind = i
    contour_L = contours_L[contour_L_ind].tolist()
    contour_L_list = []
    for i in range(len(contour_L)):
        contour_L_list.append(contour_L[i][0])
    
    revised_contour_L_list = []
    for i in range(len(contour_L_list)):
        revised_contour_L_list.append([contour_L_list[i][0], contour_L_list[i][1] - 1])
    cv2.fillConvexPoly(mask_L_R, np.array(revised_contour_L_list), 1)

    ret,thresh_R = cv2.threshold(mask_R,9,2,cv2.THRESH_BINARY)
    contours_R, hierarchy = cv2.findContours(thresh_R, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour_R_len = 0
    for i in range(len(contours_R)):
        if contours_R[i].shape[0] > contour_R_len:
            contour_R_len = contours_R[i].shape[0]
            contour_R_ind = i
    contour_R = contours_R[contour_R_ind].tolist()
    contour_R_list = []
    for i in range(len(contour_R)):
        contour_R_list.append(contour_R[i][0])
    
    cv2.fillConvexPoly(mask_L_R, np.array(contour_R_list), 1)
    mask_L_R[np.where(mask_L_R == 1)] = 10
    mask_L_R = mask_L_R.astype(np.uint8)
    ret,thresh_L_R = cv2.threshold(mask_L_R,9,2,cv2.THRESH_BINARY)
    contours_L_R, hierarchy = cv2.findContours(thresh_L_R, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour_L_R_len = 0
    for i in range(len(contours_L_R)):
        if contours_L_R[i].shape[0] > contour_L_R_len:
            contour_L_R_len = contours_L_R[i].shape[0]
            contour_L_R_ind = i
    contour_L_R = contours_L_R[contour_L_R_ind].tolist()
    contour_L_R_list = []
    for i in range(len(contour_L_R)):
        contour_L_R_list.append(contour_L_R[i][0])

    list_bound_L_R = list(set(map(tuple,contour_L_list)).intersection(set(map(tuple, contour_R_list))))
    
    contour_R_no_inter = list(set(map(tuple, contour_R_list)) - set(list_bound_L_R))
    contour_L_no_inter = list(set(map(tuple, contour_L_list)) - set(list_bound_L_R))
    revised_contour_L_no_inter_list = []
    for i in range(len(contour_L_no_inter)):
        revised_contour_L_no_inter_list.append([contour_L_no_inter[i][0], contour_L_no_inter[i][1] - 1])
    
    
    sort_list = sorted(list_bound_L_R,key=lambda l:l[1], reverse=False)
    if len(sort_list) >= 2:
        top_point = list(sort_list[0])
        bottom_point = list(sort_list[-1])
    else:
        top_point = [1, 1]
        bottom_point = [0, 0]
    
    if bottom_point[1] - top_point[1] == 0:
        slope_diameter = math.atan(math.inf)
    else:
        slope_diameter = math.atan((bottom_point[0] - top_point[0]) / (bottom_point[1] - top_point[1]))       #######oct 2021
    
    return top_point, bottom_point, slope_diameter, mask_L_R, revised_contour_L_no_inter_list, contour_R_no_inter, contour_L_R_list
       

def get_intersecs_boundary_standord(L_contour, R_contour, inner_point, slope):  ####Modified Oct 2021

    product_L = np.zeros((len(L_contour), 1))                    ##########new method developed on Oct 2021
    product_R = np.zeros((len(R_contour), 1))
    for k in range(len(L_contour)):
        if L_contour[k][1] == inner_point[1]:
            perpen_slope = math.atan(math.inf)
        else:
            perpen_slope = math.atan((L_contour[k][0] - inner_point[0]) / (L_contour[k][1] - inner_point[1]))  #[-pi/2, pi/2]
        # product_L[k] = np.abs(perpen_slope - slope - math.pi / 2)
        product_L[k] = abs(math.tan(perpen_slope - slope))
    for k in range(len(R_contour)):
        if R_contour[k][1] == inner_point[1]:
            perpen_slope = math.atan(math.inf)
        else:
            perpen_slope = math.atan((R_contour[k][0] - inner_point[0]) / (R_contour[k][1] - inner_point[1]))
        product_R[k] = abs(math.tan(perpen_slope - slope))
    pointL_index = np.argmax(product_L)
    pointR_index = np.argmax(product_R)
    return [L_contour[pointL_index][0], L_contour[pointL_index][1]], [R_contour[pointR_index][0], R_contour[pointR_index][1]]


def get_LV_area_and_volume_no_strain(whole_mask, L_val, R_val):
     
    top_point, bottom_point, slope_diameter, LV_mask, contour_L, contour_R, contour_L_R = get_top_bottom_intersect_points(whole_mask, L_val, R_val)
    area = collections.Counter(LV_mask.flatten())[1]
    interval = [(bottom_point[0] - top_point[0]) / 21, (bottom_point[1] - top_point[1]) / 21]
    interval_len = np.sqrt(np.power((bottom_point[0] - top_point[0]) / 21, 2) + 
                        np.power((bottom_point[1] - top_point[1]) / 21, 2))

    vol = 0
    inter_left_list = []
    inter_right_list = []
    for j in range(1, 21):
        intersect_left, intersect_right = get_intersecs_boundary_standord(contour_L, contour_R, [min(top_point[0] + interval[0] * j, IMG_WIDTH_SEG - 1) , min(top_point[1] + interval[1] * j, IMG_WIDTH_SEG - 1)], slope_diameter)
        inter_left_list.append(intersect_left)
        inter_right_list.append(intersect_right)
        diameter = np.sqrt(np.power((intersect_right[0] - intersect_left[0]), 2) + np.power((intersect_right[1] - intersect_left[1]), 2))
        vol += pi*np.power(diameter/2, 2) * interval_len
        
    return area, vol, LV_mask, top_point, bottom_point, np.array(inter_left_list), np.array(inter_right_list), contour_L, contour_R, contour_L_R

def equal_space_LA(LA_contour, num_points):   #########new Sep 2021   ####Modified Oct 2021
    peri = 0
    for i in range(len(LA_contour)):
        peri += np.sqrt(np.power((LA_contour[i - 1][0] - LA_contour[i][0]), 2) + \
            np.power((LA_contour[i - 1][1] - LA_contour[i][1]), 2))
    
    space = peri / (num_points)
    out_points = []
    peri_space = 0
    for i in range(len(LA_contour) - 1):
        peri_space += np.sqrt(np.power((LA_contour[i][0] - LA_contour[i + 1][0]), 2) + \
            np.power((LA_contour[i][1] - LA_contour[i + 1][1]), 2))
        if peri_space >= space:
            out_points.append([LA_contour[i + 1][0],  LA_contour[i + 1][1]])
            peri_space = 0
    # out_points.append([LA_contour[-1][0],  LA_contour[-1][1]])
    out_points.insert(0, [LA_contour[0][0],  LA_contour[0][1]])
    return out_points

def get_sorted_boundary(full_contour, intersect_points):  ####new and Modified Oct 2021
    '''
    get the boundary without intersection sorted
    inputs are lists
    '''
    min_ele = intersect_points[0]
    for ele in intersect_points:
        if ele[0] < min_ele[0]:
            min_ele = ele
    min_ele_ind = full_contour.index(min_ele)
    num_control_points = len(full_contour) - len(intersect_points)
    double_contour_LA_list = full_contour + full_contour
    contour_sorted = double_contour_LA_list[min_ele_ind  : min_ele_ind + num_control_points + 2]
    return contour_sorted

def get_MC_contour(mask, LLV, RLV, LA, MC): ###Modified Oct 2021
    
    mask_MC= np.zeros(mask.shape)
    mask_LA = np.zeros(mask.shape)
    mask_MC[np.where(mask == LLV)] = 10
    mask_MC[np.where(mask == RLV)] = 10
    mask_MC[np.where(mask == MC)] = 10
    mask_LA[np.where(mask == LA)] = 10
    mask_LA_new = np.zeros((mask.shape[0] + 1, mask.shape[1]))
    mask_LA_new[0:-1, :] = mask_LA
    mask_LA = mask_LA_new[1::, :]
    mask_MC = mask_MC.astype(np.uint8)
    mask_LA = mask_LA.astype(np.uint8)
    
    ret,thresh_MC = cv2.threshold(mask_MC,9,2,cv2.THRESH_BINARY)
    contours_MC, hierarchy = cv2.findContours(thresh_MC, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour_MC_len = 0
    for i in range(len(contours_MC)):
        if contours_MC[i].shape[0] > contour_MC_len:
            contour_MC_len = contours_MC[i].shape[0]
            contour_MC_ind = i
    contour_MC = contours_MC[contour_MC_ind].tolist()
    contour_MC_list = []
    for i in range(len(contour_MC)):
        contour_MC_list.append(contour_MC[i][0])

    ret,thresh_LA = cv2.threshold(mask_LA,9,2,cv2.THRESH_BINARY)
    contours_LA, hierarchy = cv2.findContours(thresh_LA, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour_LA_len = 0
    for i in range(len(contours_LA)):
        if contours_LA[i].shape[0] > contour_LA_len:
            contour_LA_len = contours_LA[i].shape[0]
            contour_LA_ind = i    
    if contours_LA != []:
        contour_LA = contours_LA[contour_LA_ind].tolist()

        contour_LA_list = []
        for i in range(len(contour_LA)):
            contour_LA_list.append(contour_LA[i][0])
    else:
        contour_LA_list = []
        
    contour_LA_list = equal_space_LA(contour_LA_list, 20)  ####Sep 2021
    
    # print(contour_LV_list)
    # print(contour_LA_list)
    list_bound_MC_LA = set(map(tuple,contour_MC_list)).intersection(set(map(tuple, contour_LA_list)))
    contour_LA_tuple = get_sorted_boundary(contour_LA_list, list(map(list,list_bound_MC_LA))) ###sep 2021, using this function to get the sorted boundary

    contour_MC_tuple = list(set(map(tuple,contour_MC_list)) - list_bound_MC_LA)
    
    return contour_MC_tuple, contour_LA_tuple #########sep 2021



def get_L_R_MC(top_pt, bottom_pt, MC_contour_list):      ### ###Modified Oct 2021
    A = bottom_pt[1] - top_pt[1]
    B = top_pt[0] - bottom_pt[0]
    C = bottom_pt[0]*top_pt[1] - top_pt[0]*bottom_pt[1]
    L_MC = []
    R_MC = []
    L_MC_top = []
    R_MC_top = []
    
    sorted_MC_list = sorted(MC_contour_list, key=lambda l:l[1], reverse = False)
    MC_top = sorted_MC_list[0]
    
    for i in range(1, len(sorted_MC_list)):
        dist = A*sorted_MC_list[i][0] +B*sorted_MC_list[i][1] + C  
        if dist < 0:
            L_MC.append([sorted_MC_list[i][0], sorted_MC_list[i][1]])
            if len(L_MC_top) < 20:              ####adjust the number for the smoother top MC contour
                L_MC_top.append([sorted_MC_list[i][0], sorted_MC_list[i][1], abs(dist)])
        elif dist > 0:
            R_MC.append([sorted_MC_list[i][0], sorted_MC_list[i][1]])
            if len(R_MC_top) < 20:
                R_MC_top.append([sorted_MC_list[i][0], sorted_MC_list[i][1], abs(dist)])
    L_MC_top_sort = sorted(L_MC_top, key = lambda l:l[2], reverse = False)
    R_MC_top_sort = sorted(R_MC_top, key = lambda l:l[2], reverse = False)
    for row in L_MC_top_sort:
        del row[2]
    for row in R_MC_top_sort:
        del row[2]
           
    return L_MC, R_MC, MC_top, L_MC_top_sort, R_MC_top_sort

def finetune_mask(mask, point1, point2, L_val, R_val):
    A = point2[1] - point1[1]
    B = point1[0] - point2[0]
    C = point2[0]*point1[1] - point1[0]*point2[1]
    for k in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if A * j + B * k + C <= 0 and mask[k, j] == R_val:
                mask[k, j] = L_val
            elif A * j + B * k + C > 0 and mask[k, j] == L_val:
                mask[k, j] = R_val
    return mask

def get_LV_area_and_volume_with_strain(whole_mask, L_val, R_val, LA_val, MC_val):   ###Modified Feb 2022
    
    top_point, bottom_point, slope_diameter, LV_mask, contour_L, contour_R, contour_L_R = get_top_bottom_intersect_points(whole_mask, L_val, R_val)
    whole_mask = finetune_mask(whole_mask, top_point, bottom_point, L_val, R_val)
    top_point, bottom_point, slope_diameter, LV_mask, contour_L, contour_R, contour_L_R= get_top_bottom_intersect_points(whole_mask, L_val, R_val)
    
    MC_contour, LA_contour = get_MC_contour(whole_mask, L_val, R_val, LA_val, MC_val)
    
    area = collections.Counter(whole_mask.flatten())[L_val] + collections.Counter(whole_mask.flatten())[R_val]
    interval = [(bottom_point[0] - top_point[0]) / 21, (bottom_point[1] - top_point[1]) / 21]
    interval_len = np.sqrt(np.power((bottom_point[0] - top_point[0]) / 21, 2) + 
                        np.power((bottom_point[1] - top_point[1]) / 21, 2))
    
    # MC_top_point = [int(top_point[0] - 3 * interval[0]), int(top_point[1] - 3 * interval[1])]
    # top_mid_point = [(top_point[0] + MC_top_point[0]) // 2, (top_point[1] + MC_top_point[1]) // 2]
    MC_L, MC_R, MC_top_point, inter_leftMC_list, inter_rightMC_list = get_L_R_MC(top_point, bottom_point, MC_contour) ###get the top ten points on each side of the MC
    top_mid_point = [(top_point[0] + MC_top_point[0]) // 2, (top_point[1] + MC_top_point[1]) // 2]

    #yield top_point, bottom_point, slope_diameter, interval, interval_len, frame_gray, pred, pred_triage, t

    vol = 0
    inter_leftLV_list = []
    inter_rightLV_list = []
    inter_leftMC__disc_list = []  #this list is for strain calculation
    inter_rightMC_disc_list = []
    for j in range(1, 21):
        intersect_left_LV, intersect_right_LV = get_intersecs_boundary_standord(contour_L, contour_R, [min(top_point[0] + interval[0] * j, IMG_WIDTH_SEG - 1), \
            min(top_point[1] + interval[1] * j, IMG_WIDTH_SEG - 1)], slope_diameter) 
        inter_leftLV_list.append(intersect_left_LV)
        inter_rightLV_list.append(intersect_right_LV)
        intersect_left_MC, intersect_right_MC = get_intersecs_boundary_standord(MC_L, MC_R, [min(top_point[0] + interval[0] * j, IMG_WIDTH_SEG - 1), \
            min(top_point[1] + interval[1] * j, IMG_WIDTH_SEG - 1)], slope_diameter)
        inter_leftMC__disc_list.append(intersect_left_MC)
        inter_rightMC_disc_list.append(intersect_right_MC) 
        
        if intersect_left_MC not in inter_leftMC_list:
            inter_leftMC_list.append(intersect_left_MC)      ######this list is for returning the control points
            
        if intersect_right_MC not in inter_rightMC_list:
            inter_rightMC_list.append(intersect_right_MC)
            
        diameter = np.sqrt(np.power((intersect_right_LV[0] - intersect_left_LV[0]), 2) + np.power((intersect_right_LV[1] - intersect_left_LV[1]), 2))
        vol += pi*np.power(diameter/2, 2) * interval_len
    
    intersect_left_LV_top_1, intersect_right_LV_top_1 = get_intersecs_boundary_standord(contour_L, contour_R, [min(top_point[0] + interval[0] / 2, IMG_WIDTH_SEG - 1) , \
        min(top_point[1] + interval[1] / 2, IMG_WIDTH_SEG - 1)], slope_diameter) ##OCT14 2022
        
    intersect_left_LV_vice_anchor, intersect_right_LV_vice_anchor = get_intersecs_boundary_standord(contour_L, contour_R, [min(bottom_point[0] - interval[0] / 2, IMG_WIDTH_SEG - 1) , \
        min(bottom_point[1] - interval[1] / 2, IMG_WIDTH_SEG - 1)], slope_diameter)   ##Oct 2021  generate these points to make the common line more line a straight line
    intersect_left_MC_vice_anchor, intersect_right_MC_vice_anchor = get_intersecs_boundary_standord(MC_L, MC_R, [min(bottom_point[0] - interval[0] / 2, IMG_WIDTH_SEG - 1) , \
        min(bottom_point[1] - interval[1] / 2, IMG_WIDTH_SEG - 1)], slope_diameter)   ##Oct 2021
    
    intersect_left_LV_anchor, intersect_right_LV_anchor = get_intersecs_boundary_standord(contour_L, contour_R, [min(bottom_point[0] - interval[0] / 4, IMG_WIDTH_SEG - 1) , \
        min(bottom_point[1] - interval[0] / 4, IMG_WIDTH_SEG - 1)], slope_diameter)   ##Oct 19 2022
    intersect_left_MC_anchor, intersect_right_MC_anchor = get_intersecs_boundary_standord(MC_L, MC_R, [min(bottom_point[0] - interval[0] / 4, IMG_WIDTH_SEG - 1) , \
        min(bottom_point[1] - interval[0] / 4, IMG_WIDTH_SEG - 1)], slope_diameter)   ##Oct 19 2022
    
    left_strain_list = []
    right_strain_list = []
    left_mid_point_list = []
    right_mid_point_list = []
    for i in range(20):
        left_mid_point_list.append([(inter_leftMC__disc_list[i][0] + inter_leftLV_list[i][0])//2, (inter_leftMC__disc_list[i][1] + inter_leftLV_list[i][1])//2])
        right_mid_point_list.append([(inter_rightMC_disc_list[i][0] + inter_rightLV_list[i][0])//2, (inter_rightMC_disc_list[i][1] + inter_rightLV_list[i][1])//2])
    
    ######################use mid point#############
    for i in range(19):
        left_strain_list.append(np.sqrt(np.power(left_mid_point_list[i][0] - left_mid_point_list[i + 1][0], 2)\
                                         + np.power(left_mid_point_list[i][1] - left_mid_point_list[i + 1][1], 2)))
        #########from top to bottom##############
        right_strain_list.append(np.sqrt(np.power(right_mid_point_list[i][0] - right_mid_point_list[i + 1][0], 2)\
                                         + np.power(right_mid_point_list[i][1] - right_mid_point_list[i + 1][1], 2)))
    AP_length_left = np.sqrt(np.power((top_mid_point[0] - (inter_leftMC__disc_list[0][0] + inter_leftLV_list[0][0])/2), 2)\
                                         + np.power((top_mid_point[1] - (inter_leftMC__disc_list[0][1] + inter_leftLV_list[0][1])/2), 2))
    AP_length_right = np.sqrt(np.power((top_mid_point[0] - (inter_rightMC_disc_list[0][0] + inter_rightLV_list[0][0])/2), 2)\
                                         + np.power((top_mid_point[1] - (inter_rightMC_disc_list[0][1] + inter_rightLV_list[0][1])/2), 2))
    
    A_M_boundary = 6 ###GLS OCT 2022
    M_B_boundary = 13 ###GLS OCT 2022
    
    AP_length = round(AP_length_left + AP_length_right, 3)  ###GLS OCT 2022
    AS_length = round(AP_length_left + np.sum(left_strain_list[0:A_M_boundary]), 3)
    MS_length = round(np.sum(left_strain_list[A_M_boundary:M_B_boundary]), 3)
    BS_length = round(np.sum(left_strain_list[M_B_boundary::]), 3)
    AL_length = round(AP_length_right + np.sum(right_strain_list[0:A_M_boundary]), 3)
    ML_length = round(np.sum(right_strain_list[A_M_boundary:M_B_boundary]), 3)
    BL_length = round(np.sum(right_strain_list[M_B_boundary::]), 3)
    regional_strain_list = [AS_length,MS_length,BS_length, AL_length, ML_length, BL_length, AP_length] ###GLS OCT 2022
    
    
    ###LV control points include 20 discs and two more points on top between the AP and the first disc, plus one point between the bottom and the last disc
    inter_leftLV_list.insert(0, intersect_left_LV_top_1)##OCT14 2022
    inter_rightLV_list.insert(0, intersect_right_LV_top_1)##OCT14 2022
    
    inter_leftLV_list.append(intersect_left_LV_vice_anchor)
    inter_rightLV_list.append(intersect_right_LV_vice_anchor)
    inter_leftMC_list.append(intersect_left_MC_vice_anchor)
    inter_rightMC_list.append(intersect_right_MC_vice_anchor)
    
    inter_leftLV_list.append(intersect_left_LV_anchor)
    inter_rightLV_list.append(intersect_right_LV_anchor)
    inter_leftMC_list.append(intersect_left_MC_anchor)
    inter_rightMC_list.append(intersect_right_MC_anchor)  ##Oct 2021 add the anchor points of LV and MC for the front end
        
    return area, vol, LV_mask, top_point, bottom_point, np.array(inter_leftLV_list), np.array(inter_rightLV_list), contour_L, contour_R, contour_L_R\
        ,np.array(inter_leftMC_list), np.array(inter_rightMC_list), MC_top_point, left_mid_point_list, right_mid_point_list, regional_strain_list, LA_contour

def find_ED_ES(vol_lst, win_width):   ###Modified Feb 2022)
    ED_ind = [0]
    ES_ind = [0]
    if len(vol_lst) >= win_width:  #50fps in the current case
        vol_list_window_ori = vol_lst[-win_width::]
        vol_list_window = savgol_filter(vol_list_window_ori, int(len(vol_list_window_ori) // 4 * 2) + 1, 3)  ####Added Feb 2022
        # vol_list_window = vol_list_window_ori
        diff_list = np.diff(np.array(vol_list_window))
        threshold = 3 * np.mean(np.abs(diff_list))
        if np.sum(diff_list) < threshold:
         
            if np.sum(diff_list[0:win_width // 2]) > 0 and np.sum(diff_list[win_width // 2 : win_width - 1] < 0) and \
            (vol_list_window[0] + vol_list_window[1]) < (vol_list_window[win_width // 2 - 2] + vol_list_window[win_width // 2 - 1]) and\
            (vol_list_window[-1] + vol_list_window[-2]) < (vol_list_window[win_width // 2] + vol_list_window[win_width // 2 + 1]):

                ED_ind = np.where(vol_list_window_ori==np.max(vol_list_window_ori))[0] + len(vol_lst) - win_width
            elif np.sum(diff_list[0:win_width // 2]) < 0 and np.sum(diff_list[win_width // 2 : win_width - 1] > 0) and \
            vol_list_window[0] + vol_list_window[1] > vol_list_window[win_width // 2 - 2] + vol_list_window[win_width // 2 - 1] and \
            vol_list_window[-1] + vol_list_window[-2] > vol_list_window[win_width // 2] + vol_list_window[win_width // 2 + 1]:
                ES_ind = np.where(vol_list_window_ori==np.min(vol_list_window_ori))[0] + len(vol_lst) - win_width

    if np.array(ED_ind)[0] != 0 or np.array(ES_ind)[0] != 0:
        return np.array([ED_ind[0]]), np.array([ES_ind[0]])
        # return np.array(ED_ind)[0], np.array(ES_ind)[0]
    else:
        return 0, 0
    

def get_ED_ES_list(ES_list, ED_list, vol_list, win_width, find_ED, find_ES):
    ED, ES = find_ED_ES(vol_list, win_width)
    # print(ED, ES, find_ED, find_ES)
    if ED != 0 and find_ED == True:
        ED_list.append([ED[0], vol_list[ED[0]]])
        find_ED = False
        find_ES= True
    elif ED !=0 and find_ED == False:
        if ED_list[-1][1] < vol_list[ED[0]]:
            ED_list[-1] = [ED[0], vol_list[ED[0]]]
    elif ES != 0 and find_ES == True:
        ES_list.append([ES[0], vol_list[ES[0]]])
        find_ED = True
        find_ES= False
    elif ES != 0 and find_ES == False:
        # print(ES_list)
        if ES_list[-1][1] > vol_list[ES[0]]:
            ES_list[-1] = [ES[0], vol_list[ES[0]]]
                
    return ES_list, ED_list, find_ED, find_ES

def feature_extraction(patches):
    patches = patches.reshape(patches.shape[0], -1)
    features = np.zeros((patches.shape[0], 5))
    
    features[:, 0] = patches.mean(axis=1)
    features[:, 1] = patches.std(axis=1)
    features[:, 2] = patches.max(axis=1)
    features[:, 3] = patches.min(axis=1)
#    features[:, 4] = patches.sum(axis=1)
    return features

def track_points_sequential(frames, markers, WS, SS, model=None, show_message=True):
    (oldX, oldY) = markers    
    rows = frames.shape[0]
    cols = oldX.shape[0]
    all_new_x = np.zeros((rows, cols))
    all_new_y = np.zeros((rows, cols))
    all_new_x[0] = oldX.reshape((1, -1))
    all_new_y[0] = oldY.reshape((1, -1))
    
    for i in range(0, rows-1):
        newX, newY = track_specified_points(frames[i], frames[i+1],
                     markers=(oldX, oldY), WS=WS, SS=SS, model=model, show_message=False)
        (oldX, oldY) = (newX, newY)
        all_new_x[i+1] = newX.reshape((1, -1))
        all_new_y[i+1] = newY.reshape((1, -1))
        if show_message:
            print('{:.3}'.format((i+1)/(frames.shape[0]-1)*100))
    return all_new_x, all_new_y

def track_specified_points(frame1, frame2, markers, WS, SS, model=None, show_message=False):
    ######
    ### track_point > applying block matching to track a point between 
    ###               two frames.
    ### im1 = firt frame
    ### im2 = second frame
    ### markers = inpute initial points
    ### WS = Window Size
    ### SS= Search Size
    #######
    
    if type(frame1) != np.ndarray:
           print ('Error: Input image1 should be numpy.ndarray')
           return 0
    if type(frame2) != np.ndarray:
           print ('Error: Input image2 should be numpy.ndarray')
           return 0
    X, Y = markers   
    f_size = frame1.shape
    f_rows = f_size[0]
    f_cols = f_size[1]
#    counts = markers.shape[0]
    counts = X.shape[0]
    PD = WS + SS #Padding
    
    TH = 0  #Speckle Threshold
    
    vecty = np.zeros(f_size)
    progress = 0
    x_displacements = np.zeros_like(X)
    y_displacements = np.zeros_like(Y)
    
    for i in range(0, counts):
        try:
            progress += 1
            if show_message:
                print('{:.3}'.format(progress/(counts)*100))
            col = X[i]
            row = Y[i]
            if col==0 and row==0:
                continue
            
            window = frame1[row-WS:row+WS,col-WS:col+WS]

            if model:
                feature = feature_extraction(window.reshape((1, window.shape[0], window.shape[1])))
                label = model.predict(feature)
                
                if label[0] != 1: # If it is not speckle, go to next point.
                    continue
            
            match_score = np.zeros((2*SS+1, 2*SS+1))
            cross_col = np.zeros((2*SS+1, 2*SS+1))
            
            for ii in range(-SS, SS+1):
                for jj in range(-SS, SS+1):
                    patch = frame2[row+ii-WS:row+ii+WS, col+jj-WS:col+jj+WS]
    #                    match_score[ii+SS, jj+SS] = np.sum(np.power(window-patch, 2))            
                    if window.std()*patch.std():
                        cross_col[ii+SS, jj+SS] = abs(np.mean((window-window.mean())*(patch-patch.mean()) / (window.std()*patch.std())))
            match_score = cross_col
            match_score = np.nan_to_num(match_score)

            if match_score[SS, SS] == 1 or match_score[SS, SS] != match_score[SS, SS]:
                y_displacements[i] = 0
                x_displacements[i] = 0
            else:
                a, b = np.where(match_score == np.max(match_score))
                y_displacements[i] = a[0] - (SS)
                x_displacements[i] = b[0] - (SS)
        except:
            y_displacements[i] = 0
            x_displacements[i] = 0
 
    return (x_displacements + X, y_displacements + Y)

def Regional_LS_speckle(LS, RS, AP):
    dist_AP = 0
    dist_AP += np.sqrt(np.power(LS[0,0] - AP[0,0], 2) + np.power(LS[1,0] - AP[1,0], 2))
    dist_AP += np.sqrt(np.power(RS[0,0] - AP[0,0], 2) + np.power(RS[1,0] - AP[1,0], 2))
    
    incre_L = LS.shape[1] // 3
    dist_LT = 0 ##left top
    for i in range(incre_L):
        dist_LT += np.sqrt(np.power(LS[0,i] - LS[0,i+1], 2) + np.power(LS[1,i] - LS[1,i+1], 2))
    dist_LM = 0 ##left mid
    for i in range(incre_L, 2 * incre_L):
        dist_LM += np.sqrt(np.power(LS[0,i] - LS[0,i+1], 2) + np.power(LS[1,i] - LS[1,i+1], 2))
    dist_LB = 0 ##left bottom
    for i in range(2 * incre_L, LS.shape[1] - 1):
        dist_LB += np.sqrt(np.power(LS[0,i] - LS[0,i+1], 2) + np.power(LS[1,i] - LS[1,i+1], 2))
        
    incre_R = RS.shape[1] // 3
    dist_RT = 0 ##left top
    for i in range(incre_R):
        dist_RT += np.sqrt(np.power(RS[0,i] - RS[0,i+1], 2) + np.power(RS[1,i] - RS[1,i+1], 2))
    dist_RM = 0 ##left mid
    for i in range(incre_R, 2 * incre_R):
        dist_RM += np.sqrt(np.power(RS[0,i] - RS[0,i+1], 2) + np.power(RS[1,i] - RS[1,i+1], 2))
    dist_RB = 0 ##left bottom
    for i in range(2 * incre_R, RS.shape[1] - 1):
        dist_RB += np.sqrt(np.power(RS[0,i] - RS[0,i+1], 2) + np.power(RS[1,i] - RS[1,i+1], 2))
    
    return [dist_AP, dist_LT, dist_LM, dist_LB, dist_RT, dist_RM, dist_RB]
        
