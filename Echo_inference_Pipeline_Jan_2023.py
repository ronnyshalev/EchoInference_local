import json
import pydicom
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" # 
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed

'''
I added the following environment variable because I had an error 
Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
    1. Downgrade the protobuf package to 3.20.x or lower.
    2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
'''
#os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python' 
import EchoCardiFunction as echofunctions
import numpy as np
import cv2

import sys
from matplotlib import pyplot as plt
class EchoInferenceEngine:
    
    def __init__(self):
        
        self.IMG_WIDTH_SEG = 224
        self.IMG_WIDTH_EF = 128
        self.IMG_WIDTH_SPK = 224 ##Nov 1st, 2022
        
        self.view_dict = {0:'A2C', 
                          1:'A3C',
                          2:'A4C',
                          3:'A5C',
                          4:'PLAX',
                          5:'PSAXM',
                          6:'PSAXA',
                          7:'PSAXPM',
                          8:'PSAXAP',
                          9:'Parasternal Short-Axis Mid-papilary', 
                          10:'Right Ventricular Infolow', 
                          11:'Right Ventricular Outflow',
                          12:'Subcostal Inferior Vena Cava Long-Axis', 
                          13:'Subcostal 4-Chamber', 
                          14:'Color Doppler SSN',
                          15:'Color Doppler AO Valve', 
                          16:'Color Doppler MV', 
                          17:'Color Doppler RV Inflow', 
                          18:'Color Doppler RV Outflow', 
                          19:'Color Doppler AO Valve-Chamber', 
                          20:'Color Doppler MV-Chamber', 
                          21:'Color Doppler RV Inflow-Chamber', 
                          22:'M-Mode', 
                          23:'Unknown'}
    #####################################################################################
    
    
    def ROI_video(self, pixel_array, cs):
        """
        Get the ROI of the video after running the ROI model###########
        """
        array_shape = pixel_array.shape
           
        if len(array_shape) == 4:
            x = 0
            y = 0
            w = 0
            h = 0
            array_shape = pixel_array.shape
            validate_frame = min(10, array_shape[0])
            for i in range(validate_frame):
                if cs == "YBR_FULL_422" or cs == 'YBR_FULL': ###GLS OCT 2022
                    img_array = cv2.cvtColor(pixel_array[i], cv2.COLOR_YCrCb2RGB) ######Modified Jan 2022
                elif cs == "RGB":
                    img_array = cv2.cvtColor(pixel_array[i], cv2.COLOR_BGR2RGB) ######Modified Jan 2022
                img_array = img_array.astype(np.uint8)
                xx, yy, ww, hh = echofunctions.ROI_frame(img_array)
                x += xx
                y += yy
                w += ww
                h += hh
            return [int(x // validate_frame), int(y // validate_frame), int(w //validate_frame), int(h //validate_frame)]
     
        elif len(array_shape) == 3:  #########new change
            if cs == "YBR_FULL_422" or cs == 'YBR_FULL': ###GLS OCT 2022
                img_array = cv2.cvtColor(pixel_array, cv2.COLOR_YCrCb2RGB) ######Modified Jan 2022
            elif cs == "RGB":
                img_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)         ######Modified Jan 2022
            img_array = img_array.astype(np.uint8)
            x, y, w, h = echofunctions.ROI_frame(img_array)
            return x, y, w, h
        else:
            return [0, 0, 255, 255]
        
        
    def classify_video(self, pixel_array, cs):  ###Modified March 2022
        """
        Get the view classification of the instance, check 10 frames if it is a video instance
        """
        cls_result_list = []
        view_dict = {}
        array_shape = pixel_array.shape
        validate_frame = min(30, array_shape[0])
        for i in range(validate_frame):
            if cs == "YBR_FULL_422" or cs == "YBR_FULL":###GLS OCT 2022
                img_array = cv2.cvtColor(pixel_array[i], cv2.COLOR_YCrCb2RGB) ######Modified Jan 2022
            elif cs == "RGB":
                img_array = cv2.cvtColor(pixel_array[i], cv2.COLOR_BGR2RGB)

            cls_result_frame = echofunctions.cls_frame(img_array)
            cls_result_list.append(cls_result_frame)
        unique_view = list(np.unique(np.array(cls_result_list)))
        if len(unique_view) == 1:
            print("Image view:" + str(unique_view[0]))
            return int(unique_view[0])
        else:
            view_frame_count = 0
            for view in unique_view:
                view_dict[view] = cls_result_list.count(view)
            print(view_dict)
            for key in view_dict.keys():
                if view_dict[key] >= view_frame_count:
                    view_frame_count = view_dict[key]
                    cls_result = key
            print("Image view:" + str(cls_result))
            return int(cls_result)
    
    def classify_image(self, pixel_array, cs):   ###Modified Oct 2021
        if cs == "YBR_FULL_422" or cs == "YBR_FULL": ###GLS OCT 2022
            img_array = cv2.cvtColor(pixel_array, cv2.COLOR_YCrCb2RGB) ######Modified Jan 2022
        elif cs == "RGB":
            img_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)
        img_array = img_array.astype(np.uint8)
        cls_result = echofunctions.cls_frame(img_array)
        print("Image view:" + str(cls_result))
        return int(cls_result)
    
    def inference_instance(self, pixel_array, cs):  ###Modified Oct 2021
        """
        Get the view result of an instance
        """
        array_shape = pixel_array.shape
        # print(len(array_shape))
        if len(array_shape) == 4:
            view_instance = self.classify_video(pixel_array, cs)
        elif len(array_shape) == 3:   ############new change
            view_instance = self.classify_image(pixel_array, cs)
        else:
            view_instance = 11
        return view_instance
    
    def save_segmentation_results(self, image, mask, result_path, image_ind):
        '''
        Save segmentation results to the instance result folder
        '''
        mask_resize = cv2.resize(mask.astype('uint8'), (image.shape[1], image.shape[0]))
        # mask_overlay = cv2.addWeighted(image, 0.5, cv2.cvtColor(mask_resize.astype(np.uint8), cv2.COLOR_GRAY2BGR), 0.5, 0)
        image_path = os.path.join(result_path, 'image')
        mask_path = os.path.join(result_path, 'mask')
        mask_overlay_path = os.path.join(result_path, 'mask_overlay')
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        if not os.path.exists(mask_overlay_path):
            os.makedirs(mask_overlay_path)    
        cv2.imwrite(os.path.join(image_path, str(image_ind) + '.png'), image)
        cv2.imwrite(os.path.join(mask_path, str(image_ind) + '.png'), mask_resize)
        # cv2.imwrite(os.path.join(mask_overlay_path, str(image_ind) + '.png'), mask_overlay)
        ###################################################################plt.imshow(image)
        ##################################################################plt.imshow(mask_resize, alpha = 0.6)
        plt.axis('off')
        plt.savefig(os.path.join(mask_overlay_path, str(image_ind) + '.png'))
        
        
    
    def run_inference_on_instance(self, ori_pixel_array, pixel_array, fps, select_ROI, ori_instance_shape, color_space, BSA, V_spacing, real_V_spacing, cls_result, result_path):  ###Modified Oct 2021, the function name is changed
        """
        If the instance is A4C or A2C, run the inference pipeline.
        """

        instance_shape = pixel_array.shape 
        alpha = self.IMG_WIDTH_SEG / pixel_array.shape[1]
        beta = self.IMG_WIDTH_SEG / pixel_array.shape[2]
        voxel_ratio = 1 / (alpha * beta * np.sqrt(np.power(alpha, 2) + np.power(beta, 2)))
        print("Voxel Ratio is" + str(voxel_ratio))             ##########real spacing and the image spacing after resize are different, need to get the ratio
        
        ES_list = []
        ED_list = []
        EF_list = []
        ED_vol_list = []
        ES_vol_list = []  ### Jan 2022
        # EF_list_from_model = []
        # EF_list_all_instances = []
        find_ED = True
        find_ES = 2
        ES_list_for_EF = ["Unknown"] #ES frame
        ED_list_for_EF = ["Unknown"] #ED frame
        ES_value_list_for_EF = ["Unknown"] #Volume of ES frame
        ED_value_list_for_EF = ["Unknown"] #Volume of ED frame
        EF_index = 1
        ES_index = 1
        ES_next_index = 2
        LowEF_list = ["Unknown"]
        HR_list = []
        HR_list_all_instances = []
        Arrhythmia_list = ["Unknown"]
        start_GLS_plot = False
        GLS_count = 0
        GLS_beat = []
        GLS_beat_all = []
        GLS_Min_list = []
        GLS_model_info_text = "Global Longitudinal Strain: "
        ASLS_beat = []  ###Modified Oct 2021
        MSLS_beat = []
        BSLS_beat = []
        ALLS_beat = []
        MLLS_beat = []
        BLLS_beat = []
        APLS_beat = []  ###GLS OCT 2022
        
        ASLS_beat_Min_list = []
        MSLS_beat_Min_list = []
        BSLS_beat_Min_list = []
        ALLS_beat_Min_list = []
        MLLS_beat_Min_list = []
        BLLS_beat_Min_list = []

        AILS_beat_Min_list = []       ###modified Jan 2022
        MILS_beat_Min_list = []
        BILS_beat_Min_list = []
        AALS_beat_Min_list = []
        MALS_beat_Min_list = []
        BALS_beat_Min_list = []
        
        APLS_A2C_beat_Min_list = []     ###GLS OCT 2022
        APLS_A4C_beat_Min_list = []     ###GLS OCT 2022
        regional_beat_min_list = []
        
        
        print("FPS of this instance is: " + str(fps))
        for j in range(instance_shape[0]):
            FR_result = {}
            
            frame_array_ori = pixel_array[j]
            frame_array_ctl_pts = ori_pixel_array[j]
            if color_space == "YBR_FULL_422" or color_space == 'YBR_FULL': ###GLS OCT 2022
                frame_array_ori = cv2.cvtColor(frame_array_ori, cv2.COLOR_YCrCb2RGB)
                frame_array_ctl_pts = cv2.cvtColor(frame_array_ctl_pts, cv2.COLOR_YCrCb2RGB)
            
            x1, y1, x2, y2, x3, y3 = echofunctions.get_init_from_model(frame_array_ori)   ##Nov 1st, 2022

            frame_array= cv2.resize(frame_array_ori, (self.IMG_WIDTH_SEG, self.IMG_WIDTH_SEG))
            cv2.normalize(frame_array, frame_array, 0, 255, cv2.NORM_MINMAX)
            # frame_gray = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
            if cls_result == 2:
                seg_mask, seg_mask_bin, seg_mask_proba = echofunctions.get_mask(np.expand_dims(frame_array, 0))
            elif cls_result == 0 or cls_result == 1: ###Speckle 2022
                seg_mask, seg_mask_bin, seg_mask_proba = echofunctions.get_mask_A2C(np.expand_dims(frame_array, 0))
            
            #######Save Segmentation Results#############
            self.save_segmentation_results(frame_array_ori, seg_mask_bin[0], result_path, j)
            #############################################
            
            try:
                LV_area1, LV_vol1, new_mask1, top_ap1, bottom_mid1, inter_left1, inter_right1, L_contour, R_contour, L_R_contour, \
                    inter_leftMC, inter_rightMC, top_MC, mid_leftMC, mid_rightMC, regional_strains, contour_LA = echofunctions.get_LV_area_and_volume_with_strain(seg_mask_bin[0], 32, 16, 64, 48)
            except Exception as e:
                print(e)
            
            LV_vol1 = LV_vol1 * V_spacing * voxel_ratio  # in ml
            LV_area1 = LV_area1 * np.power(V_spacing * voxel_ratio, 2/3) # in cm**2
            
            Control_points_dict = {}
            LV_points_dict = {}
            MC_points_dict = {}
            LA_points_dict = {}         #Sep 2021
            Main_points_dict = {}
            
            
            # contour_mask = np.zeros(new_mask1.shape)  ###Modified Oct 2021
            for i in range(len(inter_left1)):
                LV_points_dict["LV_l_" + str(i)] = {"x": int(inter_left1[i][0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y": int(inter_left1[i][1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            for i in range(len(inter_right1)):
                LV_points_dict["LV_r_" + str(i)] = {"x": int(inter_right1[i][0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y": int(inter_right1[i][1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            for i in range(len(inter_leftMC)):
                MC_points_dict["MC_l_" + str(i)] = {"x": int(inter_leftMC[i][0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y": int(inter_leftMC[i][1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            for i in range(len(inter_rightMC)):
                MC_points_dict["MC_r_" + str(i)] = {"x": int(inter_rightMC[i][0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y": int(inter_rightMC[i][1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            
  
            for i in range(len(contour_LA)):   ###Modified Oct 2021
                LA_points_dict["LA_" + str(i)] = {"x": int(contour_LA[i][0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), \
                    "y": int(contour_LA[i][1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}    #Sep 2021

             ###Modified Oct 2021
            Main_points_dict["LV_AP"] = {"x":int(top_ap1[0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y":int(top_ap1[1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            Main_points_dict["LV_Bottom_Mid"] = {"x":int(bottom_mid1[0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y":int(bottom_mid1[1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            Main_points_dict["MC_AP"] = {"x": int(top_MC[0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]),"y": int(top_MC[1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            Main_points_dict["Central"] = {"x": int((top_ap1[0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]+ bottom_mid1[0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]) / 2)\
                , "y": int((top_ap1[1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1] + bottom_mid1[1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1]) / 2)}
            
            Control_points_dict["LV"] = LV_points_dict
            Control_points_dict["MC"] = MC_points_dict
            Control_points_dict["Major_Points"] = Main_points_dict
            Control_points_dict["LA"] = LA_points_dict  #Sep 2021
            
            
            #########Save Control Points Results#########################
            for key in LV_points_dict.keys():
                cv2.circle(frame_array_ctl_pts, (int(LV_points_dict[key]['x']), int(LV_points_dict[key]['y'])), 3, (0, 255, 255), -1)
            for key in MC_points_dict.keys():
                cv2.circle(frame_array_ctl_pts, (int(MC_points_dict[key]['x']), int(MC_points_dict[key]['y'])), 3, (255, 0, 255), -1)
            for key in LA_points_dict.keys():
                cv2.circle(frame_array_ctl_pts, (int(LA_points_dict[key]['x']), int(LA_points_dict[key]['y'])), 3, (0, 0, 255), -1)
            for key in Main_points_dict.keys():
                cv2.circle(frame_array_ctl_pts, (int(Main_points_dict[key]['x']), int(Main_points_dict[key]['y'])), 3, (255, 0, 0), -1)  
            control_points_path = os.path.join(result_path, 'Control_points_overlay')
            if not os.path.exists(control_points_path):
                os.makedirs(control_points_path)    
            cv2.imwrite(os.path.join(control_points_path, str(j) + '.png'), frame_array_ctl_pts)
            #############################################################
                           
            ##Speckle 2022
            """
            Speckle 2022
            """
            num_of_pts = 5   ##Nov 11 2022
            Speckle_points_dict = {} 
            L_LV_Speckle_dict = {}
            R_LV_Speckle_dict = {}
            L_MC_Speckle_dict = {}
            R_MC_Speckle_dict = {}
            AP_LV_Speckle_dict = {}
            AP_MC_Speckle_dict = {}
            

            sk_leftMC = inter_leftMC[-20::2, :]##Nov 11 2022
            sk_rightMC = inter_rightMC[-20::2, :]##Nov 11 2022

            
            sk_leftLV = inter_left1[-20::2, :]##Nov 11 2022
            sk_rightLV = inter_right1[-20::2, :]##Nov 11 2022
            
            sk_leftLV = echofunctions.get_modified_points(sk_leftLV, [x2, y2], num_of_pts) ##Nov 1st, 2022
            sk_rightLV = echofunctions.get_modified_points(sk_rightLV, [x3, y3], num_of_pts) ##Nov 1st, 2022
            sk_leftMC = echofunctions.get_modified_points(sk_leftMC, [x2, y2], num_of_pts) ##Nov 1st, 2022
            sk_rightMC = echofunctions.get_modified_points(sk_rightMC, [x3, y3], num_of_pts) ##Nov 1st, 2022
            
            
            for i in range(sk_leftLV.shape[0]):
                L_LV_Speckle_dict["LV_l_" + str(i)] = {"x": int(sk_leftLV[i, 0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y": int(sk_leftLV[i, 1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
                R_LV_Speckle_dict["LV_r_" + str(i)] = {"x": int(sk_rightLV[i, 0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y": int(sk_rightLV[i, 1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
                L_MC_Speckle_dict["MC_l_" + str(i)] = {"x": int(sk_leftMC[i, 0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y": int(sk_leftMC[i, 1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
                R_MC_Speckle_dict["MC_r_" + str(i)] = {"x": int(sk_rightMC[i, 0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y": int(sk_rightMC[i, 1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            AP_LV_Speckle_dict["LV_AP"] = {"x":int(top_ap1[0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]), "y":int(top_ap1[1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            AP_MC_Speckle_dict["MC_AP"] = {"x": int(top_MC[0] * instance_shape[2] / self.IMG_WIDTH_SEG + select_ROI[0]),"y": int(top_MC[1] * instance_shape[1] / self.IMG_WIDTH_SEG + select_ROI[1])}
            
            Speckle_points_dict["LV_left"] = L_LV_Speckle_dict 
            Speckle_points_dict["LV_right"] = R_LV_Speckle_dict   
            Speckle_points_dict["MC_left"] = L_MC_Speckle_dict 
            Speckle_points_dict["MC_right"] = R_MC_Speckle_dict   
            Speckle_points_dict["LV_AP"] = AP_LV_Speckle_dict
            Speckle_points_dict["MC_AP"] = AP_MC_Speckle_dict
            """
            Speckle 2022 End
            """
            
            GLS = sum(regional_strains)
            # print(regional_strains)

            if j == 0:
                area_list = np.array([[0], [LV_area1]])
                vol_list = np.array([[0],[LV_vol1]])
                GLS_list = np.array([[0],[GLS]])
                ASLS_list = np.array([[0],[regional_strains[0]]])  ###Modified Oct 2021
                MSLS_list = np.array([[0],[regional_strains[1]]])
                BSLS_list = np.array([[0],[regional_strains[2]]])
                ALLS_list = np.array([[0],[regional_strains[3]]])
                MLLS_list = np.array([[0],[regional_strains[4]]])
                BLLS_list = np.array([[0],[regional_strains[5]]]) 
                APLS_list = np.array([[0],[regional_strains[6]]]) ###GLS OCT 2022
                
            else:
                vol_list = np.append(vol_list, [[j], [LV_vol1]], axis = 1)
                area_list = np.append(area_list, [[j], [LV_area1]], axis = 1)
                GLS_list = np.append(GLS_list, [[j], [GLS]], axis = 1)
                ASLS_list = np.append(ASLS_list, [[j], [regional_strains[0]]], axis = 1)###Modified Oct 2021
                MSLS_list = np.append(MSLS_list, [[j], [regional_strains[1]]], axis = 1)
                BSLS_list = np.append(BSLS_list, [[j], [regional_strains[2]]], axis = 1)
                ALLS_list = np.append(ALLS_list, [[j], [regional_strains[3]]], axis = 1)
                MLLS_list = np.append(MLLS_list, [[j], [regional_strains[4]]], axis = 1)
                BLLS_list = np.append(BLLS_list, [[j], [regional_strains[5]]], axis = 1)
                APLS_list = np.append(APLS_list, [[j], [regional_strains[6]]], axis = 1)  ###GLS OCT 2022
            
            
            win_width = int(fps // 2 + 1)
            
            ES_list, ED_list, find_ED, find_ES = echofunctions.get_ED_ES_list(ES_list, ED_list, vol_list[:][1], win_width, find_ED, find_ES)  
            if len(ES_list) == EF_index and len(ED_list) == EF_index:
                if ES_list_for_EF == ["Unknown"]:
                    ES_list_for_EF = []
                ES_list_for_EF.append(str(ES_list[-1][0]))
                if ED_list_for_EF == ["Unknown"]:
                    ED_list_for_EF = []
                ED_list_for_EF.append(str(ED_list[-1][0]))
                if ES_value_list_for_EF == ["Unknown"]:
                    ES_value_list_for_EF = []
                ES_value_list_for_EF.append(ES_list[-1][1])
                if ED_value_list_for_EF == ["Unknown"]:
                    ED_value_list_for_EF = []
                ED_value_list_for_EF.append(ED_list[-1][1])
        
                ##########################use LV volume###################
                if ED_list[-1][1] != 0:   #Dec 2021
                    # EF = round((ED_list[-1][1] - ES_list[-1][1])  * 100 / ED_list[-1][1], 1)
                    # EF_list.append(EF)
                    ED_vol_list.append(round(ED_list[-1][1], 2))
                    ES_vol_list.append(round(ES_list[-1][1], 2))
                else:
                    # EF_list.append(0)
                    ED_vol_list.append(round(ED_list[-1][1], 2))
                    ES_vol_list.append(round(ES_list[-1][1], 2))
                
                ED_frame = cv2.cvtColor(pixel_array[ED_list[-1][0]], cv2.COLOR_YCrCb2RGB)
                ES_frame = cv2.cvtColor(pixel_array[ES_list[-1][0]], cv2.COLOR_YCrCb2RGB)

                ED_frame = cv2.cvtColor(ED_frame, cv2.COLOR_BGR2GRAY)
                ES_frame = cv2.cvtColor(ES_frame, cv2.COLOR_BGR2GRAY)
                # print(self.ED_list[-1][0], self.ES_list[-1][0])
                mid_frame = cv2.cvtColor(pixel_array[(ED_list[-1][0] + ES_list[-1][0]) // 2], cv2.COLOR_YCrCb2RGB)
                mid_frame = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
                
                EF_from_model = round(float(echofunctions.pred_EF(ED_frame, ES_frame,mid_frame)[0]), 1)
                EF_list.append(EF_from_model)
                # EF_list_all_instances.append(EF_from_model)
                # print(EF_list_from_model)
        
        
                if EF_from_model < 55:
                    if LowEF_list == ["Unknown"]:
                        LowEF_list = []
                    LowEF_list.append("EF is too low.")
                else:
                    if LowEF_list == ["Unknown"]:
                        LowEF_list = []
                    LowEF_list.append("EF is normal.")
                EF_index += 1  
                        
                if len(ED_list) >= 2 or len(ES_list) >= 2:   #######March 2022
                    if Arrhythmia_list == ["Unknown"]:
                        Arrhythmia_list = []
                    if len(ED_list) >= 2 and ED_list[-1][0] != ED_list[-2][0]:
                        HBR = int(fps / (ED_list[-1][0] - ED_list[-2][0]) * 60)
                    elif len(ES_list) >= 2 and ES_list[-1][0] != ES_list[-2][0]:
                        HBR = int(fps / (ES_list[-1][0] - ES_list[-2][0]) * 60)
                    else:
                        HBR = 0
                    print("HR:" + str(HBR))
                    if HBR not in HR_list:
                        HR_list.append(HBR)
                        HR_list_all_instances.append(HBR)
                        # print("HBR: " + str(HBR))
                        if HBR> 100:
                            Arrhythmia_list.append("HR is too fast.")
                            print("Diagnosis: " + "HR is too fast.")
                        elif HBR < 60:
                            Arrhythmia_list.append("HR is too low.")
                            print("Diagnosis: " + "HR is too slow.")
                        elif HBR > 60 and HBR < 100:
                            Arrhythmia_list.append("HR is normal.")
                            print("Diagnosis: " + "Normal.")
                
                    ##############################Strain Calculation################################################
            if len(ES_list) == 1:
                start_GLS_plot = True
            
            if len(ES_list) == ES_index and start_GLS_plot == True:
        
                cur_ES = GLS_list[1][ES_list[ES_index - 1][0]]
                if cur_ES != 0:
                    GLS_beat.append((cur_ES - GLS) / cur_ES)
                    print("cur_ES:" + str(cur_ES))
                    FR_result["GLS"] = GLS_beat[-1]
                
                cur_ES_AS = ASLS_list[1][ES_list[ES_index - 1][0]]  ###Modified Oct 2021
                if cur_ES_AS != 0:
                    ASLS_beat.append((cur_ES_AS - regional_strains[0]) / cur_ES_AS)
                cur_ES_MS = MSLS_list[1][ES_list[ES_index - 1][0]]
                if cur_ES_MS != 0:
                    MSLS_beat.append((cur_ES_MS - regional_strains[1]) / cur_ES_MS)
                cur_ES_BS = BSLS_list[1][ES_list[ES_index - 1][0]]
                if cur_ES_BS != 0:
                    BSLS_beat.append((cur_ES_BS - regional_strains[2]) / cur_ES_BS)
                cur_ES_AL = ALLS_list[1][ES_list[ES_index - 1][0]]
                if cur_ES_AL != 0:
                    ALLS_beat.append((cur_ES_AL - regional_strains[3]) / cur_ES_AL)
                cur_ES_ML = MLLS_list[1][ES_list[ES_index - 1][0]]
                if cur_ES_ML != 0:
                    MLLS_beat.append((cur_ES_ML - regional_strains[4]) / cur_ES_ML)
                cur_ES_BL = BLLS_list[1][ES_list[ES_index - 1][0]]
                if cur_ES_BL != 0:
                    BLLS_beat.append((cur_ES_BL - regional_strains[5]) / cur_ES_BL)
                cur_ES_AP = APLS_list[1][ES_list[ES_index - 1][0]]  ###GLS OCT 2022
                if cur_ES_AP != 0:  ###GLS OCT 2022
                    APLS_beat.append((cur_ES_AP - regional_strains[6]) / cur_ES_AP)  ###GLS OCT 2022
                
            if  len(ES_list) == ES_next_index and start_GLS_plot == True:
                if GLS_beat != []:
                    min_strain = round(np.min(np.array(GLS_beat)) * 100, 2)
                    GLS_Min_list.append(min_strain)
                if ASLS_beat != []:
                    min_strain_AS = round(np.min(np.array(ASLS_beat)) * 100, 2)  ###Modified Oct 2021
                    if cls_result == 2:
                        ASLS_beat_Min_list.append(min_strain_AS)
                    elif cls_result == 0:
                        AILS_beat_Min_list.append(min_strain_AS)
                if MSLS_beat != []:
                    min_strain_MS = round(np.min(np.array(MSLS_beat)) * 100, 2)
                    if cls_result == 2:
                        MSLS_beat_Min_list.append(min_strain_MS)
                    elif cls_result == 0:
                        MILS_beat_Min_list.append(min_strain_MS)
                if BSLS_beat != []:
                    min_strain_BS = round(np.min(np.array(BSLS_beat)) * 100, 2)
                    if cls_result == 2:
                        BSLS_beat_Min_list.append(min_strain_BS)
                    elif cls_result == 0:
                        BILS_beat_Min_list.append(min_strain_MS)
                if ALLS_beat != []:
                    min_strain_AL = round(np.min(np.array(ALLS_beat)) * 100, 2)
                    if cls_result == 2:
                        ALLS_beat_Min_list.append(min_strain_AL)
                    elif cls_result == 0:
                        AALS_beat_Min_list.append(min_strain_AL)
                if MLLS_beat != []:
                    min_strain_ML = round(np.min(np.array(MLLS_beat)) * 100, 2)
                    if cls_result == 2:
                        MLLS_beat_Min_list.append(min_strain_ML)
                    elif cls_result == 0:
                        MALS_beat_Min_list.append(min_strain_ML)
                if BLLS_beat != []:
                    min_strain_BL = round(np.min(np.array(BLLS_beat)) * 100, 2)
                    if cls_result == 2:
                        BLLS_beat_Min_list.append(min_strain_BL)
                    elif cls_result == 0:
                        BALS_beat_Min_list.append(min_strain_BL)
                
                if APLS_beat != []: ###GLS OCT 2022
                    min_strain_AP = round(np.min(np.array(APLS_beat)) * 100, 2) ###GLS OCT 2022
                    if cls_result == 2: ###GLS OCT 2022
                        APLS_A4C_beat_Min_list.append(min_strain_AP) ###GLS OCT 2022
                    elif cls_result == 0: 
                        APLS_A2C_beat_Min_list.append(min_strain_AP) ###GLS OCT 2022
                ES_index += 1
                ES_next_index += 1
                # self.GLS_count == 0
                GLS_beat = []     ###Modified Oct 2021
                ASLS_beat = []
                MSLS_beat = []
                BSLS_beat = []
                ALLS_beat = []
                MLLS_beat = []
                BLLS_beat = []     
            regional_beat_min_list = [ASLS_beat_Min_list, MSLS_beat_Min_list, BSLS_beat_Min_list\
                , ALLS_beat_Min_list, MLLS_beat_Min_list, BLLS_beat_Min_list\
                , AILS_beat_Min_list, MILS_beat_Min_list, BILS_beat_Min_list\
                , AALS_beat_Min_list, MALS_beat_Min_list, BALS_beat_Min_list, APLS_A2C_beat_Min_list, APLS_A4C_beat_Min_list]  ###GLS OCT 2022
            
            FR_result["LV_AREA"] = LV_area1
            FR_result["LV_VOL"] = LV_vol1
            FR_result["LV_VOL_BSA"] = round(LV_vol1 / BSA, 3)
            if real_V_spacing:
                FR_result["Real_Spacing"] = "yes"
            else:
                FR_result["Real_Spacing"] = "no"
            FR_result["GLL"] = GLS
            FR_result["EF_list_from_volume"] = str(EF_list)               #comment for test
            # FR_result["EF_list_from_model"] = str(EF_list_from_model)
            FR_result["Low_EF"] = str(LowEF_list[-1])
            FR_result["Heart_Rate"] = str(HR_list)
            FR_result["Arrhythmia"] = str(Arrhythmia_list[-1])
            FR_result["ES_list"] = str(ES_list_for_EF)
            FR_result["ED_list"] = str(ED_list_for_EF)
            FR_result["ES_value"] = str(ES_value_list_for_EF)
            FR_result["ED_Value"] = str(ED_value_list_for_EF)
            FR_result["ASLSMIN"] = str(regional_beat_min_list[0])     ###Modified Oct 2021
            FR_result["MSLSMIN"] = str(regional_beat_min_list[1])
            FR_result["BSLSMIN"] = str(regional_beat_min_list[2])
            FR_result["ALLSMIN"] = str(regional_beat_min_list[3])
            FR_result["MLLSMIN"] = str(regional_beat_min_list[4])
            FR_result["BLLSMIN"] = str(regional_beat_min_list[5])
            FR_result["AILSMIN"] = str(regional_beat_min_list[6])     ###Modified Jan 2021
            FR_result["MILSMIN"] = str(regional_beat_min_list[7])
            FR_result["BILSMIN"] = str(regional_beat_min_list[8])
            FR_result["AALSMIN"] = str(regional_beat_min_list[9])
            FR_result["MALSMIN"] = str(regional_beat_min_list[10])
            FR_result["BALSMIN"] = str(regional_beat_min_list[11])
            FR_result["APLSMIN_A2C"] = str(regional_beat_min_list[12]) ###GLS OCT 2022
            FR_result["APLSMIN_A4C"] = str(regional_beat_min_list[13]) ###GLS OCT 2022
            FR_result["LV_Mask"] = 32
            FR_result["LV_MC_MID"] = 16
            FR_result["MC_Mask"] = 64
            FR_result["LA_Mask"] = 48
            
            FR_result_DB = {"instanceResult": FR_result, "Control_Points": Control_points_dict, "Speckle_Init_Points": Speckle_points_dict}  ##Speckle 2022

            FR_result_DB = json.dumps(FR_result_DB)
            
            result_dict = {}
            result_dict["HR_list"] = HR_list
            result_dict["EF_list"] = EF_list
            result_dict["ED_vol_list"] = ED_vol_list
            result_dict["ES_vol_list"] = ES_vol_list
            result_dict["GLS_Min_list"] = GLS_Min_list
            result_dict["regional_beat_min_list"] = regional_beat_min_list
            result_dict["ED_list_for_EF"] = ED_list_for_EF
            result_dict["ES_list_for_EF"] = ES_list_for_EF
            
            
        return result_dict

    def get_instance_result(self, HR_list, EF_list, ED_vol_list, ES_vol_list, GLS_Min_list, regional_beat_min_list, ED_frame_list, ES_frame_list):  ###Modified Oct 2021
        instance_result = {}
        if HR_list != []:
            instance_result["HR List"] = HR_list
            instance_result["HR MIN"] = str(int(np.min(np.array(HR_list))))
            instance_result["HR MAX"] = str(int(np.max(np.array(HR_list))))
            HBR = int(np.mean(np.array(HR_list)))
            instance_result["HR Mean"] = str(HBR)
            instance_result["HR STD"] = str(int(np.std(np.array(HR_list))))
            
            if HBR> 100:
                instance_result["Arrhythmia"] = "HR is too fast."
            elif HBR < 60:
                instance_result["Arrhythmia"] = "HR is too slow."
            elif HBR > 60 and HBR < 100:
                instance_result["Arrhythmia"] = "HR is normal."
        else:
            instance_result["HR List"] = []
            instance_result["HR MIN"] = "Unknown"
            instance_result["HR MAX"] = "Unknown"
            instance_result["HR Mean"] = "Unknown"
            instance_result["HR STD"] = "Unknown"
            instance_result["Arrhythmia"] = "Unknown"

        if EF_list != []:
            instance_result["EF List"] = EF_list
            instance_result["EF_vol MIN"] = str(round(np.min(np.array(EF_list)), 3))
            instance_result["EF_vol MAX"] = str(round(np.max(np.array(EF_list)), 3))
            instance_result["EF_vol Mean"] = str(round(np.mean(np.array(EF_list)), 3))
            instance_result["EF_vol STD"] = str(round(np.std(np.array(EF_list)), 3))
            
            instance_result["EF analysis"] = ""   #####need to modify
        else:
            instance_result["EF List"] = []
            instance_result["EF_vol MIN"] = "Unknown"
            instance_result["EF_vol MAX"] = "Unknown"
            instance_result["EF_vol Mean"] = "Unknown"
            instance_result["EF_vol STD"] = "Unknown"
        if ED_frame_list != []:
            instance_result["ED_frame_list"] = ED_frame_list
        else:
            instance_result["ED_frame_list"] = "Unknown"
        
        if ES_frame_list != []:
            instance_result["ES_frame_list"] = ES_frame_list
        else:
            instance_result["ES_frame_list"] = "Unknown"
        
        if ED_vol_list != []:
            instance_result["ED_Vol_List"] = ED_vol_list
            instance_result["ED_Vol Mean"] = str(round(np.max(np.array(ED_vol_list)), 3))
            
        else:
            instance_result["ED_Vol_List"] = "Unknown"
            instance_result["ED_Vol Mean"] = "Unknown"
        
        if ES_vol_list != []:
            instance_result["ES_Vol_List"] = ES_vol_list
            instance_result["ES_Vol Mean"] = str(round(np.max(np.array(ES_vol_list)), 3))
            
        else:
            instance_result["ES_Vol_List"] = "Unknown"
            instance_result["ES_Vol Mean"] = "Unknown"

        if GLS_Min_list != []:
            instance_result["GLS List"] = GLS_Min_list
            instance_result["GLS MIN"] = str(round(np.min(np.array(GLS_Min_list)), 3))
            instance_result["GLS MAX"] = str(round(np.max(np.array(GLS_Min_list)), 3))
            instance_result["GLS Mean"] = str(round(np.mean(np.array(GLS_Min_list)), 3))
            instance_result["GLS STD"] = str(round(np.std(np.array(GLS_Min_list)), 3))
        else:
            instance_result["GLS List"] = []
            instance_result["GLS MIN"] = "Unknown"
            instance_result["GLS MAX"] = "Unknown"
            instance_result["GLS Mean"] = "Unknown"
            instance_result["GLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[0] != []:   ###GLS OCT 2022
            instance_result["ASLS List"] = regional_beat_min_list[0]
            instance_result["ASLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[0])), 3))
            instance_result["ASLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[0])), 3))
            instance_result["ASLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[0])), 3))
            instance_result["ASLS STD"] = str(round(np.std(np.array(regional_beat_min_list[0])), 3))
        else:
            instance_result["ASLS List"] = []
            instance_result["ASLS MIN"] = "Unknown"
            instance_result["ASLS MAX"] = "Unknown"
            instance_result["ASLS Mean"] = "Unknown"
            instance_result["ASLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[1] != []:###GLS OCT 2022
            instance_result["MSLS List"] = regional_beat_min_list[1]
            instance_result["MSLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[1])), 3))
            instance_result["MSLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[1])), 3))
            instance_result["MSLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[1])), 3))
            instance_result["MSLS STD"] = str(round(np.std(np.array(regional_beat_min_list[1])), 3))
        else:
            instance_result["MSLS List"] = []
            instance_result["MSLS MIN"] = "Unknown"
            instance_result["MSLS MAX"] = "Unknown"
            instance_result["MSLS Mean"] = "Unknown"
            instance_result["MSLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[2] != []:###GLS OCT 2022
            instance_result["BSLS List"] = regional_beat_min_list[2]
            instance_result["BSLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[2])), 3))
            instance_result["BSLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[2])), 3))
            instance_result["BSLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[2])), 3))
            instance_result["BSLS STD"] = str(round(np.std(np.array(regional_beat_min_list[2])), 3))
        else:
            instance_result["BSLS List"] = []
            instance_result["BSLS MIN"] = "Unknown"
            instance_result["BSLS MAX"] = "Unknown"
            instance_result["BSLS Mean"] = "Unknown"
            instance_result["BSLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[3] != []:###GLS OCT 2022
            instance_result["ALLS List"] = regional_beat_min_list[3]
            instance_result["ALLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[3])), 3))
            instance_result["ALLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[3])), 3))
            instance_result["ALLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[3])), 3))
            instance_result["ALLS STD"] = str(round(np.std(np.array(regional_beat_min_list[3])), 3))
        else:
            instance_result["ALLS List"] = []
            instance_result["ALLS MIN"] = "Unknown"
            instance_result["ALLS MAX"] = "Unknown"
            instance_result["ALLS Mean"] = "Unknown"
            instance_result["ALLS STD"] = "Unknown"
            
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[4] != []:###GLS OCT 2022
            instance_result["MLLS List"] = regional_beat_min_list[4]
            instance_result["MLLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[4])), 3))
            instance_result["MLLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[4])), 3))
            instance_result["MLLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[4])), 3))
            instance_result["MLLS STD"] = str(round(np.std(np.array(regional_beat_min_list[4])), 3))
        else:
            instance_result["MLLS List"] = []
            instance_result["MLLS MIN"] = "Unknown"
            instance_result["MLLS MAX"] = "Unknown"
            instance_result["MLLS Mean"] = "Unknown"
            instance_result["MLLS STD"] = "Unknown"
            
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[5] != []:###GLS OCT 2022
            instance_result["BLLS List"] = regional_beat_min_list[5]
            instance_result["BLLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[5])), 3))
            instance_result["BLLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[5])), 3))
            instance_result["BLLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[5])), 3))
            instance_result["BLLS STD"] = str(round(np.std(np.array(regional_beat_min_list[5])), 3))
        else:
            instance_result["BLLS List"] = []
            instance_result["BLLS MIN"] = "Unknown"
            instance_result["BLLS MAX"] = "Unknown"
            instance_result["BLLS Mean"] = "Unknown"
            instance_result["BLLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[6] != []:   ###GLS OCT 2022
            instance_result["AILS List"] = regional_beat_min_list[6]
            instance_result["AILS MIN"] = str(round(np.min(np.array(regional_beat_min_list[6])), 3))
            instance_result["AILS MAX"] = str(round(np.max(np.array(regional_beat_min_list[6])), 3))
            instance_result["AILS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[6])), 3))
            instance_result["AILS STD"] = str(round(np.std(np.array(regional_beat_min_list[6])), 3))
        else:
            instance_result["AILS List"] = []
            instance_result["AILS MIN"] = "Unknown"
            instance_result["AILS MAX"] = "Unknown"
            instance_result["AILS Mean"] = "Unknown"
            instance_result["AILS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[7] != []:###GLS OCT 2022
            instance_result["MILS List"] = regional_beat_min_list[7]
            instance_result["MILS MIN"] = str(round(np.min(np.array(regional_beat_min_list[7])), 3))
            instance_result["MILS MAX"] = str(round(np.max(np.array(regional_beat_min_list[7])), 3))
            instance_result["MILS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[7])), 3))
            instance_result["MSLS STD"] = str(round(np.std(np.array(regional_beat_min_list[7])), 3))
        else:
            instance_result["MILS List"] = []
            instance_result["MILS MIN"] = "Unknown"
            instance_result["MILS MAX"] = "Unknown"
            instance_result["MILS Mean"] = "Unknown"
            instance_result["MILS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[8] != []:###GLS OCT 2022
            instance_result["BILS List"] = regional_beat_min_list[8]
            instance_result["BILS MIN"] = str(round(np.min(np.array(regional_beat_min_list[8])), 3))
            instance_result["BILS MAX"] = str(round(np.max(np.array(regional_beat_min_list[8])), 3))
            instance_result["BILS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[8])), 3))
            instance_result["BILS STD"] = str(round(np.std(np.array(regional_beat_min_list[8])), 3))
        else:
            instance_result["BILS List"] = []
            instance_result["BILS MIN"] = "Unknown"
            instance_result["BILS MAX"] = "Unknown"
            instance_result["BILS Mean"] = "Unknown"
            instance_result["BILS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[9] != []:###GLS OCT 2022
            instance_result["AALS List"] = regional_beat_min_list[9]
            instance_result["AALS MIN"] = str(round(np.min(np.array(regional_beat_min_list[9])), 3))
            instance_result["AALS MAX"] = str(round(np.max(np.array(regional_beat_min_list[9])), 3))
            instance_result["AALS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[9])), 3))
            instance_result["AALS STD"] = str(round(np.std(np.array(regional_beat_min_list[9])), 3))
        else:
            instance_result["AALS List"] = []
            instance_result["AALS MIN"] = "Unknown"
            instance_result["AALS MAX"] = "Unknown"
            instance_result["AALS Mean"] = "Unknown"
            instance_result["AALS STD"] = "Unknown"
            
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[10] != []:###GLS OCT 2022
            instance_result["MALS List"] = regional_beat_min_list[10]
            instance_result["MALS MIN"] = str(round(np.min(np.array(regional_beat_min_list[10])), 3))
            instance_result["MALS MAX"] = str(round(np.max(np.array(regional_beat_min_list[10])), 3))
            instance_result["MALS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[10])), 3))
            instance_result["MALS STD"] = str(round(np.std(np.array(regional_beat_min_list[10])), 3))
        else:
            instance_result["MALS List"] = []
            instance_result["MALS MIN"] = "Unknown"
            instance_result["MALS MAX"] = "Unknown"
            instance_result["MALS Mean"] = "Unknown"
            instance_result["MALS STD"] = "Unknown"
            
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[11] != []:###GLS OCT 2022
            instance_result["BALS List"] = regional_beat_min_list[11]
            instance_result["BALS MIN"] = str(round(np.min(np.array(regional_beat_min_list[11])), 3))
            instance_result["BALS MAX"] = str(round(np.max(np.array(regional_beat_min_list[11])), 3))
            instance_result["BALS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[11])), 3))
            instance_result["BALS STD"] = str(round(np.std(np.array(regional_beat_min_list[11])), 3))
        else:
            instance_result["BALS List"] = []
            instance_result["BALS MIN"] = "Unknown"
            instance_result["BALS MAX"] = "Unknown"
            instance_result["BALS Mean"] = "Unknown"
            instance_result["BALS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[12] != []:###GLS OCT 2022
            instance_result["APLS_A2C List"] = regional_beat_min_list[12] ###GLS OCT 2022
            instance_result["APLS_A2C MIN"] = str(round(np.min(np.array(regional_beat_min_list[12])), 3)) ###GLS OCT 2022
            instance_result["APLS_A2C MAX"] = str(round(np.max(np.array(regional_beat_min_list[12])), 3)) ###GLS OCT 2022
            instance_result["APLS_A2C Mean"] = str(round(np.mean(np.array(regional_beat_min_list[12])), 3)) ###GLS OCT 2022
            instance_result["APLS_A2C STD"] = str(round(np.std(np.array(regional_beat_min_list[12])), 3)) ###GLS OCT 2022 
        else: ###GLS OCT 2022
            instance_result["APLS_A2C List"] = [] ###GLS OCT 2022
            instance_result["APLS_A2C MIN"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A2C MAX"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A2C Mean"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A2C STD"] = "Unknown" ###GLS OCT 2022
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[13] != []:###GLS OCT 2022
            instance_result["APLS_A4C List"] = regional_beat_min_list[13] ###GLS OCT 2022
            instance_result["APLS_A4C MIN"] = str(round(np.min(np.array(regional_beat_min_list[13])), 3)) ###GLS OCT 2022
            instance_result["APLS_A4C MAX"] = str(round(np.max(np.array(regional_beat_min_list[13])), 3)) ###GLS OCT 2022
            instance_result["APLS_A4C Mean"] = str(round(np.mean(np.array(regional_beat_min_list[13])), 3)) ###GLS OCT 2022
            instance_result["APLS_A4C STD"] = str(round(np.std(np.array(regional_beat_min_list[13])), 3)) ###GLS OCT 2022
        else: ###GLS OCT 2022
            instance_result["APLS_A4C List"] = [] ###GLS OCT 2022
            instance_result["APLS_A4C MIN"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A4C MAX"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A4C Mean"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A4C STD"] = "Unknown" ###GLS OCT 2022
        
        
        return instance_result

    def get_series_study_result(self, HR_list, EF_list, ED_vol_list, ES_vol_list, GLS_Min_list, regional_beat_min_list):  ###Modified Oct 2021
        instance_result = {}
        if HR_list != []:
            instance_result["HR List"] = HR_list
            instance_result["HR MIN"] = str(int(np.min(np.array(HR_list))))
            instance_result["HR MAX"] = str(int(np.max(np.array(HR_list))))
            HBR = int(np.mean(np.array(HR_list)))
            instance_result["HR Mean"] = str(HBR)
            instance_result["HR STD"] = str(int(np.std(np.array(HR_list))))
            
            if HBR> 100:
                instance_result["Arrhythmia"] = "HR is too fast."
            elif HBR < 60:
                instance_result["Arrhythmia"] = "HR is too slow."
            elif HBR > 60 and HBR < 100:
                instance_result["Arrhythmia"] = "HR is normal."
        else:
            instance_result["HR List"] = []
            instance_result["HR MIN"] = "Unknown"
            instance_result["HR MAX"] = "Unknown"
            instance_result["HR Mean"] = "Unknown"
            instance_result["HR STD"] = "Unknown"
            instance_result["Arrhythmia"] = "Unknown"

        if EF_list != []:
            instance_result["EF List"] = EF_list
            instance_result["EF_vol MIN"] = str(round(np.min(np.array(EF_list)), 3))
            instance_result["EF_vol MAX"] = str(round(np.max(np.array(EF_list)), 3))
            instance_result["EF_vol Mean"] = str(round(np.mean(np.array(EF_list)), 3))
            instance_result["EF_vol STD"] = str(round(np.std(np.array(EF_list)), 3))
            
            instance_result["EF analysis"] = ""   #####need to modify
        else:
            instance_result["EF List"] = []
            instance_result["EF_vol MIN"] = "Unknown"
            instance_result["EF_vol MAX"] = "Unknown"
            instance_result["EF_vol Mean"] = "Unknown"
            instance_result["EF_vol STD"] = "Unknown"
        
        if ED_vol_list != []:
            instance_result["ED_Vol_List"] = ED_vol_list
            instance_result["ED_Vol Mean"] = str(round(np.max(np.array(ED_vol_list)), 3))
            
        else:
            instance_result["ED_Vol_List"] = "Unknown"
            instance_result["ED_Vol Mean"] = "Unknown"
        
        if ES_vol_list != []:
            instance_result["ES_Vol_List"] = ES_vol_list
            instance_result["ES_Vol Mean"] = str(round(np.max(np.array(ES_vol_list)), 3))
            
        else:
            instance_result["ES_Vol_List"] = "Unknown"
            instance_result["ES_Vol Mean"] = "Unknown"

        if GLS_Min_list != []:
            instance_result["GLS List"] = GLS_Min_list
            instance_result["GLS MIN"] = str(round(np.min(np.array(GLS_Min_list)), 3))
            instance_result["GLS MAX"] = str(round(np.max(np.array(GLS_Min_list)), 3))
            instance_result["GLS Mean"] = str(round(np.mean(np.array(GLS_Min_list)), 3))
            instance_result["GLS STD"] = str(round(np.std(np.array(GLS_Min_list)), 3))
        else:
            instance_result["GLS List"] = []
            instance_result["GLS MIN"] = "Unknown"
            instance_result["GLS MAX"] = "Unknown"
            instance_result["GLS Mean"] = "Unknown"
            instance_result["GLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[0] != []:   ###GLS OCT 2022
            instance_result["ASLS List"] = regional_beat_min_list[0]
            instance_result["ASLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[0])), 3))
            instance_result["ASLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[0])), 3))
            instance_result["ASLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[0])), 3))
            instance_result["ASLS STD"] = str(round(np.std(np.array(regional_beat_min_list[0])), 3))
        else:
            instance_result["ASLS List"] = []
            instance_result["ASLS MIN"] = "Unknown"
            instance_result["ASLS MAX"] = "Unknown"
            instance_result["ASLS Mean"] = "Unknown"
            instance_result["ASLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[1] != []:###GLS OCT 2022
            instance_result["MSLS List"] = regional_beat_min_list[1]
            instance_result["MSLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[1])), 3))
            instance_result["MSLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[1])), 3))
            instance_result["MSLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[1])), 3))
            instance_result["MSLS STD"] = str(round(np.std(np.array(regional_beat_min_list[1])), 3))
        else:
            instance_result["MSLS List"] = []
            instance_result["MSLS MIN"] = "Unknown"
            instance_result["MSLS MAX"] = "Unknown"
            instance_result["MSLS Mean"] = "Unknown"
            instance_result["MSLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[2] != []:###GLS OCT 2022
            instance_result["BSLS List"] = regional_beat_min_list[2]
            instance_result["BSLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[2])), 3))
            instance_result["BSLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[2])), 3))
            instance_result["BSLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[2])), 3))
            instance_result["BSLS STD"] = str(round(np.std(np.array(regional_beat_min_list[2])), 3))
        else:
            instance_result["BSLS List"] = []
            instance_result["BSLS MIN"] = "Unknown"
            instance_result["BSLS MAX"] = "Unknown"
            instance_result["BSLS Mean"] = "Unknown"
            instance_result["BSLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[3] != []:###GLS OCT 2022
            instance_result["ALLS List"] = regional_beat_min_list[3]
            instance_result["ALLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[3])), 3))
            instance_result["ALLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[3])), 3))
            instance_result["ALLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[3])), 3))
            instance_result["ALLS STD"] = str(round(np.std(np.array(regional_beat_min_list[3])), 3))
        else:
            instance_result["ALLS List"] = []
            instance_result["ALLS MIN"] = "Unknown"
            instance_result["ALLS MAX"] = "Unknown"
            instance_result["ALLS Mean"] = "Unknown"
            instance_result["ALLS STD"] = "Unknown"
            
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[4] != []:###GLS OCT 2022
            instance_result["MLLS List"] = regional_beat_min_list[4]
            instance_result["MLLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[4])), 3))
            instance_result["MLLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[4])), 3))
            instance_result["MLLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[4])), 3))
            instance_result["MLLS STD"] = str(round(np.std(np.array(regional_beat_min_list[4])), 3))
        else:
            instance_result["MLLS List"] = []
            instance_result["MLLS MIN"] = "Unknown"
            instance_result["MLLS MAX"] = "Unknown"
            instance_result["MLLS Mean"] = "Unknown"
            instance_result["MLLS STD"] = "Unknown"
            
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[5] != []:###GLS OCT 2022
            instance_result["BLLS List"] = regional_beat_min_list[5]
            instance_result["BLLS MIN"] = str(round(np.min(np.array(regional_beat_min_list[5])), 3))
            instance_result["BLLS MAX"] = str(round(np.max(np.array(regional_beat_min_list[5])), 3))
            instance_result["BLLS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[5])), 3))
            instance_result["BLLS STD"] = str(round(np.std(np.array(regional_beat_min_list[5])), 3))
        else:
            instance_result["BLLS List"] = []
            instance_result["BLLS MIN"] = "Unknown"
            instance_result["BLLS MAX"] = "Unknown"
            instance_result["BLLS Mean"] = "Unknown"
            instance_result["BLLS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[6] != []:   ###GLS OCT 2022
            instance_result["AILS List"] = regional_beat_min_list[6]
            instance_result["AILS MIN"] = str(round(np.min(np.array(regional_beat_min_list[6])), 3))
            instance_result["AILS MAX"] = str(round(np.max(np.array(regional_beat_min_list[6])), 3))
            instance_result["AILS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[6])), 3))
            instance_result["AILS STD"] = str(round(np.std(np.array(regional_beat_min_list[6])), 3))
        else:
            instance_result["AILS List"] = []
            instance_result["AILS MIN"] = "Unknown"
            instance_result["AILS MAX"] = "Unknown"
            instance_result["AILS Mean"] = "Unknown"
            instance_result["AILS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[7] != []:###GLS OCT 2022
            instance_result["MILS List"] = regional_beat_min_list[7]
            instance_result["MILS MIN"] = str(round(np.min(np.array(regional_beat_min_list[7])), 3))
            instance_result["MILS MAX"] = str(round(np.max(np.array(regional_beat_min_list[7])), 3))
            instance_result["MILS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[7])), 3))
            instance_result["MSLS STD"] = str(round(np.std(np.array(regional_beat_min_list[7])), 3))
        else:
            instance_result["MILS List"] = []
            instance_result["MILS MIN"] = "Unknown"
            instance_result["MILS MAX"] = "Unknown"
            instance_result["MILS Mean"] = "Unknown"
            instance_result["MILS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[8] != []:###GLS OCT 2022
            instance_result["BILS List"] = regional_beat_min_list[8]
            instance_result["BILS MIN"] = str(round(np.min(np.array(regional_beat_min_list[8])), 3))
            instance_result["BILS MAX"] = str(round(np.max(np.array(regional_beat_min_list[8])), 3))
            instance_result["BILS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[8])), 3))
            instance_result["BILS STD"] = str(round(np.std(np.array(regional_beat_min_list[8])), 3))
        else:
            instance_result["BILS List"] = []
            instance_result["BILS MIN"] = "Unknown"
            instance_result["BILS MAX"] = "Unknown"
            instance_result["BILS Mean"] = "Unknown"
            instance_result["BILS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[9] != []:###GLS OCT 2022
            instance_result["AALS List"] = regional_beat_min_list[9]
            instance_result["AALS MIN"] = str(round(np.min(np.array(regional_beat_min_list[9])), 3))
            instance_result["AALS MAX"] = str(round(np.max(np.array(regional_beat_min_list[9])), 3))
            instance_result["AALS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[9])), 3))
            instance_result["AALS STD"] = str(round(np.std(np.array(regional_beat_min_list[9])), 3))
        else:
            instance_result["AALS List"] = []
            instance_result["AALS MIN"] = "Unknown"
            instance_result["AALS MAX"] = "Unknown"
            instance_result["AALS Mean"] = "Unknown"
            instance_result["AALS STD"] = "Unknown"
            
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[10] != []:###GLS OCT 2022
            instance_result["MALS List"] = regional_beat_min_list[10]
            instance_result["MALS MIN"] = str(round(np.min(np.array(regional_beat_min_list[10])), 3))
            instance_result["MALS MAX"] = str(round(np.max(np.array(regional_beat_min_list[10])), 3))
            instance_result["MALS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[10])), 3))
            instance_result["MALS STD"] = str(round(np.std(np.array(regional_beat_min_list[10])), 3))
        else:
            instance_result["MALS List"] = []
            instance_result["MALS MIN"] = "Unknown"
            instance_result["MALS MAX"] = "Unknown"
            instance_result["MALS Mean"] = "Unknown"
            instance_result["MALS STD"] = "Unknown"
            
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[11] != []:###GLS OCT 2022
            instance_result["BALS List"] = regional_beat_min_list[11]
            instance_result["BALS MIN"] = str(round(np.min(np.array(regional_beat_min_list[11])), 3))
            instance_result["BALS MAX"] = str(round(np.max(np.array(regional_beat_min_list[11])), 3))
            instance_result["BALS Mean"] = str(round(np.mean(np.array(regional_beat_min_list[11])), 3))
            instance_result["BALS STD"] = str(round(np.std(np.array(regional_beat_min_list[11])), 3))
        else:
            instance_result["BALS List"] = []
            instance_result["BALS MIN"] = "Unknown"
            instance_result["BALS MAX"] = "Unknown"
            instance_result["BALS Mean"] = "Unknown"
            instance_result["BALS STD"] = "Unknown"
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[12] != []:###GLS OCT 2022
            instance_result["APLS_A2C List"] = regional_beat_min_list[12] ###GLS OCT 2022
            instance_result["APLS_A2C MIN"] = str(round(np.min(np.array(regional_beat_min_list[12])), 3)) ###GLS OCT 2022
            instance_result["APLS_A2C MAX"] = str(round(np.max(np.array(regional_beat_min_list[12])), 3)) ###GLS OCT 2022
            instance_result["APLS_A2C Mean"] = str(round(np.mean(np.array(regional_beat_min_list[12])), 3)) ###GLS OCT 2022
            instance_result["APLS_A2C STD"] = str(round(np.std(np.array(regional_beat_min_list[12])), 3)) ###GLS OCT 2022 
        else: ###GLS OCT 2022
            instance_result["APLS_A2C List"] = [] ###GLS OCT 2022
            instance_result["APLS_A2C MIN"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A2C MAX"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A2C Mean"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A2C STD"] = "Unknown" ###GLS OCT 2022
        
        if  len(regional_beat_min_list) == 14 and regional_beat_min_list[13] != []:###GLS OCT 2022
            instance_result["APLS_A4C List"] = regional_beat_min_list[13] ###GLS OCT 2022
            instance_result["APLS_A4C MIN"] = str(round(np.min(np.array(regional_beat_min_list[13])), 3)) ###GLS OCT 2022
            instance_result["APLS_A4C MAX"] = str(round(np.max(np.array(regional_beat_min_list[13])), 3)) ###GLS OCT 2022
            instance_result["APLS_A4C Mean"] = str(round(np.mean(np.array(regional_beat_min_list[13])), 3)) ###GLS OCT 2022
            instance_result["APLS_A4C STD"] = str(round(np.std(np.array(regional_beat_min_list[13])), 3)) ###GLS OCT 2022
        else: ###GLS OCT 2022
            instance_result["APLS_A4C List"] = [] ###GLS OCT 2022
            instance_result["APLS_A4C MIN"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A4C MAX"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A4C Mean"] = "Unknown" ###GLS OCT 2022
            instance_result["APLS_A4C STD"] = "Unknown" ###GLS OCT 2022
        
        
        return instance_result
    
    def list_dicom(self, study_dir):
        dcm_list = []
        for root, dirs, files in os.walk(study_dir):
            for file in files:
                dcm_list.append(os.path.join(root, file))
        return dcm_list

    def get_bsa(self, dcm_list):
        dcm_file = dcm_list[0]
        try:
            dcm_data = pydicom.dcmread(dcm_file)
        except:
            weight = 70
            height = 1.7
        else:
            try:
                weight = dcm_data.PatientWeight 
            except:
                weight = 70
            try:
                height = dcm_data.PatientSize
            except:
                height = 1.7
        bsa = np.sqrt(height * weight * 1000 / 3600)
        return bsa

    def get_study_id(self, dcm_list):
        dcm_file = dcm_list[0]
        try:
            dcm_data = pydicom.dcmread(dcm_file)
        except:
            study_id = '1.1.1234567890'
        else:
            study_id = dcm_data.StudyInstanceUID 
        return study_id
            

    def get_series_list(self, dcm_list):
        '''
        Get all series and all instances belong to each
        return a dict {seriesid:{instanceid:instancepath}}
        '''
        series_dict = {}
        for dcm_file in dcm_list:
            try:
                dcm_data = pydicom.dcmread(dcm_file)
            except:
                continue
            else:
                series_id = dcm_data.SeriesInstanceUID
                if series_id not in series_dict.keys():
                    series_dict[series_id] = [{dcm_data.SOPInstanceUID: dcm_file}]
                else:
                    series_dict[series_id].append({dcm_data.SOPInstanceUID: dcm_file})
        return series_dict
    
    def make_study_folder(self, cur_path, study_id):
        study_path = os.path.join(cur_path, study_id)
        if not os.path.exists(study_path):
            os.makedirs(study_path)
            print('Study result folder created.')
        return study_path
    
    def make_instance_folder(self, study_folder, instance_id):
        instance_path = os.path.join(study_folder, instance_id)
        if not os.path.exists(instance_path):
            os.makedirs(instance_path)
            print('Instance result folder created.')
        return instance_path
    
            
def main():   
    verbosity = False
    
    study_folder = "./Test_data/Anonymized - 01_01_2018_14_11_52_d"      #Paste the study directory here
    #study_folder = "./Test_data/20220113084357016414246828271"      #Paste the study directory here
    #study_folder = "./Test_data/202201130828320762231393317"      #Paste the study directory here

    
    engine = EchoInferenceEngine()
    work_path = os.getcwd()
    result_path = os.path.join(work_path, 'Results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if os.path.exists(study_folder):
        print("Study Path:{}".format(study_folder))
    else:
        print('Cannot find the study folder')
        return
    
    dcm_list = engine.list_dicom(study_folder)
    bsa = engine.get_bsa(dcm_list)
    study_id = engine.get_study_id(dcm_list)

    # *************************************************************************
    # *********** Rony added the next 3 lines to ignore Pydicom warnings ******
    # *************************************************************************
     
    import warnings
    warnings.filterwarnings('ignore')
    print('The script still runs.')

    series_dict = engine.get_series_list(dcm_list)
    if verbosity:
        print(study_id, series_dict)
    
    study_result_folder = engine.make_study_folder(result_path, os.path.basename(study_folder))      

    study_HR_list = []
    study_EF_list = []
    study_ED_vol_list = []
    study_ES_vol_list = []
    study_EF_list_from_model = []
    study_GLS_Min_list = []  
    study_ASLS_Min_list = []  ###Modified Oct 2021
    study_MSLS_Min_list = []
    study_BSLS_Min_list = []
    study_ALLS_Min_list = []
    study_MLLS_Min_list = []
    study_BLLS_Min_list = []
    study_AILS_Min_list = []  ###Modified Jan2022
    study_MILS_Min_list = []
    study_BILS_Min_list = []
    study_AALS_Min_list = []
    study_MALS_Min_list = []
    study_BALS_Min_list = []
    study_AP_A2C_Min_list = [] ###GLS OCT 2022
    study_AP_A4C_Min_list = [] ###GLS OCT 2022
    study_regional_LS_Min_list = []

    for series in series_dict.keys():

        series_HR_list = []
        series_EF_list = []
        series_ED_vol_list = []
        series_ES_vol_list = []
        series_EF_list_from_model = []
        series_GLS_Min_list = []
        series_ASLS_Min_list = []  ###Modified Oct 2021
        series_MSLS_Min_list = []
        series_BSLS_Min_list = []
        series_ALLS_Min_list = []
        series_MLLS_Min_list = []
        series_BLLS_Min_list = []
        series_AILS_Min_list = []  ###Modified Jan2022
        series_MILS_Min_list = []
        series_BILS_Min_list = []
        series_AALS_Min_list = []
        series_MALS_Min_list = []
        series_BALS_Min_list = []
        series_AP_A2C_Min_list = [] ###GLS OCT 2022
        series_AP_A4C_Min_list = [] ###GLS OCT 2022
        series_regional_LS_Min_list = []
        
        instance_list= series_dict[series]
        for instance in instance_list:

            instance_id = list(instance.keys())[0]
            instance_path = instance[instance_id]
            
            
            try:
                dataset = pydicom.dcmread(instance_path)
            except:
                continue
            try:
                frame_time = float(dataset.FrameTime)
            except:
                frame_time = None
                frame_rate = 30
            else:
                frame_rate = int(round(1000 / frame_time))
            try:
                ori_pixel_array = dataset.pixel_array
                colorspace = dataset.PhotometricInterpretation
            except:
                print("Unable to load the pixel array")
                continue
    
            ###############get the ROI of the video
            ori_shape = ori_pixel_array.shape
            ROI = engine.ROI_video(ori_pixel_array, colorspace)
            print("ROI of Instance {} is {}".format(dataset.InstanceNumber, ROI))
            
            ################view classification
            try:           ##########Dec 2021
                if ROI == [0, 0, 255, 255]:
                    instance_view = 23
                elif len(ori_shape) == 4:
                    pixel_array = ori_pixel_array[:, ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2], :]
                    instance_view = engine.inference_instance(pixel_array, colorspace)   ###Modified Oct 2021
                elif len(ori_shape) == 3:      ##########new change
                    pixel_array = ori_pixel_array[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2], :]
                    instance_view = engine.inference_instance(pixel_array, colorspace)   ###Modified Oct 2021
            except:
                continue 
            
            #######Save ROI and View Classification Results############
            ROI_cls_dict = {}
            ROI_cls_dict['ROI'] = {'x': ROI[0], 'y': ROI[1], 'width': ROI[2], 'height': ROI[3]}     
            ROI_cls_dict['View Classification'] = engine.view_dict[instance_view]
            instance_result_path = engine.make_instance_folder(study_result_folder, os.path.basename(instance_path) + '-' + engine.view_dict[instance_view])
            with open(os.path.join(instance_result_path, 'ROI_and_View_result.json'), 'w') as f:
                json.dump(ROI_cls_dict, f) 
            ###########################################################
                     
            if len(ori_shape) == 4:
                
                if instance_view == 2 or instance_view == 0 or instance_view == 1:  #####new change  ###Modified Oct 2022
                    try:
                        spacing_list = list(dataset.PixelSpacing)   ##get spacing
                        real_spacing = True
                        print("Use real spacing")
                    except:
                        spacing_list = [0.4, 0.4]
                        real_spacing = False
                        print("Use fake spacing")
                    
                    spacing = (spacing_list[0] + spacing_list[1]) / 2 #mm
                    spacing_voxel = np.power((spacing / 10), 3)  #ml
                    print("Spacing: " + str(spacing_voxel))
                                  
                    try:
                        result_dict = engine.run_inference_on_instance(ori_pixel_array, pixel_array, frame_rate, ROI, ori_shape, colorspace, bsa, spacing_voxel, real_spacing, instance_view, instance_result_path)
                    except Exception as e:
                        print(e)
                        print("Bad image quality, skip this instance.")
            
                    else:
                        series_HR_list += result_dict["HR_list"]       ##Modified Feb 2022
                        series_EF_list += result_dict["EF_list"]
                        series_ED_vol_list += result_dict["ED_vol_list"]
                        series_ES_vol_list += result_dict["ES_vol_list"]
                        series_GLS_Min_list += result_dict["GLS_Min_list"]
                        series_ASLS_Min_list += result_dict["regional_beat_min_list"][0]  ###Modified Oct 2021
                        series_MSLS_Min_list += result_dict["regional_beat_min_list"][1]
                        series_BSLS_Min_list += result_dict["regional_beat_min_list"][2]
                        series_ALLS_Min_list += result_dict["regional_beat_min_list"][3]
                        series_MLLS_Min_list += result_dict["regional_beat_min_list"][4]
                        series_BLLS_Min_list += result_dict["regional_beat_min_list"][5]
                        series_AILS_Min_list += result_dict["regional_beat_min_list"][6]  ###Modified Oct 2021
                        series_MILS_Min_list += result_dict["regional_beat_min_list"][7]
                        series_BILS_Min_list += result_dict["regional_beat_min_list"][8]
                        series_AALS_Min_list += result_dict["regional_beat_min_list"][9]
                        series_MALS_Min_list += result_dict["regional_beat_min_list"][10]
                        series_BALS_Min_list += result_dict["regional_beat_min_list"][11]
                        series_AP_A2C_Min_list += result_dict["regional_beat_min_list"][12] ###GLS OCT 2022
                        series_AP_A4C_Min_list += result_dict["regional_beat_min_list"][13] ###GLS OCT 2022
                        series_regional_LS_Min_list = [series_ASLS_Min_list, series_MSLS_Min_list, series_BSLS_Min_list, \
                            series_ALLS_Min_list, series_MLLS_Min_list, series_BLLS_Min_list, \
                                series_AILS_Min_list, series_MILS_Min_list, series_BILS_Min_list, \
                            series_AALS_Min_list, series_MALS_Min_list, series_BALS_Min_list, series_AP_A2C_Min_list, series_AP_A4C_Min_list] ###GLS OCT 2022
                        instance_result = engine.get_instance_result(result_dict["HR_list"], result_dict["EF_list"], result_dict["ED_vol_list"],\
                            result_dict["ES_vol_list"], result_dict["GLS_Min_list"], result_dict["regional_beat_min_list"], \
                                result_dict["ED_list_for_EF"], result_dict["ES_list_for_EF"])  ##Modified Feb 2022
                        ##########Write Study Result Json#######################
                        instance_result_dict = {}
                        instance_result_dict["instanceResult"] = instance_result
                        with open(os.path.join(instance_result_path, 'Inference_Result.json'), 'w') as f:
                            json.dump(instance_result_dict, f) 
                        ########################################################

        study_HR_list += series_HR_list
        study_EF_list += series_EF_list
        study_ED_vol_list += series_ED_vol_list
        study_ES_vol_list += series_ES_vol_list
        study_GLS_Min_list += series_GLS_Min_list 
        study_ASLS_Min_list += series_ASLS_Min_list###Modified Oct 2021
        study_MSLS_Min_list += series_MSLS_Min_list
        study_BSLS_Min_list += series_BSLS_Min_list
        study_ALLS_Min_list += series_ALLS_Min_list
        study_MLLS_Min_list += series_MLLS_Min_list
        study_BLLS_Min_list += series_BLLS_Min_list
        study_AILS_Min_list += series_AILS_Min_list###Modified Jan 2022
        study_MILS_Min_list += series_MILS_Min_list
        study_BILS_Min_list += series_BILS_Min_list
        study_AALS_Min_list += series_AALS_Min_list
        study_MALS_Min_list += series_MALS_Min_list
        study_BALS_Min_list += series_BALS_Min_list
        study_AP_A2C_Min_list += series_AP_A2C_Min_list###GLS OCT 2022
        study_AP_A4C_Min_list += series_AP_A4C_Min_list###GLS OCT 2022
        study_regional_LS_Min_list = [study_ASLS_Min_list, study_MSLS_Min_list, study_BSLS_Min_list, study_ALLS_Min_list, \
            study_MLLS_Min_list, study_BLLS_Min_list, \
                study_AILS_Min_list, study_MILS_Min_list, study_BILS_Min_list, study_AALS_Min_list, \
            study_MALS_Min_list, study_BALS_Min_list, study_AP_A2C_Min_list, study_AP_A4C_Min_list] ###GLS OCT 2022
    ###############study results

    study_result = engine.get_series_study_result(study_HR_list, study_EF_list, study_ED_vol_list, study_ES_vol_list, study_GLS_Min_list, study_regional_LS_Min_list)
    ##########Write Study Result Json#####################
    study_result_dict = {}
    study_result_dict["studyResult"] = study_result
    with open(os.path.join(study_result_folder, 'Study_Inference_Result.json'), 'w') as f:
        json.dump(study_result_dict, f)
    print(study_result_dict)
    ######################################################
if __name__ == "__main__":
    main()