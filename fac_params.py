# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:27:57 2023

@author: Rui Pinto
"""

import math


def edit_scaling(scal_horiz_val, scal_vert_val, camera):
    
    scal_horiz_norm = round(scal_horiz_val/1920, 2)
    scal_vert_norm = round(scal_vert_val/1080, 2)
    
    print("Set normalized horizontal scaling value to " + str(scal_horiz_norm))
    print("Set normalized vertical scaling value to " + str(scal_vert_norm))
    
 ##   camera.ScalingHorizontalAbs.SetValue(scal_horiz_norm)
 ##   camera.ScalingVerticalAbs.SetValue(scal_vert_norm)

def is_perfect_square(number: int) -> bool:
    """
    Returns True if the provided number is a
    perfect square (and False otherwise).
    """
    return math.isqrt(number) ** 2 == number
     
def use_extra_params(params_extra, Camera):
    
    Camera.TLParamsLocked = False

    if len(params_extra) == 62:
        
        dev_manuf_code = params_extra[0]
        
        try:
            Camera.DeviceManufacturerInfo.SetValue(str(dev_manuf_code))
           
        except Exception:
            print("Not configuring DeviceManufacturerInfo") 
            print(Camera.DeviceManufacturerInfo.this.SetValue(str(dev_manuf_code)))   ## __name__['this']
        
        dev_scan_type = params_extra[1]
        Camera.DeviceScanType.SetValue(str(dev_scan_type)) 
        
        dev_vend_name = params_extra[2]  
        Camera.DeviceVendorName.SetValue(str(dev_vend_name))
        
        timer_sel = params_extra[3] 
        Camera.TimerSelector.SetValue("TimerSelector_" + str(timer_sel))        
        
        timer_trig_source = params_extra[4]
        Camera.TimerTriggerSource.SetValue(str(timer_trig_source))        
        
        timer_delay = params_extra[5]
        Camera.TimerDelayAbs.SetValue(int(timer_delay))
        
        timer_delay_timebase = params_extra[6]
        Camera.TimerDelayTimebaseAbs.SetValue(int(timer_delay_timebase))
        
        timer_duration = params_extra[7]
        Camera.TimerDurationAbs.SetValue(int(timer_duration))
        
        timer_duration_timebase = params_extra[8]
        Camera.TimerDurationTimebaseAbs.setValue(int(timer_duration_timebase))
        
        color_transf_sel = params_extra[9]
        Camera.ColorTransformationSelector.SetValue("ColorTransformationSelector_" + str(color_transf_sel))
        
        color_transf_matrix_factor = params_extra[10] 
        Camera.ColorTransformationMatrixFactor.SetValue(float(color_transf_matrix_factor))
        
        color_transf_value_sel = params_extra[11]
        Camera.ColorTransformationValueSelector.SetValue("ColorTransformationValueSelector_" + str(color_transf_value_sel))
        
        color_transf_value = params_extra[12]
        Camera.ColorTransformationValue.SetValue(float(color_transf_value))
        
        color_adjust_en = params_extra[13]
        Camera.ColorAdjustmentEnable.SetValue(bool(color_adjust_en))
        
        color_adjust_sel = params_extra[14]
        Camera.ColorAdjustmentSelector.SetValue("ColorAdjustmentSelector_" + str(color_adjust_sel))
        
        color_adjust_hue = params_extra[15]
        Camera.ColorAdjustmentHue.SetValue(float(color_adjust_hue))       
        
        color_adjust_sat = params_extra[16]
        Camera.ColorAdjustmentSaturation.SetValue(float(color_adjust_sat))  
        
        stacked_zone_imag_index = params_extra[17]
        Camera.StackedZoneImagingIndex.SetValue(int(stacked_zone_imag_index))
        
        stacked_zone_imag_zone_en = params_extra[18]
        Camera.StackedZoneImagingZoneEnable.SetValue(bool(stacked_zone_imag_zone_en))
        
        stacked_zone_imag_zone_offset_x = params_extra[19]
        Camera.StackedZoneImagingZoneOffsetX.SetValue(int(stacked_zone_imag_zone_offset_x))
        
        stacked_zone_imag_zone_offset_y = params_extra[20]
        Camera.StackedZoneImagingZoneOffsetY.SetValue(int(stacked_zone_imag_zone_offset_y))
        
        stacked_zone_imag_zone_height = params_extra[21]
        Camera.StackedZoneImagingZoneHeight.SetValue(int(stacked_zone_imag_zone_height))
        
        stacked_zone_imag_zone_width = params_extra[22]
        Camera.StackedZoneImagingZoneWidth.SetValue(int(stacked_zone_imag_zone_width))       
        
        acq_mode = params_extra[23]
        Camera.AcquisitionMode.SetValue("AcquisitionMode_" + str(acq_mode))
        
        exp_overlap_time_max = params_extra[24]
        Camera.ExposureOverlapTimeMaxAbs.SetValue(int(exp_overlap_time_max))        
        
        field_output_mode = params_extra[25]
        Camera.FieldOutputMode.SetValue(str(field_output_mode))    
        
        global_reset_release_mode_en = params_extra[26]
        Camera.GlobalResetReleaseModeEnable.SetValue(bool(global_reset_release_mode_en))
        
        acq_status_sel = params_extra[27]
        Camera.AcquisitionStatusSelector.SetValue("AcquisitionStatusSelector_" + str(acq_status_sel))
        
        bal_ratio_sel = params_extra[28]
        Camera.BalanceRatioSelector.SetValue("BalanceRatioSelector_" + str(bal_ratio_sel))
        
        bal_ratio = params_extra[29]
        Camera.BalanceRatioAbs.SetValue(float(bal_ratio))
        
        process_raw_en = params_extra[30] 
        Camera.ProcessedRawEnable.SetValue(bool(process_raw_en))
        
        light_source_sel = params_extra[31]
        Camera.LightSourceSelector.SetValue("LightSourceSelector_" + str(light_source_sel))
        
        seq_control_sel = params_extra[32]
        Camera.SequenceControlSelector.SetValue("SequenceControlSelector_" + str(seq_control_sel))        
        
        seq_address_bit_sel = params_extra[33]
        Camera.SequenceAddressBitSelector.SetValue("SequenceAddressBitSelector_" + str(seq_address_bit_sel))
        
        seq_control_source = params_extra[34]
        Camera.SequenceControlSource.SetValue("SequenceControlSource_" + str(seq_control_source))
        
        seq_address_bit_source = params_extra[35]
        Camera.SequenceBitSource.SetValue("SequenceAddressBitSource_" + str(seq_address_bit_source))
        
        horiz_decim = params_extra[36]
        Camera.DecimationHorizontal.SetValue(int(horiz_decim))
        
        vert_decim = params_extra[37]        
        Camera.DecimationVertical.SetValue(int(vert_decim))
        
        StreamGrabber = Camera.GetStreamGrabberParams() 
        
        stream_grabber_en_resend = params_extra[38]
        StreamGrabber.EnableResend.SetValue(bool(stream_grabber_en_resend))
        
        stream_grabber_packet_timeout = params_extra[39]
        StreamGrabber.PacketTimeout.SetValue(int(stream_grabber_packet_timeout))
         
        stream_grabber_frame_retention = params_extra[40]
        StreamGrabber.FrameRetention.SetValue(int(stream_grabber_frame_retention))
        
        stream_grabber_resend_req_thresh = params_extra[41]
        StreamGrabber.ResendRequestThreshold.SetValue(int(stream_grabber_resend_req_thresh))
        
        stream_grabber_rec_window_size = params_extra[42]
        StreamGrabber.ReceiveWindowSize.SetValue(int(stream_grabber_rec_window_size))
        
        stream_grabber_resend_req_bat = params_extra[43]
        StreamGrabber.ResendRequestBatching.SetValue(int(stream_grabber_resend_req_bat))
        
        stream_grabber_resend_timeout = params_extra[44]
        StreamGrabber.ResendTimeout.SetValue(int(stream_grabber_resend_timeout))
        
        stream_grabber_resend_req_resp_timeout = params_extra[45]
        StreamGrabber.ResendRequestResponseTimeout.SetValue(int(stream_grabber_resend_req_resp_timeout))
        
        stream_grabber_max_num_resend_reqs = params_extra[46]
        StreamGrabber.MaximumNumberResendRequests.SetValue(int(stream_grabber_max_num_resend_reqs))
        
        ## Transport Layer
        
        TlParams = Camera.GetTLParams()
        
        t1_param_read_timeout = params_extra[47]
        TlParams.ReadTimeout.SetValue(int(t1_param_read_timeout))
        
        t1_param_write_timeout = params_extra[48]
        TlParams.WriteTimeout.SetValue(int(t1_param_write_timeout))
        
        t1_param_heartbeat_timeout = params_extra[49]
        TlParams.HeartbeatTimeout.SetValue(int(t1_param_heartbeat_timeout))
         
        sync_user_output_sel = params_extra[50]
        Camera.SyncUserOutputSelector.SetValue("SyncUserOutputSelector_" + str(sync_user_output_sel))
        
        sync_user_output_value = params_extra[51]
        Camera.SyncUserOutputValue.SetValue(bool(sync_user_output_value))
        
        user_set_sel = params_extra[52]
        Camera.UserSetSelector.SetValue("UserSetSelector_" + str(user_set_sel))
        
        default_set_sel = params_extra[53]
        Camera.DefaultSetSelector.SetValue("DefaultSetSelector_" + str(default_set_sel))
        
        user_set_default_sel = params_extra[54]
        Camera.UserSetDefaultSelector.SetValue("UserSetDefaultSelector_" + str(user_set_default_sel))
        
        params_sel = params_extra[55]
        Camera.ParameterSelector.SetValue("ParameterSelector_" + str(params_sel))
        
        remove_limits = params_extra[56]
        Camera.RemoveLimits.SetValue(bool(remove_limits))
        
        auto_func_aoi_usage_intens = params_extra[57]
        Camera.AutoFunctionAOIUsageIntensity.SetValue(bool(auto_func_aoi_usage_intens))
        
        auto_func_aoi_usage_white_bal = params_extra[58]
        Camera.AutoFunctionAOIUsageWhiteBalance.SetValue(bool(auto_func_aoi_usage_white_bal))        
        
        median_filter = params_extra[59]
        Camera.MedianFilter.SetValue(bool(median_filter))        
        
        chunck_sel = params_extra[60]
        Camera.ChunkSelector.SetValue("ChunkSelector_" + str(chunck_sel))
        
        chunck_en = params_extra[61]
        Camera.ChunkEnable.SetValue(bool(chunck_en))        
        
    else:
        print("Wrong number of extra parameters")
    
    Camera.TLParamsLocked = True
    

def distorsion_models_gui():
    
    import PySimpleGUI as sg
    
    repInit = True
    
    layout = [
        [sg.Text("Camera distortion model: ")],
        [sg.T("         "), sg.Checkbox('Kannala-Brandt Model', default=False, key="-IN1-")],
        [sg.T("         "), sg.Checkbox('Brown-Conrady Model', default=True, key="-IN2-")],
        [sg.T("         "), sg.Checkbox('No distortion model', default=False, key="-IN3-")]
    ]
    
    window = sg.Window('Distortion Models GUI', layout)
    
    code_distortion_model = -1
    
    while repInit == True:
       event, values = window.read()
       if event == "Exit" or event == sg.WIN_CLOSED:
           break
       
       if event == "Next":
           if values["-IN1-"] == True and values["-IN2-"] == False and values["-IN3-"] == False:
               code_distortion_model = 1
               repInit = False
               break
           elif values["-IN2-"] == True and values["-IN1-"] == False and values["-IN3-"] == False:
               code_distortion_model = 2
               repInit = False
               break
           elif values["-IN3-"] == True and values["-IN1-"] == False and values["-IN2-"] == False:
               code_distortion_model = 0
               repInit = False
               break
           else:
               if (values["-IN1-"] == True and values["-IN2-"] == True) or (values["-IN1-"] == True and values["-IN3-"] == True) or (values["-IN2-"] == True and values["-IN3-"] == True):
                   print("Only pick one option !!!")                   
                   repInit = True           
                   continue
               
    window.close()
    
    return code_distortion_model                
                

def vig_cor_to_jpeg(dir_img, filename_img, final_image_filename):
    
    import cv2
    import numpy as np
    
    if not('.jpg' in filename_img):
        img = cv2.imread(dir_img + filename_img)
        
        file_parts = filename_img.rpartition('.')
        
        for d in file_parts:
            if len(d) > 5:
                filename_without_ext = d
                break
            
        cv2.imwrite(dir_img + filename_without_ext + '.jpg', img)
        
    else:
        filename_without_ext = filename_img
    
    img = cv2.imread(dir_img + filename_without_ext + '.jpg') #load rgb image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
        
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))
        
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        
    if not('.jpg' in final_image_filename) and not('.' in final_image_filename):
       final_image_filename += '.jpg'          
            
    cv2.imwrite(final_image_filename, img)  


def vignetic_correction():
    
    import PySimpleGUI as sg
    
    img_after_vig_cor = True
    
    windowx = sg.Window('Choose image directory', [[sg.Text('Filename')], [sg.InputText(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
    (keyword, dict_dir) = windowx                
    dir_path = dict_dir['Browse']
    
    if '/' in dir_path:
        dir_parts = dir_path.split('/')
    elif "\\" in dir_path:
        dir_parts = dir_path.split("\\")     
    
    dir_img = dir_parts[:-1]
    filename_img = dir_parts[-1]  

    while img_after_vig_cor == True:
    
        final_image_filename = input('Please specify final image filename (after vignetic correction process): ')
        
        if len(final_image_filename) < 5:
            img_after_vig_cor = True
        else:
            img_after_vig_cor = False
            break
            
    if '.png' in final_image_filename:
        final_image_parts = final_image_filename.split('.png')
        
        for d in final_image_parts:
            if len(d) > 0:
                final_image_filename = d
    
    elif '.tiff' in final_image_filename:
        final_image_parts = final_image_filename.split('.tiff')
        
        for d in final_image_parts:
            if len(d) > 0:
                final_image_filename = d
                
    elif '.jpeg' in final_image_filename:
        final_image_parts = final_image_filename.split('.jpeg')
        
        for d in final_image_parts:
            if len(d) > 0:
                final_image_filename = d
    
    
 ##   dir_img = ""
 ##   filename_img = ""
 ##   final_image_filename = ""
     
    
    vig_cor_to_jpeg(dir_img, filename_img, final_image_filename)
    

def scaling_gui(width, height):
    
    import PySimpleGUI as sg
    
    racio = round(height/width, 2)
    
    scal_horiz_val = 0
    scal_vert_val = 0 
    
    layout_dims = [
        [sg.Text("Width: "), sg.Text(str(width))],
        [sg.Text("Height: "), sg.Text(str(height))]      
    ]
    
    layout_racio = [
        [sg.Text("Racio: "), sg.Text(str(racio))] 
    ]
    
    layout_dims_all = [
      [
        sg.Column(layout_dims),
        sg.VSeparator(),
        sg.Column(layout_racio)        
      ]
    ]
    
    layout = [
        [layout_dims_all],
        [sg.Text('Scaling Horizontal Value: '), sg.InputText(default_text= "", size=(19, 1), key="SCAL_HORIZ")],
        [sg.Text('Scaling Vertical Value: '), sg.InputText(default_text= "", size=(19, 1), key="SCAL_VERT")],
        [sg.Button("Save"), sg.Button("Break")]        
    ]
    
    window = sg.Window('Scaling Info', layout) 

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Back":
            break
        if event == "Save":           
            
            scal_horiz_val = 1920
            scal_vert_val = 1080
            
            if not values['SCAL_HORIZ'] and values['SCAL_VERT']:
                
                scal_vert_val = int(values['SCAL_VERT'])                
                print("Setting a value for SCAL_HORIZ, based on the SCAL_VERT one ...")
                scal_horiz_val = (1/racio)*scal_vert_val
                
                scal_horiz_val = int(scal_horiz_val)
                print("Scal vert: " + str(scal_horiz_val))
                
            elif not values['SCAL_VERT'] and values['SCAL_HORIZ']:
                
                scal_horiz_val = int(values['SCAL_HORIZ'])
                print("Setting a value for SCAL_VERT, based on the SCAL_HORIZ one ...")
                scal_vert_val = racio*scal_horiz_val
                
                scal_vert_val = int(scal_vert_val)                
                print("Scal vert: " + str(scal_vert_val))
                
            elif not values['SCAL_HORIZ'] and not values['SCAL_VERT']:
                
                print("Setting default values for scaling ...")
                scal_horiz_val = 1920
                scal_vert_val = 1080
               
                
            break
        
    window.close() 
    
    scal_data = [scal_horiz_val, scal_vert_val]
    
    return scal_data    

def constant_info(ChunkInfo, AcqManagInfo, SensorDimsInfo, StreamParamInfo, DevInfoAdd, StreamGrabberStatistics, DriverOpts):   
     
    import PySimpleGUI as sg 
    
    stop_first = False    
    stop_sec = False
    stop_third = False
    stop_fourth = False   
    stop_five = False   
    stop_six = False    
    
    
    for x in ChunkInfo:
        if x is None:
            stop_first = True
            break
    
    for x in AcqManagInfo:
        if x is None:
            stop_sec = True
            break
            
    for x in SensorDimsInfo:
        if x is None:
            stop_third = True
            break
    
    for x in StreamParamInfo:
        if x is None:
            stop_fourth = True
            break
    
    for x in DevInfoAdd:
        if x is None:
            stop_five = True
            break
    
    for x in StreamGrabberStatistics:
        if x is None:
            stop_six = True
            break 
    
    for x in DriverOpts:
        if x is None:
            stop_seven = True
            break
            
            
    if stop_first == False and stop_sec == False and stop_third == False and stop_fourth == False and stop_five == False and stop_six == False and stop_seven == False:
        
        
    
        ChunkDynamicRangeMin = ChunkInfo[0]
        ChunkDynamicRangeMax = ChunkInfo[1]
        ChunkPixelFormat = ChunkInfo[2]
        ChunkOffsetX = ChunkInfo[3]
        ChunkOffsetY = ChunkInfo[4]
        ChunkWidth = ChunkInfo[5]
        ChunkHeight = ChunkInfo[6]
        ChunkFramecounter = ChunkInfo[7]
        ChunkTimestamp = ChunkInfo[8]
        ChunkTriggerinputcounter = ChunkInfo[9]
        ChunkLineStatusAll = ChunkInfo[10]
        ChunkSequenceSetIndex = ChunkInfo[11]  
        
        ReadoutTimeAbs = AcqManagInfo[0]
        ResultingFrameRateAbs = AcqManagInfo[1]
        LastError = AcqManagInfo[2]
        PatternRemovalAuto = AcqManagInfo[3]
        
        SensorWidth = SensorDimsInfo[0]
        SensorHeight = SensorDimsInfo[1]
        WidthMax = SensorDimsInfo[2]
        HeightMax = SensorDimsInfo[3]
        
        PayloadSize = StreamParamInfo[0]
        GevSCFJM = StreamParamInfo[1]
        GevSCDMT = StreamParamInfo[2]
        GevSCDCT = StreamParamInfo[3]
        DeviceLinkCurrentThroughput = StreamParamInfo[4]
        BandwidthAssigned = StreamParamInfo[5]
        
        dev_user_id = DevInfoAdd[0]
        dev_indic_mode = DevInfoAdd[1]
        dev_link_sel = DevInfoAdd[2]
        dev_link_speed = DevInfoAdd[3]
        dev_link_throughput_limit_mode = DevInfoAdd[4]
        dev_sfnc_version_major = DevInfoAdd[5]
        dev_sfnc_version_minor = DevInfoAdd[6]
        dev_sfnc_version_sub_minor = DevInfoAdd[7]  
        dev_family_name = DevInfoAdd[8]       
        
   
        stream_grabber_statistic_buffer_underrun = StreamGrabberStatistics[0]
        stream_grabber_statistic_failed_buffer_count = StreamGrabberStatistics[1]
        stream_grabber_statistic_failed_packet_count = StreamGrabberStatistics[2]
        stream_grabber_statistic_last_block_id = StreamGrabberStatistics[3]
        stream_grabber_statistic_last_failed_buffer_status = StreamGrabberStatistics[4]
        stream_grabber_statistic_last_failed_buffer_status_text = StreamGrabberStatistics[5]
        stream_grabber_statistic_missed_frame_count = StreamGrabberStatistics[6]
        stream_grabber_statistic_resend_request_count = StreamGrabberStatistics[7]
        stream_grabber_statistic_resend_packet_count = StreamGrabberStatistics[8]
        stream_grabber_statistic_resynchronization_count = StreamGrabberStatistics[9]
        stream_grabber_statistic_total_buffer_count = StreamGrabberStatistics[10]
        stream_grabber_statistic_total_packet_count = StreamGrabberStatistics[11]
        
        
        windows_perf_driver_available = DriverOpts[0]
        windows_filter_driver_available = DriverOpts[1]
        soc_driver_available = DriverOpts[2]
     
        
        layout_chunck = [
            [sg.Text('Chunk Dynamic Range Min. : '), sg.Text(str(ChunkDynamicRangeMin))],
            [sg.Text('Chunk Dynamic Range Max. : '), sg.Text(str(ChunkDynamicRangeMax))],
            [sg.Text('Chunk Pixel Format: '), sg.Text(str(ChunkPixelFormat))],
            [sg.Text('Chunk Offset X: '), sg.Text(str(ChunkOffsetX))],
            [sg.Text('Chunk Offset Y: '), sg.Text(str(ChunkOffsetY))],
            [sg.Text('Chunk Width: '), sg.Text(str(ChunkWidth))],
            [sg.Text('Chunk Height: '), sg.Text(str(ChunkHeight))],
            [sg.Text('Chunk Frame Counter: '), sg.Text(str(ChunkFramecounter))],
            [sg.Text('Chunk Timestamp: '), sg.Text(str(ChunkTimestamp))],
            [sg.Text('Chunk Trigger Input Counter: '), sg.Text(str(ChunkTriggerinputcounter))],
            [sg.Text('Chunk Line Status All: '), sg.Text(str(ChunkLineStatusAll))],
            [sg.Text('Chunk Sequence Set Index: '), sg.Text(str(ChunkSequenceSetIndex))]    
        ]
        
        layout_acq_management = [
            [sg.Text('Readout Time: '), sg.Text(str(ReadoutTimeAbs))],
            [sg.Text('Resulting Frame Rate: '), sg.Text(str(ResultingFrameRateAbs))],
            [sg.Text('Last Error: '), sg.Text(str(LastError))],
            [sg.Text('Pattern Removal Auto: '), sg.Text(str(PatternRemovalAuto))]        
        ]
        
        
        layout_sensor_dims = [
            [sg.Text('Sensor Width: '), sg.Text(str(SensorWidth))],
            [sg.Text('Sensor Height: '), sg.Text(str(SensorHeight))],
            [sg.Text('Width Max. : '), sg.Text(str(WidthMax))],
            [sg.Text('Height Max. : '), sg.Text(str(HeightMax))]       
        ]
        
        layout_stream_params = [
            [sg.Text('Payload Size: '), sg.Text(str(PayloadSize))],
            [sg.Text('Frame Jitter Max. : '), sg.Text(str(GevSCFJM))],
            [sg.Text('Device Max. Throughput: '), sg.Text(str(GevSCDMT))],
            [sg.Text('Device Current Throughput: '), sg.Text(str(GevSCDCT))],
            [sg.Text('Device Link Current Throughput: '), sg.Text(str(DeviceLinkCurrentThroughput))],
            [sg.Text('Bandwidth Assigned: '), sg.Text(str(BandwidthAssigned))]
        ]
        
        layout_device_info_params = [
            [sg.Text('Device User ID: '), sg.Text(str(dev_user_id))],
            [sg.Text('Device Family Name: '), sg.Text(str(dev_family_name))],
            [sg.Text('Device Indicator Mode: '), sg.Text(str(dev_indic_mode))],
            [sg.Text('Device Link Selector: '), sg.Text(str(dev_link_sel))],
            [sg.Text('Device Link Speed: '), sg.Text(str(dev_link_speed))],
            [sg.Text('Device Link Throughput Limit: '), sg.Text(str(dev_link_throughput_limit_mode))],
            [sg.Text('Device SFNC version major: '), sg.Text(str(dev_sfnc_version_major))],
            [sg.Text('Device SFNC version minor: '), sg.Text(str(dev_sfnc_version_minor))],
            [sg.Text('Device SFNC version sub-minor: '), sg.Text(str(dev_sfnc_version_sub_minor))]
        ]  

        layout_stream_grabber_statistics = [
            [sg.Text('Statistic Buffer Underrun Count: '), sg.Text(str(stream_grabber_statistic_buffer_underrun))],
            [sg.Text('Statistic Failed Buffer Count: '), sg.Text(str(stream_grabber_statistic_failed_buffer_count))],
            [sg.Text('Statistic Failed Packet Count: '), sg.Text(str(stream_grabber_statistic_failed_packet_count))],
            [sg.Text('Statistic Last Block ID: '), sg.Text(str(stream_grabber_statistic_last_block_id))],
            [sg.Text('Statistic Last Failed Buffer Status: '), sg.Text(str(stream_grabber_statistic_last_failed_buffer_status))],
            [sg.Text('Statistic Last Failed Buffer Status Text: '), sg.Text(str(stream_grabber_statistic_last_failed_buffer_status_text))],
            [sg.Text('Statistic Missed Frame Count: '), sg.Text(str(stream_grabber_statistic_missed_frame_count))],
            [sg.Text('Statistic Resend Request Count: '), sg.Text(str(stream_grabber_statistic_resend_request_count))],
            [sg.Text('Statistic Resend Packet Count: '), sg.Text(str(stream_grabber_statistic_resend_packet_count))],
            [sg.Text('Statistic Resynchronization Count: '), sg.Text(str(stream_grabber_statistic_resynchronization_count))],
            [sg.Text('Statistic Total Buffer Count: '), sg.Text(str(stream_grabber_statistic_total_buffer_count))],
            [sg.Text('Statistic Total Packet Count: '), sg.Text(str(stream_grabber_statistic_total_packet_count))]        
        ]
        
        layout_driver_opts = [
            [sg.Text('Windows Intel Performance Driver Available: '), sg.Text(str(windows_perf_driver_available))],
            [sg.Text('Windows Filter Driver Available: '), sg.Text(str(windows_filter_driver_available))],
            [sg.Text('Socket Driver Available: '), sg.Text(str(soc_driver_available))]
        ]
        
        
        tabgrp = [[sg.TabGroup([[sg.Tab('Chunck Details', layout_chunck),
                        sg.Tab('Acquisition Management Details', layout_acq_management),
                        sg.Tab('Sensor Details', layout_sensor_dims),
                        sg.Tab('Stream Parameters Details', layout_stream_params),
                        sg.Tab('Device Info Parameters Details', layout_device_info_params),
                        sg.Tab('Stream Grabber Statistics', layout_stream_grabber_statistics),
                        sg.Tab('Driver Details', layout_driver_opts)]])
        ]]
        
        
        layout_opt = [
            [sg.Button("Back")]
        ]
        
        
        layout = [
            [tabgrp], 
            [layout_opt]       
        ]
        
        
        window = sg.Window('Extra Parameters Info', layout)
        
        while True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               break
           if event == "Back": 
               break
           
           
        window.close()  
    else:
        print("Error receving constant info to show afterwards")


def get_constant_info(Camera):
    ChunkInfo = []
    AcqManagInfo = []     
    SensorDimsInfo = []
    StreamParamInfo = []
    DevInfoAdd = []
    StreamGrabberStatistics = []
    DriverOpts = []
    
    dynamicRangeMin = Camera.ChunkDynamicRangeMin.GetValue()
    if dynamicRangeMin is not None:    
        ChunkInfo.append(dynamicRangeMin)
    else:
        ChunkInfo.append(None)
        
    dynamicRangeMax = Camera.ChunkDynamicRangeMax.GetValue()   
    if dynamicRangeMax is not None: 
        ChunkInfo.append(dynamicRangeMax)
    else:
        ChunkInfo.append(None)     
        
    pixelFormat = Camera.ChunkPixelFormat.GetValue()   
    if pixelFormat is not None: 
        ChunkInfo.append(pixelFormat)
    else:
        ChunkInfo.append(None)
        
    offsetX = Camera.ChunkOffsetX.GetValue()
    if offsetX is not None: 
        ChunkInfo.append(offsetX)
    else:
        ChunkInfo.append(None)
        
    offsetY = Camera.ChunkOffsetY.GetValue()
    if offsetY is not None: 
        ChunkInfo.append(offsetY)
    else:
        ChunkInfo.append(None)
        
    width = Camera.ChunkWidth.GetValue()        
    if width is not None: 
        ChunkInfo.append(width)
    else:
        ChunkInfo.append(None)
        
    height = Camera.ChunkHeight.GetValue()    
    if height is not None: 
        ChunkInfo.append(height)
    else:
        ChunkInfo.append(None)
        
    frameCounter = Camera.ChunkFramecounter.GetValue() 
    if frameCounter is not None: 
        ChunkInfo.append(frameCounter)
    else:
        ChunkInfo.append(None)
        
    timeStamp = Camera.ChunkTimestamp.GetValue()    
    if timeStamp is not None: 
        ChunkInfo.append(timeStamp)
    else:
        ChunkInfo.append(None)
        
    triggerinputCounter = Camera.ChunkTriggerinputcounter.GetValue()
    if triggerinputCounter is not None: 
        ChunkInfo.append(triggerinputCounter)
    else:
        ChunkInfo.append(None)
        
    lineStatusAll = Camera.ChunkLineStatusAll.GetValue()    
    if lineStatusAll is not None: 
        ChunkInfo.append(lineStatusAll)
    else:
        ChunkInfo.append(None)
        
    timeStamp_index = Camera.ChunkSequenceSetIndex.GetValue()      
    if timeStamp_index is not None: 
        ChunkInfo.append(timeStamp_index)
    else:
        ChunkInfo.append(None)
        
        
    ReadoutTime = Camera.ReadoutTimeAbs.GetValue()    
    if ReadoutTime is not None: 
        AcqManagInfo.append(ReadoutTime)
    else:
        AcqManagInfo.append(None)
        
    resultingFps = Camera.ResultingFrameRateAbs.GetValue()    
    if resultingFps is not None: 
        AcqManagInfo.append(resultingFps)
    else:
        AcqManagInfo.append(None)
        
    lasterror = Camera.LastError.GetValue()    
    if lasterror is not None: 
        AcqManagInfo.append(lasterror)
    else:
        AcqManagInfo.append(None)
        
    e = Camera.PatternRemovalAuto.GetValue()    ## PatternRemovalAutoEnums  
    if e is not None: 
        AcqManagInfo.append(e)
    else:
        AcqManagInfo.append(None)
    
    sensorWidth = Camera.SensorWidth.GetValue()    
    if sensorWidth is not None: 
        SensorDimsInfo.append(sensorWidth)
    else:
        SensorDimsInfo.append(None)
        
    sensorHeight = Camera.SensorHeight.GetValue()       
    if sensorHeight is not None: 
        SensorDimsInfo.append(sensorHeight)
    else:
        SensorDimsInfo.append(None)
        
    maxWidth = Camera.WidthMax.GetValue()
    if maxWidth is not None: 
        SensorDimsInfo.append(maxWidth)
    else:
        SensorDimsInfo.append(None)
        
    maxHeight = Camera.HeightMax.GetValue()    
    if maxHeight is not None: 
        SensorDimsInfo.append(maxHeight)
    else:
        SensorDimsInfo.append(None)
    
    payloadSize = Camera.PayloadSize.GetValue()
    if payloadSize is not None: 
        StreamParamInfo.append(payloadSize)
    else:
        StreamParamInfo.append(None)
        
    jitterMax = Camera.GevSCFJM.GetValue()
    if jitterMax is not None: 
        StreamParamInfo.append(jitterMax)
    else:
        StreamParamInfo.append(None)     
        
    maxThroughput = Camera.GevSCDMT.GetValue()
    if maxThroughput is not None: 
        StreamParamInfo.append(maxThroughput)
    else:
        StreamParamInfo.append(None)
        
    currentThroughput = Camera.GevSCDCT.GetValue()    
    if currentThroughput is not None: 
        StreamParamInfo.append(currentThroughput)  
    else:
        StreamParamInfo.append(None)
        
    ## Include the two below ones afterwards
    deviceLinkCurrentThroughput = Camera.BslDeviceLinkCurrentThroughput.GetValue()
    if deviceLinkCurrentThroughput is not None: 
        StreamParamInfo.append(deviceLinkCurrentThroughput)  
    else:
        StreamParamInfo.append(None)
    
    bandwidth_assigned = Camera.GevSCBWA.GetValue()
    if bandwidth_assigned is not None: 
        StreamParamInfo.append(bandwidth_assigned)  
    else:
        StreamParamInfo.append(None) 
        
    dev_user_id = Camera.DeviceUserID.GetValue()
    if dev_user_id is not None: 
        DevInfoAdd.append(dev_user_id)  
    else:
        DevInfoAdd.append(None)

    dev_indic_mode = Camera.DeviceIndicatorMode.GetValue()
    if dev_indic_mode is not None: 
        DevInfoAdd.append(dev_indic_mode)  
    else:
        DevInfoAdd.append(None) 
    
    dev_link_sel = Camera.DeviceLinkSelector.GetValue()
    if dev_link_sel is not None: 
        DevInfoAdd.append(dev_link_sel)  
    else:
        DevInfoAdd.append(None) 
    
    dev_link_speed = Camera.DeviceLinkSpeed.GetValue()
    if dev_link_speed is not None: 
        DevInfoAdd.append(dev_link_speed)  
    else:
        DevInfoAdd.append(None)  

    dev_link_thoughput_limit_mode = Camera.DeviceLinkThroughputLimitMode.GetValue()
    if dev_link_thoughput_limit_mode is not None: 
        DevInfoAdd.append(dev_link_thoughput_limit_mode)  
    else:
        DevInfoAdd.append(None)
        
    dev_SFNC_version_major = Camera.DeviceSFNCVersionMajor.GetValue()
    if dev_SFNC_version_major is not None: 
        DevInfoAdd.append(dev_SFNC_version_major)  
    else:
        DevInfoAdd.append(None)
    
    dev_SFNC_version_minor = Camera.DeviceSFNCVersionMinor.GetValue()
    if dev_SFNC_version_minor is not None: 
        DevInfoAdd.append(dev_SFNC_version_minor)  
    else:
        DevInfoAdd.append(None)
    
    dev_SFNC_version_sub_minor = Camera.DeviceSFNCVersionSubMinor.GetValue()
    if dev_SFNC_version_sub_minor is not None: 
        DevInfoAdd.append(dev_SFNC_version_sub_minor)  
    else:
        DevInfoAdd.append(None)
        
    dev_family_name = Camera.DeviceFamilyName.GetValue()
    if dev_family_name is not None: 
        DevInfoAdd.append(dev_family_name)  
    else:
        DevInfoAdd.append(None) 

    buffer_underrun_count = Camera.GetStreamGrabberParams().Statistic_Buffer_Underrun_Count.GetValue() 
    if buffer_underrun_count is not None: 
        StreamGrabberStatistics.append(buffer_underrun_count)  
    else:
        StreamGrabberStatistics.append(None)
    
    failed_buffer_count = Camera.GetStreamGrabberParams().Statistic_Failed_Buffer_Count.GetValue() 
    if failed_buffer_count is not None: 
        StreamGrabberStatistics.append(failed_buffer_count)  
    else:
        StreamGrabberStatistics.append(None)
        
    failed_packet_count = Camera.GetStreamGrabberParams().Statistic_Failed_Packet_Count.GetValue() 
    if failed_packet_count is not None: 
        StreamGrabberStatistics.append(failed_packet_count)  
    else:
        StreamGrabberStatistics.append(None)
    
    last_block_id = Camera.GetStreamGrabberParams().Statistic_Last_Block_Id.GetValue() 
    if last_block_id is not None: 
        StreamGrabberStatistics.append(last_block_id)  
    else:
        StreamGrabberStatistics.append(None)
    
    last_failed_buffer_status = Camera.GetStreamGrabberParams().Statistic_Last_Failed_Buffer_Status.GetValue() 
    if last_failed_buffer_status is not None: 
        StreamGrabberStatistics.append(last_failed_buffer_status)  
    else:
        StreamGrabberStatistics.append(None)
    
    last_failed_buffer_status_text = Camera.GetStreamGrabberParams().Statistic_Last_Failed_Buffer_Status_Text.GetValue() 
    if last_failed_buffer_status_text is not None: 
        StreamGrabberStatistics.append(last_failed_buffer_status_text)  
    else:
        StreamGrabberStatistics.append(None) 
    
    missed_frame_count = Camera.GetStreamGrabberParams().Statistic_Missed_Frame_Count.GetValue() 
    if missed_frame_count is not None: 
        StreamGrabberStatistics.append(missed_frame_count)  
    else:
        StreamGrabberStatistics.append(None)    
    
    resend_req_count = Camera.GetStreamGrabberParams().Statistic_Resend_Request_Count.GetValue() 
    if resend_req_count is not None: 
        StreamGrabberStatistics.append(resend_req_count)  
    else:
        StreamGrabberStatistics.append(None) 
    
    resend_packet_count = Camera.GetStreamGrabberParams().Statistic_Resend_Packet_Count.GetValue() 
    if resend_packet_count is not None: 
        StreamGrabberStatistics.append(resend_packet_count)  
    else:
        StreamGrabberStatistics.append(None)
    
    resync_count = Camera.GetStreamGrabberParams().Statistic_Resynchronization_Count.GetValue() 
    if resync_count is not None: 
        StreamGrabberStatistics.append(resync_count)  
    else:
        StreamGrabberStatistics.append(None)
        
    total_buffer_count = Camera.GetStreamGrabberParams().Statistic_Total_Buffer_Count.GetValue() 
    if total_buffer_count is not None: 
        StreamGrabberStatistics.append(total_buffer_count)  
    else:
        StreamGrabberStatistics.append(None) 

    total_packet_count = Camera.GetStreamGrabberParams().Statistic_Total_Packet_Count.GetValue() 
    if total_packet_count is not None: 
        StreamGrabberStatistics.append(total_packet_count)  
    else:
        StreamGrabberStatistics.append(None)
        
    ## Driver Opts
    
    windows_perf_driver_available = Camera.GetStreamGrabberParams().TypeIsWindowsIntelPerformanceDriverAvailable().GetValue()
    if windows_perf_driver_available is not None: 
        DriverOpts.append(windows_perf_driver_available)  
    else:
        DriverOpts.append(None)
    
    windows_filter_driver_availale = Camera.GetStreamGrabberParams().TypeIsWindowsFilterDriverAvailable().GetValue()
    if windows_filter_driver_availale is not None: 
        DriverOpts.append(windows_filter_driver_availale)  
    else:
        DriverOpts.append(None)  
    
    socket_driver_available = Camera.GetStreamGrabberParams().TypeIsSocketDriverAvailable().GetValue()
    if socket_driver_available is not None: 
        DriverOpts.append(socket_driver_available)  
    else:
        DriverOpts.append(None)     
        
    
    constant_info(ChunkInfo, AcqManagInfo, SensorDimsInfo, StreamParamInfo, DevInfoAdd, StreamGrabberStatistics, DriverOpts)   
    
    

def extra_params_gui(camera):
    
    import PySimpleGUI as sg
    import math    
    
    dev_manuf_code = ""
    dev_scan_type = ""
    dev_vend_name = ""
    
    timer_sel = ""
    timer_trig_source = ""
    timer_delay = ""
    timer_delay_timebase = ""
    timer_duration = ""
    timer_duration_timebase = ""
    
    color_transf_sel = ""
    color_transf_matrix_factor = ""
    color_transf_value_sel = ""
    color_transf_value = ""
    color_adjust_en = ""
    color_adjust_sel = ""
    color_adjust_hue = ""
    color_adjust_sat = ""
    
    stacked_zone_imag_index = ""
    stacked_zone_imag_zone_en = ""
    stacked_zone_imag_zone_offset_x = ""
    stacked_zone_imag_zone_offset_y = ""
    stacked_zone_imag_zone_height = ""
    stacked_zone_imag_zone_width = ""
    
    acq_mode = ""
    exp_overlap_time_max = ""
    field_output_mode = ""
    global_reset_release_mode_en = ""
    acq_status_sel = ""
    bal_ratio_sel = ""
    bal_ratio = ""
    process_raw_en = ""
    light_source_sel = ""
    
    seq_control_sel = ""
    seq_address_bit_sel = ""
    seq_control_source = ""
    seq_address_bit_source = ""
    horiz_decim = ""
    vert_decim = ""
    
    stream_grabber_access_mode = ""
    stream_grabber_auto_packet_size = ""
    stream_grabber_max_buffer_size = ""
    stream_grabber_max_num_buffer = ""
    stream_grabber_max_transfer_size = ""
    stream_grabber_num_max_queuedUrbs = ""
    stream_grabber_rec_thread_priority_override = ""
    stream_grabber_rec_thread_priority = ""
    stream_grabber_socket_buffer_size = ""
    stream_grabber_status = ""
    stream_grabber_transfer_loop_thread_priority = ""
    stream_grabber_type_gige_vision_filter_driver = ""
    stream_grabber_firewall_traversal_interval = ""
    get_event_grabber_firewall_traversal_interval = ""
    stream_grabber_transmission_type = ""
    stream_grabber_destination_port = ""
    
    stream_grabber_en_resend = ""
    stream_grabber_packet_timeout = "" 
    stream_grabber_frame_retention = ""
    stream_grabber_resend_req_thresh = ""
    stream_grabber_rec_window_size = ""
    stream_grabber_resend_req_bat = ""
    stream_grabber_resend_timeout = ""
    stream_grabber_resend_req_resp_timeout = ""
    stream_grabber_max_num_resend_reqs = ""    
    
    t1_param_read_timeout = ""
    t1_param_write_timeout = ""
    t1_param_heartbeat_timeout = ""
    
    sync_user_output_sel = ""
    sync_user_output_value = ""
    user_set_sel = ""
    default_set_sel = ""
    user_set_default_sel = ""
    params_sel = ""
    remove_limits = ""
    auto_func_aoi_usage_intens = ""
    auto_func_aoi_usage_white_bal = ""
    median_filter = ""
    chunck_sel = ""
    chunck_en = ""  
    
    bw_res_mode = ""
    dev_link_throughput_lim = ""
    gevHeartbeatTimeout = ""   
    
    timer_delay_ok = False
    
    basler_name = "Basler"
    
    layout_headers = [
        [sg.Text('Device Manufacturer Info: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="DEV_MANUF_INFO")],
        [sg.Text('Device Scan Type: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="DEV_SCAN_TYPE")],
        [sg.Text('Device Vendor Name: '), sg.InputText(disabled=True, default_text= basler_name, size=(19, 1), key="DEV_VENDOR_NAME")],
        [sg.Button('Save Header Details')]
    ]    
    
    layout_params_first = [
        [sg.Text('Timer Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="TIMER_SEL")],
        [sg.Text('Timer Trigger Source: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="TIMER_TRIG_SOURCE")],
        [sg.Text('Timer Delay: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="TIMER_DELAY")],
        [sg.Text('Timer Delay Timebase: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="TIMER_DELAY_TIMEBASE")],
        [sg.Text('Timer Duration: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="TIMER_DURATION")],
        [sg.Text('Timer Duration Timebase: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="TIMER_DURATION_TIMEBASE")],
        [sg.Button('Save Timer Details')]        
    ]
    
    layout_params_sec = [
        [sg.Text('Color Transformation Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="COL_TRANSF_SEL")],
        [sg.Text('Color Transformation Matrix Factor: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="COL_TRANSF_MFACTOR")],
        [sg.Text('Color Transformation Value Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="COL_TRANSF_VAL_SEL")],
        [sg.Text('Color Transformation Value: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="COL_TRANSF_VAL")],
        [sg.Text('Color Adjustment Enable: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="COL_ADJUST_EN")],
        [sg.Text('Color Adjustment Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="COL_ADJUST_SEL")],
        [sg.Text('Color Adjustment Hue: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="COL_ADJUST_HUE")],
        [sg.Text('Color Adjustment Saturation: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="COL_ADJUST_SAT")],
        [sg.Button('Save Color Details')] 
    ] 
    
    layout_params_third = [
        [sg.Text('Stacked Zone Imaging Index: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STACK_ZONE_IMAG_IND")],
        [sg.Text('Stacked Zone Imaging Zone Enable: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STACK_ZONE_IMAG_ZONE_EN")],
        [sg.Text('Stacked Zone Imaging Zone Offset X: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STACK_ZONE_IMAG_ZONE_OFFSET_X")],
        [sg.Text('Stacked Zone Imaging Zone Offset Y: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STACK_ZONE_IMAG_ZONE_OFFSET_Y")],
        [sg.Text('Stacked Zone Imaging Zone Height: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STACK_ZONE_IMAG_ZONE_HEIGHT")],
        [sg.Text('Stacked Zone Imaging Zone Width: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STACK_ZONE_IMAG_ZONE_WIDTH")],
        [sg.Button('Save Stacked Zone Imaging Details')]  
    ]
    
    layout_params_fourth = [    ## Acquisition & Exposure
        [sg.Text('Acquisition Mode: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="ACQ_MODE")],
        [sg.Text('Exposure Overlap Time Max. : '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="EXP_OVERLAP_TIME_MAX")],
        [sg.Text('Field Output Mode: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="FIELD_OUT_MODE")],
        [sg.Text('Global Reset Release Mode Enable: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="GL_RES_REL_MODE_EN")],
        [sg.Text('Acquisition Status Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="ACQ_STATUS_SEL")],
        [sg.Text('Balance Ratio Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="BAL_RATIO_SEL")],
        [sg.Text('Balance Ratio: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="BAL_RATIO")],
        [sg.Text('Process Raw Enable: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="PROC_RAW_EN")],
        [sg.Text('Light Source Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="LIGHT_SOURCE_SEL")],
        [sg.Button('Save Acquisition & Exposure Details')] 
    ] 
     
    layout_params_fifth = [       ## Sequence & Decimation
        [sg.Text('Sequence Control Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="SEQ_CONTROL_SEL")],
        [sg.Text('Sequence Address Bit Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="SEQ_ADDRESS_BIT_SEL")],
        [sg.Text('Sequence Control Source: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="SEQ_CONTROL_SOURCE")],
        [sg.Text('Sequence Address Bit Source: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="SEQ_ADDRESS_BIT_SOURCE")],
        [sg.Text('Horizontal Decimation: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="HORIZ_DECIM")],
        [sg.Text('Vertical Decimation: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="VERT_DECIM")],
        [sg.Button('Save Sequence & Decimation Details')]
    ]
    
    layout_params_stream_grabber = [
        [sg.Text('Stream Grabber Access Mode: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_ACCESS_MODE")],
        [sg.Text('Stream Grabber Auto Packet Size: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_AUTO_PACKET_SIZE")],
        [sg.Text('Stream Grabber Max. Buffer Size: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_MAX_BUFFER_SIZE")],
        [sg.Text('Stream Grabber Max. Num. Buffer: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_MAX_NUM_BUFFER")],
        [sg.Text('Stream Grabber Max. Transfer Size: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_MAX_TRANSFER_SIZE")],
        [sg.Text('Stream Grabber Num. Max. Queued Urbs: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_NUM_MAX_QUEUED_URBS")],
        [sg.Text('Stream Grabber Receive Thread Priority Override: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_REC_THREAD_PRIO_OVERRIDE")],
        [sg.Text('Stream Grabber Receive Thread Priority: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_REC_THREAD_PRIO")],
        [sg.Text('Stream Grabber Socket Buffer Size: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_SOC_BUFFER_SIZE")],
        [sg.Text('Stream Grabber Status: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_STATUS")],
        [sg.Text('Stream Grabber Transfer Loop Thread Priority: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_TRANSFER_LOOP_THREAD_PRIO")],
        [sg.Text('Stream Grabber Type GigE Vision Filter Driver: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_TYPE_GIGE_VISION_FILTER_DRIVER")],
        [sg.Text('Stream Grabber Firewall Traversal Interval: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_FIREWALL_TRAV_INTER")],
        [sg.Text('Get Event Grabber Firewall Traversal Interval: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="GET_EV_GRAB_FIREWALL_TRAV_INTER")],
        [sg.Text('Stream Grabber Transmission Type: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_TRANSMIS_TYPE")],
        [sg.Text('Stream Grabber Destination Port: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_DEST_PORT")],
        [sg.Button('Save Stream Grabber Details - 1st Part')]
    ]
    
    layout_params_stream_grabber_sec = [
        [sg.Text('Stream Grabber Enable Resend: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_EN_RESEND")],
        [sg.Text('Stream Grabber Packet Timeout: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_PAC_TIMEOUT")],
        [sg.Text('Stream Grabber Frame Retention: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_FRAME_RET")],
        [sg.Text('Stream Grabber Resend Request Threshold: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_RESEND_REQ_THRESH")],
        [sg.Text('Stream Grabber Receive Window Size: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_REC_WINDOW_SIZE")],
        [sg.Text('Stream Grabber Resend Request Batching: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_RESEND_REQ_BAT")],
        [sg.Text('Stream Grabber Resend Timeout: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_RESEND_TIMEOUT")],
        [sg.Text('Stream Grabber Resend Request Response Timeout: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_RESEND_REQ_RESP_TIMEOUT")],
        [sg.Text('Stream Grabber Max. Number Resend Requests: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="STREAM_GRAB_MAX_NUM_RESEND_REQS")],
        [sg.Button('Save Stream Grabber Details - 2nd Part')]
    ]
    
    layout_params_t1Par = [
        [sg.Text('T1 Parameters Read Timeout: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="T1_PAR_READ_TIMEOUT")],
        [sg.Text('T1 Parameters Write Timeout: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="T1_PAR_WRITE_TIMEOUT")],
        [sg.Text('T1 Parameters Heartbeat Timeout: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="T1_PAR_HEARTBEAT_TIMEOUT")],
        [sg.Button('Save T1 Parameters Details')]
    ]
    
    layout_other_opt = [
        [sg.Text('Sync. User Output Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="SYNC_USER_OUT_SEL")],
        [sg.Text('Sync. User Output Value: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="SYNC_USER_OUT_VAL")],
        [sg.Text('User Set Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="USER_SET_SEL")],
        [sg.Text('Default Set Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="DEFAULT_SET_SEL")],
        [sg.Text('User Set Default Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="USER_SET_DEFAULT_SEL")],        
        [sg.Text('Parameter Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="PARAM_SEL")], 
        [sg.Text('Remove Limits: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="REMOVE_LIMS")],
        [sg.Text('Auto Function AOI Usage Intensity: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="AUTO_FUNC_AOI_USAGE_INTENS")],
        [sg.Text('Auto Function AOI Usage White Balance: '), sg.InputText(default_text= "", size=(19, 1), key="AUTO_FUNC_AOI_USAGE_WHITE_BAL")],
        [sg.Text('Median Filter: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="MEDIAN_FILTER")], 
        [sg.Text('Chunck Selector: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="CHUNCK_SEL")], 
        [sg.Text('Chunck Enable: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="CHUNCK_EN")],
        [sg.Button('Save Further Details')]
    ]          
    
    layout_bandwidth = [
        [sg.Text('Bandwidth Reserve Mode: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="BANDWIDTH_RESERV_MODE")],
        [sg.Text('Device Link Throughput Limit: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="DEV_LINK_THROUGHPUT_LIM")],
        [sg.Text('Gev. Heartbeat Timeout: '), sg.InputText(disabled=True, default_text= "", size=(19, 1), key="GEV_HEARTBEAT_TIMEOUT")],
        [sg.Button('Save Bandwidth Details')]
    ]    
    
    tabgrp = [[sg.TabGroup([[sg.Tab('Headers Details', layout_headers),
                    sg.Tab('Timer Details', layout_params_first),
                    sg.Tab('Color Details', layout_params_sec),
                    sg.Tab('Stacked Zone Imaging Details', layout_params_third),
                    sg.Tab('Acquisition & Exposure Details', layout_params_fourth),
                    sg.Tab('Sequence & Decimation Details', layout_params_fifth),
                    sg.Tab('Stream Grabber Details - 1st part', layout_params_stream_grabber),
                    sg.Tab('Stream Grabber Details - 2nd part', layout_params_stream_grabber_sec),
                    sg.Tab('T1 Parameters Details', layout_params_t1Par),
                    sg.Tab('Bandwidth details', layout_bandwidth),
                    sg.Tab('Others', layout_other_opt)]])
    ]]
    
       
     
    layout_opt = [
        [sg.Button("Back"), sg.Button("Scaling")]
    ]
    
    
    layout = [
        [tabgrp], 
        [layout_opt]       
    ]
    
    ev_extra = False    
    
    window = sg.Window('Extra properties', layout)
    
    again = True
    
    while again == True:
       event, values = window.read()
       if event == "Exit" or event == sg.WIN_CLOSED:
           break
       if event == "Back":
           break
       
       if event == "Save Header Details":
           ev_extra = True
           if not values['DEV_MANUF_INFO']:
               again = False
               print("Empty field for device manufacturer info")
           else:
               print("Going to use provided device manufacturer info")
               dev_manuf_code = values['DEV_MANUF_INFO']
           
           if not values['DEV_SCAN_TYPE']:
               again = False
               print("Empty field for device scan type")
           else:
               print("Going to use provided device scan type")
               dev_scan_type = values['DEV_SCAN_TYPE']
           
           if not values['DEV_VENDOR_NAME']:
               again = False
               print("Empty field for device vendor name")
           else:
               print("Going to use provided device vendor name")
               dev_vend_name = values['DEV_VENDOR_NAME']              
       
       
       if event == "Save Timer Details":
           ev_extra = True
           if not values['TIMER_SEL']:
               again = False
               print("Empty field for timer selector")
           else:
               print("Going to use provided timer selector")
               timer_sel = values['TIMER_SEL'] 
               
               if not('Timer' in timer_sel):
                again = True
                print("Timer selector must include the key 'Timer' ")
               
           if not values['TIMER_TRIG_SOURCE']:
               again = False
               print("Empty field for timer trigger source")
           else:
               print("Going to use provided timer trigger source")
               timer_trig_source = values['TIMER_TRIG_SOURCE'] 
               
               if "TimerTriggerSource" in timer_trig_source:
               
                   if timer_trig_source != "ExposureStart" and timer_trig_source != "FlashWindowStart":
                    again = True
                    print("Timer trigger source parameter should be set to one of the following trigger source events: \n \t ExposureStart \n \t FlashWindowStart")
               else:
                    again = True
                    print("Timer trigger source must include the key 'TimerTriggerSource' ")
               
           if not values['TIMER_DELAY']: 
               again = False
               print("Empty field for timer delay")
           else:
               print("Going to use provided timer delay")
               timer_delay = values['TIMER_DELAY'] 
               
               if timer_delay.isdigit():
                   timer_delay_int = int(timer_delay)
                   
                   if int(timer_delay) < 4095:
                        again = True
                        print("Timer delay should be equal or higher than 4095")     
                   else:
                        print("Timer delay right")
                        timer_delay_ok = True
                        
               else:
                    again = True
                    print("Timer delay - Write only numbers !!!")
                
               
           if not values['TIMER_DELAY_TIMEBASE']:
               again = False
               print("Empty field for timer delay timebase")
           else:
               print("Going to use provided timer delay timebase")
               
               if timer_delay_ok == True:
                    timer_delay_ok = False
                    timer_delay_int = int(timer_delay)
                    
                    ratio_delay = int(math.ceil(timer_delay_int/4095))    

                    timer_delay_timebase = str(ratio_delay)
                    timer_delay_timebase_val = values['TIMER_DELAY_TIMEBASE']                
                    
                    if int(timer_delay_timebase_val) != ratio_delay:
                        print("Wrong timer delay timebase value ...")
                        again = True
                    
               else:                    
                    timer_delay_timebase = values['TIMER_DELAY_TIMEBASE']                
               
           if not values['TIMER_DURATION']: 
               again = False
               print("Empty field for timer duration")
           else:
               print("Going to use provided timer duration")
               timer_duration = values['TIMER_DURATION']
               
               if timer_duration.isdigit():

                   timer_duration_int = int(timer_duration)
                   
                   if timer_duration_int < 4095:
                        again = True
                        print("Timer duration be equal or higher than 4095")
                    
                   else:
                        print("Timer duration right")
                        timer_duration_ok = True    
               else:
                   again = True
                   print("Timer duration should contain only numbers")
               
             
               
           if not values['TIMER_DURATION_TIMEBASE']: 
               again = False
               print("Empty field for timer duration timebase")         
               
               
           else:
               print("Going to use provided timer duration timebase")            
               timer_duration_int = int(timer_duration)
               
               if timer_duration_ok == True:
                    timer_duration_ok = False
               
                    ratio_duration_timebase_int = int(math.floor(timer_duration_int/4095))
                    
                    timer_duration_timebase = str(ratio_duration_timebase_int)
                    timer_delay_timebase_val = values['TIMER_DURATION_TIMEBASE']                
                    
                    if int(timer_delay_timebase_val) != ratio_delay:
                        print("Wrong timer delay timebase value ...")
                        again = True
               else:                    
                    timer_duration_timebase = values['TIMER_DURATION_TIMEBASE']   
               
           
       if event == "Save Color Details":
           ev_extra = True
           if not values['COL_TRANSF_SEL']:
               again = False
               print("Empty field for color transformation selector")
           else:
               print("Going to use provided color transformation selector")
               color_transf_sel = values['COL_TRANSF_SEL']
               
               if "ColorTransformationSelector" in color_transf_sel:
               
                   if not("RGBtoRGB" in color_transf_sel) and not("RGBtoYUV" in color_transf_sel) and not("YUVtoRGB" in color_transf_sel):
                      
                      again = True
                      print("Color transformation selector should contain one of the following transformations: \t \n 'RGBtoRGB' \t \n 'RGBtoYUV' \t \n 'YUVtoRGB'")
               else:
                    again = True
                    print("Color transformation selector should contain keyword 'ColorTransformationSelector' ")
               
           if not values['COL_TRANSF_MFACTOR']:
               again = False
               print("Empty field for color transformation matrix factor")
           else:
               print("Going to use provided color transformation matrix factor")   
               color_transf_matrix_factor = values['COL_TRANSF_MFACTOR']
               
               if color_transf_matrix_factor.isdigit():
                    color_transf_matrix_factor_float = float(color_transf_matrix_factor)
                    
                    if color_transf_matrix_factor_float < 0 or color_transf_matrix_factor_float > 1:
                        again = True
                        print("Colo transformation matrix factor should be in the interval: \n \t [0, 1]")
                    
               else:
                    again = True
                    print("Clor transformation matrix factor should only contain numbers")
               
           if not values['COL_TRANSF_VAL_SEL']:
               again = False
               print("Empty field for color transformation value selector")
           else:
               print("Going to use provided color transformation value selector") 
               color_transf_value_sel = values['COL_TRANSF_VAL_SEL']
               
               if "ColorTransformationValueSelector" in color_transf_value_sel:
                    if not("Gain_" in color_transf_value_sel):
                        if (color_transf_value_sel[-1].isdigit()) and (color_transf_value_sel[-2].isdigit()):
                        
                            if int(color_transf_value_sel[-1]) < 0 or int(color_transf_value_sel[-2]) < 0 or int(color_transf_value_sel[-1]) > 2 or int(color_transf_value_sel[-2]) > 2:
                                print("Final integer numbers should be within the range [0,2]")
                            else:
                                print("Color transformation value selector right")
                        else:
                            again = True
                            print("Color transformation value selector should end with two numbers, related to its identifier")
                    else:
                        again = True
                        print("Color transformation value selector should contain keyword 'Gain_' ")
               else:
                    again = True
                    print("Color transformation value selector should contain keyword 'ColorTransformationValueSelector' ")
           
           if not values['COL_TRANSF_VAL']:
               again = False
               print("Empty field for color transformation value")
           else:
               print("Going to use provided color transformation value")
               color_transf_value = values['COL_TRANSF_VAL']
               
               if color_transf_value.isdigit():
                    color_transf_value_int = int(color_transf_value)
                    
                    if color_transf_value_int < -8 or color_transf_value_int > 7.96875:
                        again = True
                        print("Color transformation value should be within the range [-8 7.96875]")
                    else:
                        print("Color transformation value right")
               else:
                    again = True
                    print("Color transformation value should only contains numbers")
               
           if not values['COL_ADJUST_EN']:
               again = False
               print("Empty field for color adjustment enable")
           else:
               print("Going to use provided color adjustment enable") 
               color_adjust_en = values['COL_ADJUST_EN']
               
               if color_adjust_en != "true" and color_adjust_en != "True" and color_adjust_en != "false" and color_adjust_en != "False":
                   
                   again = True
                   print("Color adjustment enable should be boolean type")
               else:
                   print("Color adjustment enable right")                   
         
           if not values['COL_ADJUST_SEL']:
               again = False
               print("Empty field for color adjustment selector")
           else:
               print("Going to use provided color adjustment selector") 
               color_adjust_sel = values['COL_ADJUST_SEL']
               
               if "ColorAdjustmentSelector_" in color_adjust_sel:
                    if not("Red" in color_adjust_sel) and not("Cyan" in color_adjust_sel) and not("Green" in color_adjust_sel) and not("Blue" in color_adjust_sel) and not("Yellow" in color_adjust_sel) and not("Magenta" in color_adjust_sel):
                       
                       print("Wrong color - Choose one of the primary or secondary colours: \n Primary: Red, Green, Blue; \n Secondary: Yellow, Cyan, Magenta")
                       again = True
                    else:
                       print("Color adjustment selector right")                       
               else:
                    again = True
                    print("Color adjustment selector should contains the keyword 'ColorAdjustmentSelector' ")
                    
           if not values['COL_ADJUST_HUE']:
               again = False
               print("Empty field for color adjustment hue")
           else:
               print("Going to use provided color adjustment hue") 
               color_adjust_hue = values['COL_ADJUST_HUE']
               
               if color_adjust_hue.isdigit():
                    color_adjut_hue_float = float(color_adjust_hue)
                    
                    if color_adjut_hue_float < -4 or color_adjut_hue_float > 3.96875:
                        again = True
                        print("Color adjustment hue should be within the range [-4, 3.96875]")
                    else:
                        print("Color adjustment hue right")
               else:
                    again = True
                    print("Color adjustment hue should contains only numbers")
               
           if not values['COL_ADJUST_SAT']:
               again = False
               print("Empty field for color adjustment saturation")
           else:
               print("Going to use provided color adjustment saturation") 
               color_adjust_sat = values['COL_ADJUST_SAT']
               
               if color_adjust_sat.isdigit():
                  color_adjust_sat_float = float(color_adjust_sat)
                  
                  if color_adjust_sat_float < 0 or color_adjust_sat_float > 1.99219:
                      again = True
                      print("Color adjustment saturation should be within the range [0, 1.99219]")
                  else:
                      print("Color adjustment saturation right")
               else:              
                  again = True
                  print("Color adjustment saturation should contains only numbers")
                    
       if event == "Save Stacked Zone Imaging Details":
           ev_extra = True
           if not values['STACK_ZONE_IMAG_IND']:
               again = False
               print("Empty field for stacked zone imaging index")
           else:
               print("Going to use provided stacked zone imaging index") 
               stacked_zone_imag_index = values['STACK_ZONE_IMAG_IND']
               
               if stacked_zone_imag_index.isdigit():
                    if '.' or ',' in stacked_zone_imag_index:
                        again = True
                        print("Stacked Zone Imaging Index should be an integer")
                    else:
                        stacked_zone_imag_index_int = int(stacked_zone_imag_index)
                        
                        if stacked_zone_imag_index_int != 1 and stacked_zone_imag_index_int != 2 and stacked_zone_imag_index_int != 3:
                           
                           again = True
                           print("Stacked Zone Imaging Index should be within the range [1,3]")
                        else:
                           print("Stacked Zone Imaging Index right")
                       
               else:
                   again = True
                   print("Stacked Zone Imaging Index should only contains numbers")
           
           if not values['STACK_ZONE_IMAG_ZONE_EN']:
               again = False
               print("Empty field for stacked zone imaging zone enable")
           else:
               print("Going to use provided stacked zone imaging zone enable")
               stacked_zone_imag_zone_en = values['STACK_ZONE_IMAG_ZONE_EN']
                
               if stacked_zone_imag_zone_en != "true" and stacked_zone_imag_zone_en != "True" and stacked_zone_imag_zone_en != "false" and stacked_zone_imag_zone_en != "False":
                  
                  again = True
                  print("Stacked Zone Imaging Zone Enable should be boolean type")
               else:
                   print("Stacked Zone Imaging Zone Enable right")
                  
           if not values['STACK_ZONE_IMAG_ZONE_OFFSET_X']:
               again = False
               print("Empty field for stacked zone imaging zone offset X")
           else:
               print("Going to use provided stacked zone imaging zone offset X") 
               stacked_zone_imag_zone_offset_x = values['STACK_ZONE_IMAG_ZONE_OFFSET_X'] 
               
               if stacked_zone_imag_zone_offset_x.isdigit():
                    stacked_zone_imag_zone_offset_x_int = int(stacked_zone_imag_zone_offset_x)
                    
                    if stacked_zone_imag_zone_offset_x_int > 1920:
                        again = True
                        print("Stacked Zone Imaging Zone Offset X must not exceed maximum image dimensions")
                    else:
                        print("Stacked Zone Imaging Zone Offset X right") 
                    
               else:
                    again = True
                    print("Stacked Zone Imaging Zone Offset X should only contains numbers")
               
           if not values['STACK_ZONE_IMAG_ZONE_OFFSET_Y']:
               again = False
               print("Empty field for stacked zone imaging zone offset Y")
           else:
               print("Going to use provided stacked zone imaging zone offset Y") 
               stacked_zone_imag_zone_offset_y = values['STACK_ZONE_IMAG_ZONE_OFFSET_Y'] 
               
               if stacked_zone_imag_zone_offset_y.isdigit():
                    stacked_zone_imag_zone_offset_y_int = int(stacked_zone_imag_zone_offset_y)
                    
                    if stacked_zone_imag_zone_offset_y_int > 1920:
                        again = True
                        print("Stacked Zone Imaging Zone Offset Y must not exceed maximum image dimensions")
                    else:
                        print("Stacked Zone Imaging Zone Offset Y right")
               else:
                    again = True
                    print("Stacked Zone Imaging Zone Offset Y should only contains numbers")
           
           if not values['STACK_ZONE_IMAG_ZONE_HEIGHT']:
               again = False
               print("Empty field for stacked zone imaging zone height")
           else:
               print("Going to use provided stacked zone imaging zone height")
               stacked_zone_imag_zone_height = values['STACK_ZONE_IMAG_ZONE_HEIGHT'] 
               
               if stacked_zone_imag_zone_height.isdigit():
                    stacked_zone_imag_zone_height_int = int(stacked_zone_imag_zone_height)
                    
                    if stacked_zone_imag_zone_height_int > 1920:
                        again = True
                        print("Stacked Zone Imaging Zone Height must not exceed maximum image dimensions")
                    else:
                        print("Stacked Zone Imaging Zone Height right")
               else:
                    again = True
                    print("Stacked Zone Imaging Zone Height should only contains numbers")
           
           if not values['STACK_ZONE_IMAG_ZONE_WIDTH']: 
               again = False
               print("Empty field for stacked zone imaging zone width")
           else:
               print("Going to use provided stacked zone imaging zone width")
               stacked_zone_imag_zone_width = values['STACK_ZONE_IMAG_ZONE_WIDTH'] 
               
               if stacked_zone_imag_zone_width.isdigit():
                    stacked_zone_imag_zone_width_int = int(stacked_zone_imag_zone_width)
                    
                    if stacked_zone_imag_zone_width_int > 1920:
                        again = True
                        print("Stacked Zone Imaging Zone Width must not exceed maximum image dimensions")
                    else:
                        print("Stacked Zone Imaging Zone Width right")
               else:
                    again = True
                    print("Stacked Zone Imaging Zone Width should only contains numbers")
       
       if event == "Save Acquisition & Exposure Details":
           ev_extra = True
           if not values['ACQ_MODE']:
               again = False
               print("Empty field for acquisition mode")
           else:
               print("Going to use provided acquisition mode")
               acq_mode = values['ACQ_MODE'] 
               
               if "AcquisitionMode_" in acq_mode:
                    if not("SingleFrame" in acq_mode) and not("Continuous" in acq_mode):
                       
                       again = True
                       print("Acquisition mode should be one of the following: \n \t SingleFrame \n \t Continuous")
                       
                    else:
                        print("Acquisition mode right")                    
               else:
                    again = True
                    print("Acquisition mode should contain keyword 'AcquisitionMode_' ")
           
           if not values['EXP_OVERLAP_TIME_MAX']:
               again = False
               print("Empty field for exposition overlap time max. ")
           else:
               print("Going to use provided exposition overlap time max. ")
               exp_overlap_time_max = values['EXP_OVERLAP_TIME_MAX']
               
               if exp_overlap_time_max.isdigit():
                    exp_overlap_time_max_int = int(exp_overlap_time_max)
                    
                    if exp_overlap_time_max_int <= 0 or exp_overlap_time_max_int >= 35000:
                        again = True
                        print("Exposure Overlap Time Max. should be positive, and less than 35000 us")
                    else:
                        print("Exposure Overlap Time Max. right")                        
               else:
                    again = True
                    print("Exposure Overlap Time Max. should contain only numbers")
               
           if not values['FIELD_OUT_MODE']:
               again = False
               print("Empty field for field output mode")
           else:
               print("Going to use provided field output mode")               
               field_output_mode = values['FIELD_OUT_MODE'] 
               
               if (not "Field0" in field_output_mode) and (not "Field1" in  field_output_mode) and (not "ConcatenatedNewFields" in field_output_mode) and (not "DeinterlacedNewFields" in field_output_mode):
                  
                  again = True
                  print("Field output mode should be one of the following: \n \t 'Field0' \n \t 'Field1' \n \t 'ConcatenatedNewFields' \n \t 'DeinterlacedNewFields' ")
               else:
                    print("Field output mode right")                
           
           if not values['GL_RES_REL_MODE_EN']:
               again = False
               print("Empty field for global reset release mode enable")
           else:
               print("Going to use provided global reset release mode enable")
               global_reset_release_mode_en = values['GL_RES_REL_MODE_EN']

               if global_reset_release_mode_en != "true" and global_reset_release_mode_en != "True" and global_reset_release_mode_en != "false" and global_reset_release_mode_en != "False":
                  
                  again = True
               else:
                  print("Global reset release mode enable right") 
           
           if not values['ACQ_STATUS_SEL']: 
               again = False
               print("Empty field for acquisition status selector")
           else:
               print("Going to use provided acquisition status selector")   
               acq_status_sel = values['ACQ_STATUS_SEL'] 
               
               if "AcquisitionStatusSelector_" in acq_status_sel:
                    if not("AcquisitionActive" in acq_status_sel) and not("AcquisitionIdle" in acq_status_sel) and not("ExposureActive" in acq_status_sel) and not("ExposureTriggerWait" in acq_status_sel) and not("FrameBurstActive" in acq_status_sel) and not("AcquisitionTriggerWait" in acq_status_sel) and not("FrameTriggerWait" in acq_status_sel):
                       
                       again = True
                       print("Acquisition status selector should be one of the following")
                       print("\t AcquisitionActive")
                       print("\t AcquisitionIdle")
                       print("\t ExposureActive")
                       print("\t ExposureTriggerWait")
                       print("\t FrameBurstActive")
                       print("\t AcquisitionTriggerWait")
                       print("\t FrameTriggerWait")
                       
                    else:
                       print("Acquisition status selector right")                       
               else:
                    again = True
                    print("Acquisition status selector should contain keyword 'AcquisitionStatusSelector_' ")
               
           if not values['BAL_RATIO_SEL']:
               again = False
               print("Empty field for balance ratio selector")
           else:
               print("Going to use provided balance ratio selector")  
               bal_ratio_sel = values['BAL_RATIO_SEL'] 
               
               if "BalanceRatioSelector_" in bal_ratio_sel:
                    if not("Red" in bal_ratio_sel) and not("Green"in bal_ratio_sel) and not("Blue" in bal_ratio_sel):
                       
                       again = True
                       print("Balance RatiO Selector should be one of the following: \n \t 'Red' \n \t 'Green' \n \t 'Blue' ")  
                    else:
                        print("Balance Ratio Selector right")
                        
               else:
                    again = True
                    print("Balance Ratio Selector should contain keyword 'BalanceRatioSelector_' ")
           
           if not values['BAL_RATIO']:
               again = False
               print("Empty field for balance ratio")
           else:
               print("Going to use provided balance ratio")
               bal_ratio = values['BAL_RATIO'] 
               
               if bal_ratio.isdigit():
                    bal_ratio_float = float(bal_ratio)
                    
                    if bal_ratio < 1 or bal_ratio > 15.9844:
                        again = True
                        print("Balance Ratio should be within the range [1, 15.9844]")
                    else:
                        print("Balance Ratio right")
                    
               else:
                    again = True
                    print("Balance Ratio should only contain numbers")
               
           if not values['PROC_RAW_EN']: 
               again = False
               print("Empty field for processed raw enable")
           else:
               print("Going to use provided processed raw enable") 
               process_raw_en = values['PROC_RAW_EN'] 
               
               if process_raw_en != "true" and process_raw_en != "True" and process_raw_en != "false" and process_raw_en != "False":
                  
                  again = True
               else:
                  print("Processed Raw Enable right") 
               
           if not values['LIGHT_SOURCE_SEL']:
               again = False
               print("Empty field for light source selector")
           else:
               print("Going to use provided light source selector")   
               light_source_sel = values['LIGHT_SOURCE_SEL'] 
               
               if "LightSourceSelector_" in light_source_sel:
                    if not("Off" in light_source_sel) and not("Tungsten" in light_source_sel) and not("Daylight" in light_source_sel) and not("Daylight6500K" in light_source_sel):
                       
                       again = True
                       print("Light source selector should one of the following: \n \t 'Off' \n \t 'Tungsten' \n \t 'Daylight' \n \t 'Daylight6500K' ")
                    else:
                        print("Light source selector right")
               else:
                    again = True
                    print("Light source selector should contain keyword 'LightSourceSelector_' ")               
            
       if event == "Save Sequence & Decimation Details": 
           ev_extra = True
           if not values['SEQ_CONTROL_SEL']:
               again = False
               print("Empty field for sequence control selector")
           else:
               print("Going to use provided sequence control selector") 
               seq_control_sel = values['SEQ_CONTROL_SEL']
               
               if "SequenceControlSelector_" in seq_control_sel:
                    if not("Advance" in seq_control_sel) and not("Restart" in seq_control_sel):
                       
                       again = True
                       print("Sequence Control Selector must stand to one of the following modes:")
                       print("\t Advance")
                       print("\t Restart")
                    else:
                        print("Sequence Control Selector right")
               else:
                    again = True
                    print("Sequence control selector should contain keyword 'SequenceControlSelector_' ")
                
           if not values['SEQ_ADDRESS_BIT_SEL']:
               again = False
               print("Empty field for sequence address bit selector")
           else:
               print("Going to use provided sequence address bit selector") 
               seq_address_bit_sel = values['SEQ_ADDRESS_BIT_SEL'] 
               
               if "SequenceAddressBitSelector_" in seq_address_bit_sel:
                    if "Bit" in seq_address_bit_sel:
                        if seq_address_bit_sel[-1].isdigit():
                            print("Sequence address bit selector right")
                        else:
                            again = True
                            print("Last element should be a number, to indicate the selected bit")
               else:
                   again = True
                   print("Sequence address bit selector should contain keyword 'SequenceAddressBitSelector_' ")
           
           if not values['SEQ_CONTROL_SOURCE']:
               again = False
               print("Empty field for sequence control source")
           else:
               print("Going to use provided sequence control source")
               seq_control_source = values['SEQ_CONTROL_SOURCE']
               
               opts_seq_control_source = ['AlwaysActive', 'CC1', 'CC2', 'CC3', 'CC4', 'Disabled', 'Line1', 'Line2', 'Line3', 'Line4',
                                          'Line5', 'Line6', 'Line7', 'Line8', 'VInput1', 'VInput2', 'VInput3', 'VInput4', 'VInputDecActive']

               if "SequenceControlSource_" in seq_control_source:               
                    not_element = 0
                    for opt in opts_seq_control_source:
                        if(not opt in seq_control_source):
                            not_element += 1
                            
                    if not_element == len(opts_seq_control_source):
                        again = True
                        print("Sequence Control Source must stand to one of the following modes:")
                        
                        for opt in opts_seq_control_source:
                            print("\t " + opt)
                    else:
                        print("Sequence Control Source right")
               else:
                    again = True
                    print("Sequence Control Source should contain keyword 'SequenceControlSource_' ")
           
           if not values['SEQ_ADDRESS_BIT_SOURCE']:
               again = False
               print("Empty field for sequence address bit source")
           else:
               print("Going to use provided sequence address bit source")
               seq_address_bit_source =  values['SEQ_ADDRESS_BIT_SOURCE'] 
               
               if "SequenceAddressBitSource_" in seq_address_bit_source:
                    if "Line" in seq_address_bit_source:
                        if seq_address_bit_source[-1].isdigit():
                            print("Sequence Address Bit Source right")
                        else:
                            again = True
                            print("Line specification should be a number")
                        
                    else:
                        again = True
                        print("Sequence Address Bit Source must stand to a specific line")
               else:
                    again = True
                    print("Sequence Address Bit Source should contain keyword 'SequenceAddressBitSource_' ")
               
           if not values['HORIZ_DECIM']:
               again = False
               print("Empty field for horizontal decimation") 
           else:
               print("Going to use provided horizontal decimation")
               horiz_decim = values['HORIZ_DECIM'] 
               
               if horiz_decim.isdigit():
                    horiz_decim_int = int(horiz_decim)
                    
                    if horiz_decim_int == 1:
                        print("Disabling decimation for this axis ...")
                    else:
                        print("Horizontal decimation right")
               else:
                    again = True
                    print("Horizontal decimation should be a number")
               
           if not values['VERT_DECIM']:
               again = False
               print("Empty field for vertical decimation")
           else:
               print("Going to use provided vertical decimation")
               vert_decim = values['VERT_DECIM'] 
               
               if vert_decim.isdigit():
                    vert_decim_int = int(vert_decim)
                    
                    if vert_decim_int == 1:
                        print("Disabling decimation for this axis ...")
                    else:
                        print("Vertical decimation right")
                    
               else:
                    again = True
                    print("Vertical decimation should be a number")                    
               
       if event == "Scaling":
           w = False
           h = False   
           
           width_int = 0
           height_int = 0
           
           while w == False:
           
               width = input('Width: ')
               
               if width.isdigit():
                   
                   width_int = int(width)
                   
                   if width_int <= 1920:
                       w = True
                       break
                   else:
                       print("Width must be equal or less than 1920")
                       print("Try again ...")
               else:
                   print("Insert only numbers !!!")
                   
           while h == False:
           
               height = input('Height: ')
               
               if height.isdigit():
                   
                   height_int = int(height)
                    
                   if height_int <= 1920:
                       h = True
                       break
                   else:
                       print("Height must be equal or less than 1080")
                       print("Try again ...")
               else:
                   print("Insert only numbers !!!")              
          
           scal_data = scaling_gui(width_int, height_int) 
           scal_horiz_val = scal_data[0] 
           scal_vert_val = scal_data[1]
           
           edit_scaling(scal_horiz_val, scal_vert_val, camera)  
           
      #     break  
           
       if event == "Save Stream Grabber Details - 1st Part":          
             ev_extra = True
             if not values['STREAM_GRAB_ACCESS_MODE']:
                   again = False
                   print("Empty field for stream grabber access mode")
             else:
                   print("Going to use provided stream grabber access mode")
                   stream_grabber_access_mode = values['STREAM_GRAB_ACCESS_MODE'] 
             
             if not values['STREAM_GRAB_AUTO_PACKET_SIZE']:
                   again = False
                   print("Empty field for stream grabber auto packet size")
             else:
                   print("Going to use provided stream grabber auto packet size")
                   stream_grabber_auto_packet_size = values['STREAM_GRAB_AUTO_PACKET_SIZE'] 
             
             if not values['STREAM_GRAB_MAX_BUFFER_SIZE']:
                   again = False
                   print("Empty field for stream grabber max. buffer size")
             else:
                   print("Going to use provided stream grabber max. buffer size")
                   stream_grabber_max_buffer_size = values['STREAM_GRAB_MAX_BUFFER_SIZE']  
             
             if not values['STREAM_GRAB_MAX_NUM_BUFFER']:
                   again = False
                   print("Empty field for stream grabber max. num. buffer")
             else:
                   print("Going to use provided stream grabber max. num. buffer")
                   stream_grabber_max_num_buffer = values['STREAM_GRAB_MAX_NUM_BUFFER']  
             
             if not values['STREAM_GRAB_MAX_TRANSFER_SIZE']:
                   again = False
                   print("Empty field for stream grabber max. transfer size")
             else:
                   print("Going to use provided stream grabber max.transfer size")
                   stream_grabber_max_transfer_size = values['STREAM_GRAB_MAX_TRANSFER_SIZE']  
             
             if not values['STREAM_GRAB_NUM_MAX_QUEUED_URBS']:
                   again = False
                   print("Empty field for stream grabber num. max. queued urbs.")
             else:
                   print("Going to use provided stream grabber num. max. queued urbs.")
                   stream_grabber_num_max_queued_urbs = values['STREAM_GRAB_NUM_MAX_QUEUED_URBS']
             
             if not values['STREAM_GRAB_REC_THREAD_PRIO_OVERRIDE']:
                   again = False
                   print("Empty field for stream grabber receive thread priority override")
             else:
                   print("Going to use provided stream grabber receive thread priority override.")
                   stream_grabber_rec_thread_prio_override = values['STREAM_GRAB_REC_THREAD_PRIO_OVERRIDE']            
             
             if not values['STREAM_GRAB_REC_THREAD_PRIO']:
                   again = False
                   print("Empty field for stream grabber receive thread priority")
             else:
                   print("Going to use provided stream grabber receive thread priority.")
                   stream_grabber_rec_thread_prio = values['STREAM_GRAB_REC_THREAD_PRIO']
             
             if not values['STREAM_GRAB_SOC_BUFFER_SIZE']:
                   again = False
                   print("Empty field for stream grabber socket buffer size")
             else:
                   print("Going to use provided stream grabber socket buffer size.")
                   stream_grabber_soc_buffer_size = values['STREAM_GRAB_SOC_BUFFER_SIZE']
             
             if not values['STREAM_GRAB_STATUS']:
                   again = False
                   print("Empty field for stream grabber status")
             else:
                   print("Going to use provided stream grabber status.")
                   stream_grabber_status = values['STREAM_GRAB_STATUS']
             
             if not values['STREAM_GRAB_TRANSFER_LOOP_THREAD_PRIO']:
                   again = False
                   print("Empty field for stream grabber transfer loop thread priority")
             else:
                   print("Going to use provided stream grabber transfer loop thread priority.")
                   stream_grabber_transfer_loop_thread_priority = values['STREAM_GRAB_TRANSFER_LOOP_THREAD_PRIO']
                   
             if not values['STREAM_GRAB_TYPE_GIGE_VISION_FILTER_DRIVER']:
                   again = False
                   print("Empty field for stream grabber type GigE vision filter driver")
             else:
                   print("Going to use provided stream grabber type GigE vision filter driver.")
                   stream_grabber_type_gige_vision_filter_driver = values['STREAM_GRAB_TYPE_GIGE_VISION_FILTER_DRIVER']

             if not values['STREAM_GRAB_FIREWALL_TRAV_INTER']:
                   again = False
                   print("Empty field for stream grabber firewall traversal interval")
             else:
                   print("Going to use provided stream grabber firewall traversal interval.")
                   stream_grabber_firewall_traversal_interval = values['STREAM_GRAB_FIREWALL_TRAV_INTER']
                   
             if not values['GET_EV_GRAB_FIREWALL_TRAV_INTER']:
                    again = False
                    print("Empty field for get event grabber firewall traversal interval")
             else:
                   print("Going to use provided get event grabber firewall traversal interval.")
                   get_event_grabber_firewall_traversal_interval = values['GET_EV_GRAB_FIREWALL_TRAV_INTER']
             
             if not values['STREAM_GRAB_TRANSMIS_TYPE']:
                    again = False
                    print("Empty field for stream grabber transmission type")
             else:
                   print("Going to use provided stream grabber transmission type.")
                   stream_grabber_transmission_type = values['STREAM_GRAB_TRANSMIS_TYPE']
             
             if not values['STREAM_GRAB_DEST_PORT']:
                    again = False
                    print("Empty field for stream grabber destination port")
             else:
                   print("Going to use provided stream grabber destination port.")
                   stream_grabber_destination_port = values['STREAM_GRAB_DEST_PORT']
        
      
       if event == "Save Stream Grabber Details - 2nd Part":
           ev_extra = True
           if not values['STREAM_GRAB_EN_RESEND']:
               again = False
               print("Empty field for stream grabber enable resend")
           else:
               print("Going to use provided stream grabber enable resend")
               stream_grabber_en_resend = values['STREAM_GRAB_EN_RESEND'] 
               
               if stream_grabber_en_resend != "true" and stream_grabber_en_resend != "True" and stream_grabber_en_resend != "false" and stream_grabber_en_resend != "False":
                  
                  again = True
               else:
                  print("Stream Grabber Enable Resend right") 
               
           if not values['STREAM_GRAB_PAC_TIMEOUT']:
               again = False
               print("Empty field for stream grabber packet timeout")
           else:
               print("Going to use provided stream grabber packet timeout")
               stream_grabber_packet_timeout = values['STREAM_GRAB_PAC_TIMEOUT'] 
               
               if stream_grabber_packet_timeout.isdigit():
                    stream_grabber_packet_timeout = int(stream_grabber_packet_timeout)
                    
                    if stream_grabber_packet_timeout <= 10000:
                        again = True
                        print("Stream Grabber Packet Timeout should be higher than the time interval set for the inter-packet delay")
                    else:
                        print("Stream Grabber Packet Timeout right")
               else:
                    again = True
                    print("Stream Grabber Packet Timeout should be a number")
               
           if not values['STREAM_GRAB_FRAME_RET']:
               again = False
               print("Empty field for stream grabber frame retention")
           else:
               print("Going to use provided stream grabber frame retention")
               stream_grabber_frame_retention = values['STREAM_GRAB_FRAME_RET']
               
               if stream_grabber_frame_retention.isdigit():
                    stream_grabber_frame_retention_int = int(stream_grabber_frame_retention)
                    
                    if stream_grabber_frame_retention_int <= 0:
                        again = True
                        print("Stream grabber frame retention must be positive")
                    else:
                        print("Stream grabber frame retention right")                    
               else:
                    again = True
                    print("Stream grabber frame retention should be a number")
               
           if not values['STREAM_GRAB_RESEND_REQ_THRESH']: 
               again = False
               print("Empty field for stream grabber resend request threshold")
           else:
               print("Going to use provided stream grabber resend request threshold")
               stream_grabber_resend_req_thresh = values['STREAM_GRAB_RESEND_REQ_THRESH']
               
               if stream_grabber_resend_req_thresh.isdigit():
                    stream_grabber_resend_req_thresh_int = int(stream_grabber_resend_req_thresh)
                    
                    if stream_grabber_resend_req_thresh_int <= 0:
                        again = True
                        print("Stream grabber resend request threshold must be positive")
                    else:
                        print("Stream grabber resend request threshold right")
               else:
                    again = True
                    print("Stream grabber resend request threshold should be a number")               
                
           if not values['STREAM_GRAB_REC_WINDOW_SIZE']:
               again = False
               print("Empty field for stream grabber receive window size")
           else:
               print("Going to use provided stream grabber receive window size")
               stream_grabber_rec_window_size = values['STREAM_GRAB_REC_WINDOW_SIZE']
               
               if stream_grabber_rec_window_size.isdigit():
                    stream_grabber_rec_window_size_int = int(stream_grabber_rec_window_size)
                    
                    resp_perf_square = is_perfect_square(stream_grabber_rec_window_size_int)
                    
                    if resp_perf_square == False:
                        again = True
                        print("Stream grabber receive window size should be a perfect square (e.g. 4, 9, 16, 25)")                        
                    else:
                        print("Stream grabber receive window size right")
               else:
                    again = True
                    print("Stream grabber receive window size should be a number")
               
           if not values['STREAM_GRAB_RESEND_REQ_BAT']:
               again = False
               print("Empty field for stream grabber resend request batching")
           else:
               print("Going to use provided stream grabber resend request batching")
               stream_grabber_resend_req_bat = values['STREAM_GRAB_RESEND_REQ_BAT']
               
               if stream_grabber_resend_req_bat.isdigit():
                    stream_grabber_resend_req_bat_int = int(stream_grabber_resend_req_bat)
                    
                    if stream_grabber_resend_req_bat_int <= 0:
                        again = True
                        print("Stream grabber resend request batching must be positive")
                    else:
                        print("Stream grabber resend request batching right")
               else:
                   again = True
                   print("Stream grabber resend request batching should be a number") 
               
           if not values['STREAM_GRAB_RESEND_TIMEOUT']:
               again = False
               print("Empty field for stream grabber resend timeout")              
           else:
               print("Going to use provided stream grabber resend timeout")
               stream_grabber_resend_timeout = values['STREAM_GRAB_RESEND_TIMEOUT']
               
               if stream_grabber_resend_timeout.isdigit():
                    stream_grabber_resend_timeout_int = int(stream_grabber_resend_timeout)
                    
                    if stream_grabber_resend_timeout_int <= 0:
                        again = True
                        print("Stream grabber resend timeout must be positive")         
                    else:
                        print("Stream grabber resend timeout right")
               else:
                    again = True
                    print("Stream grabber resend timeout should be a number") 
               
           if not values['STREAM_GRAB_RESEND_REQ_RESP_TIMEOUT']:
               again = False
               print("Empty field for stream grabber resend request response timeout")
           else:
               print("Going to use provided stream grabber resend request response timeout")
               stream_grabber_resend_req_resp_timeout = values['STREAM_GRAB_RESEND_REQ_RESP_TIMEOUT']
               
               if stream_grabber_resend_req_resp_timeout.isdigit():
                    stream_grabber_resend_req_resp_timeout_int = int(stream_grabber_resend_req_resp_timeout)
                    
                    if stream_grabber_resend_req_resp_timeout_int <= 0:
                        again = True
                        print("Stream grabber resend request response timeout must be positive")
                    else:
                        print("Stream grabber resend request response timeout right")
               else:
                    again = True
                    print("Stream grabber resend request response timeout should be a number")
           
           if not values['STREAM_GRAB_MAX_NUM_RESEND_REQS']:
               again = False
               print("Empty field for stream grabber max. number of resend requests")
           else:
               print("Going to use provided stream grabber max. number of resend requests")  
               stream_grabber_max_num_resend_reqs = values['STREAM_GRAB_MAX_NUM_RESEND_REQS']
               
               if stream_grabber_max_num_resend_reqs.isdigit():
                    stream_grabber_max_num_resend_reqs_int = int(stream_grabber_max_num_resend_reqs)
                    
                    if stream_grabber_max_num_resend_reqs_int <= 0:
                        again = True
                        print("Stream grabber max. number of resend requests must be positive")
                    else:
                        print("Stream grabber max. number of resend requests right")
               else:
                    again = True
                    print("Stream grabber max. number of resend requests should be a number")
           
       
       if event == "Save T1 Parameters Details":   
           ev_extra = True
           if not values['T1_PAR_READ_TIMEOUT']:
               again = False
               print("Empty field for T1 parameters read timeout")
           else:
               print("Going to use provided T1 parameters read timeout")
               t1_param_read_timeout = values['T1_PAR_READ_TIMEOUT']
               
               if t1_param_read_timeout.isdigit():
                    t1_param_read_timeout_int = int(t1_param_read_timeout)
                    
                    if t1_param_read_timeout_int <= 0:
                        again = True
                        print("T1 parameters read timeout must be positive")
                    else:
                        print("T1 parameters read timeout right")
               else:
                    again = True
                    print("T1 parameters read timeout should be a number")
           
           if not values['T1_PAR_WRITE_TIMEOUT']:
               again = False 
               print("Empty field for T1 parameters write timeout")
           else: 
               print("Going to use provided T1 parameters write timeout")
               t1_param_write_timeout = values['T1_PAR_WRITE_TIMEOUT']
               
               if t1_param_write_timeout.isdigit():
                    t1_param_write_timeout_int = int(t1_param_write_timeout)
                    
                    if t1_param_write_timeout_int <= 0:
                        again = True
                        print("T1 parameters write timeout must be positive")
                    else:
                        print("T1 parameters write timeout right")
               else:
                    again = True
                    print("T1 parameters write timeout should be a number")
               
           if not values['T1_PAR_HEARTBEAT_TIMEOUT']:
               again = False
               print("Empty field for T1 parameters heartbeat timeout")
           else:
               print("Going to use provided T1 parameters heartbeat timeout")
               t1_param_heartbeat_timeout = values['T1_PAR_HEARTBEAT_TIMEOUT']
               
               if t1_param_heartbeat_timeout.isdigit():
                    t1_param_heartbeat_timeout_int = int(t1_param_heartbeat_timeout)
                    
                    if t1_param_heartbeat_timeout_int <= 0:
                        again = True
                        print("T1 parameters heartbeat timeout must be positive")
                    else:
                        print("T1 parameters heartbeat timeout right")
               else:
                    again = True
                    print("T1 parameters heartbeat timeout should be a number")           
           
       if event == "Save Bandwidth Details":
           ev_extra = True
           if not values['BANDWIDTH_RESERV_MODE']:
               again = False
               print("Empty field for bandwidth reserve mode")
           else:
               print("Going to use provided bandwidth reserve mode")
               bw_res_mode = values['BANDWIDTH_RESERV_MODE']
               
               if "BandwidthReserveMode_" in bw_res_mode:
                    if not ("Standard" in bw_res_mode) and not ("Performance" in bw_res_mode) and not ("Manual" in bw_res_mode):
                       
                       again = True
                       print("Bandwidth reserve mode must stnad to one of the following modes: ")
                       print("\t Standard")
                       print("\t Performance")
                       print("\t Manual")
                    else:
                        print("Bandwidth reserve mode right")
               else:
                    again = True
                    print("Bandwidth reserve mode should contain keyword 'BandwidthReserveMode_' ")
           
           if not values['DEV_LINK_THROUGHPUT_LIM']:
               again = False
               print("Empty field for device link throughput limit")
           else:
               print("Going to use provided device link throughput limit")
               dev_link_throughput_lim = values['DEV_LINK_THROUGHPUT_LIM']
               
               if "DeviceLinkThroughputLimit_" in dev_link_throughput_lim:
                    if not ("On" in dev_link_throughput_lim) and not ("Off" in dev_link_throughput_lim):
                       
                       again = True
                       print("Device link thoughput limit must stand to one of the following modes: ")
                       print("\t On")
                       print("\t Off")
                    else:
                       print("Device link thoughput limit right")
               else:
                    again = True
                    print("Device link thoughput limit should contain keyword 'DeviceLinkThroughputLimit_' ")
           
           if not values['GEV_HEARTBEAT_TIMEOUT']:
               again = False
               print("Empty field for gev. heartbeat timeout")
           else:
               print("Going to use provided gev. heartbeat timeout")
               gevHeartbeatTimeout = values['GEV_HEARTBEAT_TIMEOUT']
               
               if gevHeartbeatTimeout.isdigit():
                    gevHeartbeatTimeout_int = int(gevHeartbeatTimeout)
                    
                    if gevHeartbeatTimeout_int < 3000:
                        again = True
                        print("Gev. heartbeat timeout must be higher than the default heartbeat timeout ( > 3000 ms)")
                    else:
                        print("Gev. heartbeat timeout right")
               else:
                    again = True
                    print("Gev. heartbeat timeout should only contain numbers")  
                    
                    
       if event == "Save Further Details": 
           ev_extra = True
           if not values['SYNC_USER_OUT_SEL']:
               again = False
               print("Empty field for sync. user output selector")
           else:
               print("Going to use provided sync. user output selector")
               sync_user_output_sel = values['SYNC_USER_OUT_SEL']
               
               if "SyncUserOutputSelector_" in sync_user_output_sel:
                    if ("SyncUserOutput" in sync_user_output_sel) and sync_user_output_sel[-1].isdigit():
                        print("Sync. user output selector right")
                    else:
                        again = True
                        print("Last element must be a number, to indicate the sync. user output number")                        
               else:
                    again = True
                    print("Sync. user output selector should contain keyword 'SyncUserOutputSelector_' ")
               
           if not values['SYNC_USER_OUT_VAL']:
               again = False
               print("Empty field for sync. user output value")
           else:
               print("Going to use provided sync. user output value")
               sync_user_output_value = values['SYNC_USER_OUT_VAL']
               
               if sync_user_output_value != "true" and sync_user_output_value != "True" and sync_user_output_value != "false" and sync_user_output_value != "False":
                  
                  again = True
               else:
                  print("Sync. user output value right") 
               
           if not values['USER_SET_SEL']:
               again = False
               print("Empty field for user set selector")
           else:
               print("Going to use provided user set selector")               
               user_set_sel = values['USER_SET_SEL'] 
               
               if "UserSetSelector_" in user_set_sel:
                    if ("UserSet" in user_set_sel) and user_set_sel[-1].isdigit():
                        print("")
                    else:
                        again = True
                        print("Last element must be a number, to indicate the user set number")
               else:
                    again = True
                    print("User set selector should contain keyword 'UserSetSelector_' ")
               
           if not values['DEFAULT_SET_SEL']:
               again = False
               print("Empty field for default set selector")
           else:
               print("Going to use provided default set selector")
               default_set_sel = values['DEFAULT_SET_SEL']
               
               if "DefaultSetSelector_" in default_set_sel:
                    if not("Standard" in default_set_sel) and not("HighGain" in default_set_sel) and not("AutoFunctions" in default_set_sel) and not("Color" in default_set_sel):
                       
                       again = True
                       print("Default set selector must stand to one of the following modes:")
                       print("\t Standard")
                       print("\t HighGain")
                       print("\t AutoFunctions")
                       print("\t Color")
                    else:
                        print("Default set selector right")
               else:
                    again = True
                    print("Default set selector should contain keyword 'DefaultSetSelector_' ")
               
           if not values['USER_SET_DEFAULT_SEL']:
               again = False
               print("Empty field for user set default selector")
           else:
               print("Going to use provided user set default selector")
               user_set_default_sel = values['USER_SET_DEFAULT_SEL']
               
               if "UserSetDefaultSelector_" in user_set_default_sel:
                    user_set_default_sel_parts = user_set_default_sel.split('_')                    
                    user_set_default_sel_rem = user_set_default_sel_parts[-1]
                    
                    if not("Default" in user_set_default_sel_rem):
                        again = True
                        print("User set default selector must stand to the 'Default' one")
                    else:
                        print("User set default selector right")
               else:
                    again = True
                    print("User set default selector should contain keyword 'UserSetDefaultSelector_' ")
               
           if not values['PARAM_SEL']:
               again = False
               print("Empty field for parameter selector")
           else:
               print("Going to use provided parameter selector")
               params_sel = values['PARAM_SEL']
               
               if "ParameterSelector_" in params_sel:
                    if params_sel[-1] == '_':
                        again = True
                        print("Parameter not specified")
                    else:
                        print("Parameter selector right")
               else:
                    again = True
                    print("Parameter selector should contain keyword 'ParameterSelector_'")
               
           if not values['REMOVE_LIMS']: 
               again = False
               print("Empty field for remove limits")
           else:
               print("Going to use provided remove limits")
               remove_limits = values['REMOVE_LIMS']
               
               if remove_limits != "true" and remove_limits != "True" and remove_limits != "false" and remove_limits != "False":
                  
                  again = True
               else:
                  print("Remove limits right")              
          
           if not values['AUTO_FUNC_AOI_USAGE_INTENS']:
               again = False
               print("Empty field for auto function AOI usage intensity")
           else:
               print("Going to use provided auto function AOI usage intensity")
               auto_func_aoi_usage_intens = values['AUTO_FUNC_AOI_USAGE_INTENS']
               
               if auto_func_aoi_usage_intens != "true" and auto_func_aoi_usage_intens != "True" and auto_func_aoi_usage_intens != "false" and auto_func_aoi_usage_intens != "False":
                  
                  again = True
               else:
                  print("Auto function AOI usage intensity right")
               
           if not values['AUTO_FUNC_AOI_USAGE_WHITE_BAL']:
               again = False
               print("Empty field for auto function AOI usage white balance")
           else:
               print("Going to use provided auto function AOI usage white balance")
               auto_func_aoi_usage_white_bal = values['AUTO_FUNC_AOI_USAGE_WHITE_BAL']
               
               if auto_func_aoi_usage_white_bal != "true" and auto_func_aoi_usage_white_bal != "True" and auto_func_aoi_usage_white_bal != "false" and auto_func_aoi_usage_white_bal != "False":
                  
                  again = True
               else:
                  print("Auto function AOI usage white balance right")
               
           if not values['MEDIAN_FILTER']:
               again = False
               print("Empty field for median filter")
           else:
               print("Going to use provided median filter") 
               median_filter = values['MEDIAN_FILTER']
               
               if median_filter != "true" and median_filter != "True" and median_filter != "false" and median_filter != "False":
                  
                  again = True
               else:
                  print("Mean filter option right")               
               
           if not values['CHUNCK_SEL']:
               again = False
               print("Empty field for chunck selector")
           else:
               print("Going to use provided chunck selector")
               chunck_sel = values['CHUNCK_SEL']
               
               if 'ChunkSelector_' in chunck_sel:
                    if chunck_sel[-1] == '_':
                        again = True
                        print("Parameter for chunck selector not specified")
                    else:
                        print("Chunck selector right")
               else:
                    again = True
                    print("Chunck selector should contain keyword 'ChunkSelector_' ")
               
           if not values['CHUNCK_EN']:
               again = False
               print("Empty field for chunck enable")
           else:
               print("Going to use provided chunck enable") 
               chunck_en = values['CHUNCK_EN']
               
               if chunck_en != "true" and chunck_en != "True" and chunck_en != "false" and chunck_en != "False":
                  
                  again = True
               else: 
                  print("Chunck Enable right")           
    
                  
    if ev_extra == True:
        
        window.close()
               
        params_extra = [dev_manuf_code, dev_scan_type, dev_vend_name,
                        timer_sel, timer_trig_source, timer_delay, timer_delay_timebase, timer_duration, timer_duration_timebase,
                        color_transf_sel, color_transf_matrix_factor, color_transf_value_sel, color_transf_value, color_adjust_en, color_adjust_sel, color_adjust_hue, color_adjust_sat,
                        stacked_zone_imag_index, stacked_zone_imag_zone_en, stacked_zone_imag_zone_offset_x, stacked_zone_imag_zone_offset_y, stacked_zone_imag_zone_height, stacked_zone_imag_zone_width,
                        acq_mode, exp_overlap_time_max, field_output_mode, global_reset_release_mode_en, acq_status_sel, bal_ratio_sel, bal_ratio, process_raw_en, light_source_sel,
                        seq_control_sel, seq_address_bit_sel, seq_control_source, seq_address_bit_source, horiz_decim, vert_decim, 
                        stream_grabber_en_resend, stream_grabber_packet_timeout, stream_grabber_frame_retention, stream_grabber_resend_req_thresh, stream_grabber_rec_window_size, stream_grabber_resend_req_bat, stream_grabber_resend_timeout, stream_grabber_resend_req_resp_timeout, stream_grabber_max_num_resend_reqs,
                        t1_param_read_timeout, t1_param_write_timeout, t1_param_heartbeat_timeout,
                        sync_user_output_sel, sync_user_output_value, user_set_sel, default_set_sel, user_set_default_sel, params_sel, remove_limits, auto_func_aoi_usage_intens, auto_func_aoi_usage_white_bal, median_filter, chunck_sel, chunck_en]
        
        use_extra_params(params_extra, camera) 
        
        return 1
        
    else:
        params_extra = None  
        window.close()
        
        return 0
        
    

## extra_params_gui()
    
    
    