# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:46:26 2022

@author: marco
"""

from study_calib_params_aux import calib_camera_part, write_params_to_metadata_file_part, ask_user_metadata_filename_part, write_to_csv_calib_params_database_part, write_data_to_csv_file_part, gui_png_file_part
from optional_extra_prop import optional_prop_norm
import keyboard
from soft_single_test import videoAnalysis_single

def whole_processing_software(frame_rate, packet_size, inter_packet_delay, bandwidth_resv, bandwidth_resv_acc, gain_raw, exp_time, rec_time, image_height, image_width, decisorLevel, dir_path, curTest, numberTests, gap_bet_tests):

    print("Starting acquisition and processing steps ...")
    
    import cv2 
    from pypylon import pylon
    import numpy as np 
    import glob 
    import string 
    import time 
    import os  
    import keyboard  
    from countdown_timer import countdown_timer_display
 ##   from pre_processing import type_img
##    from find_dice_coe_along_image_sequence import dice_coeff
    
    numberImages = rec_time*frame_rate
    
    numberImages += 1
     
    print("Main imports gone nice")
    
    from softwareImageProcessingForLaserSpeckleAnalysisFinal import videoAnalysis
    
    print("OK 1")
    
    from getCurrentDateAndTime import getDateTimeStrMarker  
     
    print("OK 2")
    
    from show_outputToWeb import showLiveImageGUI
    
    import threading
    
    print("OK 3")
     
    print("Project imports gone nice")  
    
    enable_save_timestamps = optional_prop_norm()
    
    alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase)
    
    counterTest = 0
    key = False
    key_not_pressed = 0
    
    unique_test = False
    dat_unique = []
    
    time_gap_min = 30    
     
    run_again = True
    
    # def show_window(closeProp):
        
    #     import PySimpleGUI as sg
        
    #     window = sg.Window('Alert', [[sg.Text("Press any leter from the keyboard to continue ...")]], element_justification='c')
        
    #     if not closeProp:          
    #         event, values = window.read()
        
    #     if closeProp:
    #         window.close()
    
    # import tkinter as tk
    
    # root = tk.Tk()
    # message = tk.Label(root, text = "Press any leter from the keyboard to continue ...")
    # message.pack()   
        
 ##       window.close()
 #   popup = sg.popup("Press any leter from the keyboard to continue ...", auto_close_duration=0)   ## , auto_close=False
    
##    print("Press any leter from the keyboard to continue ...")
    
    import PySimpleGUI as sg
    
    window = sg.Window('Acquisition process starter ...', [[sg.Button("Continue ...")]], element_justification='c')
    
       
    # thread = threading.Thread(target=show_window)
    # thread.start()
    
    while True:
        
        print("A")
        
        event, values = window.read()
        
        print("B")
        
        if event == sg.WIN_CLOSED:
            continue
        if event == "Continue ...":
            print("Continue button trigger ...")
            key = True
   #         window.close()
        if key == True:    
        
            while curTest < numberTests:
                    key_not_pressed = 0
                    
                    base_path = dir_path
                    
                    ## base_path = "C:/Users/Other/"                
                    
                    
                   ## base_path2 = 'C:/Users/Other/'
                    base_path2 = base_path
                  ##  base_path3 = "C:\\Users\\Other\\"        
                  
                    base_path3 = base_path.replace("/", "\\")
                    
                    nodeFile = base_path + "feature_data.pfs"
                    
                    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                    converter = pylon.ImageFormatConverter() 
                    
                    camera.Open()
                 
                        
                    camera.CenterX=False
                    camera.CenterY=False
                    
                   
                    
                    # Set the upper limit of the camera's frame rate to 30 fps
                    camera.AcquisitionFrameRateEnable.SetValue(True)
                    camera.AcquisitionFrameRateAbs.SetValue(frame_rate)
                    
                    camera.GevSCPSPacketSize.SetValue(packet_size)
                    
                    # Inter-Packet Delay            
                    camera.GevSCPD.SetValue(inter_packet_delay)
                    
                    # Bandwidth Reserve 
                    camera.GevSCBWR.SetValue(bandwidth_resv)
                    
                    # Bandwidth Reserve Accumulation
                    camera.GevSCBWRA.SetValue(bandwidth_resv_acc)    
                    
                    ## Save feature data to .pfs file
                  ##  pylon.FeaturePersistence.Save(nodeFile, camera.GetNodeMap())            
                
                    # demonstrate some feature access
                    new_width = camera.Width.GetValue() - camera.Width.GetInc()
                    if new_width >= camera.Width.GetMin():
                        camera.Width.SetValue(new_width)
                        
                    camera.Width.SetValue(image_width) 
                    camera.Height.SetValue(image_height)
                    
                    numberOfImagesToGrab = 100
                    
                    camera.GainRaw=gain_raw
                    camera.ExposureTimeRaw=exp_time
                    
                    camera.StartGrabbingMax(numberOfImagesToGrab)
                    
                    run_again = True
             #       camera.Open()
                    
                  
                            
                    print("Max gain",camera.GainRaw.Max)
                    print("Min gain",camera.GainRaw.Min) 
                    print("Max ExposureTime",camera.ExposureTimeRaw.Max)
                    print("Min ExposureTime",camera.ExposureTimeRaw.Min)
                    i=0
                    j=0
                    
                   
                    counter = 0               
                    
                    list_img = []
                    
                    initTime = 0
                    initTimeStamp = 0
                    
                    
                    import csv
                    
                    def write_csv_line(filename, data):
                        with open(filename, 'a', newline = '') as file:
                            writer = csv.writer(file)
                            writer.writerow(data) 
                    
                    def get_csv_filename_timestamps():
                        import PySimpleGUI as sg
                        
                        csv_timestamps_filename = sg.popup_get_text("Specify the CSV filename to save the timestamps along the next acquisition: ")
                        
                        return csv_timestamps_filename        
                    
                    csv_timestamps_filename = ""
                    
                    if enable_save_timestamps:
                    
                        while len(csv_timestamps_filename) == 0:                        
                            csv_timestamps_filename = get_csv_filename_timestamps()
                        
                  ##      csv_timestamps_filename = input("Specify the CSV filename to save the timestamps along the next acquisition: ")
                        csv_timestamps_filename += '.csv'
                        
                        title_line = ["Number of image", "Acquisition time (sec.)", "Time since start of acquisition"]
                        
                        write_csv_line(csv_timestamps_filename, title_line)
                        
                        time_x = 0 
                    
                    while camera.IsGrabbing():
                        
                     ##   if counter < numberImages:
                            againT = True
                            
                            while againT:
                                try:
                                    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                                    againT = False
                                    break
                                except Exception:
                                    print("Trying again ...")
                                    time.sleep(3)
                                    againT = True
                                
                            if grabResult.GrabSucceeded():   
    
                              
                                if counter > 0:                      
                                    
                        ##            print("Image " + str(counter))
                                    image = converter.Convert(grabResult)
                                    
                                    if enable_save_timestamps:
                                    
                                        ## unixTimeAcq = grabResult.GetTimeStamp()
                                        unixTimeAcq = (time.time())*(1e6)
                                    ##    time_unit = grabResult.GetTimeStampModule().GetTimeUnit()
                                        if counter > 2:                                  
                                            print("Time X:" + str(time_x))
                                            acquisitionTime = round(((unixTimeAcq- time_x)/1e6),5)
                                            initTime += acquisitionTime  
                                         
                                                
                                            
                                            
                                            print(f"Acquisition time: {acquisitionTime:.3f} seconds")
                                            print("Time since start of acquisition: " + str(round(initTime,5)))
                                            
                                            data = [counter, acquisitionTime, round(initTime,5)] 
                                            
                                            write_csv_line(csv_timestamps_filename, data)
                                        
                                        time_x = unixTimeAcq 
                                    
                                    img = image.GetArray()                               
                                      
                                    list_img.append(img)                            
                                    
                                  ##  cv2.imshow("image",img) 
                                  
                  ##                  shown_check = showLiveImageGUI(img, counter)
                 ##                   print("Image shown: " + str(shown_check))
                                     
                ##                    cv2.imwrite(base_path + "/image" + str(curTest) + "_" + str(counter) + ".tiff", img)                               
                                   
                                    
                                counter += 1 
                   #     else: 
                  #          break
                              
                     ##       cv2.waitKey(1)
                    
                    
                    for ind_img, img in enumerate(list_img):
                        cv2.imwrite(base_path + "/image" + str(curTest) + "_" + str(ind_img) + ".tiff", img)
                    
                    folderThisTest = base_path + '/Test_' + str(curTest) + '/'
                    
                    if not os.path.exists(folderThisTest):
                        os.mkdir(os.path.join(folderThisTest))
                    
                    for ind_img, img in enumerate(list_img):
                        cv2.imwrite(folderThisTest + "image" + str(ind_img) + ".tiff", img)
                    
                           
                        
                     
         ##           image_graph_dice_coeff = dice_coeff(np.array([list_img])[0])
                     
                    print("End of image acquisition")
                    
                    grabResult.Release()
                    camera.Close()
                    
                    print("Converting to video ... ")
                    
                  
                            
                    addingCurTest = str(curTest) + "_"  
                    
                    
                    img_array = []
                    ind = 0
                    for filename in glob.glob(base_path2 + '/*tiff'):      ## base_path2 + 'files_python/Img_track/*.tiff'
                        print("A")
                        if ("image" in filename) and (addingCurTest in filename):   ## if ("image" and addingCurTest) in filename           ## addingCurTest = str(curTest) + "_"  
                            print("B")
                            img = cv2.imread(filename)
                            height, width, layers = img.shape 
                            size = (width,height) 
                            print(size)
                            img_array.append(img) 
                            print("Image " + str(ind) + " drained to set")
                            ind += 1
                            
                            im = cv2.imread(filename)
                            im_parts = filename.split('.tiff')
                            for d in im_parts:
                                if len(d) != 0:
                                    im_rem = d
                                
                            dir_png_path = im_rem + '.png'        
                            cv2.imwrite(dir_png_path, im)
                            
                            key_calib = False
                            print("Press a key ...")
                            
                            counter_for_key = 0
                            
                           
                    
                    out = cv2.VideoWriter('test_' + '00' + str(counterTest) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                     
                    for i in range(len(img_array)):
                        out.write(img_array[i])
                    out.release()
                    
                    print("Conversion done")
                     
                    #%%
                    
                    dateTimeMarker = getDateTimeStrMarker()
                    
                    sequence_name = counterTest
                    dest_path = base_path2 + '/Image_Processing/'
                    dest_path2 = base_path3 + "\\Image_Processing\\" 
                     
                    count_folder_ex = 0
                    
                    exists = True
                    
                    while exists == True:          
                        
                        if os.path.exists(dest_path) == True:                   
                            new_str_folder = '/Image_Processing' + str(count_folder_ex) + '/'
                            new_str_folder_sec = "\\Image_Processing" + str(count_folder_ex) + "\\" 
                            dest_path = base_path2 + new_str_folder 
                            if os.path.exists(dest_path) == True:
                                exists = True
                                count_folder_ex += 1 
                            else:
                                dest_path2 = base_path3 + new_str_folder_sec                        
                                dest_path = os.path.join(dest_path)
                                os.mkdir(dest_path)   
                                exists = False
                                break
                        else: 
                            dest_path = os.path.join(dest_path)
                            os.mkdir(dest_path)
                            exists = False
                            break                     
                            
                    
                    mainPathVideoData = dest_path + 'VideoData_' + dateTimeMarker + '/'
                    mainPathVideo = os.path.join(mainPathVideoData)
                    os.mkdir(mainPathVideo) 
                    mtsVideoPath = dest_path + 'VideosAlmostLaserSpeckle/' 
                    mtsVideoPath = os.path.join(mtsVideoPath)
                    os.mkdir(mtsVideoPath) 
                    mp4VideoFile = dest_path2 + 'VideosAlmostLaserSpeckle' + dateTimeMarker + "\\"      ## change '/' to "\\" 
                    mp4VideoFilePath = os.path.join(mp4VideoFile)                                     ## path_out (ffmpeg)
                    os.mkdir(mp4VideoFilePath)   
                    # mtsVideoPathP = os.path.join(mtsVideoPath) 
                    # os.mkdir(mtsVideoPathP)   
                    IFVP = dest_path + 'DataSequence_'
                    locationMP4_file = "FilesFor_"
                    roiPath = 'Approach' 
                    newRoiPath = 'Approach_new'  
                    pathRoiStart = dest_path + 'modRoisFirstMom_'
                    pathRoiEnd = dest_path + 'modRoisSecMom_'
                    first_clustering_storing_output = "Quality_kMeans_Clustering_real_"
                    
                    dir_parts = dir_path.split('/')
                    dir_parts.remove(dir_parts[-2])
                    pathPythonFile = '/'.join(dir_parts)  
                    pathPythonFile = pathPythonFile.rstrip('/')                 
         
                    IFVP = IFVP + dateTimeMarker 
                    locationMP4_file = locationMP4_file + dateTimeMarker  
                    roiPath = roiPath + dateTimeMarker 
                    newRoiPath = newRoiPath + dateTimeMarker  
                    pathRoiStart = pathRoiStart + dateTimeMarker 
                    pathRoiEnd = pathRoiEnd + dateTimeMarker 
                    first_clustering_storing_output = first_clustering_storing_output + dateTimeMarker
                    
                    print(" - - - Total number of tests: " + str(numberTests))
                    print(" - - - Current test: " + str(curTest+1))
                    
                    folderThisTest = base_path + '/Test_' + str(curTest) + '/'
                    
                    if not os.path.exists(folderThisTest):                        
                        os.mkdir(os.path.join(folderThisTest))
                    
                    flag = True 
                    
                    while flag:
                        if os.path.isdir(folderThisTest):
                            flag = False
                        else:
                            print("Directory not created")
                            
                    data_to_save = []
    
                    print("Tests here ... ")                  
                    
                    
                    if numberTests == 1:  
                        
                        print("Tests here  A ... ")                  
                        
                        import PySimpleGUI as sg                    
                        
                        infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile)
                        
                        
                        layout = [[sg.ProgressBar(1000, orientation='h', size = (20,20), key = 'progressbar')]]
    
                        window = sg.Window('Processing ...', layout)
                       
                        while True:
                            
                            event, values = window.read(timeout=1000)
                            
                            if event == sg.WIN_CLOSED:
                                break
                            
                            if True:
                                
                                clusteringInfData, executionTime, totCountImages = videoAnalysis_single(infi, folderThisTest, ind_img)
                                datax = (clusteringInfData, executionTime, totCountImages)  
                            #    gui_show_results(clusteringInfData, executionTime, totCountImages) 
                               
                                data_to_save.append(datax)
                        
                                # clusteringInfData, executionTime, totCountImages = videoAnalysis(infi)               
                                # datax = (clusteringInfData, executionTime, totCountImages)                  
                                
                                # data_to_save.append(datax)
                                
                                break 
                        
                        window.close()
                        
                        unique_test = True
                        
                        print("Here")
                        
                        break
                        
            ##            return data_to_save 
                        
                    else: 
    
                        print("Tests here  B ... ")                     
                    
                        if curTest < numberTests-1: 
                            print(" - - - Going to save data ...")
                            if curTest == 0:
                                infx = []
                                data_to_save = [] 
                                
                            infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile))  ## , pathRoiStart, pathRoiEnd
                            
                            nextTest = countdown_timer_display(gap_bet_tests)    ## in minutes
                        ##    time.sleep(gap_bet_tests*60)
                            
                            print("Ready for next test: " + str(nextTest))
                            
                        elif curTest == numberTests-1:    
                             
                            if curTest == 0:
                                infx = []
                                data_to_save = []
                            
                            infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile))  ## , pathRoiStart, pathRoiEnd
                            
                            import PySimpleGUI as sg
                            
                            data_from_tests = []
                            
                            print("Length of infx: " + str(len(infx)))
                            
                            for indI, infi in enumerate(infx):                         
                                
                                  layout = [[sg.ProgressBar(1000, orientation='h', size = (20,20), key = 'progressbar')]]
                                
                                  window = sg.Window('Processing ...', layout)
                                  
                                  while True:
                                      event, values = window.read(timeout=1000)
                                      
                                      if event == sg.WIN_CLOSED:
                                          break
                                      
                                      if True:
                                          
                                          if indI < len(infx) - 1:   
                                              print("\n\n HERE " + str(indI) + "\n\n")
                                              data_from_tests = videoAnalysis(indI, numberTests, infi, data_from_tests)
                                          else:
                                              print("\n\n HERE FINALLY " + str(indI) + "\n\n")
                                              clusteringInfData, executionTime, totCountImages = videoAnalysis(indI, numberTests, infi, data_from_tests)               
                                              datax = (clusteringInfData, executionTime, totCountImages)                  
                                              
                                              data_to_save.append(datax)       
                                      break
                                  
                            window.close()
                               
                            counterTest += 1                    
                    #        break 
                        curTest += 1  
                        print("\n Got here \n")
                        
            window.close()
            break
                        # import sys
                        # sys.exit() 
        else:
            key_not_pressed += 1
            
            if key_not_pressed == 1:
                startTime = time.time() 
            else: 
                if key_not_pressed > 1:
                    if (time.time() - startTime) > 60*time_gap_min:
                        print("Ending tests set ...")
                        break
        
        if unique_test == True:
            dat_unique = data_to_save
            data_to_save = dat_unique
            unique_test = False 
            break
    return data_to_save 
 
## with advanced options

def whole_processing_software_adv(frame_rate, packet_size, inter_packet_delay, bandwidth_resv, bandwidth_resv_acc, gain_raw, exp_time, rec_time, image_height, image_width, decisorLevel, dir_path, curTest, numberTests, gap_bet_tests, list_values_adv, list_keys_adv):

    print("Starting acquisition and processing steps ...")
    
    import cv2 
    from pypylon import pylon
    import numpy as np
    import glob 
    import string 
    import time
    import os  
    import keyboard 
    from countdown_timer import countdown_timer_display
    from pre_processing import type_img
##    from find_dice_coe_along_image_sequence import dice_coeff
    
    numberImages = rec_time*frame_rate
    
    numberImages += 1
     
    print("Main imports gone nice")
    
    from softwareImageProcessingForLaserSpeckleAnalysisFinal import videoAnalysis
    
    print("OK 1")
    
    from getCurrentDateAndTime import getDateTimeStrMarker  
     
    print("OK 2")
    
    from show_outputToWeb import showLiveImageGUI
    
    print("OK 3")
     
    print("Project imports gone nice")  
    
    alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase)
    
    counterTest = 0
    key = False
    key_not_pressed = 0
    
    unique_test = False
    dat_unique = []
    
    time_gap_min = 30    
     
    run_again = True
    
 ##   print("Press any leter from the keyboard to continue ...")
    
    import PySimpleGUI as sg
    
    window = sg.Window('Acquisition process starter ...', [[sg.Button("Continue ...")]], element_justification='c')
    
       
    # thread = threading.Thread(target=show_window)
    # thread.start()
    
    while True:
        
        print("A")
        
        event, values = window.read()
        
        print("B")
        
        if event == sg.WIN_CLOSED:
            continue
        if event == "Continue ...":
            print("Continue button trigger ...")
            key = True
            window.close()
            
        
   
   
        # for letter in alphabet:
        #     if keyboard.is_pressed(letter):
        #         print("key pressed") 
        #         key = True
        
        if key == True:
            
            while True:
                key_not_pressed = 0
                
                base_path = dir_path
                
                ## base_path = "C:/Users/Other/"                
                
                
               ## base_path2 = 'C:/Users/Other/'
                base_path2 = base_path
              ##  base_path3 = "C:\\Users\\Other\\"        
              
                base_path3 = base_path.replace("/", "\\")
                
                nodeFile = base_path + "feature_data.pfs"
                
                camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                converter = pylon.ImageFormatConverter()               
                
                while run_again == True:

                    try:               
                        camera.Open()
                        run_again = False
                    except RuntimeError:
                        print(" -- Please check the wire connection !!!")
                        print("Trying again in 10 seconds ...")
                        
                        time.sleep(10)
                        
                        run_again = True
                        
                camera.CenterX=False
                camera.CenterY=False
                
                
                ##################################################################
                
                pixf = False
                 
                for ind_k, key in enumerate(list_keys_adv):
                    
                    if key == 'GENAPI_V':
                        camera.DeviceFirmwareVersion.SetValue(list_values_adv[ind_k])
                    if key == 'DEV_V':
                        camera.DeviceVersion.SetValue(list_values_adv[ind_k])
                    if key == 'DEV_NAME':
                        camera.DeviceModelName.SetValue(list_values_adv[ind_k]) 
                    if key == 'INIT_CODE':
                        camera.DeviceSerialNumber.SetValue(list_values_adv[ind_k])                    
                    if key == 'PROD_GUID':
                        camera.DeviceID.SetValue(list_values_adv[ind_k])
                    if key == 'PROD_V_GUID':                        
                        camera.DeviceUserID.SetValue(list_values_adv[ind_k]) 
                        
                    if key == 'SEQ_TOT':
                        camera.SequenceSetTotalNumber.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'SEQ_INDEX':
                        camera.SequenceSetIndex.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'SEQ_EXEC':
                        camera.SequenceSetExecutions.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'SEQ_ADV':
                        camera.SequenceEnable.SetValue(False)
                        camera.SequenceAdvanceMode.SetValue("SequenceAdvanceMode_" + str(list_values_adv[ind_k]))
                    
                    elif key == 'GAIN_AUTO':
                        camera.GainAuto.SetValue("GainAuto_" + str(list_values_adv[ind_k]))
                    elif key == 'GAIN_SEL':
                        camera.GainSelector.SetValue("GainSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'BLACK_LEV_SEL':
                        camera.BlackLevelSelector.setValue("BlackLevelSelector_" + str(list_values_adv[ind_k]))
                        
                    elif key == 'GAMMA_EN':
                        camera.GammaEnable.SetValue(bool(list_values_adv[ind_k]))
                    elif key == 'GAMMA_SEL':
                        camera.GammaSelector.SetValue("GammaSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'GAMMA':
                        camera.Gamma.SetValue(float(list_values_adv[ind_k]))
                    
                    elif key == 'DIG_SHIFT':
                        camera.DigitalShift.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'PIX_FORMAT':  
                        pixf = True
                        camera.PixelFormat.SetValue(str(list_values_adv[ind_k]))
                    elif key == 'REV_X':
                        camera.ReverseX.SetValue(bool(int(list_values_adv[ind_k])))
                    # elif key == 'REV_Y': 
                    #     camera.ReverseY.SetValue(bool())
                        
                    elif key == 'TEST_IMAGE_SEL':
                        camera.TestImageSelector.SetValue("TestImageSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'OFFSET_X':
                        camera.OffsetX.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'OFFSET_Y':
                        camera.OffsetY.SetValue(int(list_values_adv[ind_k]))   
                    elif key == 'CENTER_X':
                        camera.CenterX.SetValue(bool(int(list_values_adv[ind_k])))
                    elif key == 'CENTER_Y':
                        camera.CenterY.SetValue(bool(int(list_values_adv[ind_k])))               
                    elif key == 'BIN_MODE_H':
                        camera.BinningHorizontalMode.SetValue("BinningHorizontalMode_" + str(list_values_adv[ind_k]))                        
                    elif key == 'BIN_H':
                        camera.BinningHorizontal.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'BIN_MODE_V':
                        camera.BinningVerticalMode.SetValue("BinningVerticalMode_" + str(list_values_adv[ind_k]))
                    elif key == 'BIN_V':
                        camera.BinningVertical.SetValue(int(list_values_adv[ind_k]))  
                        
                    elif key == 'ACQ_FRAME_COUNT':
                        camera.AcquisitionFrameCount.SetValue(int(list_values_adv[ind_k]))                        
                    elif key == 'TRIG_SEL_ONE':
                        camera.TriggerSelector.SetValue("TriggerSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'TRIGGER_MODE':
                        camera.TriggerMode.SetValue("TriggerMode_" + str(list_values_adv[ind_k]))
                    elif key == 'ACQ_FR_EN':
                        camera.AcquisitionFrameRateEnable.SetValue(bool(int(list_values_adv[ind_k])))
                    elif key == 'TRIG_SEL_TWO':
                        camera.TriggerSelector.SetValue("TriggerSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'TRIG_SOURCE':
                        camera.TriggerSource.SetValue("TriggerSource_" + str(list_values_adv[ind_k]))
                    elif key == 'TRIG_ACTIV':
                        camera.TriggerActivation.SetValue("TriggerActivation_" + str(list_values_adv[ind_k]))
                    elif key == 'TRIG_DELAY_ABS':
                        camera.TriggerDelayAbs.SetValue(int(list_values_adv[ind_k]))                        
                    elif key == 'EXP_MODE':
                        camera.ExposureMode.SetValue("ExposureMode_" + str(list_values_adv[ind_k]))
                    elif key == 'EXP_AUTO':                       
                        camera.ExposureAuto.SetValue("ExposureAuto_" + str(list_values_adv[ind_k]))                    
                    elif key == 'SHUTTER_MODE':
                        camera.SensorShutterMode.SetValue("SensorShutterMode_" + str(list_values_adv[ind_k]))
                     
                    
                    elif key == 'LINE_SEL_ONE':
                        camera.LineSelector.SetValue("LineSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'LINE_SEL_TWO':
                        camera.LineSelector.SetValue("LineSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'LINE_MODE_ONE':
                        camera.LineMode.SetValue("LineMode_" + str(list_values_adv[ind_k]))
                    elif key == 'LINE_MODE_TWO':
                        camera.LineMode.SetValue("LineMode_" + str(list_values_adv[ind_k]))    
                    elif key == 'LINE_FORMAT':
                        camera.LineFormat.SetValue("LineFormat_" + str(list_values_adv[ind_k]))                        
                    elif key == 'LINE_SOURCE':
                        camera.LineSource.SetValue("LineSource_" + str(list_values_adv[ind_k]))
                    elif key == 'LINE_INV':                     
                        camera.LineInverter.SetValue(bool(int(list_values_adv[ind_k])))
                    elif key == 'LINE_DEB_TIME':                        
                        camera.LineDebouncerTime.SetValue(float(list_values_adv[ind_k]))
                    elif key == 'LINE_MINOUT_PULSE':           
                     #   camera.LineMinimumOutputPulseWidth.SetValue(float(list_values_adv[ind_k]))
                        camera.MinOutPulseWidthAbs.SetValue(float(list_values_adv[ind_k]))
                    
                    elif key == 'COUNTER_SEL_ONE':                        
                        camera.CounterSelector.SetValue("CounterSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'COUNTER_SEL_TWO':                        
                        camera.CounterSelector.SetValue("CounterSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'COUNT_EV_SOURCE_ONE':                        
                        camera.CounterEventSource.SetValue("CounterEventSource_" + str(list_values_adv[ind_k]))
                    elif key == 'COUNT_EV_SOURCE_TWO':                        
                        camera.CounterEventSource.SetValue("CounterEventSource_" + str(list_values_adv[ind_k])) 
                    elif key == 'COUNTER_RESET_SOURCE':                        
                        camera.CounterEventSource.SetValue("CounterResetSource_" + str(list_values_adv[ind_k]))     
                     
                    elif key == 'LUT_SEL':
                        camera.LUTSelector.SetValue("LUTSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'LUT_VAL':
                        
                        n_bits_dig_sensor_reading = 12
                        
                        if pixf == True:
                            pixFormat = str(list_values_adv['PIX_FORMAT'])
                            
                        n_bits_list = [int(i) for i in pixFormat.split() if i.isdigit()]
                        
                        n_bits_str = ""
                        
                        for n in n_bits_list: 
                            n_bits_str.append(n)
                        
                        n_bits = int(n_bits_str)    
                        tot_size = 2^n_bits_dig_sensor_reading
                        
                        for i in range(0, tot_size, n_bits):
                            camera.LUTIndex.SetValue(i)
                            camera.LUTValue.SetValue((tot_size-1)-i)
                    elif key == 'LUT_EN':
                        camera.LUTEnable.SetValue(bool(list_values_adv[ind_k])) 
                    elif key == 'GEVS_CHANNEL_SEL':
                        camera.GevStreamChannelSelector.SetValue("GevStreamChannelSelector_" + str(list_values_adv[ind_k]))                    
                    elif key == 'AUTO_TARGET':
                        camera.AutoTargetValue.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'GREY_ADJUST':
                        camera.GrayValueAdjustmentDampingRaw.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'BAL_ADJUST':
                        camera.BalanceWhiteAdjustmentDampingRaw.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'GAIN_LOW':                         
                        camera.AutoGainRawLowerLimit.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'GAIN_UP':    
                        camera.AutoGainRawUpperLimit.SetValue(int(list_values_adv[ind_k]))                      
                    elif key == 'EXP_LOW':
                        camera.AutoExposureTimeAbsLowerLimit.SetValue(float(list_values_adv[ind_k]))
                    elif key == 'EXP_UP':
                        camera.AutoExposureTimeAbsUpperLimit.SetValue(float(list_values_adv[ind_k]))
                     
                    elif key == 'AUTO_FUNC_PROF':
                        camera.AutoFunctionProfile.SetValue("AutoFunctionProfile_" + str(list_values_adv[ind_k]))                    
                    elif key == 'AOI_WIDTH':
                        camera.AutoFunctionAOIWidth.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'AOI_HEIGHT':
                        camera.AutoFunctionAOIHeight.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'AOI_OFFSETX':
                        camera.AutoFunctionAOIOffsetX.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'AOI_OFFSETY':
                        camera.AutoFunctionAOIOffsetY.SetValue(int(list_values_adv[ind_k]))                   
                    elif key == 'NAME_DEF_VALUE1':
                        camera.UserDefinedValueSelector.SetValue("UserDefinedValueSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'NAME_DEF_VALUE2':  
                        camera.UserDefinedValueSelector.SetValue("UserDefinedValueSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'NAME_DEF_VALUE3':
                        camera.UserDefinedValueSelector.SetValue("UserDefinedValueSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'NAME_DEF_VALUE4':
                        camera.UserDefinedValueSelector.SetValue("UserDefinedValueSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'NAME_DEF_VALUE5':
                        camera.UserDefinedValueSelector.SetValue("UserDefinedValueSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'DEF_VALUE1':
                        camera.UserDefinedValue.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'DEF_VALUE2':
                        camera.UserDefinedValue.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'DEF_VALUE3':
                        camera.UserDefinedValue.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'DEF_VALUE4':
                        camera.UserDefinedValue.SetValue(int(list_values_adv[ind_k]))
                    elif key == 'DEF_VALUE5':
                        camera.UserDefinedValue.SetValue(int(list_values_adv[ind_k]))
                   
                    elif key == 'CHUNK_MODE':  
                        camera.ChunkModeActive.SetValue(bool(int(list_values_adv[ind_k])))
                    elif key == 'EV_SEL_ONE':
                        camera.EventSelector.SetValue("EventSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_SEL_TWO':
                        camera.EventSelector.SetValue("EventSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_SEL_THREE':
                        camera.EventSelector.SetValue("EventSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_SEL_FOUR':
                        camera.EventSelector.SetValue("EventSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_SEL_FIVE':
                        camera.EventSelector.SetValue("EventSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_SEL_SIX':
                        camera.EventSelector.SetValue("EventSelector_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_NOTIF_ONE':
                        camera.EventNotification.SetValue("EventNotification_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_NOTIF_TWO':
                        camera.EventNotification.SetValue("EventNotification_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_NOTIF_THREE':
                        camera.EventNotification.SetValue("EventNotification_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_NOTIF_FOUR':
                        camera.EventNotification.SetValue("EventNotification_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_NOTIF_FIVE':
                        camera.EventNotification.SetValue("EventNotification_" + str(list_values_adv[ind_k]))
                    elif key == 'EV_NOTIF_SIX': 
                        camera.EventNotification.SetValue("EventNotification_" + str(list_values_adv[ind_k]))
                        
                     
                ##################################################################
                
                
               
                
                # Set the upper limit of the camera's frame rate to 30 fps
                camera.AcquisitionFrameRateEnable.SetValue(True)
                camera.AcquisitionFrameRateAbs.SetValue(frame_rate)
                
                camera.GevSCPSPacketSize.SetValue(packet_size)
                
                # Inter-Packet Delay            
                camera.GevSCPD.SetValue(inter_packet_delay)
                
                # Bandwidth Reserve 
                camera.GevSCBWR.SetValue(bandwidth_resv)
                
                # Bandwidth Reserve Accumulation
                camera.GevSCBWRA.SetValue(bandwidth_resv_acc)    
                
                ## Save feature data to .pfs file
              ##  pylon.FeaturePersistence.Save(nodeFile, camera.GetNodeMap())            
            
                # demonstrate some feature access
                new_width = camera.Width.GetValue() - camera.Width.GetInc()
                if new_width >= camera.Width.GetMin():
                    camera.Width.SetValue(new_width)
                    
                camera.Width.SetValue(image_width) 
                camera.Height.SetValue(image_height)
                
                numberOfImagesToGrab = 100
                camera.StartGrabbing()
                
                run_again = True
                while run_again == True:

                    try:               
                        camera.Open()
                        run_again = False
                    except RuntimeError:
                        print(" -- Please check the wire connection !!!")
                        print("Trying again in 5 seconds ...")
                        
                        time.sleep(5)
                        
                        run_again = True
                        
                print("Max gain",camera.GainRaw.Max)
                print("Min gain",camera.GainRaw.Min) 
                print("Max ExposureTime",camera.ExposureTimeRaw.Max)
                print("Min ExposureTime",camera.ExposureTimeRaw.Min)
                i=0
                j=0
                 
                camera.GainRaw=gain_raw
                camera.ExposureTimeRaw=exp_time
                counter = 0               
                
                list_img = []
                
                while camera.IsGrabbing():
                    
                    if counter < numberImages:
                        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        
                        if grabResult.GrabSucceeded():   

                            # if counter == 0:
                            
                            #     print("Image " + str(counter))
                            #     image = converter.Convert(grabResult)
                            #     img = image.GetArray()
                                
                            #     filename = base_path + "/image" + "_" + "test" + ".tiff"
                                
                            #     cv2.imwrite(filename, img)                           
    
                            #     code_img = type_img(filename, base_path, 1)
                                
                            #     if code_img == 0:
                            #         print("Error classifying image !!!")
                            #     elif code_img == 1:
                            #         print("Laser Speckle Image detected !!!")
                            #     elif code_img == 2:
                            #         print("Chess Image detected !!!")
                            
                            if counter > 0:                     
                                
                                print("Image " + str(counter))
                                image = converter.Convert(grabResult)
                                img = image.GetArray()                               
                                
                                list_img.append(img)                            
                                
                              ##  cv2.imshow("image",img)
                              
                                shown_check = showLiveImageGUI(img, counter)
                                print("Image shown: " + str(shown_check))
                                 
                                cv2.imwrite(base_path + "/image" + str(curTest) + "_" + str(counter) + ".tiff", img)                               
                               
                                
                            counter += 1 
                    else: 
                        break
                          
                 ##       cv2.waitKey(1)
                 
     ##           image_graph_dice_coeff = dice_coeff(np.array([list_img])[0])
                 
                print("End of image acquisition")
                
                grabResult.Release()
                camera.Close()
                
                print("Converting to video ... ")
                
                # count_diff_imgs = 0                
                
                # for ind_im, im in enumerate(list_img):
                #     if ind_im == 0:
                #         diff_img = np.zeros_like(im)
                #     else:
                #         for b in range(len(diff_img[0])):
                #             for a in range(len(diff_img)):
                #                 if ind_im > 1:
                #                     diff_img[a,b] = abs(im[a,b]-list_img[ind_im-1][0][a,b])
                        
                #         cv2.imwrite(base_path + "/image" + str(curTest) + "_" + str(counter) + ".tiff", diff_img)
                #         count_diff_imgs += 1
                        
                addingCurTest = str(curTest) + "_"  
                
                
                img_array = []
                ind = 0
                for filename in glob.glob(base_path2 + '/*tiff'):      ## base_path2 + 'files_python/Img_track/*.tiff'
                    print("A")
                    if ("image" in filename) and (addingCurTest in filename):   ## if ("image" and addingCurTest) in filename           ## addingCurTest = str(curTest) + "_"  
                        print("B")
                        img = cv2.imread(filename)
                        height, width, layers = img.shape 
                        size = (width,height) 
                        print(size)
                        img_array.append(img) 
                        print("Image " + str(ind) + " drained to set")
                        ind += 1
                        
                        im = cv2.imread(filename)
                        im_parts = filename.split('.tiff')
                        for d in im_parts:
                            if len(d) != 0:
                                im_rem = d
                            
                        dir_png_path = im_rem + '.png'        
                        cv2.imwrite(dir_png_path, im)
                        
                        key_calib = False
                        print("Press a key ...")
                        
                        counter_for_key = 0
                        
                        # while key_calib == False:
                            
                        #     if counter_for_key < 10:
                            
                        #         print("Keep waiting for key input ...")
                        
                        #     for letter in alphabet:
                        #         if keyboard.is_pressed(letter):
                        #             key_calib = True 
                        #             break
                                
                        #     counter_for_key += 1
                            
                            
                            
                        # code_ret = calib_camera_part(True, "acA1920-25gm", dir_png_path)  
                                  
                        # if code_ret == 2:  
                        #      print(" -- Trouble !!!")
                        #      break
                
                out = cv2.VideoWriter('test_' + '00' + str(counterTest) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                 
                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()
                
                print("Conversion done")
                 
                #%%
                
                dateTimeMarker = getDateTimeStrMarker()
                
                sequence_name = counterTest
                dest_path = base_path2 + '/Image_Processing/'
                dest_path2 = base_path3 + "\\Image_Processing\\" 
                 
                count_folder_ex = 0
                
                exists = True
                
                while exists == True:          
                    
                    if os.path.exists(dest_path) == True:                   
                        new_str_folder = '/Image_Processing' + str(count_folder_ex) + '/'
                        new_str_folder_sec = "\\Image_Processing" + str(count_folder_ex) + "\\" 
                        dest_path = base_path2 + new_str_folder 
                        if os.path.exists(dest_path) == True:
                            exists = True
                            count_folder_ex += 1 
                        else:
                            dest_path2 = base_path3 + new_str_folder_sec                        
                            dest_path = os.path.join(dest_path)
                            os.mkdir(dest_path)   
                            exists = False
                            break
                    else: 
                        dest_path = os.path.join(dest_path)
                        os.mkdir(dest_path)
                        exists = False
                        break                     
                        
                
                mainPathVideoData = dest_path + 'VideoData_' + dateTimeMarker + '/'
                mainPathVideo = os.path.join(mainPathVideoData)
                os.mkdir(mainPathVideo) 
                mtsVideoPath = dest_path + 'VideosAlmostLaserSpeckle/' 
                mtsVideoPath = os.path.join(mtsVideoPath)
                os.mkdir(mtsVideoPath) 
                mp4VideoFile = dest_path2 + 'VideosAlmostLaserSpeckle' + dateTimeMarker + "\\"      ## change '/' to "\\" 
                mp4VideoFilePath = os.path.join(mp4VideoFile)                                     ## path_out (ffmpeg)
                os.mkdir(mp4VideoFilePath)   
                # mtsVideoPathP = os.path.join(mtsVideoPath) 
                # os.mkdir(mtsVideoPathP)   
                IFVP = dest_path + 'DataSequence_'
                locationMP4_file = "FilesFor_"
                roiPath = 'Approach'
                newRoiPath = 'Approach_new'  
                pathRoiStart = dest_path + 'modRoisFirstMom_'
                pathRoiEnd = dest_path + 'modRoisSecMom_'
                first_clustering_storing_output = "Quality_kMeans_Clustering_real_"
                
                dir_parts = dir_path.split('/')
                dir_parts.remove(dir_parts[-2])
                pathPythonFile = '/'.join(dir_parts)  
                pathPythonFile = pathPythonFile.rstrip('/')                 
     
                IFVP = IFVP + dateTimeMarker 
                locationMP4_file = locationMP4_file + dateTimeMarker  
                roiPath = roiPath + dateTimeMarker 
                newRoiPath = newRoiPath + dateTimeMarker  
                pathRoiStart = pathRoiStart + dateTimeMarker 
                pathRoiEnd = pathRoiEnd + dateTimeMarker 
                first_clustering_storing_output = first_clustering_storing_output + dateTimeMarker
                
                print(" - - - Total number of tests: " + str(numberTests))
                print(" - - - Current test: " + str(curTest+1))
                
                data_to_save = []

                print("Tests here ... ")                  
                
                
                if numberTests == 1:  
                    
                    print("Tests here  A ... ")                  
                    
                    
                    infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile)
                                        
                    clusteringInfData, executionTime, totCountImages = videoAnalysis(infi)               
                    datax = (clusteringInfData, executionTime, totCountImages)                  
                    
                    data_to_save.append(datax)   
                    
                    unique_test = True
                    
                    print("Here")
                    
                    break
                    
        ##            return data_to_save 
                    
                else: 

                    print("Tests here  B ... ")                     
                
                    if curTest < numberTests-1: 
                        print(" - - - Going to save data ...")
                        if curTest == 0:
                            infx = []
                            data_to_save = [] 
                            
                        infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile))
                        
                        nextTest = countdown_timer_display(gap_bet_tests)    ## in minutes
                    ##    time.sleep(gap_bet_tests*60)
                        
                        print("Ready for next test: " + str(nextTest))
                        
                    elif curTest == numberTests-1:    
                        
                        if curTest == 0:
                            infx = []
                            data_to_save = []
                        
                        infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile))
                        
                        for infi in infx:            
                            
                              clusteringInfData, executionTime, totCountImages = videoAnalysis(infi)               
                              datax = (clusteringInfData, executionTime, totCountImages)                  
                              
                              data_to_save.append(datax)                   
                          
                        counterTest += 1                    
                        break 
                    curTest += 1
        else:
            key_not_pressed += 1
            
            if key_not_pressed == 1:
                startTime = time.time() 
            else: 
                if key_not_pressed > 1:
                    if (time.time() - startTime) > 60*time_gap_min:
                        print("Ending tests set ...")
                        break
        
        if unique_test == True:
            dat_unique = data_to_save
            data_to_save = dat_unique
            unique_test = False 
            break
    return data_to_save 





        




