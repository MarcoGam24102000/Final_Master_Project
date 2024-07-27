# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:11:30 2022

@author: marco
"""

import os
import numpy as np
import cv2
from pypylon import pylon
import glob
from show_outputToWeb import showLiveImageGUI 
from info_gui import theory_info 
from parse_pfs_filename_to_get_dateAndTime import dateAndTime_from_pfs_file_full, dateAndTime_from_pfs_file
from deadlines import time_configs, deadline_pfs_file_gui
from texts_info import write_to_txt_file_tests_info
from countdown_timer import countdown_timer_display

import PySimpleGUI as sg
from PIL import Image
import io
import time
import webbrowser  


def get_diff_in_secs(dateAndTimeInfo, actual_dateAndTimeInfo, time_guidelines):
    
    default_sec_min = time_guidelines[0]
    default_min_hour = time_guidelines[1] 
    default_hour_day = time_guidelines[2] 
    default_days_week = time_guidelines[3]
    default_days_month = time_guidelines[4]
    default_days_year = time_guidelines[5]  
    
    
    (date_pfs, time_pfs) = dateAndTimeInfo 
    (actual_date, actual_time) = actual_dateAndTimeInfo
    
    cur_day = int(actual_date[0:2]) 
    cur_month = int(actual_date[2:4])
    cur_year = int(actual_date[4:6])    
    cur_hours = int(actual_time[0:2])
    cur_mins = int(actual_time[2:4]) 
    cur_secs = int(actual_time[4:6])
    
    curSecs = (cur_year*default_days_year + cur_month*default_days_month + cur_day)*default_hour_day*(default_sec_min*default_min_hour) + cur_hours*(default_sec_min*default_min_hour) + cur_mins*default_sec_min + cur_secs
    
    pfs_day = int(date_pfs[0:2]) 
    pfs_month = int(date_pfs[2:4]) 
    pfs_year = int(date_pfs[4:6])   
    pfs_hours = int(date_pfs[0:2])
    pfs_mins = int(date_pfs[2:4])
    pfs_secs = int(date_pfs[4:6])    
    
    pfsSecs = (pfs_year*default_days_year + pfs_month*default_days_month + pfs_day)*default_hour_day*(default_sec_min*default_min_hour) + pfs_hours*(default_sec_min*default_min_hour) + pfs_mins*default_sec_min + pfs_secs
    
    diff_secs = abs(curSecs - pfsSecs)
    
    return diff_secs   
    

def dateAndTimeConfirm(dir_path, dirStorage):

    default_sec_min = 60
    default_min_hour = 60
    default_hour_day = 24
    default_days_week = 7 
    default_days_month = 30
    default_days_year = 365
    
    changes_timing = time_configs()
    
    print("changes_timing")
    print(changes_timing)    
    
    if changes_timing is not None:
    
        if changes_timing is not None:
            if len(changes_timing) > 0:
        
                for timing_key in changes_timing:
                    if timing_key == 'SEC_MIN':
                        default_sec_min = changes_timing['SEC_MIN']
                    if timing_key == 'MIN_HOUR':
                        default_min_hour = changes_timing['MIN_HOUR']
                    if timing_key == 'HOUR_DAY':
                        default_hour_day = changes_timing['HOUR_DAY']
                    if timing_key == 'DAY_WEEK':
                        default_days_week = changes_timing['DAY_WEEK']
                    if timing_key == 'DAY_MONTH':
                        default_days_month = changes_timing['DAY_MONTH']
                    if timing_key == 'DAY_YEAR':
                        default_days_year = changes_timing['DAY_YEAR']
                
        time_guidelines = [default_sec_min, default_min_hour, default_hour_day, default_days_week, default_days_month, default_days_year]
        
        print("DirPath: " + dir_path)
        
        inter_str_par = "parameters_"    
        confirm = False    
        adv_file = False  
        
        
        if '\\' in dir_path:    
            dir_parts = dir_path.split('\\') 
        elif '/' in dir_path:
            dir_parts = dir_path.split('/') 
        
        if '\\' in dirStorage:    
            dir_parts_store = dirStorage.split('\\') 
        elif '/' in dirStorage:
            dir_parts_store = dirStorage.split('/') 
            
    ##    folder_name = dir_parts_store[-1]   
            
        print("DirSplitted: " + dir_path)
        
        pfs_filename = dir_parts[-1]
        pfs_parts = pfs_filename.split('.')
        pfs_filename_without_ext = pfs_parts[0]
        
        print("PFS filename: " + str(pfs_filename)) 
        
        pfs_file_parts = pfs_filename_without_ext.split('p')
        
        folder_name = pfs_file_parts[0]   
        
        if 'full' in pfs_filename:
            full_str = "_full"
            dateAndTimeInfo = dateAndTime_from_pfs_file_full(pfs_filename_without_ext, folder_name, inter_str_par, full_str)
            adv_file = True
            confirm = False  
        else:
            dateAndTimeInfo = dateAndTime_from_pfs_file(pfs_filename_without_ext, folder_name, inter_str_par)
            confirm = True 
            
        ## Compare actual date and time with the given result got before
        
        from datetime import datetime 
        now = datetime.now()     
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)
        
        dt_string_splitted = dt_string.split(' ')
        date_info = dt_string_splitted[0]
        time_info = dt_string_splitted[1]
        
        actual_date_str = ""
        actual_time_str = ""
        
        date_info_splitted = date_info.split('/')
        
        for ind_datSplit, datSplit in enumerate(date_info_splitted):
            
            if ind_datSplit < 2:        
                actual_date_str += datSplit
            else:
                year_info = datSplit
                year_info = year_info[2:]    ## 2022 -> 22
                actual_date_str += year_info
        
        time_info_splitted = time_info.split(':')
        
        for timeSplit in time_info_splitted:
            actual_time_str += timeSplit
        
        actual_dateAndTimeInfo = (actual_date_str, actual_time_str)
        
        print(dateAndTimeInfo)
        print(actual_dateAndTimeInfo)
        
        secs_out = get_diff_in_secs(dateAndTimeInfo, actual_dateAndTimeInfo, time_guidelines)  
    
        deadline_period = deadline_pfs_file_gui(time_guidelines) 
    
        if secs_out >= deadline_period:
            print("PFS file out of date")
            confirm = False
        else:
            print("PFS file ready")
            
            confirm = True         
    else:
        confirm = False
     
    return confirm 
    

def listFeatures(list_features):
    
    print("List Features function")
    df = open('list_features.txt', 'w')
    
    df.write("List of features: ")
    df.write('\n\n')
    for d in list_features:
        df.write(d)
        df.write('\n')
        
    df.close()
    
    contents = open('list_features.txt', 'r')
    
    with open("list_features.html", "w") as e:
        for lines in contents.readlines():
            
            lines = lines[:-1]   
            
            if not ("List of features:" in lines):
                lines = lines.replace(" ", "")
            
            print(lines)
    
            e.write(lines + "<br>\n")
            
    webbrowser.open('list_features.html', new=2)  

def second_gui_show_results(firstLoaded):
    
    MAX_SIZE = (200, 200)
    
    print("Second Gui")
    
    image_viewer_second_graph = [
        [sg.Text("Distance to the centroid, for the first cluster")],
        [sg.Image(key = "-DIST_FIRST_CLUSTER-")],
        [sg.Button("Distance to the centroid, for the first cluster")]
    ]
     
    image_viewer_third_graph = [
        [sg.Text("Distance to the centroid, for the second cluster")],
        [sg.Image(key = "-DIST_SECOND_CLUSTER-")],
        [sg.Button("Distance to the centroid, for the second cluster")]
    ]
    
    layout = [ 
        [
            sg.Column(image_viewer_second_graph),
            sg.VSeparator(),
            sg.Column(image_viewer_third_graph)
        ]
    ]
    
    window = sg.Window("Output Results", layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))   ## web_port=2219,
    
    thisDir = os.getcwd()
    dirResultsOutput = thisDir + '\\GraphsOutput\\'    
   
    secondLoaded = False
    
    while(True):
        event, values = window.read()    ## timeout = 1000 * 10
        
        if event == 'Exit' or event == sg.WIN_CLOSED: 
            break
    
        if firstLoaded == True and event == 'Distance to the centroid, for the first cluster':
            if os.path.exists(dirResultsOutput):
                if os.path.isfile(dirResultsOutput + "distances_firstCluster.png"):
                    
               #      image = Image.open(dirResultsOutput + "distances_firstCluster.png")
               #      image.thumbnail(MAX_SIZE)
                    
               #      bio = io.BytesIO()
               # ##     bio = io       ## .save(bio, format="PNG")
               #      image.save(bio, format="PNG")
               #      window['-DIST_FIRST_CLUSTER-'].update(data = bio.getvalue())
                    
               #      firstLoaded = True  
               #      firstLoaded == False
               #      secondLoaded = True
                    
                    img = cv2.imread(dirResultsOutput + "distances_firstCluster.png")
                    cv2.imshow('Distance to the centroid, for the first cluster', img)
                    cv2.waitKey(0)
                                         
                    firstLoaded = True
                    secondLoaded = True
                    print("One")
                
        if secondLoaded == True and event == 'Distance to the centroid, for the second cluster':
            if os.path.exists(dirResultsOutput):
                if os.path.isfile(dirResultsOutput + "distances_secondCluster.png"):
          #           image = Image.open(dirResultsOutput + "distances_secondCluster.png")
          #           image.thumbnail(MAX_SIZE)
                    
          #           bio = io.BytesIO() 
          # ##          bio = io           ## .save(bio, format="PNG") 
          
          #           image.save(bio, format="PNG")
          #           window['-DIST_SECOND_CLUSTER-'].update(data = bio.getvalue())    
    
                    img = cv2.imread(dirResultsOutput + "distances_secondCluster.png")
                    cv2.imshow('Distance to the centroid, for the second cluster', img)
                    cv2.waitKey(0)
                    
                    secondLoaded = False
                    
                    time.sleep(1)
                    
                    break
        else:
             print("Unknown event")
##    window.close()
    

### Reduzir o tamanho da janela e ver possiblidade de ajustar com o rato para ficar maior ou menor (autoajuste)
def gui_show_results(clusteringRes, execTime, numberImg):
     
    execTime = int(execTime)
    
    print("Gui for results")
    
    MAX_SIZE = (200, 200)
    
    classAFurther = clusteringRes[0]
    classBFurther = clusteringRes[1]
    nclusters = clusteringRes[2]
    number_recommended_clusters = clusteringRes[3]
    remainingMetricsToClustering = clusteringRes[4]  
    
    
    lst = [str(feature) for feature in remainingMetricsToClustering]
    
    print(" -- Features:")  
    
    for l in lst:
        print(l)
  
    layout_inf = [
        [sg.Text("Number of images generated, for the selected test video: ")],
        [sg.Text("", size = (10,2), key='-TOT_IMG-')],
        [sg.Text("Execution time for the selected test video (sec): ")],
        [sg.Text("", size = (10,2), key='-EXEC_TIME-')],
        [sg.Text("Number of clusters: ")],
        [sg.Text("", size = (10,2), key='-NUMBER_CLUSTERS-')],
        [sg.Text("Number of recomended clusters: ")],
        [sg.Text("", size = (10,2), key='-NUMBER_REC_CLUSTERS-')],
        [sg.Text("Final list of features, for the current test video: ")],
   ##     [sg_py.Listbox(values=[], size = (10, 20), key='-LISTBOX_FEATURES-')],   ## no_scrollbar=True
        [sg.Button("Data Info"), sg.Button("Read more ...")]
    
    ]    
    
    image_viewer_first_graph = [
        [sg.Text("PCA Results")],
        [sg.Image(key = "-PCA-", size=(200,200))],
        [sg.Button("PCA graph output for the first two features")],
        [sg.Button("Distances to the centroid")]
    ]    

    layout = [ 
        [
            sg.Column(layout_inf),
            sg.VSeparator(),
            sg.Column(image_viewer_first_graph)
      ##      sg_py.VSeparator(),
     ##       sg_py.Column(image_viewer_second_graph, image_viewer_third_graph)
      ##      sg_py.VSeparator(),
      ##      sg_py.Column(image_viewer_third_graph)         
         
        ]
    ]
    
    window = sg.Window("Output Results", layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))   ## web_port=2219,
    
    thisDir = os.getcwd()
    dirResultsOutput = thisDir + '\\GraphsOutput\\'
    
    firstLoaded = False
    secondLoaded = False
##    thirdLoaded = False

    
    print("Loop Gui for results")
    
    while(True):
        event, values = window.read()    ## timeout = 1000 * 10
        
        print("Hear")
        
        if event == 'Exit' or event == sg.WIN_CLOSED: 
            print("Exit")
            break
        if event == 'Read more ...':
            theory_info() 
        if event == 'Data Info':
            print("Data Info") 
            
            window['-TOT_IMG-'].update(value = numberImg)
            window['-EXEC_TIME-'].update(value = execTime) 
            window['-NUMBER_CLUSTERS-'].update(value = nclusters)
            window['-NUMBER_REC_CLUSTERS-'].update(value = number_recommended_clusters)
  ##          window.FindElement['-LISTBOX_FEATURES-'].update(values = lst)
  
            listFeatures(lst) 
            
        if event == 'PCA graph output for the first two features':  
            
            print("PCA graph output for the first two features")
        
            if os.path.exists(dirResultsOutput):
                
                print("Confirmed") 
                
                if os.path.isfile(dirResultsOutput + "pca_graph.png"):
                    image = Image.open(dirResultsOutput + "pca_graph.png")
                    image.thumbnail(MAX_SIZE)
                     
                    bio = io.BytesIO()
            ##        bio = io     ## .save(bio, format="PNG") 
                    image.save(bio, format="PNG")
                    window['-PCA-'].update(data = bio.getvalue())
                   
                    firstLoaded = True  
        if firstLoaded == True and event == 'Distances to the centroid':
             print("Distances to the centroid")
             second_gui_show_results(firstLoaded)  
             
             time.sleep(3)
             
             break



# import pypylon.pylon as py

def gui_recTime(repeat):
    import PySimpleGUI as sg
    import time 
    
    sliderPos = False
    rec_time = 0     
    
    layout = [
        [sg.Text('Recording Time:')],
        [sg.Slider(default_value = 5, orientation ='horizontal', key='recTime', range=(1,100)),
             sg.Text(size=(5,2), key='-SECONDS-'), sg.Text(" seconds")], [sg.Button('Next')]
    ]
    
    window = sg.Window('GUI', layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))  
    
    f = False
    
    while repeat == True:             # Event Loop
            event, values = window.read()
            print(event, values)         
            
            if event == 'Next':    ## and sliderPos == True
            
       #             if sliderPos == True:
           
                        rec_time = int(values['recTime'])
               
                        print("Recording time catched: " + str(rec_time))  
                        f = True
                        break
     #               else: 
    #                    rec_time = 5
    #                    print("Recording time catched: " + str(rec_time)) 
    #                    f = True
    #                    break
            if f == True:
                break
            
            if event == 'recTime':
                print("Event up to slider triggered !!!")
                window_sliderVal = sg.Window('', [sg.Text("", size=(0, 1), key='OUTPUT')]).read(close=True)
                
                window_sliderVal.bind('<FocusOut>', '+FOCUS OUT+')
    
                window_sliderVal['-BUTTON-'].bind('<Button-3>', '+RIGHT CLICK+')
                window_sliderVal['-TEXT-'].bind('<Enter>', '+MOUSE OVER+')
                window_sliderVal['-TEXT-'].bind('<Leave>', '+MOUSE AWAY+')
                window_sliderVal['-IN-'].bind('<FocusIn>', '+INPUT FOCUS+')
                
                eventSlider, valSlider = window_sliderVal.read()
                if eventSlider == sg.WINDOW_CLOSED:
                    break
                else:
                    window_sliderVal['OUTPUT'].update(value=values['recTime'])   
                    
                    rec_time = int(values['recTime'])
                    
           
           ##     time.sleep(2) 
                
                sliderPos = True
                 
                window_sliderVal.close()
                
    window.close() 

    return rec_time          
    

def basler_configs(gap_bet_tests, repeat, dict_data_basler, dirStorage, numberTests, startTime, enab_timestamps, rec_Time):
    
   from countdown_timer import countdown_timer_display
   from getCurrentDateAndTime import getDateTimeStrMarker  
   from softwareImageProcessingForLaserSpeckleAnalysisFinal import videoAnalysis
   import time
   import string 
   import keyboard
     
##   rec_Time = gui_recTime(repeat)    
    
   frame_rate = int(dict_data_basler['AcquisitionFrameRateAbs'])
   
   print("Rec. time: " + str(rec_Time))
   print("Frame rate: " + str(frame_rate))   
   
   numberImages = rec_Time*frame_rate
   
   print("Number of images: " + str(numberImages))
   
   counterTest = 0 
   curTest = 0
   
   indT = 0
    
   base_path = dirStorage
   
   alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase)
  
   counterTest = 0
   key = False
   key_not_pressed = 0
   
   time_gap_min = 30 
   
   print("numberTests: " + str(numberTests))   
   
   
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
           break
           
 
        
       # for letter in alphabet:
       #     if keyboard.is_pressed(letter): 
       #         print("key pressed") 
       #         key = True
       
   if key == True:
       if True:
           
           from video_25_4_23 import dynamic_video_repr
           
           again = True
           
 ##          cv2.namedWindow('image', cv2.WINDOW_NORMAL)
           
           if True: 
           
               again = dynamic_video_repr(int(dict_data_basler['Width']),
                                int(dict_data_basler['Height']),
                                int(dict_data_basler['ExposureTimeRaw']),
                                int(dict_data_basler['GainRaw']))
          
           while curTest < numberTests:
               
               print("Current test: " + str(curTest))
               print("Number tests: " + str(numberTests))
               
               key_not_pressed = 0  
    
               camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
               converter = pylon.ImageFormatConverter()               
               
        #       if not camera.IsOpen:               
               
               camera.Open()   
             
                   
                 
            ##   camera.CenterX = int(dict_data_basler['CenterX'])
               camera.CenterX = bool(int(dict_data_basler['CenterX']))
               camera.CenterY = bool(int(dict_data_basler['CenterY']))             
                            
               # Set the upper limit of the camera's frame rate to 30 fps      
               
               
               camera.AcquisitionFrameRateEnable.SetValue(True)
               camera.AcquisitionFrameRateAbs.SetValue(int(dict_data_basler['AcquisitionFrameRateAbs']))
                       
               camera.GevSCPSPacketSize.SetValue(int(dict_data_basler['GevSCPSPacketSize']))
                          
               # Inter-Packet Delay            
               camera.GevSCPD.SetValue(int(dict_data_basler['GevSCPD']))
                            
               # Bandwidth Reserve 
               camera.GevSCBWR.SetValue(int(dict_data_basler['GevSCBWR']))
                            
               # Bandwidth Reserve Accumulation
               camera.GevSCBWRA.SetValue(int(dict_data_basler['GevSCBWRA']) )
               
               new_width = camera.Width.GetValue() - camera.Width.GetInc()
               if new_width >= camera.Width.GetMin():
                   camera.Width.SetValue(new_width)
             
               camera.Width.SetValue(int(dict_data_basler['Width']))
               camera.Height.SetValue(int(dict_data_basler['Height']))
               
               
        ##       camera.Open()   
              
               camera.GainRaw = int(dict_data_basler['GainRaw'])
               camera.ExposureTimeRaw = int(dict_data_basler['ExposureTimeRaw'])
               
           #    camera.StartGrabbing()
               camera.StartGrabbingMax(numberImages)
               
               counter = 0  
               
         ##      camera.ImageCompressionMode = "BaslerCompressionBeyond"
          ##     camera.ImageCompressionRateOption = "Lossless"
          ##     camera.BslImageCompressionRatio = 30.0
               
         ##      print(str(camera.IsGrabbing()))
         
         
               buf_imgs = []
               
               
               initTime = 0
               initTimeStamp = 0
               
               
               import csv 
               
               def write_csv_line(filename, data):
                   with open(filename, 'a', newline = '') as file:
                       writer = csv.writer(file)
                       writer.writerow(data) 
                       
               csv_timestamps_filename = ""
                       
               def get_csv_filename_timestamps():
               
                   import PySimpleGUI as sg
                   
                   csv_timestamps_filename = sg.popup_get_text("Specify the CSV filename to save the timestamps along the next acquisition: ")
                   
                   return csv_timestamps_filename
               
               if enab_timestamps:
               
                   while len(csv_timestamps_filename) == 0:
                       csv_timestamps_filename = get_csv_filename_timestamps()
              #     csv_timestamps_filename = input("Specify the CSV filename to save the timestamps along the next acquisition: ")
                   csv_timestamps_filename += '.csv'
                   
                   title_line = ["Number of image", "Acquisition time (sec.)", "Time since start of acquisition"]
                   
                   write_csv_line(csv_timestamps_filename, title_line)
                   
                   time_x = 0
                   
               times = time.time()               
                
               
               while camera.IsGrabbing():
            ##       print("Grabbing")
         ##          print(str(numberImages))
          #         if counter < numberImages:
                       grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                       
           #            print("Grab Result: " + str(grabResult.GrabSucceeded()))
                       
                       if grabResult.GrabSucceeded():                               
            
                           # i+=1
                           # j+=1
                           # if i==700:
                           #     print(i) 
                               #camera.Open()
                               # camera.Height.SetValue( 200)
                               # camera.Width.SetValue( 300)
                               
                               
                           #     i=0 
                           # if j==500:
                               #camera.Open()
                               # print(j)
                               
                               # camera.Height.SetValue(1088)
                               # camera.Width.SetValue( 2048)
                               
                               # camera.Gain=20
                               # camera.ExposureTime=8000
                               # j=0
                           # Access the image data.
                           
                          ## print("AMAR :",camera.IsOpen())
               ##            print("Image " + str(counter))
                           image = converter.Convert(grabResult)
                           img = image.GetArray()
                           
                           ########################
                           
                    #       img = Image.fromarray(img)
                           
                    #       output_buffer = io.BytesIO()
                    #       img.save(output_buffer, format='JPEG', quality=80)
                           
                    #       img = np.asarray(img)
                           
                           ########################
                           
                           
            #               if counter > 0:
                           
                           buf_imgs.append(img)
                               
                       #         if enab_timestamps:
                                
                       #             unixTimeAcq = (time.time())*(1e6)
                       #         ##    time_unit = grabResult.GetTimeStampModule().GetTimeUnit()
                       #             if counter > 2:                                  
                       #    #             print("Time X:" + str(time_x))
                       #                 acquisitionTime = round(((unixTimeAcq- time_x)/1e6),5)
                       #                 initTime += acquisitionTime  
                                    
                                           
                                       
                                       
                       # #                print(f"Acquisition time: {acquisitionTime:.3f} seconds")
                       # #                print("Time since start of acquisition: " + str(round(initTime,5)))
                                       
                       #                 data = [counter, acquisitionTime, round(initTime,5)] 
                                       
                       #                 write_csv_line(csv_timestamps_filename, data)
                                   
                       #             time_x = unixTimeAcq 
                           # if counter == numberImages:
                           #        break
                            
                         ##  cv2.imshow("image",img)
                         
              ##             shown_check = showLiveImageGUI(img, counter)
            ##               print("Image shown: " + str(shown_check))
                           
                           
                            
          ##                     cv2.imwrite(base_path + "/image" + str(counter) + ".tiff", img)
                           
                   #        counter += 1  
                   # else: 
                   #     break
                        
                ##       cv2.waitKey(1)
                
               timef = time.time()
               
              
               
               tx = time.time()
                
               for ind_img , img in enumerate(buf_imgs):
                    cv2.imwrite(base_path + "/image" + str(ind_img) + ".tiff", img)
               
               folderThisTest = base_path + '/Test_' + str(curTest) + '/'
                
               os.mkdir(os.path.join(folderThisTest))
                
               flag = True
                
               while flag: 
                    if os.path.isdir(folderThisTest):
                        flag = False
                    else:
                        print("Directory not created")
             
                
         #      time.sleep(1000)
                        
                 
               for ind_img , img in enumerate(buf_imgs):
             ##        cv2.imwrite(base_path + "/image" + str(ind_img) + ".tiff", img)
                    
                     cv2.imwrite(folderThisTest + 'image' + str(ind_img) + ".tiff", img)
                     
                    
                
               print("End of image acquisition")
               
               grabResult.Release()
               camera.Close()
               
               print("Converting to video ... ")
               
              
               
               img_array = []
               ind = 0
               for filename in glob.glob(base_path + '/*.tiff'):      ## base_path2 + 'files_python/Img_track/*.tiff'
                   print("A")
                   if "image" in filename:
                       print("B")
                       img = cv2.imread(filename)
                       height, width, layers = img.shape 
                       size = (width,height) 
                       print(size)
                       img_array.append(img)
                       print("Image " + str(ind) + " drained to set")
                       ind += 1
               
               out = cv2.VideoWriter('test_' + '00' + str(counterTest) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                
               for i in range(len(img_array)):
                   out.write(img_array[i])
               out.release() 
               
               folderThisTest_h = folderThisTest[:-1]
               
               img_array = []
               ind = 0
               for filename in glob.glob(folderThisTest_h + '/*.tiff'):      ## base_path2 + 'files_python/Img_track/*.tiff'
                   print("A")
                   if "image" in filename:
                       print("B")
                       img = cv2.imread(filename)
                       height, width, layers = img.shape 
                       size = (width,height) 
                       print(size)
                       img_array.append(img)
                       print("Image " + str(ind) + " drained to set")
                       ind += 1
               
               out = cv2.VideoWriter(folderThisTest + 'test_' + '00' + str(counterTest) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                
               for i in range(len(img_array)):
                   out.write(img_array[i])
               out.release()
               
               print("Conversion done")
                
               #%%
                
               dateTimeMarker = getDateTimeStrMarker()
               
               base_path3 = base_path.replace("/", "\\")
               
               sequence_name = counterTest
               dest_path = base_path + '/Image_Processing/'
               dest_path2 = base_path3 + "\\Image_Processing\\"  
               
               count_folder_ex = 0
               
               exists = True
               
               while exists == True:               
                   
                   if os.path.exists(dest_path) == True:                   
                       new_str_folder = '/Image_Processing' + str(count_folder_ex) + '/'
                       new_str_folder_sec = "\\Image_Processing" + str(count_folder_ex) + "\\" 
                       dest_path = base_path + new_str_folder 
                         
        
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
          #     pathRoiStart = dest_path + 'modRoisFirstMom_'
          #     pathRoiEnd = dest_path + 'modRoisSecMom_'
               first_clustering_storing_output = "Quality_kMeans_Clustering_real_"
               pathPythonFile = dest_path + "SpeckleTraining/ffmpeg-5.0.1-full_build/bin"
               
               ## Adding date and time to which one of the paths shown above
               IFVP = IFVP + dateTimeMarker
               locationMP4_file = locationMP4_file + dateTimeMarker  
               roiPath = roiPath + dateTimeMarker 
               newRoiPath = newRoiPath + dateTimeMarker 
 #              pathRoiStart = pathRoiStart + dateTimeMarker 
  #             pathRoiEnd = pathRoiEnd + dateTimeMarker 
               first_clustering_storing_output = first_clustering_storing_output + dateTimeMarker
               
               decisorLevel = 0
               
               ##################  
               
               base_name = "config_dirs"
               
               filename = base_name + getDateTimeStrMarker()
               
               with open(base_path + "_" + filename + ".txt", "w") as f:
                   f.write("Configs")
                   f.write("\n") 
                   f.write("\n")
                   f.write("Decisor Level: \t\t\t\t\t\t" + str(decisorLevel))
                   f.write("\n")
                   f.write("Main Path: \t\t\t\t\t\t" + str(mainPathVideoData))
                   f.write("\n")
                   f.write("Sequence Name: \t\t\t\t\t\t" + str(indT))
                   f.write("\n")
                   f.write("Destination Path: \t\t\t\t\t" + str(dest_path))
                   f.write("\n")
                   f.write("MTS video file path: \t\t\t\t\t" + str(mtsVideoPath))
                   f.write("\n")                 
                   f.write("MP4 Video Filename: \t\t\t\t\t" + str(mp4VideoFile))
                   f.write("\n")
                   f.write("Storage folder for this sequence: \t\t\t" + str(IFVP))
                   f.write("\n")
                   f.write("MP4 File Location: \t\t\t\t\t" + str(locationMP4_file))
                   f.write("\n")
                   f.write("ROI Path - First Stage: \t\t\t\t" + str(roiPath))
                   f.write("\n")
                   f.write("ROI Path - Second Stage: \t\t\t\t" + str(newRoiPath))
                   f.write("\n")
                   f.write("First Clustering Storage Output: \t\t\t" + str(first_clustering_storing_output))
                   f.write("\n")
                   f.write("Python File Path: \t\t\t\t\t" + str(pathPythonFile))
                   f.write("\n")  
                   
               print("Configs written")
               
               gap_between_tests = gap_bet_tests*60
               
               ty = time.time()
                
               print("Time between: " + str(round((abs(ty-tx)),3)))
               print("Time between, in aquisition: " + str(round((abs(timef-times)),3)))
               
               if curTest == 0: 
               
                   write_to_txt_file_tests_info(base_path, [numberTests, rec_Time, gap_between_tests]) 
                   print("Tests info written") 
                   
               if curTest != numberTests-1: 
                   
                   t_another = time.time()                  
                   
                   diff_time_extra = abs(t_another-tx)
               
                   nextTest = countdown_timer_display(gap_bet_tests)   ## -diff_time_extra   ## in minutes         
                   nextTest = True
                   print("Ready for next test: " + str(nextTest))
               
               curTest += 1
               
               indT += 1
                
               ###################
                   
                   # print("Test number " + str(curTest+1))         
                   
                   
                   # if numberTests == 1:
                   #      data_to_save = []  
                        
                                        
                   #      layout = [[sg.ProgressBar(1000, orientation='h', size = (20,20), key = 'progressbar')]]
    
                   #      window = sg.Window('Processing ...', layout)
                       
                   #      while True:
                            
                   #          event, values = window.read(timeout=1000)
                            
                   #          if event == sg.WIN_CLOSED:
                   #              break
                            
                   #          if True:
                   #              data_from_tests = []
                        
                   #              infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile)
                                                    
                   #              clusteringInfData, executionTime, totCountImages = videoAnalysis(curTest, numberTests, infi, data_from_tests)               
                   #              datax = (clusteringInfData, executionTime, totCountImages)  
                                
                   #              gui_show_results(clusteringInfData, executionTime, totCountImages) 
                                
                   #              data_to_save.append(datax)  
                   #              break
                            
                   #      window.close()
                        
                   #      break
                         
                   # else:   
                   
                   
                   
                   #     if curTest < numberTests-1: 
                            
                   #         if curTest == 0:
                   #             infx = []
                   #             data_to_save = []
                               
                   #         decisorLevel = 0
                               
                   #         infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile))
                           
                   #         nextTest = countdown_timer_display(gap_bet_tests)    ## in minutes         
                           
                   #         print("Ready for next test: " + str(nextTest))
                           
                   #     elif curTest == numberTests-1:               
                           
                   #         infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile))
                           
                   #         data_from_tests = []
                           
                   #         for ind_inf, infi in enumerate(infx):
                               
                   #             layout = [[sg.ProgressBar(1000, orientation='h', size = (20,20), key = 'progressbar')]]
    
                   #             window = sg.Window('Processing ...', layout)
                              
                   #             while True: 
                                   
                   #                 event, values = window.read(timeout=1000)
                                   
                   #                 if event == sg.WIN_CLOSED:
                   #                     break
                                   
                   #                 if True:
                                       
                   #                     print("Looking for info " + str(ind_inf) + " ...")
                                       
                   #                     time.sleep(10)
                                       
                   #                     if ind_inf != len(infx) - 1:
                               
                   #                         data_from_tests = videoAnalysis(curTest, numberTests, infi, data_from_tests) 
                                           
                   #                         print("Length for data_from_tests: " + str(len(data_from_tests)))
                                           
                   #                         if len(data_from_tests) > 0:
                   #                             for d in data_from_tests:
                   #                                 print("Data: ")
                   #                                 for x in d:
                   #                                     print(str(x) + " \t")
                                       
                   #                     else:
                   #                         clustering_output = videoAnalysis(curTest, numberTests, infi, data_from_tests) 
                                           
                   #                         for c in clustering_output:                                 
                   #                             datax = c
                                       
                   #                             data_to_save.append(datax) 
                   #                     break 
                                   
                   #             window.close() 
                            
                   #         counterTest += 1 
                           
                   #         if len(data_to_save) > 0: 
            
                   #           for ind_data, data in enumerate(data_to_save):
                                 
                   #                  print("Length of Data after: " + str(len(data))) 
                                    
                   #                  if len(data) != 3: 
                   #                      print("Check length data ...")
                   #                  if len(data) == 3:
                                           
                   #                      clusteringRes, execTime, numberImg = data       
                                                   
                   #                      gui_show_results(clusteringRes, execTime, numberImg) 
                                                   
                   #                      print(" -- Showing GUI with results for test number " + str(ind_data+1))
                                                   
                   #                      time.sleep(5)       
                                  
                                        
                   #         else:
                   #           print("Output results not available !!!")                  
                
                
                   #                 ###################
                
                   #         executionTime = (time.time() - startTime)
                   #         print('Whole execution time in seconds: ' + str(executionTime))
                    
                    
                   #         break 
                   #     curTest += 1  
   else:
             key_not_pressed += 1
          
             if key_not_pressed == 1:
                 startTime = time.time() 
             else:
                 if key_not_pressed > 1:
                     if (time.time() - startTime) > 60*time_gap_min:
                         print("Ending tests set ...")
       ##                  break
       

      
def read_pfs_file(gap_bet_tests, numberTests, repeat, startTime, enab_val_pfs, enab_timestamps):
    
    import PySimpleGUI as sg
    
    print("PFS file option selected ...")
    
    dirStorage = ""
    
    if repeat == True:  
        
 ##       for i in range(0,numberTests-1):    
            
            
            valid_pfs = False
            
            rec_Time = gui_recTime(repeat) 
            
            
            while( valid_pfs == False):
            
                windowx = sg.Window('Choose path to PFS file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
                (keyword, dict_dir) = windowx                 
            
                dir_path = dict_dir['Browse'] 
                
                if enab_val_pfs:
                    valid_pfs = dateAndTimeConfirm(dir_path, dirStorage)
                else:
                    valid_pfs = True               
                
            
            print("Path for data storing, by PFS file option: " + dir_path) 
            
            windowx = sg.Window('Choose directory folder', [[sg.Text('Folder name')], [sg.Input(), sg.FolderBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
            (keyword, dict_dir) = windowx                
         
            dir_path_b = dict_dir['Browse'] 
             
            dirStorage = dir_path_b
            
            print("Dir Path: " + dir_path_b)
            
            print("Reading PFS file ...")
            
            f = open(dir_path) 
            
            data = f.read() 
            
            real_data = []
            
            numberLines = 0
            
        ##    with open("file.pfs", "r") as input:
            
            with open(dir_path, "r") as input:        
                
                for ind_line, line in enumerate(input):
                    numberLines += 1
                    
                    if '#' in line or '000000000000000' in line:
                        print("Discarding line " + str(ind_line))
                    else:
                        real_data.append(line)
            
            feature_names = [] 
            features_data= []
            
            for feature_line in real_data:
                print("Feature Line: " + feature_line)
                featureInfo = feature_line.split()
                
                print(featureInfo)
                
                feature_names.append(featureInfo[0])
                features_data.append(featureInfo[1])
                
            listIndex = []
            listValuesFeatures = []
            listNamesFeatures = []
                
            featuresForPylon = ['CenterX', 
                                'CenterY', 
                                'AcquisitionFrameRateEnable',
                                'AcquisitionFrameRateAbs',
                                'GevSCPSPacketSize',
                                'GevSCPD',
                                'GevSCBWR',
                                'GevSCBWRA',
                                'Width',
                                'Height',
                                'GainRaw',
                                'ExposureTimeRaw']         
            
            for ind_featName, featName in enumerate(featuresForPylon):
                indexFeat = feature_names.index(featName)
                listIndex.append(indexFeat)
           ##     listNamesFeatures.append()
                
            for indx in listIndex:
                listValuesFeatures.append(features_data[indx])
                listNamesFeatures.append(feature_names[indx])
                     
            # for ind_featureName, featureName in enumerate(featuresForPylon):
            #     if ind_featureName in listIndex:
            #         listNamesFeatures.append(featureName)   
                    
            featuresFromPFS_file = [listNamesFeatures, listValuesFeatures]
                 
            featuresFromPFS_file = np.array(featuresFromPFS_file).T.tolist()
                    
            dict_data_pfs = dict(featuresFromPFS_file)
            ## Get only the desired features and put all the code above inside def block 
            
             
            basler_configs(gap_bet_tests, repeat, dict_data_pfs, dirStorage, numberTests, startTime, enab_timestamps, rec_Time)           
        # -*- coding: utf-8 -*-

