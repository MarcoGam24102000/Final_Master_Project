# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:03:49 2022

@author: Marco Gameiro
""" 
 

respx = False
  
import PySimpleGUIWeb as sg
import PySimpleGUI as sg_py 
from pypylon import pylon   
import numpy as np   
import io   
import os     
from PIL import Image  
## from paramsPylon import setParametersToPypylon 
from full_code import whole_processing_software, whole_processing_software_adv
from open_form_gui import exp_control  
from countdown_timer import countdown_timer_display 
from info_gui import theory_info  
from pfs_input import read_pfs_file 
# from pfs_input_acq_step_only import read_pfs_file
from getCurrentDateAndTime import getDateTimeStrMarker 
import keyboard       
import time     
import webbrowser      
from fac_params import extra_params_gui 
from gui_advanced import adv_params_gui 
from softwareImageProcessingForLaserSpeckleAnalysisFinal import videoAnalysis
from check_basler_camera import confirm_basler
from live_camera_image import acq_image_camera 
from check_centered_image import help_camera_center, get_ok, set_ok
from test_image_preview import get_test_image
from optional_extra_prop import optional_prop, optional_prop_norm  
from gen_gui_working import control_gui 
from common import splitfn  
import threading 
## from threading import Event 
import multiprocessing  
from threading import Thread 
import glob
import cv2 
import subprocess 
import functools    
import shlex  
import imageio   
import sys
## import queue

# from queue import PriorityQueue

# queue = PriorityQueue()
 
startTime = time.time() 

adv_written = False 

itemHeight = ["135", "240", "270", "480", "540", "960", "1080", "1920"]


def repeat_loop_proc():
    
    import PySimpleGUI as sg
    
    layout = [
        [sg.Text("One more ?")],
        [sg.T("         "), sg.Checkbox('Yes', default=False, key="-IN1-")],
        [sg.T("         "), sg.Checkbox('No', default=True, key="-IN2-")],
        [sg.Button("Exit"), sg.Button("Next")]
    ]
    
    window = sg.Window("Repeat Loop", layout)
    
    again = True
    proc_again = 0
    
    while again == True:
        event, values = window.read()
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "Next":
            if values["-IN1-"] and values["-IN2-"]:
                print("Please select only one option ...")
                again = True
            elif not values["-IN1-"] and not values["-IN2-"]:
                print("Select one option ...")
                again = True
            else:
                if values["-IN1-"] and not values["-IN2-"]:
                    print("One more ...")
                    proc_again = 2
                elif values["-IN2-"] and not values["-IN1-"]:
                    print("Ending ...")
                    proc_again = 1
                    
                break
            
    window.close() 
    
    return proc_again   
    
 
def stop_button_cam_imgs_layout(window):
   
##    start_time = time.time()

    import PySimpleGUI as sg 
    
    resp = False

    while True:
   
        event, values = window.read() 
        print(event, values)        
              
                 
        if event == "Exit" or event == sg.WIN_CLOSED:
            resp = False
            break
                
        if event == 'Live camera images routine control':
            resp = True
            break
            
      ##      window.close()    
            
            # process = multiprocessing.current_process()            
            # process.terminate()
            break
        
        print(resp)
    
    window.close()
    
    # if resp:
    #     respx = resp 
                
    #     item = (priority, int(resp))
            
    #     queue.put(item)    
 
    return resp
  
    #     else:
    #         break 
        
    
    
  ##  return resp 



class CustomThread(Thread):
    # constructor
    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
        
        Thread.setDaemon(self, True)       
       
        # set a default value
        self.value = None
 
    # function executed in a new thread
    def run(self):
        # block for a moment
   ##     time.sleep(1)
        # store data in an instance variable
        
        resp = stop_button_cam_imgs_layout()
        
        time.sleep(1)
        
        self.value = str(resp)       
         
        
        import multiprocessing
        
        process = multiprocessing.current_process()        
        process.terminate()
        
def get_basic_stream_params():
    
    import PySimpleGUI as sg
    
    
    
    packet_size = 1500
    inter_packet_delay = 5000
    bw_resv_acc = 4
    bw_resv = 10  


    layout = [
        [sg.Text('Packet Size:')], 
        [sg.InputText(default_text= "1500", size=(19, 1), key="PACKET_SIZE")],     ## , sg.Button('Copy')
        [sg.Text('Inter-Packet Delay:')],
        [sg.Input(default_text= "5000", size=(19, 1), key="INTER_PACKET_DELAY")],
        [sg.Text('Bandwidth Reserve Accumulation:')],
        [sg.Input(default_text= "4", size=(19, 1), key="BANDWIDTH_RESV_ACC")],
        [sg.Text('Bandwidth Reserve:')],
        [sg.Input(default_text= "10", size=(19, 1), key="BANDWIDTH_RESV")],            
        [sg.Button('Save'), sg.Button('Next')]
    ]  
    
    window = sg.Window('Basic Streaming parameters', layout)
    
    valid = False
    save_opt = False
    
    while valid == False:             # Event Loop
            event, values = window.read()
            print(event, values)        
          
             
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "Save":
                
                save_opt = True
                
                if len(values["PACKET_SIZE"]) == 0:                 
                    valid = True 
                else:
                    if int(values["PACKET_SIZE"]) <= 0:
                        sg.popup_error('Packet size must be positive')
                        valid = False
                        print("Failed to packet size range")
                    else:
                        valid = True                        
                        packet_size = int(values["PACKET_SIZE"])
                                                
                
                if len(values["INTER_PACKET_DELAY"]) == 0:                    
                    valid = True 
                else:    
                    if int(values["INTER_PACKET_DELAY"]) < 0 or int(values["INTER_PACKET_DELAY"]) > 10000:
                        sg.popup_error('Inter-packet delay must be within [0,10000] interval') 
                        valid = False  
                        print("Failed to inter-packet delay range") 
                    else:
                        valid = True
                        inter_packet_delay = int(values["INTER_PACKET_DELAY"])
                
                if len(values["BANDWIDTH_RESV_ACC"]) == 0:                    
                    valid = True
                else:
                    if int(values["BANDWIDTH_RESV_ACC"]) < 0 or int(values["BANDWIDTH_RESV_ACC"]) > 10:
                        sg.popup_error('Bandwidth reserve accumulation must be within [0,100] interval')  
                        valid = False
                        print("Failed to bandwidth reserve accumalation range")
                    else:
                        valid = True
                        bw_resv_acc = int(values["BANDWIDTH_RESV_ACC"])
                         
                if len(values["BANDWIDTH_RESV"]) == 0:                     
                    valid = True
                else:                     
                    if int(values["BANDWIDTH_RESV"]) < 0 or int(values["BANDWIDTH_RESV"]) > 100:
                        sg.popup_error('Bandwidth reserve must be within [0,10] interval')   
                        valid = False
                        print("Failed to bandwidth reserve range")
                    else:
                        valid = True          
                        bw_resv = int(values["BANDWIDTH_RESV"])
                
                
            if event == "Next" and save_opt:
                save_opt = False
                break
            
            
    window.close()
    
    return packet_size, inter_packet_delay, bw_resv_acc, bw_resv
     

def gui_calib_params(fx, fy, ppx, ppy):
    
        import PySimpleGUI as sg
    
    
        layout = [
            [sg.Text("Calibration parameters:")],
            [sg.Text('Fx:'), sg.Input(default_text= str(fx), key="Fx")],
            [sg.Text('Fy:'), sg.Input(default_text= str(fy), key="Fy")],
            [sg.Text('PPx:'), sg.Input(default_text= str(ppx), key="PPx")],
            [sg.Text('PPy:'), sg.Input(default_text= str(ppy), key="PPy")]
        ]
        
        
        layout_opt = [
            [sg.Button("Back")]
        ]
        
        
        layout = [ 
            [
                sg.Column(layout),
                sg.VSeparator(),                            
                sg.Column(layout_opt)
            ]
        ]
        
        window = sg.Window('Camera calibration results', layout)
        
        code_ret = -1
        
        while True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               code_ret = 0
               break
           if event == "Back":
               code_ret = 1
               break
           
        window.close()
        
        return code_ret 

def gui_png_file(): 
     
     import PySimpleGUI as sg
     
     again = True
     
     while again == True:
         
         print("Here")
     
         windowx = sg.Window('Choose path to png file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ])
         
         event, values = window.read()
         
         if event == sg.WINDOW_CLOSED or event == 'Cancel':
             # Exit the loop when the window is closed or "Cancel" is pressed
             again = True 
         else:
             (keyword, dict_dir) = windowx                
         
             dir_png_path = dict_dir['Browse'] 
             
             if dir_png_path is None:
                 again = True
             else:
             
                 if '/' in dir_png_path:
                     dir_png_parts = dir_png_path.split('/')
                 elif "\\" in dir_png_path:
                     dir_png_parts = dir_png_path.split("\\")
                 
                 png_filename = dir_png_parts[-1]    ## Check if it is a bag file
                 
                 if not('.png' in png_filename or '.jpg' in png_filename or '.tiff' in png_filename):
                     again = True
                 else:
                     again = False
                     filepath_parts = dir_png_parts[:-1]
                     
                     if '.jpg' in png_filename:
                          imx = cv2.imread(dir_png_path)
                          
                          dir_parts = dir_png_path.split('.jpg')
                          
                          for d in dir_parts:
                              if len(d) != 0:
                                 dir_without_ext = d
                                 
                          cv2.imwrite(dir_without_ext + '.png', imx)
                          
                     if '.tiff' in png_filename:
                         imx = cv2.imread(dir_png_path)
                         
                         dir_parts = dir_png_path.split('.tiff')
                         
                         for d in dir_parts:
                             if len(d) != 0:
                                dir_without_ext = d
                                
                         cv2.imwrite(dir_without_ext + '.png', imx)
    
     windowx.close()
     
     base_dir_png = ''
     
     for ind_d, d in enumerate(dir_png_parts):
          
         if ind_d < len(dir_png_parts) - 1: 
             base_dir_png += d + '/'
      
     png_info = [base_dir_png, png_filename]
      
     return dir_png_path
 

def write_data_to_csv_file(data_lab, metadata_csv_filename):
    
    with open(metadata_csv_filename, 'a', encoding='utf-8') as f:
        
        for data in data_lab: 
            line = ', '.join(data)
            f.write(line + '\n')
            
            print("One more line written to csv file ...") 
            
 
def write_to_csv_calib_params_database(calib_params, csv_database_filename, counter):
    
    if counter == 0:
        headers = ["Fx", "Fy", "PPx", "PPy"]
    
    ## Write calib_params line to csv file   
    
    with open(csv_database_filename, 'a', encoding='utf-8') as f:
        line = ', '.join(calib_params)
        f.write(line + '\n')
        
        print("One more set of calibration parameters written to csv file ...")       
        
 
def ask_user_metadata_filename():
    
   repeat = True   
   
   layout=[[sg_py.Text('Output metadata filename:'), sg_py.Input(default_text= "", size=(19, 1), key="META_INPUT")],
           [sg_py.Button("Next")]
          ]
   
   window = sg_py.Window("Save calibration parameters", layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))
  
   while repeat == True:
      event, values = window.read()
      if event == "Exit" or event == sg.WIN_CLOSED:
          break
      if event == 'Next':
          metadata_filename = values['META_INPUT']
          
          if not metadata_filename:
              repeat = True
          else:
              repeat = False
          
          break
          
   window.close()
   
   if not('.csv' in metadata_filename):
       metadata_filename += getDateTimeStrMarker() + '.csv'
   else:
       meta_parts = metadata_filename.split('.csv')
       
       for d in meta_parts:
           if len(d) != 0: 
               metadata_without = d
       
       metadata_filename = metadata_without + getDateTimeStrMarker() + '.csv'                
   
   return metadata_filename     
    
    
def write_params_to_metadata_file(params_calib, metadata_csv_filename, model_name, counter_for_params, width, height): 
    
   
    print("Writing headers ...")   ## Define which headers and then write them
        
    frame_number = counter_for_params
    resolution_x = width
    resolution_y = height 
    bytes_per_pixel = 1    ## 8 bits
        
    title_one = "Frame Info: "
    type_line = ["Type", "Basler " + str(model_name)]   ## Get name of camera and add here  ## "Basler ..."
    format_line = ["Format", "Y" + str(bytes_per_pixel*8)]       
    frame_number_line = ["Frame Number", str(counter_for_params+1)]
    resolution_x_line = ["Resolution x", str(width)]
    resolution_y_line = ["Resolution y", str(height)]
    bytes_p_pix_line = ["Bytes per pixel", str(bytes_per_pixel)]
    
    empty_line = "" 
        
    fx = params_calib[0]
    fy = params_calib[1]
    ppx = params_calib[2]
    ppy = params_calib[3]  
    
    title_two = "Intrinsic:"
    title_two = [title_two, ""]
    
    fx_line = ["Fx", round(fx,6)]
    fy_line = ["Fy", round(fy,6)]
    ppx_line = ["PPx", round(ppx,6)]  
    ppy_line = ["PPy", round(ppy,6)]
    
    distorsion = "Brown Conrady"     ## search for it and explain it on the report 
        
    data_lab = [title_one, type_line, format_line, frame_number_line, 
                frame_number_line, resolution_x_line, resolution_y_line, 
                bytes_p_pix_line, empty_line, title_two, fx_line, fy_line, 
                ppx_line, ppy_line, distorsion]
    
    write_data_to_csv_file(data_lab, metadata_csv_filename)

def calib_camera(continuous, model_name):
    
    print("Here on calib_camera")
    
    counter_for_params = 0
    
    def ask_user_metadata_filename():
        
       repeat = True 
       
       import PySimpleGUI as sg_py
       from getCurrentDateAndTime import getDateTimeStrMarker
       
       layout=[[sg_py.Text('Output metadata filename:'), sg_py.Input(default_text= "", size=(19, 1), key="META_INPUT")],
               [sg_py.Button("Next")]
              ]
       
       window = sg_py.Window("Save calibration parameters", layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))
      
       while repeat == True:
          event, values = window.read()
          if event == "Exit" or event == sg.WIN_CLOSED:
              break
          if event == 'Next':
              metadata_filename = values['META_INPUT']
              
              if not metadata_filename:
                  repeat = True
              else:
                  repeat = False
              
              break
              
       window.close()
       
       if not('.csv' in metadata_filename):
           metadata_filename += getDateTimeStrMarker() + '.csv'
       else:
           meta_parts = metadata_filename.split('.csv')
           
           for d in meta_parts:
               if len(d) != 0: 
                   metadata_without = d
           
           metadata_filename = metadata_without + getDateTimeStrMarker() + '.csv'                
       
       return metadata_filename     
        
    
    def gui_png_file(): 
         
         import PySimpleGUI as sg
         
         again = True
         
         while again == True:
             
             print("Here")
         
             windowx = sg.Window('Choose path to png file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ])
             
             event, values = windowx.read()
            
             if event == sg.WINDOW_CLOSED or event == 'Cancel':
                 # Exit the loop when the window is closed or "Cancel" is pressed
                 windowx.close()
                 again = True 
             else:
                 
                 (keyword, dict_dir) = windowx                
             
                 dir_png_path = dict_dir['Browse'] 
                 
                 if dir_png_path is None:
                     again = True
                 else:
                 
                     if '/' in dir_png_path:
                         dir_png_parts = dir_png_path.split('/')
                     elif "\\" in dir_png_path:
                         dir_png_parts = dir_png_path.split("\\")
                     
                     png_filename = dir_png_parts[-1]    ## Check if it is a bag file
                     
                     if not('.png' in png_filename or '.jpg' in png_filename or '.tiff' in png_filename):
                         again = True
                     else:
                         again = False
                         filepath_parts = dir_png_parts[:-1]
                         
                         if '.jpg' in png_filename:
                              imx = cv2.imread(dir_png_path)
                              
                              dir_parts = dir_png_path.split('.jpg')
                              
                              for d in dir_parts:
                                  if len(d) != 0:
                                     dir_without_ext = d
                                     
                              cv2.imwrite(dir_without_ext + '.png', imx)
                              
                         if '.tiff' in png_filename:
                             imx = cv2.imread(dir_png_path)
                             
                             dir_parts = dir_png_path.split('.tiff')
                             
                             for d in dir_parts:
                                 if len(d) != 0:
                                    dir_without_ext = d
                                    
                             cv2.imwrite(dir_without_ext + '.png', imx)
         windowx.close()    
         
         base_dir_png = ''
         
         for ind_d, d in enumerate(dir_png_parts):
              
             if ind_d < len(dir_png_parts) - 1: 
                 base_dir_png += d + '/'
          
         png_info = [base_dir_png, png_filename]
          
         return dir_png_path

    def exec_script():
        
        print("Here on exec_script")

        import sys
        import getopt
        from glob import glob 
        import cv2
        import os
        import numpy as np
        
        def write_data_to_csv_file(data_lab, metadata_csv_filename):
            
            with open(metadata_csv_filename, 'a', encoding='utf-8') as f:
                
                for data in data_lab: 
                    line = ', '.join(data)
                    f.write(line + '\n')
                    
                    print("One more line written to csv file ...") 
                    
         
        def write_to_csv_calib_params_database(calib_params, csv_database_filename, counter):
            
            if counter == 0:
                headers = ["Fx", "Fy", "PPx", "PPy"]
            
            ## Write calib_params line to csv file   
            
            with open(csv_database_filename, 'a', encoding='utf-8') as f:
                line = ', '.join(calib_params)
                f.write(line + '\n')
                
                print("One more set of calibration parameters written to csv file ...")       
        
        def write_params_to_metadata_file(params_calib, metadata_csv_filename, model_name, counter_for_params, width, height): 
            
           
            print("Writing headers ...")   ## Define which headers and then write them
                
            frame_number = counter_for_params
            resolution_x = width
            resolution_y = height 
            bytes_per_pixel = 1    ## 8 bits
                
            title_one = "Frame Info: "
            type_line = ["Type", "Basler " + str(model_name)]   ## Get name of camera and add here  ## "Basler ..."
            format_line = ["Format", "Y" + str(bytes_per_pixel*8)]       
            frame_number_line = ["Frame Number", str(counter_for_params+1)]
            resolution_x_line = ["Resolution x", str(width)]
            resolution_y_line = ["Resolution y", str(height)]
            bytes_p_pix_line = ["Bytes per pixel", str(bytes_per_pixel)]
            
            empty_line = "" 
                
            fx = params_calib[0]
            fy = params_calib[1]
            ppx = params_calib[2]
            ppy = params_calib[3]  
            
            title_two = "Intrinsic:"
            title_two = [title_two, ""]
            
            fx_line = ["Fx", round(fx,6)]
            fy_line = ["Fy", round(fy,6)]
            ppx_line = ["PPx", round(ppx,6)]  
            ppy_line = ["PPy", round(ppy,6)]
            
            distorsion = "Brown Conrady"     ## search for it and explain it on the report 
                
            data_lab = [title_one, type_line, format_line, frame_number_line, 
                        frame_number_line, resolution_x_line, resolution_y_line, 
                        bytes_p_pix_line, empty_line, title_two, fx_line, fy_line, 
                        ppx_line, ppy_line, distorsion]
            
            write_data_to_csv_file(data_lab, metadata_csv_filename)
        
        def gui_calib_params(fx, fy, ppx, ppy):
            
                import PySimpleGUI as sg
            
            
                layout = [
                    [sg.Text("Calibration parameters:")],
                    [sg.Text('Fx:'), sg.Input(default_text= str(fx), key="Fx")],
                    [sg.Text('Fy:'), sg.Input(default_text= str(fy), key="Fy")],
                    [sg.Text('PPx:'), sg.Input(default_text= str(ppx), key="PPx")],
                    [sg.Text('PPy:'), sg.Input(default_text= str(ppy), key="PPy")]
                ]
                
                
                layout_opt = [
                    [sg.Button("Back")]
                ]
                
                
                layout = [ 
                    [
                        sg.Column(layout),
                        sg.VSeparator(),                            
                        sg.Column(layout_opt)
                    ]
                ]
                
                window = sg.Window('Camera calibration results', layout)
                
                code_ret = -1
                
                while True:
                   event, values = window.read()
                   if event == "Exit" or event == sg.WIN_CLOSED:
                       code_ret = 0
                       break
                   if event == "Back":
                       code_ret = 1
                       break
                   
                window.close()
                
                return code_ret 
        
        if continuous == False:
            dir_png_path = gui_png_file()  
        
        im_inter = cv2.imread(dir_png_path)     
         
        counter_pre_resize = 0
        
        print("Size of image: (" + str(len(im_inter)) + "," + str(len(im_inter[0])) + "," + str(len(im_inter[1])) + ")")
        
        im_inter = cv2.cvtColor(im_inter, cv2.COLOR_BGR2GRAY)
        
        while len(im_inter) > 300 or len(im_inter[0]) > 300:
            
            dim = ((int(len(im_inter[0])/2)), int(len(im_inter)/2))
            print("Dims of input image: (" + str(dim[0]) + ' , ' + str(dim[1]) + ")")
                        
            im_inter = cv2.resize(im_inter, dim, interpolation = cv2.INTER_AREA)
            
            print(str(counter_pre_resize) + " time resizing input image (to calibration)")
            
            print("Shape of image after resizing: " + str(im_inter.shape))
             
            counter_pre_resize += 1 
            
        im_inter_bin = im_inter.copy()
            
        # for b in range(len(im_inter[0])):
        #     for a in range(len(im_inter)):
        #         if im_inter_bin[a,b] >= 125:
        #             im_inter_bin[a,b] = 255
        #         else:
        #             im_inter_bin[a,b] = 0
        
        
        
        for b in range(len(im_inter_bin[0])):
            for a in range(len(im_inter_bin)):
                im_inter_bin[a,b] *= 4
        
        cv2.imwrite(dir_png_path, im_inter)
        print("Written again ...") 
        

        args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])
        args = dict(args)
        args.setdefault('--debug', './output/')
        args.setdefault('--square_size', 1.0)
        args.setdefault('--threads', 4)
        if not img_mask:
            img_mask = dir_png_path      
            
        else:
            img_mask = img_mask[0]  
        
        img_other = cv2.imread(img_mask)
        
 
        if len(img_other[0]) >= len(img_other):
            width = len(img_other[0])
            height = len(img_other)
        else:
            width = len(img_other)
            height = len(img_other[0])       
        
     
        img_names = glob(img_mask)  
        debug_dir = args.get('--debug')
        if debug_dir and not os.path.isdir(debug_dir): 
            os.mkdir(debug_dir)
            
        square_size = float(args.get('--square_size'))  
     
      
        pattern_size = (7, 11)    ## (9,6)
        

        obj_points = []
        img_points = []
        h, w = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results

        def processImage(fn):
            print('processing %s... ' % fn)
            
            from common import splitfn  
            
            img = cv2.imread(fn, 0)   ## 0            
            
     #       img = cv.imread("C:\\VisComp\\PL\\Project_2\\Data\\Im_L_1.png") 
            
            print("Shape of image going to be processed: " + str(img.shape))
            
       #     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            
            if img is None: 
                print("Failed to load", fn)
                return None

            assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
            pattern_size = (9, 6)  
            pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
            pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) 
            pattern_points *= square_size
            
            print("Searching for the best window ...")
            
            min_window_i = 3                       ## 3
            min_window_j = min_window_i
            
            max_window_i = 40                      ## 12
            max_window_j = max_window_i 
            
            go = False
            
            for j in range(min_window_j,max_window_j):
                for i in range(min_window_i, max_window_i):
                    if i != j and go == False:
                        pattern_size = (i,j)
                        found, corners = cv2.findChessboardCorners(img, pattern_size)
                        if found:
                            i_worked = i
                            j_worked = j
                            
                            print("Found")                            
                            go = True
                            
                            break                          
                        else: 
                            print("Not found yet")   
            
            pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
            pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) 
            pattern_points *= square_size 
            
            
      #      found, corners = cv.findChessboardCorners(img, pattern_size)
            
      ##      pattern_size = (7, 11)       
            
            
            
            
            print("Found: " + str(found))
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            if debug_dir:
                if len(img.shape) == 1:
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    vis = img
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                _path, name, _ext = splitfn(fn)
                outfile = os.path.join(debug_dir, name + '_chess.png')
                cv2.imwrite(outfile, vis)

            if not found:
                print('chessboard not found')
                return None 

            print('           %s... OK' % fn)
            return (corners.reshape(-1, 2), pattern_points)

        threads_num = int(args.get('--threads'))
         
        print("Number of threads: " + str(threads_num))
        
    ##    if threads_num <= 1:
        if True:
            chessboards = [processImage(fn) for fn in img_names]
        # else:
        #     print("Run with %d threads..." % threads_num)
        #     from multiprocessing.dummy import Pool as ThreadPool
        #     pool = ThreadPool(threads_num)
        #     chessboards = pool.map(processImage, img_names)

        chessboards = [x for x in chessboards if x is not None]
        for (corners, pattern_points) in chessboards:
            img_points.append(corners)
            obj_points.append(pattern_points)

        print("Length of img_points: " + str(len(img_points)))
        print("Length of obj_points: " + str(len(obj_points)))
        
        
        if len(obj_points) != 0 and len(img_points) != 0:

            # calculate camera distortion 
            rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
        
            print("\nRMS:", rms)
            print("camera matrix:\n", camera_matrix)
            print("distortion coefficients: ", dist_coefs.ravel())
            
            
        
            # undistort the image with the calibration
            print('')
            for fn in img_names if debug_dir else []:
                print("fn: ")
                print(fn)
##                from comma import splitfn

                _path, filename = os.path.split(fn)
                name, _ext = os.path.splitext(filename)                                                   ## splitfn
                img_found = os.path.join(debug_dir, name + '_chess.png')
                outfile = os.path.join(debug_dir, name + '_undistorted.png')
        
                img = cv2.imread(img_found) 
                if img is None:
                    continue
        
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        
                dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        
                # crop and save the image
                x, y, w, h = roi 
                dst = dst[y:y+h, x:x+w]
        
                print('Undistorted image written to: %s' % outfile)  
                cv2.imwrite(outfile, dst)
                
                
        
            print('Done')
            
            return newcameramtx, dir_png_path, width, height
    

    newcameramtx, dir_png_path, width, height = exec_script()

    if type(newcameramtx) == int:
        print("Check img_points and obj_points")
    else:

        fx_calib = newcameramtx[0,0]
        fy_calib = newcameramtx[1,1]
        
        ppx_calib = newcameramtx[0,2]
        ppy_calib = newcameramtx[1,2]   
        
        params_calib = [fx_calib, fy_calib, ppx_calib, ppy_calib]
        
        if continuous == True:
            metadata_title_name = "metadata" + getDateTimeStrMarker() + ".csv"
        else:
            metadata_title_name = ask_user_metadata_filename() 
        
        if '/' in dir_png_path:        
            dir_splitted = dir_png_path.split('/')
        elif "\\" in dir_png_path:        
            dir_splitted = dir_png_path.split("\\")
            
        last_str = dir_splitted[-1]        
        chr_last = last_str[-1]
        
        if chr_last == '/' or chr_last == "\\":            
            dir_splitted = dir_splitted[:-2]
        else:
            dir_splitted = dir_splitted[:-1]
        
        dir_metadata = ''
        
        for d in dir_splitted:
            dir_metadata += d + '/' 
        
        metadata_csv_filename = dir_metadata + metadata_title_name
        
 #       metadata_csv_filename = meta_title_info + getDateTimeStrMarker()      
        
        def write_data_to_csv_file(data_lab, metadata_csv_filename):
            
            with open(metadata_csv_filename, 'a', encoding='utf-8') as f:
                
                for data in data_lab: 
                    line = ', '.join(data)
                    f.write(line + '\n')
                    
                    print("One more line written to csv file ...") 
                    
         
        def write_to_csv_calib_params_database(calib_params, csv_database_filename, counter):
            
            if counter == 0:
                headers = ["Fx", "Fy", "PPx", "PPy"]
            
            ## Write calib_params line to csv file   
            
            with open(csv_database_filename, 'a', encoding='utf-8') as f:
                line = ', '.join(calib_params)
                f.write(line + '\n')
                
                print("One more set of calibration parameters written to csv file ...")       
        
        def write_params_to_metadata_file(params_calib, metadata_csv_filename, model_name, counter_for_params, width, height): 
            
           
            print("Writing headers ...")   ## Define which headers and then write them
                
            frame_number = counter_for_params
            resolution_x = width
            resolution_y = height 
            bytes_per_pixel = 1    ## 8 bits
                
            title_one = "Frame Info: "
            type_line = ["Type", "Basler " + str(model_name)]   ## Get name of camera and add here  ## "Basler ..."
            format_line = ["Format", "Y" + str(bytes_per_pixel*8)]       
            frame_number_line = ["Frame Number", str(counter_for_params+1)]
            resolution_x_line = ["Resolution x", str(width)]
            resolution_y_line = ["Resolution y", str(height)]
            bytes_p_pix_line = ["Bytes per pixel", str(bytes_per_pixel)]
            
            empty_line = "" 
                
            fx = params_calib[0]
            fy = params_calib[1]
            ppx = params_calib[2]
            ppy = params_calib[3]  
            
            title_two = "Intrinsic:"
            title_two = [title_two, ""]
            
            fx_line = ["Fx", round(fx,6)]
            fy_line = ["Fy", round(fy,6)]
            ppx_line = ["PPx", round(ppx,6)]  
            ppy_line = ["PPy", round(ppy,6)]
            
            distorsion = "Brown Conrady"     ## search for it and explain it on the report 
                
            data_lab = [title_one, type_line, format_line, frame_number_line, 
                        frame_number_line, resolution_x_line, resolution_y_line, 
                        bytes_p_pix_line, empty_line, title_two, fx_line, fy_line, 
                        ppx_line, ppy_line, distorsion]
            
            write_data_to_csv_file(data_lab, metadata_csv_filename)
        
        def gui_calib_params(fx, fy, ppx, ppy):
            
                import PySimpleGUI as sg
            
            
                layout = [
                    [sg.Text("Calibration parameters:")],
                    [sg.Text('Fx:'), sg.Input(default_text= str(fx), key="Fx")],
                    [sg.Text('Fy:'), sg.Input(default_text= str(fy), key="Fy")],
                    [sg.Text('PPx:'), sg.Input(default_text= str(ppx), key="PPx")],
                    [sg.Text('PPy:'), sg.Input(default_text= str(ppy), key="PPy")]
                ]
                
                
                layout_opt = [
                    [sg.Button("Back")]
                ]
                
                
                layout = [ 
                    [
                        sg.Column(layout),
                        sg.VSeparator(),                            
                        sg.Column(layout_opt)
                    ]
                ]
                
                window = sg.Window('Camera calibration results', layout)
                
                code_ret = -1
                
                while True:
                   event, values = window.read()
                   if event == "Exit" or event == sg.WIN_CLOSED:
                       code_ret = 0
                       break
                   if event == "Back":
                       code_ret = 1
                       break
                   
                window.close()
                
                return code_ret 
            
            
        code_ret = gui_calib_params(fx_calib, fy_calib, ppx_calib, ppy_calib) 
        
        if continuous == True: 
            
            # width_standard = 1920
            # height_standard = 1080
            
            width_standard = width
            height_standard = height
           
            write_params_to_metadata_file(params_calib, metadata_csv_filename, model_name, counter_for_params, width_standard, height_standard)
            
            csv_database_filename = "database_calib.csv" 
            
            write_to_csv_calib_params_database(params_calib, csv_database_filename, counter_for_params)
            
            counter_for_params += 1   
        
    return code_ret     
     


def order_all_params_pfs_adv(params_changed):
    
    print("Going to order all the parameters inside .pfs file, with the ones also related to advanced properties/options")
    
    new_list_params = ["",""] * 180
    
     
    ## new_list_params.insert 
    
    if True:
        
        new_list_params.insert(0, ['GENAPI_V', params_changed['GENAPI_V']])
        new_list_params.insert(1, ['DEV_V', params_changed['DEV_V']])
        new_list_params.insert(2, ['DEV_NAME', params_changed['DEV_NAME']])
        new_list_params.insert(3, ['INIT_CODE', params_changed['INIT_CODE']])
        new_list_params.insert(4, ['PROD_GUID', params_changed['PROD_GUID']])
        new_list_params.insert(5, ['PROD_V_GUID', params_changed['PROD_V_GUID']])        
        new_list_params.insert(6, ['SEQ_TOT', params_changed['SEQ_TOT']])
        new_list_params.insert(7, ['SEQ_INDEX', params_changed['SEQ_INDEX']])
        new_list_params.insert(8, ['SEQ_EXEC', params_changed['SEQ_EXEC']])
        new_list_params.insert(9, ['SEQ_ADV', params_changed['SEQ_ADV']])
        new_list_params.insert(10, ['GAIN_AUTO', params_changed['GAIN_AUTO']])
        new_list_params.insert(11, ['GAIN_SEL', params_changed['GAIN_SEL']])
        new_list_params.insert(12, ['GAIN', params_changed['GAIN']])          
        new_list_params.insert(13, ['GAIN_SEL', params_changed['GAIN_SEL']])
        new_list_params.insert(14, ['BLACK_LEV_SEL', params_changed['BLACK_LEV_SEL']])
        new_list_params.insert(15, ['BLACK_THRESH', params_changed['BLACK_THRESH']])
        new_list_params.insert(16, ['BLACK_LEV_SEL', params_changed['BLACK_LEV_SEL']])
        new_list_params.insert(17, ['GAMMA_EN', params_changed['GAMMA_EN']])
        new_list_params.insert(18, ['GAMMA_SEL', params_changed['GAMMA_SEL']])
        new_list_params.insert(19, ['GAMMA', params_changed['GAMMA']])
        new_list_params.insert(20, ['DIG_SHIFT', params_changed['DIG_SHIFT']])
        new_list_params.insert(21, ['PIX_FORMAT', params_changed['PIX_FORMAT']])
        new_list_params.insert(22, ['REV_X', params_changed['REV_X']])
        new_list_params.insert(23, ['TEST_IMAGE_SEL', params_changed['TEST_IMAGE_SEL']])
        new_list_params.insert(24, ['FIRST', params_changed['FIRST']])
        new_list_params.insert(25, ['SEC', params_changed['SEC']])
        new_list_params.insert(26, ['OFFSET_X', params_changed['OFFSET_X']])
        new_list_params.insert(27, ['OFFSET_Y', params_changed['OFFSET_Y']])
        new_list_params.insert(28, ['CENTER_X', params_changed['CENTER_X']])
        new_list_params.insert(29, ['CENTER_Y', params_changed['CENTER_Y']])
        new_list_params.insert(30, ['BIN_MODE_H', params_changed['BIN_MODE_H']])
        new_list_params.insert(31, ['BIN_H', params_changed['BIN_H']])
        new_list_params.insert(32, ['BIN_MODE_V', params_changed['BIN_MODE_V']])
        new_list_params.insert(33, ['BIN_V', params_changed['BIN_V']])
        new_list_params.insert(34, ['ACQ_FRAME_COUNT', params_changed['ACQ_FRAME_COUNT']])
        new_list_params.insert(35, ['TRIG_SEL_ONE', params_changed['TRIG_SEL_ONE']])
        new_list_params.insert(36, ['TRIGGER_MODE', params_changed['TRIGGER_MODE']])
        new_list_params.insert(37, ['TRIG_SEL_TWO', params_changed['TRIG_SEL_TWO']])
        new_list_params.insert(38, ['TRIGGER_MODE', params_changed['TRIGGER_MODE']])
        new_list_params.insert(39, ['TRIG_SEL_TWO', params_changed['TRIG_SEL_TWO']])
        new_list_params.insert(40, ['TRIG_SEL_ONE', params_changed['TRIG_SEL_ONE']])
        new_list_params.insert(41, ['TRIG_SOURCE', params_changed['TRIG_SOURCE']])
        new_list_params.insert(42, ['TRIG_SEL_TWO', params_changed['TRIG_SEL_TWO']])
        new_list_params.insert(43, ['TRIG_SOURCE', params_changed['TRIG_SOURCE']])
        new_list_params.insert(44, ['TRIG_SEL_TWO', params_changed['TRIG_SEL_TWO']])
        new_list_params.insert(45, ['TRIG_SEL_ONE', params_changed['TRIG_SEL_ONE']])
        new_list_params.insert(46, ['TRIG_ACTIV', params_changed['TRIG_ACTIV']])
        new_list_params.insert(47, ['TRIG_SEL_TWO', params_changed['TRIG_SEL_TWO']])
        new_list_params.insert(48, ['TRIG_ACTIV', params_changed['TRIG_ACTIV']])
        new_list_params.insert(49, ['TRIG_SEL_TWO', params_changed['TRIG_SEL_TWO']])
        new_list_params.insert(50, ['TRIG_SEL_ONE', params_changed['TRIG_SEL_ONE']])
        new_list_params.insert(51, ['TRIG_DELAY_ABS', params_changed['TRIG_DELAY_ABS']])
        new_list_params.insert(52, ['TRIG_SEL_TWO', params_changed['TRIG_SEL_TWO']])
        new_list_params.insert(53, ['TRIG_DELAY_ABS', params_changed['TRIG_DELAY_ABS']])
        new_list_params.insert(54, ['TRIG_SEL_TWO', params_changed['TRIG_SEL_TWO']])
        new_list_params.insert(55, ['EXP_MODE', params_changed['EXP_MODE']])
        new_list_params.insert(56, ['EXP_AUTO', params_changed['EXP_AUTO']])
        new_list_params.insert(57, ['EXP_TIME', params_changed['EXP_TIME']])
        new_list_params.insert(58, ['SHUTTER_MODE', params_changed['SHUTTER_MODE']])
        new_list_params.insert(59, ['ACQ_FR_EN', params_changed['ACQ_FR_EN']])    
        new_list_params.insert(60, ['FRAME_RATE', params_changed['FRAME_RATE']])    
        new_list_params.insert(61, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(62, ['LINE_MODE_ONE', params_changed['LINE_MODE_ONE']])
        new_list_params.insert(63, ['LINE_SEL_TWO', params_changed['LINE_SEL_TWO']])
        new_list_params.insert(64, ['LINE_MODE_TWO', params_changed['LINE_MODE_TWO']])
        new_list_params.insert(65, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(66, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(67, ['LINE_FORMAT', params_changed['LINE_FORMAT']])
        new_list_params.insert(68, ['LINE_SEL_TWO', params_changed['LINE_SEL_TWO']])
        new_list_params.insert(69, ['LINE_FORMAT', params_changed['LINE_FORMAT']])
        new_list_params.insert(70, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(71, ['LINE_SEL_TWO', params_changed['LINE_SEL_TWO']])
        new_list_params.insert(72, ['LINE_SOURCE', params_changed['LINE_SOURCE']])
        new_list_params.insert(73, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(74, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(75, ['LINE_INV', params_changed['LINE_INV']])
        new_list_params.insert(76, ['LINE_SEL_TWO', params_changed['LINE_SEL_TWO']])
        new_list_params.insert(77, ['LINE_INV', params_changed['LINE_INV']])
        new_list_params.insert(78, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(79, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(80, ['LINE_DEB_TIME', params_changed['LINE_DEB_TIME']])
        new_list_params.insert(81, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(82, ['LINE_SEL_TWO', params_changed['LINE_SEL_TWO']])
        new_list_params.insert(83, ['LINE_MINOUT_PULSE', params_changed['LINE_MINOUT_PULSE']])
        new_list_params.insert(84, ['LINE_SEL_ONE', params_changed['LINE_SEL_ONE']])
        new_list_params.insert(85, ['UserOutputValueAll', "0"])  ## UserOutputValueAll  0
        new_list_params.insert(86, ['SyncUserOutputValueAll', "0"])  ## SyncUserOutputValueAll	0
        new_list_params.insert(87, ['COUNTER_SEL_ONE', params_changed['COUNTER_SEL_ONE']])
        new_list_params.insert(88, ['COUNT_EV_SOURCE_ONE', params_changed['COUNT_EV_SOURCE_ONE']])
        new_list_params.insert(89, ['COUNTER_SEL_TWO', params_changed['COUNTER_SEL_TWO']])
        new_list_params.insert(90, ['COUNT_EV_SOURCE_TWO', params_changed[ 'COUNT_EV_SOURCE_TWO']])
        new_list_params.insert(91, ['COUNTER_SEL_ONE', params_changed['COUNTER_SEL_ONE']])
        new_list_params.insert(92, ['COUNTER_SEL_ONE', params_changed['COUNTER_SEL_ONE']])
        new_list_params.insert(93, ['COUNTER_RESET_SOURCE', params_changed['COUNTER_RESET_SOURCE']])
        new_list_params.insert(94, ['COUNTER_SEL_TWO', params_changed['COUNTER_SEL_TWO']])
        new_list_params.insert(95, ['COUNTER_RESET_SOURCE', params_changed['COUNTER_RESET_SOURCE']])
        new_list_params.insert(96, ['COUNTER_SEL_ONE', params_changed['COUNTER_SEL_ONE']])
        new_list_params.insert(97, ['LUT_SEL', params_changed['LUT_SEL']])
        new_list_params.insert(98, ['LUT_EN', params_changed['LUT_EN']])
        new_list_params.insert(99, ['LUT_SEL', params_changed['LUT_SEL']]) 
        new_list_params.insert(100, ['LUT_SEL', params_changed['LUT_SEL']])
        new_list_params.insert(101, ['LUT_VAL', params_changed['LUT_VAL']])
        new_list_params.insert(102, ['LUT_SEL', params_changed['LUT_SEL']])
        new_list_params.insert(103, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']])        
        new_list_params.insert(104, ['PACKET_SIZE', params_changed['PACKET_SIZE']])
        new_list_params.insert(105, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']])
        new_list_params.insert(106, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']])
        new_list_params.insert(107, ['INTER_PACKET_DELAY', params_changed['INTER_PACKET_DELAY']])
        new_list_params.insert(108, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']])
        new_list_params.insert(109, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']])        
        new_list_params.insert(110, ['GevSCFTD	', '0'])    ## GevSCFTD	0         
        new_list_params.insert(111, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']]) 
        new_list_params.insert(112, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']]) 
        new_list_params.insert(113, ['BANDWIDTH_RESV', params_changed['BANDWIDTH_RESV']])
        new_list_params.insert(114, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']])
        new_list_params.insert(115, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']])
        new_list_params.insert(116, ['BANDWIDTH_RESV_ACC', params_changed['BANDWIDTH_RESV_ACC']])
        new_list_params.insert(117, ['GEVS_CHANNEL_SEL', params_changed['GEVS_CHANNEL_SEL']])
        new_list_params.insert(118, ['AUTO_TARGET', params_changed['AUTO_TARGET']])
        new_list_params.insert(119, ['GREY_ADJUST', params_changed['GREY_ADJUST']])
        new_list_params.insert(120, ['BAL_ADJUST', params_changed['BAL_ADJUST']])
        new_list_params.insert(121, ['GAIN_LOW', params_changed['GAIN_LOW']])
        new_list_params.insert(122, ['GAIN_UP', params_changed['GAIN_UP']])
        new_list_params.insert(123, ['EXP_LOW', params_changed['EXP_LOW']])
        new_list_params.insert(124, ['EXP_UP', params_changed['EXP_UP']])
        new_list_params.insert(125, ['AUTO_FUNC_PROF', params_changed['AUTO_FUNC_PROF']])
        new_list_params.insert(126, ['AOI_SEL_ONE', 'AOI1']) 
        new_list_params.insert(127, ['AOI_WIDTH', params_changed['AOI_WIDTH']])
        new_list_params.insert(128, ['AOI_SEL_TWO', 'AOI2'])
        new_list_params.insert(129, ['AOI_WIDTH', params_changed['AOI_WIDTH']])
        new_list_params.insert(130, ['AOI_SEL_ONE', 'AOI1']) 
        new_list_params.insert(131, ['AOI_SEL_ONE', 'AOI1'])
        new_list_params.insert(132, ['AOI_HEIGHT', params_changed['AOI_HEIGHT']])
        new_list_params.insert(133, ['AOI_SEL_TWO', 'AOI2'])
        new_list_params.insert(134, ['AOI_HEIGHT', params_changed['AOI_HEIGHT']])
        new_list_params.insert(135, ['AOI_SEL_ONE', 'AOI1']) 
        new_list_params.insert(136, ['AOI_SEL_ONE', 'AOI1']) 
        new_list_params.insert(137, ['AOI_OFFSETX', params_changed['AOI_OFFSETX']])
        new_list_params.insert(138, ['AOI_SEL_TWO', 'AOI2']) 
        new_list_params.insert(139, ['AOI_OFFSETX', params_changed['AOI_OFFSETX']])
        new_list_params.insert(140, ['AOI_SEL_ONE', 'AOI1']) 
        new_list_params.insert(141, ['AOI_SEL_ONE', 'AOI1'])
        new_list_params.insert(142, ['AOI_OFFSETY', params_changed['AOI_OFFSETY']])
        new_list_params.insert(143, ['AOI_SEL_TWO', 'AOI2'])  
        new_list_params.insert(144, ['AOI_OFFSETY', params_changed['AOI_OFFSETY']])
        new_list_params.insert(145, ['AOI_SEL_ONE', 'AOI1']) 
        new_list_params.insert(146, ['NAME_DEF_VALUE1', params_changed['NAME_DEF_VALUE1']])
        new_list_params.insert(147, ['DEF_VALUE1', params_changed['DEF_VALUE1']])
        new_list_params.insert(148, ['NAME_DEF_VALUE2', params_changed['NAME_DEF_VALUE2']])
        new_list_params.insert(149, ['DEF_VALUE2', params_changed['DEF_VALUE2']])
        new_list_params.insert(150, ['NAME_DEF_VALUE3', params_changed['NAME_DEF_VALUE3']])
        new_list_params.insert(151, ['DEF_VALUE3', params_changed['DEF_VALUE3']])
        new_list_params.insert(152, ['NAME_DEF_VALUE4', params_changed['NAME_DEF_VALUE4']])
        new_list_params.insert(153, ['DEF_VALUE4', params_changed['DEF_VALUE4']])
        new_list_params.insert(154, ['NAME_DEF_VALUE5', params_changed['NAME_DEF_VALUE5']])
        new_list_params.insert(155, ['DEF_VALUE5', params_changed['DEF_VALUE5']])
        new_list_params.insert(156, ['NAME_DEF_VALUE1', params_changed['NAME_DEF_VALUE1']])
        new_list_params.insert(157, ['CHUNK_MODE', params_changed['CHUNK_MODE']])
        new_list_params.insert(158, ['EV_SEL_ONE', params_changed['EV_SEL_ONE']])
        new_list_params.insert(159, ['EV_NOTIF_ONE', params_changed['EV_NOTIF_ONE']])
        new_list_params.insert(160, ['EV_SEL_TWO', params_changed['EV_SEL_TWO']])
        new_list_params.insert(161, ['EV_NOTIF_TWO', params_changed['EV_NOTIF_TWO']])
        new_list_params.insert(162, ['EV_SEL_THREE', params_changed['EV_SEL_THREE']])
        new_list_params.insert(163, ['EV_NOTIF_THREE', params_changed['EV_NOTIF_THREE']])
        new_list_params.insert(164, ['EV_SEL_FOUR', params_changed['EV_SEL_FOUR']])
        new_list_params.insert(165, ['EV_NOTIF_FOUR', params_changed['EV_NOTIF_FOUR']])
        new_list_params.insert(166, ['EV_SEL_FIVE', params_changed['EV_SEL_FIVE']])
        new_list_params.insert(167, ['EV_NOTIF_FIVE', params_changed['EV_NOTIF_FIVE']])
        new_list_params.insert(168, ['EV_SEL_SIX', params_changed['EV_SEL_SIX']])
        new_list_params.insert(169, ['EV_NOTIF_SIX', params_changed['EV_NOTIF_SIX']])
        new_list_params.insert(170, ['EV_SEL_ONE', params_changed['EV_SEL_ONE']]) 
       
    return new_list_params  
  
   
def gui_video_rec():    
    
  #  def main():
    
    
        # if threading.currentThread() == threading.main_thread():
        #     print("Main thread")
        # else:
        #     threading.currentThread().setName(threading.main_thread())   
    
        import PySimpleGUI as sg
         
   ##     sg.change_look_and_feel('Dark')
         
        
           
        video_rec = False 
        repInit = True
        
        print("PA")
        
        layout = [
            [sg.Text("Do you want to: ")],      ## Use a pre-recorded video    ## Perform a new image acquisition process
            [sg.T("         "), sg.Checkbox('Use a pre-recorded video', default=False, key="-IN1-")],
            [sg.T("         "), sg.Checkbox('Perform a new image acquisition process', default=True, key="-IN2-")],
            [sg.Button("Next")]  
        ]
        
        print("PB")
         
        window = sg.Window('Initial GUI', layout) 
         
          
        while repInit:
    ##    if repInit:
           print("PC1") 
           
           
           event, values = window.read()
           
           # try:           
           #     event, values = window.read()
           # except:
           #     repInit = True
           #     continue
           
           
           print("PC2")
           if event == "Exit" or event == sg.WIN_CLOSED:
               break 
           
           if event == "Next":
               if values["-IN1-"] == True and values["-IN2-"] == False:
                   print("Opt1")
                   video_rec = False
                   repInit = False
                   break
               elif values["-IN2-"] == True and values["-IN1-"] == False:
                   print("Opt2")
                   video_rec = True
                   repInit = False
                   break
               else:
                   if values["-IN1-"] == True and values["-IN2-"] == True:
                       print("Only pick one option !!!")                   
                       repInit = True           
                       continue
           
        print("\n\n Here after \n\n")
        window.close()        
        
        return video_rec 
    
    # if __name__ == '__main__':
    #     main()

# def simple_with_first():
     
#     print("Simple with first")
    
#     again = True
    
#     while again == True:
    
#         windowx = sg_py.Window('Choose path to video filename', [[sg_py.Text('File name')], [sg_py.Input(), sg_py.FileBrowse()], [sg_py.OK(), sg_py.Cancel()] ]).read(close=True)
#         (keyword, dict_dir) = windowx                
    
#         dir_path = dict_dir['Browse'] 
        
#         print("Filename saved !!!")
        
        
#         if os.path.isfile(dir_path): 
            
#             if not '.avi' in dir_path and not '.mp4' in dir_path:
#                 print("Please insert a filename with one of the following extensions: \n {.avi \t .mp4}")
#                 again = True
#             else:
#                     again = False
#                     count_folder_ex = 0
#                     counterTest = 0
                    
#                     dir_parts = dir_path.split('/')
#                     filename_whole = dir_parts[-1]
#                     base_path = ''
                    
#                     for ind_d, d in enumerate(dir_parts):                        
#                         if ind_d < len(dir_parts) - 1:
#                             base_path += d + '/'                                                                  
                 
                    
#                     print("Conversion done")
                     
#                     #%%
                    
#                     dateTimeMarker = getDateTimeStrMarker()
                    
#                     base_path3 = base_path.replace("/", "\\")
                    
#                     sequence_name = counterTest
#                     dest_path = base_path + '/Image_Processing/'
#                     dest_path2 = base_path3 + "\\Image_Processing\\"                    
                    
#                     exists = True
                    
#                     while exists == True:               
                        
#                         if os.path.exists(dest_path) == True:                   
#                             new_str_folder = '/Image_Processing' + str(count_folder_ex) + '/'
#                             new_str_folder_sec = "\\Image_Processing" + str(count_folder_ex) + "\\" 
#                             dest_path = base_path + new_str_folder 
                             
             
#                             if os.path.exists(dest_path) == True:
#                                 exists = True
#                                 count_folder_ex += 1 
#                             else:
                                
#                                 dest_path2 = base_path3 + new_str_folder_sec        
#                                 dest_path = os.path.join(dest_path)
#                                 os.mkdir(dest_path)
#                                 exists = False
#                                 break              
                         
#                         else: 
#                              dest_path = os.path.join(dest_path)
#                              os.mkdir(dest_path)
#                              exists = False
#                              break                  
                           
                     
#                     mainPathVideoData = dest_path + 'VideoData_' + dateTimeMarker + '/'
#                     mainPathVideo = os.path.join(mainPathVideoData)
#                     os.mkdir(mainPathVideo) 
#                     mtsVideoPath = dest_path + 'VideosAlmostLaserSpeckle/' 
#                     mtsVideoPath = os.path.join(mtsVideoPath)
#                     os.mkdir(mtsVideoPath) 
#                     mp4VideoFile = dest_path2 + 'VideosAlmostLaserSpeckle' + dateTimeMarker + "\\"      ## change '/' to "\\" 
#                     mp4VideoFilePath = os.path.join(mp4VideoFile)                                     ## path_out (ffmpeg)
#                     os.mkdir(mp4VideoFilePath)   
#                     # mtsVideoPathP = os.path.join(mtsVideoPath) 
#                     # os.mkdir(mtsVideoPathP)   
#                     IFVP = dest_path + 'DataSequence_'
#                     locationMP4_file = "FilesFor_" 
#                     roiPath = 'Approach'
#                     newRoiPath = 'Approach_new' 
#                     pathRoiStart = dest_path + 'modRoisFirstMom_'
#                     pathRoiEnd = dest_path + 'modRoisSecMom_'
#                     first_clustering_storing_output = "Quality_kMeans_Clustering_real_"
#                     pathPythonFile = dest_path + "SpeckleTraining/ffmpeg-5.0.1-full_build/bin"
                    
#                     ## Adding date and time to which one of the paths shown above
#                     IFVP = IFVP + dateTimeMarker
#                     locationMP4_file = locationMP4_file + dateTimeMarker  
#                     roiPath = roiPath + dateTimeMarker 
#                     newRoiPath = newRoiPath + dateTimeMarker 
#                     pathRoiStart = pathRoiStart + dateTimeMarker 
#                     pathRoiEnd = pathRoiEnd + dateTimeMarker 
#                     first_clustering_storing_output = first_clustering_storing_output + dateTimeMarker
                    
                    
#                     if True:
#                          data_to_save = [] 
                         
#                          decisorLevel = 0
                         
#                          infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile)
                                             
#                          clusteringInfData, executionTime, totCountImages = videoAnalysis(infi)               
#                          datax = (clusteringInfData, executionTime, totCountImages)  
                         
#                          gui_show_results(clusteringInfData, executionTime, totCountImages) 
                         
#                          data_to_save.append(datax) 
                         
#                          break
                     
#         else:
#            print("Please insert a valid directory !!!")
#            again = True
           
           
# def diff_with_first():
#     print("Compute new video, from the input, with images differences between each couple")
    
#     again = True
    
#     while again == True:
    
#         windowx = sg_py.Window('Choose path to video filename', [[sg_py.Text('File name')], [sg_py.Input(), sg_py.FileBrowse()], [sg_py.OK(), sg_py.Cancel()] ]).read(close=True)
#         (keyword, dict_dir) = windowx                
    
#         dir_path = dict_dir['Browse'] 
        
#         print("Filename saved !!!")
        
        
#         if os.path.isfile(dir_path): 
            
#             if not '.avi' in dir_path and not '.mp4' in dir_path:
#                 print("Please insert a filename with one of the following extensions: \n {.avi \t .mp4}")
#                 again = True
#             else:
#                     again = False
#                     count_folder_ex = 0
#                     counterTest = 0
                    
#                     dir_parts = dir_path.split('/')
#                     filename_whole = dir_parts[-1]
#                     base_path = ''
                    
#                     for ind_d, d in enumerate(dir_parts):                        
#                         if ind_d < len(dir_parts) - 1:
#                             base_path += d + '/' 

#                     vidcap = cv2.VideoCapture(dir_path) 
                    
#                     fps_out = 50    
#                     index_in = -1
#                     index_out = -1    
#                     reader = imageio.get_reader(dir_path)
#                     fps_in = reader.get_meta_data()['fps']
                    
#                     count = 0 
                    
#                     image_buffer = []
                   
#                     while(True): 
                        
#                         success, image = vidcap.read()
#                         if success: 
                            
#                             if count == 0:
#                                 image_ref = np.zeros_like(image)
#                                 diff_img = np.zeros_like(image)
                             
#                             cv2.imwrite("init_image%d.jpg" % count, image)
#                             image_buffer.append(image)
#                             count += 1
#                         else:  
#                             break  
                    
#                     count_diff = 0
                    
#                     buffer_diffs = []
                    
#                     for x in range(0, count):
                        
#                         imx = image_buffer[x]
                        
#                         s = imx.shape
#                         print("Shape of image: ")
#                         print(s)             
                        
#                         for c in range(0, s[2]):
#                             for b in range(0, s[1]):
#                                 for a in range(0, s[0]):                        
#                                     diff_img[a,b,c] = abs(imx[a,b,c]-image_ref[a,b,c])
                                
#                         cv2.imwrite("diff_image%d.jpg" % count_diff, diff_img)
#                         buffer_diffs.append(diff_img)
                        
#                         count_diff += 1                        
#                         image_ref = imx.copy()  
                    
#             ##        os.remove(dir_path)   ## remove video filepath
                     
#                     size = (len(imx), len(imx[0]))
                    
#                     out = cv2.VideoWriter('testd_' + '00' + str(counterTest) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                     
#                     for i in range(len(buffer_diffs)):
#                         out.write(buffer_diffs[i]) 
#                     out.release()
                    
#                     print("Conversion done")
                    
#                     dateTimeMarker = getDateTimeStrMarker()
                    
#                     base_path3 = base_path.replace("/", "\\")
                    
#                     sequence_name = counterTest
#                     dest_path = base_path + '/Image_Processing/'
#                     dest_path2 = base_path3 + "\\Image_Processing\\"                    
                    
#                     exists = True
                    
#                     while exists == True:               
                        
#                         if os.path.exists(dest_path) == True:                   
#                             new_str_folder = '/Image_Processing' + str(count_folder_ex) + '/'
#                             new_str_folder_sec = "\\Image_Processing" + str(count_folder_ex) + "\\" 
#                             dest_path = base_path + new_str_folder 
                             
             
#                             if os.path.exists(dest_path) == True:
#                                 exists = True
#                                 count_folder_ex += 1 
#                             else:
                                
#                                 dest_path2 = base_path3 + new_str_folder_sec        
#                                 dest_path = os.path.join(dest_path)
#                                 os.mkdir(dest_path)
#                                 exists = False
#                                 break              
                         
#                         else: 
#                              dest_path = os.path.join(dest_path)
#                              os.mkdir(dest_path)
#                              exists = False
#                              break                  
                           
                    
#                     mainPathVideoData = dest_path + 'VideoData_' + dateTimeMarker + '/'
#                     mainPathVideo = os.path.join(mainPathVideoData)
#                     os.mkdir(mainPathVideo) 
#                     mtsVideoPath = dest_path + 'VideosAlmostLaserSpeckle/' 
#                     mtsVideoPath = os.path.join(mtsVideoPath)
#                     os.mkdir(mtsVideoPath) 
#                     mp4VideoFile = dest_path2 + 'VideosAlmostLaserSpeckle' + dateTimeMarker + "\\"      ## change '/' to "\\" 
#                     mp4VideoFilePath = os.path.join(mp4VideoFile)                                     ## path_out (ffmpeg)
#                     os.mkdir(mp4VideoFilePath)   
#                     # mtsVideoPathP = os.path.join(mtsVideoPath) 
#                     # os.mkdir(mtsVideoPathP)   
#                     IFVP = dest_path + 'DataSequence_'
#                     locationMP4_file = "FilesFor_" 
#                     roiPath = 'Approach'
#                     newRoiPath = 'Approach_new' 
#                     pathRoiStart = dest_path + 'modRoisFirstMom_'
#                     pathRoiEnd = dest_path + 'modRoisSecMom_'
#                     first_clustering_storing_output = "Quality_kMeans_Clustering_real_"
#                     pathPythonFile = dest_path + "SpeckleTraining/ffmpeg-5.0.1-full_build/bin"
                    
#                     ## Adding date and time to which one of the paths shown above
#                     IFVP = IFVP + dateTimeMarker
#                     locationMP4_file = locationMP4_file + dateTimeMarker  
#                     roiPath = roiPath + dateTimeMarker 
#                     newRoiPath = newRoiPath + dateTimeMarker 
#                     pathRoiStart = pathRoiStart + dateTimeMarker 
#                     pathRoiEnd = pathRoiEnd + dateTimeMarker 
#                     first_clustering_storing_output = first_clustering_storing_output + dateTimeMarker
                    
                     
#                     if True:
#                          data_to_save = [] 
                         
#                          decisorLevel = 0
                         
#                          infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile)
                                             
#                          clusteringInfData, executionTime, totCountImages = videoAnalysis(infi)               
#                          datax = (clusteringInfData, executionTime, totCountImages)  
                         
#                          gui_show_results(clusteringInfData, executionTime, totCountImages) 
                         
#                          data_to_save.append(datax) 
                         
#                          break
                     
#         else:
#            print("Please insert a valid directory !!!")
#            again = True 


def demo_gui_menu():
    
    resp_demo = False
    
    import PySimpleGUI as sg 
    from just_demo_purposes import simple_with_first, diff_with_first
    
    layout = [
        [sg.Button("Images itselves, with first image")],
        [sg.Button("Difference of consecutive images, with first image")],
        [sg.Button("Back")]
    ]
    
    window = sg.Window('Demo GUI', layout)
    
    while True:
       event, values = window.Read()
       if event == "Exit" or event == sg.WIN_CLOSED:
           break
       
       if event == "Back":
           resp_demo = True
           window.close()
           break
       
       if event == "Images itselves, with first image":
           simple_with_first(window)         
           
       if event == "Difference of consecutive images, with first image":
           diff_with_first(window) 
    
    return resp_demo
    
def is_imported(module):
    return module in sys.modules

def load_core_packages():
    
    ok = False
    
    import time
    
    # while is_imported('time') == False:
    #     import time 
        
    import math
    
    # while is_imported('math') == False:
    #     import math 
        
    import os
    
    # while is_imported('os') == False:
    #     import os 
        
    import cv2
    
    # while is_imported('cv2') == False:
    #     import cv2 
        
    import imageio 
    
    # while is_imported('imageio') == False:
    #     import imageio 
        
    import shutil
    
    # while is_imported('shutil') == False:
    #     import shutil 
        
    import sys
    
    # while is_imported('sys') == False:
    #     import sys 
        
    import subprocess
    
    # while is_imported('subprocess') == False:
    #     import subprocess 
        
    import shlex
    
    # while is_imported('shlex') == False:
    #     import shlex 
        
    import numpy as np 
    
    # while is_imported('numpy') == False:
    #     import numpy as np
        
    import pandas as pd
    
    # while is_imported('pandas') == False:
    #     import pandas as pd
        
    import xlsxwriter
    
    # while is_imported('xlsxwriter') == False:
    #     import xlsxwriter
        
    import configparser 
    
    # while is_imported('configparser') == False:
    #     import configparser
    
    time.sleep(1)
    
    sobel_error = True
    
    while sobel_error == True:
    
        try:
            from skimage.filters import sobel
        except ModuleNotFoundError:
            sobel_error = True
                 
        else:
            sobel_error = False            
        
    time.sleep(1)    
         
    
    import matplotlib.pyplot as plt 
    
   
        
    import skimage.morphology as morphology
    
    # while is_imported('skimage') == False:
    #     import skimage.morphology as morphology
        
    import scipy.ndimage as ndi  
    
    # while is_imported('scipy') == False:
    #     import scipy.ndimage as ndi
        
    from skimage.color import label2rgb 
    
    # while is_imported('skimage') == False:
    #     from skimage.color import label2rgb
        
    from skimage.segmentation import watershed
    
    # while is_imported('skimage') == False:
    #     from skimage.segmentation import watershed
        
    ## from _watershed import watershed
    from sklearn.model_selection import train_test_split  
    
    # while is_imported('sklearn') == False:
    #     from sklearn.model_selection import train_test_split
        
    from sklearn.cluster import KMeans  
    
    # while is_imported('sklearn') == False:
    #     from sklearn.cluster import KMeans
        
    from sklearn.metrics import silhouette_samples, silhouette_score 
    
    # while is_imported('sklearn') == False:
    #     from sklearn.metrics import silhouette_samples
        
    # while is_imported('sklearn') == False: 
    #     from sklearn.metrics import silhouette_score  
        
    from sklearn.decomposition import PCA
    
    # while is_imported('sklearn') == False: 
    #     from sklearn.decomposition import PCA
        
    from scipy.spatial.distance import pdist, squareform 
    
    # while is_imported('scipy') == False: 
    #     from scipy.spatial.distance import pdist

    # while is_imported('scipy') == False: 
    #     from scipy.spatial.distance import squareform
        
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    
    # while is_imported('scipy') == False: 
    #     from scipy.cluster.hierarchy import dendrogram
        
    # while is_imported('scipy') == False: 
    #     from scipy.cluster.hierarchy import linkage 
    
    # while is_imported('scipy') == False: 
    #     from scipy.cluster.hierarchy import fcluster
        
    import csv
    
    # while is_imported('csv') == False: 
    #     import csv
        
    from spyder_kernels.utils.iofuncs import load_dictionary   
    
    # while is_imported('spyder_kernels') == False: 
    #     from spyder_kernels.utils.iofuncs import load_dictionary
        
    from spyder_kernels.utils.iofuncs import save_dictionary 
    
    # while is_imported('spyder_kernels') == False: 
    #     from spyder_kernels.utils.iofuncs import save_dictionary 
        
    import pickle
    
    # while is_imported('pickle') == False: 
    #     import pickle
        
    import tarfile
    
    # while is_imported('tarfile') == False: 
    #     import tarfile
        
    from sewar.full_ref import mse, rmse, _rmse_sw_single, rmse_sw, psnr, _uqi_single, uqi, ssim, ergas, scc, rase, sam, msssim, vifp, psnrb; 
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import mse
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import rmse
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import _rmse_sw_single
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import rmse_sw
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import psnr 
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import _uqi_single
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import uqi
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import ssim
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import ergas
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import scc
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import rase
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import sam
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import msssim
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import vifp
    
    # while is_imported('sewar') == False: 
    #     from sewar.full_ref import psnrb
      
    ok = True
    
    return ok

def write_to_pfs_adv_opt(adv_opts):
    print("Write advanced properties to info to PFS file") 
    adv_written = True 
    
 ##   adv_opts_list = list(adv_opts.items()) 
 
    adv_opts_list = adv_opts
    
    print("--")    
    print(" --- Lenght of adv options: " + str(len(adv_opts_list)))
    print("--")
    
    labels_all_params = []
    counter_lines = 0
    
    pfs_struct = [] 
    pfs_ok = False
    
    with open('paramsAll.txt') as f:
        for line in f:
            if counter_lines > 0:
                labels_all_params.append(str(line))     ## replace each label by the name of each parameter
            
            counter_lines += 1 
            
            
    init_code = ""        
    version_genapi = ""  
    dev_version = ""
    dev_name = ""
    guid_code = ""
    guid_version_code = ""       
    
            
    for ind_adv_p, adv_p in enumerate(adv_opts_list):  
        if ind_adv_p >= 6:
            
            if len(adv_p) > 0:
                [name_p, val_p] = adv_p 
           #     val_p = adv_p            
                
                lab_param = labels_all_params[ind_adv_p].replace("\n", "")
                
                print("Label: " + lab_param) 
           
                pfs_struct.append(lab_param + "\t" + str(val_p))  
        else:     ## Header info 
         
            if ind_adv_p == 0: 
                [name_p, val_p] = adv_p     
         #       val_p = adv_p 
                version_genapi = val_p 
            elif ind_adv_p == 1:    
                [name_p, val_p] = adv_p 
            #    val_p = adv_p
                dev_version  = val_p 
            elif ind_adv_p == 2:     
                [name_p, val_p] = adv_p 
            #    val_p = adv_p
                dev_name  = val_p 
            elif ind_adv_p == 3:    
                [name_p, val_p] = adv_p 
            #    val_p = adv_p
                init_code  = val_p
            elif ind_adv_p == 4:    
                [name_p, val_p] = adv_p
            #    val_p = adv_p
                guid_code  = val_p 
            elif ind_adv_p == 5:    
                [name_p, val_p] = adv_p
            #    val_p = adv_p
                guid_version_code  = val_p              
                
                first_line = "# " + "{" + init_code + "}"
                sec_line = "# " + "Gen Api persistence file (version " + version_genapi + ")"
                third_line = "# " + "Device = " + dev_name + " -- Basler generic GigEVision camera interface -- " + "Device version = " + dev_version + " -- Product GUID = " + guid_code + " -- Product version GUID = " + guid_version_code
           ##     Basler::GigECamera
                pfs_struct.append(first_line)
                pfs_struct.append(sec_line) 
                pfs_struct.append(third_line)            
            
                pfs_ok = True
         
    
    
    return pfs_ok, pfs_struct 
        


def save_to_pfs(dir_path, gain_raw, black_level_raw, exp_time_raw, acq_frame_rate, packet_size, inter_packet_delay, bandwidth_resv, bandwidth_resv_acc, width, height, adv_written):

    if adv_written == False:
        init_code = "05D8C294-F295-4dfb-9D01-096BD04049F405D8C294-F295-4dfb-9D01-096BD04049F4" 
        version_genapi = "3.1.0"
        dev_version = "3.7.0"
        guid_code = "1F3C6A72-7842-4edd-9130-E2E90A2058BA"
        guid_version_code = "6512A424-1B05-4C68-99D6-91F0E31857CD"
        
        first_line = "# " + "{" + init_code + "}"
        sec_line = "# " + "Gen Api persistence file (version " + version_genapi + ")"
        third_line = "# " + "Device = Basler::GigECamera -- Basler generic GigEVision camera interface -- " + "Device version = " + dev_version + " -- Product GUID = " + guid_code + " -- Product version GUID = " + guid_version_code
        
        header = [first_line, sec_line, third_line]
    
        #############################################
        seq_totNumber = 2
        seq_index = 0
        seqSet_exec = 1
        adv_mode = "Auto"  
        
        SequenceSetTotalNumberStr = "SequenceSetTotalNumber" + "\t" + str(seq_totNumber)
        SequenceSetIndexStr = "SequenceSetIndexStr" + "\t" + str(seq_index)
        SequenceSetExecutionsStr = "SequenceSetExecution" + "\t" + str(seqSet_exec)
        SequenceAdvanceModeStr = "SequenceAdvanceMode" + "\t" + adv_mode
        
        seq_espec = [SequenceSetTotalNumberStr, SequenceSetIndexStr, SequenceSetExecutionsStr, SequenceAdvanceModeStr]
        
        #############################################
        gain_auto = "Off" 
        gain_sel = "All"
        
        GainAutoStr = "GainAuto" + "\t" + gain_auto 
        GainSelectorStr = "GainSelector" + "\t" + gain_sel
        
    GainRawStr = "GainRaw" + "\t" + str(gain_raw)
        
    gain_opts = [GainAutoStr, GainSelectorStr, GainRawStr, GainSelectorStr]
    
    #############################################
    
    if adv_written == False:
        
        black_level_sel = "All"
        
        BlackLevelSelectorStr = "BlackLevelSelector" + "\t" + black_level_sel
    BlackLevelRawStr = "BlackLevelRaw" + "\t" + str(black_level_raw)
    
    blackLevel_opts = [BlackLevelSelectorStr, BlackLevelRawStr, BlackLevelSelectorStr]
    
    #############################################
    
    if adv_written == False:
        
        gammaEnableValue = 0
        gamma_sel = "User"
        gamma_value = 1
        
        GammaEnableStr = "GammaEnable" + "\t" + str(gammaEnableValue)
        GammaSelectorStr = "GammaSelector" + "\t" + gamma_sel
        GammaStr = "Gamma" + "\t" + str(gamma_value)
        
        gamma_opts = [GammaEnableStr, GammaSelectorStr, GammaStr]
        
        #############################################
        digital_shift = 0    
        DigitalShiftStr = "DigitalShift" + "\t" + str(digital_shift)
        
        pix_format = "Mono8"
        PixelFormatStr = "PixelFormat" + "\t" + pix_format
        
        reverse_x = 0
        ReverseX_Str = "ReverseX" + "\t" + str(reverse_x)
        
        test_image_sel = "Off"
        TestImageSelectorStr = "TestImageSelector" + "\t" + test_image_sel
        
        other_opt = [DigitalShiftStr, PixelFormatStr, ReverseX_Str, TestImageSelectorStr]
    
    ############################################# 
    WidthStr = "Width" + "\t" + str(width)
    HeightStr = "Height" + "\t" + str(height)
    
    img_dim = [WidthStr, HeightStr]
    
    #############################################
    
    if adv_written == False:
        
        offset_x = 0
        offset_y = 0
        center_x = 0
        center_y = 0
        
        OffsetX_Str = "OffsetX" + "\t" + str(offset_x)
        OffsetY_str = "OffsetY" + "\t" + str(offset_y)
        CenterX_Str = "CenterX" + "\t" + str(center_x)
        CenterY_Str = "CenterY" + "\t" + str(center_y)
        
        ax_params = [OffsetX_Str, OffsetY_str, CenterX_Str, CenterY_Str]
        
        #############################################
        binning_mode_horiz = "Summing"
        binning_horiz = 1
        binning_mode_vert = "Summing"
        binning_vert = 1
        
        BinningModeHorizontalStr = "BinningModeHorizontal" + "\t" + binning_mode_horiz
        BinningHorizontalStr = "BinningHorizontal" + "\t" + str(binning_horiz)
        BinningModeVerticalStr = "BinningModeVertical" + "\t" + binning_mode_vert
        BinningVerticalStr = "BinningVertical" + "\t" + str(binning_vert)
           
        binning = [BinningModeHorizontalStr, BinningHorizontalStr, BinningModeVerticalStr, BinningVerticalStr]
        
        #############################################
        acquisition_frame_count = 1
        
        AcquisitionFrameCountStr = "AcquisitionFrameCount" + "\t" + str(acquisition_frame_count)
       
        #############################################
        trig_sel_one = "AcquisitionStart"
        trig_mode_one = "Off"
        trig_sel_two = "FrameStart"
        trig_source = "Line1"
        trig_activate = "RisingEdge"
        
        trig_delay = 0
        
        TriggerSelectorOneStr = "TriggerSelector" + "\t" + trig_sel_one
        TriggerModeStr = "TriggerMode" + "\t" + trig_mode_one
        TriggerSelectorTwoStr = "TriggerSelector" + "\t" + trig_sel_two
        TriggerSourceStr = "TriggerSource" + "\t" + trig_source
        TriggerActivationStr = "TriggerActivation" + "\t" + trig_activate
        TriggerDelayAbs = "TriggerDelayAbs" + "\t" + str(trig_delay)
               
        trigger_opts = [TriggerSelectorOneStr, TriggerModeStr, 
                        TriggerSelectorTwoStr, TriggerModeStr, 
                        TriggerSelectorTwoStr, TriggerSelectorOneStr,
                        TriggerSourceStr, TriggerSelectorTwoStr,
                        TriggerSourceStr, TriggerSelectorTwoStr,
                        TriggerSelectorOneStr, TriggerActivationStr,
                        TriggerSelectorTwoStr, TriggerActivationStr,
                        TriggerSelectorTwoStr, TriggerSelectorOneStr,
                        TriggerDelayAbs, TriggerSelectorTwoStr,
                        TriggerDelayAbs, TriggerSelectorTwoStr]
        
        #############################################
        exp_mode = "Timed" 
        exp_auto = "Off"
        
        ExposureModeStr = "ExposureMode" + "\t" + exp_mode
        ExposureAutoStr = "ExposureAuto" + "\t" + exp_auto
    ExposureTimeRawStr = "ExposureTimeRaw" + "\t" + str(exp_time_raw)  
    
    exp_opts = [ExposureModeStr, ExposureAutoStr, ExposureTimeRawStr]
    
    #############################################
    
    def_shutter_mode = "GlobalResetRelease"
    
    ShutterModeStr = "ShutterMode" + "\t" + def_shutter_mode
     
    if adv_written == False:
        
            acq_frame_rate_enable = 0
            
            AcquisitionFrameRateEnableStr = "AcquisitionFrameRateEnable" + "\t" + str(acq_frame_rate_enable)
            AcquisitionFrameRateAbsStr = "AcquisitionFrameRateAbs" + "\t" + str(acq_frame_rate)
             
            #############################################
            line_sel = "Line1"
            line_sel_sec = "Out1"
            line_mode_input = "Input"
            line_mode_output = "Output"
            line_format = "OctoCoupled"
            line_source = "UserOutput"
            line_inv = 0
            line_deb_time = 10000
            min_out_pulse_width = 0
            
            LineSelectorStr = "LineSelector" + "\t" + line_sel
            LineSelectorSecStr = "LineSelector" + "\t" + line_sel_sec 
            LineModeInputStr = "LineMode" + "\t" + line_mode_input 
            LineModeOutputStr = "LineMode" + "\t" + line_mode_output
            LineFormatStr = "LineFormat" + "\t" + line_format
            LineSourceStr = "LineSource" + "\t" + line_source
            LineInvStr = "LineInverter" + "\t" + str(line_inv)
            LineDebouncerTimeRawStr = "LineDebouncerTimeRaw" + "\t" + str(line_deb_time)
            MinOutPulseWidthRawStr = "MinOutPulseWidthRaw" + "\t" + str(min_out_pulse_width)
                
            line_opts = [LineSelectorStr, LineModeInputStr,
                         LineSelectorSecStr, LineModeOutputStr,
                         LineSelectorStr, LineSelectorStr,
                         LineFormatStr, LineSelectorSecStr,
                         LineFormatStr, LineSelectorStr,
                         LineSelectorSecStr, LineSourceStr,
                         LineSelectorStr, LineSelectorStr,
                         LineInvStr, LineSelectorSecStr,
                         LineInvStr, LineSelectorStr,
                         LineSelectorStr, LineDebouncerTimeRawStr,
                         LineSelectorStr, LineSelectorSecStr,
                         MinOutPulseWidthRawStr, LineSelectorStr]
            
            #############################################
            user_output_value = 0
            sync_user_output_value = 0
            counter_sel_one = "Counter1"
            counter_sel_two = "Counter2"
            counter_ev_source_one = "FrameTrigger"
            counter_ev_source_two = "FrameStart"
            counter_reset_source = "Off"
            
            UserOutputValueAllStr = "UserOutputValueAll" + "\t" + str(user_output_value)
            SyncUserOutputValueAllStr = "SyncUserOutputValueAll" + "\t" + str(sync_user_output_value)
            CounterSelectorOneStr = "CounterSelector" + "\t" + counter_sel_one
            CounterSelectorTwoStr = "CounterSelector" + "\t" + counter_sel_two
            CounterEventSourceOneStr = "CounterEventSource" + "\t" + counter_ev_source_one
            CounterEventSourceTwoStr = "CounterEventSource" + "\t" + counter_ev_source_two
            CounterResetSourceStr = "CounterResetSource" + "\t" + counter_reset_source
            
            couter_opts = [UserOutputValueAllStr, SyncUserOutputValueAllStr,
                           CounterSelectorOneStr, CounterEventSourceOneStr,
                           CounterSelectorTwoStr, CounterEventSourceTwoStr,
                           CounterSelectorOneStr, CounterSelectorOneStr,
                           CounterResetSourceStr, CounterSelectorTwoStr,
                           CounterResetSourceStr, CounterSelectorOneStr]
           
            #############################################
            lut_sel = "Luminance" 
            lut_en = 0
            lut_value_all = "0x0000000000000000000000000000000000000000000000000000000000000000000000080000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001800000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000280000000000000000000000000000000000000000000000000000000000000030000000000000000000000000000000000000000000000000000000000000003800000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000048000000000000000000000000000000000000000000000000000000000000005000000000000000000000000000000000000000000000000000000000000000580000000000000000000000000000000000000000000000000000000000000060000000000000000000000000000000000000000000000000000000000000006800000000000000000000000000000000000000000000000000000000000000700000000000000000000000000000000000000000000000000000000000000078000000000000000000000000000000000000000000000000000000000000008000000000000000000000000000000000000000000000000000000000000000880000000000000000000000000000000000000000000000000000000000000090000000000000000000000000000000000000000000000000000000000000009800000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000000a800000000000000000000000000000000000000000000000000000000000000b000000000000000000000000000000000000000000000000000000000000000b800000000000000000000000000000000000000000000000000000000000000c000000000000000000000000000000000000000000000000000000000000000c800000000000000000000000000000000000000000000000000000000000000d000000000000000000000000000000000000000000000000000000000000000d800000000000000000000000000000000000000000000000000000000000000e000000000000000000000000000000000000000000000000000000000000000e800000000000000000000000000000000000000000000000000000000000000f000000000000000000000000000000000000000000000000000000000000000f80000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000010800000000000000000000000000000000000000000000000000000000000001100000000000000000000000000000000000000000000000000000000000000118000000000000000000000000000000000000000000000000000000000000012000000000000000000000000000000000000000000000000000000000000001280000000000000000000000000000000000000000000000000000000000000130000000000000000000000000000000000000000000000000000000000000013800000000000000000000000000000000000000000000000000000000000001400000000000000000000000000000000000000000000000000000000000000148000000000000000000000000000000000000000000000000000000000000015000000000000000000000000000000000000000000000000000000000000001580000000000000000000000000000000000000000000000000000000000000160000000000000000000000000000000000000000000000000000000000000016800000000000000000000000000000000000000000000000000000000000001700000000000000000000000000000000000000000000000000000000000000178000000000000000000000000000000000000000000000000000000000000018000000000000000000000000000000000000000000000000000000000000001880000000000000000000000000000000000000000000000000000000000000190000000000000000000000000000000000000000000000000000000000000019800000000000000000000000000000000000000000000000000000000000001a000000000000000000000000000000000000000000000000000000000000001a800000000000000000000000000000000000000000000000000000000000001b000000000000000000000000000000000000000000000000000000000000001b800000000000000000000000000000000000000000000000000000000000001c000000000000000000000000000000000000000000000000000000000000001c800000000000000000000000000000000000000000000000000000000000001d000000000000000000000000000000000000000000000000000000000000001d800000000000000000000000000000000000000000000000000000000000001e000000000000000000000000000000000000000000000000000000000000001e800000000000000000000000000000000000000000000000000000000000001f000000000000000000000000000000000000000000000000000000000000001f80000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000020800000000000000000000000000000000000000000000000000000000000002100000000000000000000000000000000000000000000000000000000000000218000000000000000000000000000000000000000000000000000000000000022000000000000000000000000000000000000000000000000000000000000002280000000000000000000000000000000000000000000000000000000000000230000000000000000000000000000000000000000000000000000000000000023800000000000000000000000000000000000000000000000000000000000002400000000000000000000000000000000000000000000000000000000000000248000000000000000000000000000000000000000000000000000000000000025000000000000000000000000000000000000000000000000000000000000002580000000000000000000000000000000000000000000000000000000000000260000000000000000000000000000000000000000000000000000000000000026800000000000000000000000000000000000000000000000000000000000002700000000000000000000000000000000000000000000000000000000000000278000000000000000000000000000000000000000000000000000000000000028000000000000000000000000000000000000000000000000000000000000002880000000000000000000000000000000000000000000000000000000000000290000000000000000000000000000000000000000000000000000000000000029800000000000000000000000000000000000000000000000000000000000002a000000000000000000000000000000000000000000000000000000000000002a800000000000000000000000000000000000000000000000000000000000002b000000000000000000000000000000000000000000000000000000000000002b800000000000000000000000000000000000000000000000000000000000002c000000000000000000000000000000000000000000000000000000000000002c800000000000000000000000000000000000000000000000000000000000002d000000000000000000000000000000000000000000000000000000000000002d800000000000000000000000000000000000000000000000000000000000002e000000000000000000000000000000000000000000000000000000000000002e800000000000000000000000000000000000000000000000000000000000002f000000000000000000000000000000000000000000000000000000000000002f80000000000000000000000000000000000000000000000000000000000000300000000000000000000000000000000000000000000000000000000000000030800000000000000000000000000000000000000000000000000000000000003100000000000000000000000000000000000000000000000000000000000000318000000000000000000000000000000000000000000000000000000000000032000000000000000000000000000000000000000000000000000000000000003280000000000000000000000000000000000000000000000000000000000000330000000000000000000000000000000000000000000000000000000000000033800000000000000000000000000000000000000000000000000000000000003400000000000000000000000000000000000000000000000000000000000000348000000000000000000000000000000000000000000000000000000000000035000000000000000000000000000000000000000000000000000000000000003580000000000000000000000000000000000000000000000000000000000000360000000000000000000000000000000000000000000000000000000000000036800000000000000000000000000000000000000000000000000000000000003700000000000000000000000000000000000000000000000000000000000000378000000000000000000000000000000000000000000000000000000000000038000000000000000000000000000000000000000000000000000000000000003880000000000000000000000000000000000000000000000000000000000000390000000000000000000000000000000000000000000000000000000000000039800000000000000000000000000000000000000000000000000000000000003a000000000000000000000000000000000000000000000000000000000000003a800000000000000000000000000000000000000000000000000000000000003b000000000000000000000000000000000000000000000000000000000000003b800000000000000000000000000000000000000000000000000000000000003c000000000000000000000000000000000000000000000000000000000000003c800000000000000000000000000000000000000000000000000000000000003d000000000000000000000000000000000000000000000000000000000000003d800000000000000000000000000000000000000000000000000000000000003e000000000000000000000000000000000000000000000000000000000000003e800000000000000000000000000000000000000000000000000000000000003f000000000000000000000000000000000000000000000000000000000000003f80000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000040800000000000000000000000000000000000000000000000000000000000004100000000000000000000000000000000000000000000000000000000000000418000000000000000000000000000000000000000000000000000000000000042000000000000000000000000000000000000000000000000000000000000004280000000000000000000000000000000000000000000000000000000000000430000000000000000000000000000000000000000000000000000000000000043800000000000000000000000000000000000000000000000000000000000004400000000000000000000000000000000000000000000000000000000000000448000000000000000000000000000000000000000000000000000000000000045000000000000000000000000000000000000000000000000000000000000004580000000000000000000000000000000000000000000000000000000000000460000000000000000000000000000000000000000000000000000000000000046800000000000000000000000000000000000000000000000000000000000004700000000000000000000000000000000000000000000000000000000000000478000000000000000000000000000000000000000000000000000000000000048000000000000000000000000000000000000000000000000000000000000004880000000000000000000000000000000000000000000000000000000000000490000000000000000000000000000000000000000000000000000000000000049800000000000000000000000000000000000000000000000000000000000004a000000000000000000000000000000000000000000000000000000000000004a800000000000000000000000000000000000000000000000000000000000004b000000000000000000000000000000000000000000000000000000000000004b800000000000000000000000000000000000000000000000000000000000004c000000000000000000000000000000000000000000000000000000000000004c800000000000000000000000000000000000000000000000000000000000004d000000000000000000000000000000000000000000000000000000000000004d800000000000000000000000000000000000000000000000000000000000004e000000000000000000000000000000000000000000000000000000000000004e800000000000000000000000000000000000000000000000000000000000004f000000000000000000000000000000000000000000000000000000000000004f80000000000000000000000000000000000000000000000000000000000000500000000000000000000000000000000000000000000000000000000000000050800000000000000000000000000000000000000000000000000000000000005100000000000000000000000000000000000000000000000000000000000000518000000000000000000000000000000000000000000000000000000000000052000000000000000000000000000000000000000000000000000000000000005280000000000000000000000000000000000000000000000000000000000000530000000000000000000000000000000000000000000000000000000000000053800000000000000000000000000000000000000000000000000000000000005400000000000000000000000000000000000000000000000000000000000000548000000000000000000000000000000000000000000000000000000000000055000000000000000000000000000000000000000000000000000000000000005580000000000000000000000000000000000000000000000000000000000000560000000000000000000000000000000000000000000000000000000000000056800000000000000000000000000000000000000000000000000000000000005700000000000000000000000000000000000000000000000000000000000000578000000000000000000000000000000000000000000000000000000000000058000000000000000000000000000000000000000000000000000000000000005880000000000000000000000000000000000000000000000000000000000000590000000000000000000000000000000000000000000000000000000000000059800000000000000000000000000000000000000000000000000000000000005a000000000000000000000000000000000000000000000000000000000000005a800000000000000000000000000000000000000000000000000000000000005b000000000000000000000000000000000000000000000000000000000000005b800000000000000000000000000000000000000000000000000000000000005c000000000000000000000000000000000000000000000000000000000000005c800000000000000000000000000000000000000000000000000000000000005d000000000000000000000000000000000000000000000000000000000000005d800000000000000000000000000000000000000000000000000000000000005e000000000000000000000000000000000000000000000000000000000000005e800000000000000000000000000000000000000000000000000000000000005f000000000000000000000000000000000000000000000000000000000000005f80000000000000000000000000000000000000000000000000000000000000600000000000000000000000000000000000000000000000000000000000000060800000000000000000000000000000000000000000000000000000000000006100000000000000000000000000000000000000000000000000000000000000618000000000000000000000000000000000000000000000000000000000000062000000000000000000000000000000000000000000000000000000000000006280000000000000000000000000000000000000000000000000000000000000630000000000000000000000000000000000000000000000000000000000000063800000000000000000000000000000000000000000000000000000000000006400000000000000000000000000000000000000000000000000000000000000648000000000000000000000000000000000000000000000000000000000000065000000000000000000000000000000000000000000000000000000000000006580000000000000000000000000000000000000000000000000000000000000660000000000000000000000000000000000000000000000000000000000000066800000000000000000000000000000000000000000000000000000000000006700000000000000000000000000000000000000000000000000000000000000678000000000000000000000000000000000000000000000000000000000000068000000000000000000000000000000000000000000000000000000000000006880000000000000000000000000000000000000000000000000000000000000690000000000000000000000000000000000000000000000000000000000000069800000000000000000000000000000000000000000000000000000000000006a000000000000000000000000000000000000000000000000000000000000006a800000000000000000000000000000000000000000000000000000000000006b000000000000000000000000000000000000000000000000000000000000006b800000000000000000000000000000000000000000000000000000000000006c000000000000000000000000000000000000000000000000000000000000006c800000000000000000000000000000000000000000000000000000000000006d000000000000000000000000000000000000000000000000000000000000006d800000000000000000000000000000000000000000000000000000000000006e000000000000000000000000000000000000000000000000000000000000006e800000000000000000000000000000000000000000000000000000000000006f000000000000000000000000000000000000000000000000000000000000006f80000000000000000000000000000000000000000000000000000000000000700000000000000000000000000000000000000000000000000000000000000070800000000000000000000000000000000000000000000000000000000000007100000000000000000000000000000000000000000000000000000000000000718000000000000000000000000000000000000000000000000000000000000072000000000000000000000000000000000000000000000000000000000000007280000000000000000000000000000000000000000000000000000000000000730000000000000000000000000000000000000000000000000000000000000073800000000000000000000000000000000000000000000000000000000000007400000000000000000000000000000000000000000000000000000000000000748000000000000000000000000000000000000000000000000000000000000075000000000000000000000000000000000000000000000000000000000000007580000000000000000000000000000000000000000000000000000000000000760000000000000000000000000000000000000000000000000000000000000076800000000000000000000000000000000000000000000000000000000000007700000000000000000000000000000000000000000000000000000000000000778000000000000000000000000000000000000000000000000000000000000078000000000000000000000000000000000000000000000000000000000000007880000000000000000000000000000000000000000000000000000000000000790000000000000000000000000000000000000000000000000000000000000079800000000000000000000000000000000000000000000000000000000000007a000000000000000000000000000000000000000000000000000000000000007a800000000000000000000000000000000000000000000000000000000000007b000000000000000000000000000000000000000000000000000000000000007b800000000000000000000000000000000000000000000000000000000000007c000000000000000000000000000000000000000000000000000000000000007c800000000000000000000000000000000000000000000000000000000000007d000000000000000000000000000000000000000000000000000000000000007d800000000000000000000000000000000000000000000000000000000000007e000000000000000000000000000000000000000000000000000000000000007e800000000000000000000000000000000000000000000000000000000000007f000000000000000000000000000000000000000000000000000000000000007f80000000000000000000000000000000000000000000000000000000000000800000000000000000000000000000000000000000000000000000000000000080800000000000000000000000000000000000000000000000000000000000008100000000000000000000000000000000000000000000000000000000000000818000000000000000000000000000000000000000000000000000000000000082000000000000000000000000000000000000000000000000000000000000008280000000000000000000000000000000000000000000000000000000000000830000000000000000000000000000000000000000000000000000000000000083800000000000000000000000000000000000000000000000000000000000008400000000000000000000000000000000000000000000000000000000000000848000000000000000000000000000000000000000000000000000000000000085000000000000000000000000000000000000000000000000000000000000008580000000000000000000000000000000000000000000000000000000000000860000000000000000000000000000000000000000000000000000000000000086800000000000000000000000000000000000000000000000000000000000008700000000000000000000000000000000000000000000000000000000000000878000000000000000000000000000000000000000000000000000000000000088000000000000000000000000000000000000000000000000000000000000008880000000000000000000000000000000000000000000000000000000000000890000000000000000000000000000000000000000000000000000000000000089800000000000000000000000000000000000000000000000000000000000008a000000000000000000000000000000000000000000000000000000000000008a800000000000000000000000000000000000000000000000000000000000008b000000000000000000000000000000000000000000000000000000000000008b800000000000000000000000000000000000000000000000000000000000008c000000000000000000000000000000000000000000000000000000000000008c800000000000000000000000000000000000000000000000000000000000008d000000000000000000000000000000000000000000000000000000000000008d800000000000000000000000000000000000000000000000000000000000008e000000000000000000000000000000000000000000000000000000000000008e800000000000000000000000000000000000000000000000000000000000008f000000000000000000000000000000000000000000000000000000000000008f80000000000000000000000000000000000000000000000000000000000000900000000000000000000000000000000000000000000000000000000000000090800000000000000000000000000000000000000000000000000000000000009100000000000000000000000000000000000000000000000000000000000000918000000000000000000000000000000000000000000000000000000000000092000000000000000000000000000000000000000000000000000000000000009280000000000000000000000000000000000000000000000000000000000000930000000000000000000000000000000000000000000000000000000000000093800000000000000000000000000000000000000000000000000000000000009400000000000000000000000000000000000000000000000000000000000000948000000000000000000000000000000000000000000000000000000000000095000000000000000000000000000000000000000000000000000000000000009580000000000000000000000000000000000000000000000000000000000000960000000000000000000000000000000000000000000000000000000000000096800000000000000000000000000000000000000000000000000000000000009700000000000000000000000000000000000000000000000000000000000000978000000000000000000000000000000000000000000000000000000000000098000000000000000000000000000000000000000000000000000000000000009880000000000000000000000000000000000000000000000000000000000000990000000000000000000000000000000000000000000000000000000000000099800000000000000000000000000000000000000000000000000000000000009a000000000000000000000000000000000000000000000000000000000000009a800000000000000000000000000000000000000000000000000000000000009b000000000000000000000000000000000000000000000000000000000000009b800000000000000000000000000000000000000000000000000000000000009c000000000000000000000000000000000000000000000000000000000000009c800000000000000000000000000000000000000000000000000000000000009d000000000000000000000000000000000000000000000000000000000000009d800000000000000000000000000000000000000000000000000000000000009e000000000000000000000000000000000000000000000000000000000000009e800000000000000000000000000000000000000000000000000000000000009f000000000000000000000000000000000000000000000000000000000000009f80000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000000a080000000000000000000000000000000000000000000000000000000000000a100000000000000000000000000000000000000000000000000000000000000a180000000000000000000000000000000000000000000000000000000000000a200000000000000000000000000000000000000000000000000000000000000a280000000000000000000000000000000000000000000000000000000000000a300000000000000000000000000000000000000000000000000000000000000a380000000000000000000000000000000000000000000000000000000000000a400000000000000000000000000000000000000000000000000000000000000a480000000000000000000000000000000000000000000000000000000000000a500000000000000000000000000000000000000000000000000000000000000a580000000000000000000000000000000000000000000000000000000000000a600000000000000000000000000000000000000000000000000000000000000a680000000000000000000000000000000000000000000000000000000000000a700000000000000000000000000000000000000000000000000000000000000a780000000000000000000000000000000000000000000000000000000000000a800000000000000000000000000000000000000000000000000000000000000a880000000000000000000000000000000000000000000000000000000000000a900000000000000000000000000000000000000000000000000000000000000a980000000000000000000000000000000000000000000000000000000000000aa00000000000000000000000000000000000000000000000000000000000000aa80000000000000000000000000000000000000000000000000000000000000ab00000000000000000000000000000000000000000000000000000000000000ab80000000000000000000000000000000000000000000000000000000000000ac00000000000000000000000000000000000000000000000000000000000000ac80000000000000000000000000000000000000000000000000000000000000ad00000000000000000000000000000000000000000000000000000000000000ad80000000000000000000000000000000000000000000000000000000000000ae00000000000000000000000000000000000000000000000000000000000000ae80000000000000000000000000000000000000000000000000000000000000af00000000000000000000000000000000000000000000000000000000000000af80000000000000000000000000000000000000000000000000000000000000b000000000000000000000000000000000000000000000000000000000000000b080000000000000000000000000000000000000000000000000000000000000b100000000000000000000000000000000000000000000000000000000000000b180000000000000000000000000000000000000000000000000000000000000b200000000000000000000000000000000000000000000000000000000000000b280000000000000000000000000000000000000000000000000000000000000b300000000000000000000000000000000000000000000000000000000000000b380000000000000000000000000000000000000000000000000000000000000b400000000000000000000000000000000000000000000000000000000000000b480000000000000000000000000000000000000000000000000000000000000b500000000000000000000000000000000000000000000000000000000000000b580000000000000000000000000000000000000000000000000000000000000b600000000000000000000000000000000000000000000000000000000000000b680000000000000000000000000000000000000000000000000000000000000b700000000000000000000000000000000000000000000000000000000000000b780000000000000000000000000000000000000000000000000000000000000b800000000000000000000000000000000000000000000000000000000000000b880000000000000000000000000000000000000000000000000000000000000b900000000000000000000000000000000000000000000000000000000000000b980000000000000000000000000000000000000000000000000000000000000ba00000000000000000000000000000000000000000000000000000000000000ba80000000000000000000000000000000000000000000000000000000000000bb00000000000000000000000000000000000000000000000000000000000000bb80000000000000000000000000000000000000000000000000000000000000bc00000000000000000000000000000000000000000000000000000000000000bc80000000000000000000000000000000000000000000000000000000000000bd00000000000000000000000000000000000000000000000000000000000000bd80000000000000000000000000000000000000000000000000000000000000be00000000000000000000000000000000000000000000000000000000000000be80000000000000000000000000000000000000000000000000000000000000bf00000000000000000000000000000000000000000000000000000000000000bf80000000000000000000000000000000000000000000000000000000000000c000000000000000000000000000000000000000000000000000000000000000c080000000000000000000000000000000000000000000000000000000000000c100000000000000000000000000000000000000000000000000000000000000c180000000000000000000000000000000000000000000000000000000000000c200000000000000000000000000000000000000000000000000000000000000c280000000000000000000000000000000000000000000000000000000000000c300000000000000000000000000000000000000000000000000000000000000c380000000000000000000000000000000000000000000000000000000000000c400000000000000000000000000000000000000000000000000000000000000c480000000000000000000000000000000000000000000000000000000000000c500000000000000000000000000000000000000000000000000000000000000c580000000000000000000000000000000000000000000000000000000000000c600000000000000000000000000000000000000000000000000000000000000c680000000000000000000000000000000000000000000000000000000000000c700000000000000000000000000000000000000000000000000000000000000c780000000000000000000000000000000000000000000000000000000000000c800000000000000000000000000000000000000000000000000000000000000c880000000000000000000000000000000000000000000000000000000000000c900000000000000000000000000000000000000000000000000000000000000c980000000000000000000000000000000000000000000000000000000000000ca00000000000000000000000000000000000000000000000000000000000000ca80000000000000000000000000000000000000000000000000000000000000cb00000000000000000000000000000000000000000000000000000000000000cb80000000000000000000000000000000000000000000000000000000000000cc00000000000000000000000000000000000000000000000000000000000000cc80000000000000000000000000000000000000000000000000000000000000cd00000000000000000000000000000000000000000000000000000000000000cd80000000000000000000000000000000000000000000000000000000000000ce00000000000000000000000000000000000000000000000000000000000000ce80000000000000000000000000000000000000000000000000000000000000cf00000000000000000000000000000000000000000000000000000000000000cf80000000000000000000000000000000000000000000000000000000000000d000000000000000000000000000000000000000000000000000000000000000d080000000000000000000000000000000000000000000000000000000000000d100000000000000000000000000000000000000000000000000000000000000d180000000000000000000000000000000000000000000000000000000000000d200000000000000000000000000000000000000000000000000000000000000d280000000000000000000000000000000000000000000000000000000000000d300000000000000000000000000000000000000000000000000000000000000d380000000000000000000000000000000000000000000000000000000000000d400000000000000000000000000000000000000000000000000000000000000d480000000000000000000000000000000000000000000000000000000000000d500000000000000000000000000000000000000000000000000000000000000d580000000000000000000000000000000000000000000000000000000000000d600000000000000000000000000000000000000000000000000000000000000d680000000000000000000000000000000000000000000000000000000000000d700000000000000000000000000000000000000000000000000000000000000d780000000000000000000000000000000000000000000000000000000000000d800000000000000000000000000000000000000000000000000000000000000d880000000000000000000000000000000000000000000000000000000000000d900000000000000000000000000000000000000000000000000000000000000d980000000000000000000000000000000000000000000000000000000000000da00000000000000000000000000000000000000000000000000000000000000da80000000000000000000000000000000000000000000000000000000000000db00000000000000000000000000000000000000000000000000000000000000db80000000000000000000000000000000000000000000000000000000000000dc00000000000000000000000000000000000000000000000000000000000000dc80000000000000000000000000000000000000000000000000000000000000dd00000000000000000000000000000000000000000000000000000000000000dd80000000000000000000000000000000000000000000000000000000000000de00000000000000000000000000000000000000000000000000000000000000de80000000000000000000000000000000000000000000000000000000000000df00000000000000000000000000000000000000000000000000000000000000df80000000000000000000000000000000000000000000000000000000000000e000000000000000000000000000000000000000000000000000000000000000e080000000000000000000000000000000000000000000000000000000000000e100000000000000000000000000000000000000000000000000000000000000e180000000000000000000000000000000000000000000000000000000000000e200000000000000000000000000000000000000000000000000000000000000e280000000000000000000000000000000000000000000000000000000000000e300000000000000000000000000000000000000000000000000000000000000e380000000000000000000000000000000000000000000000000000000000000e400000000000000000000000000000000000000000000000000000000000000e480000000000000000000000000000000000000000000000000000000000000e500000000000000000000000000000000000000000000000000000000000000e580000000000000000000000000000000000000000000000000000000000000e600000000000000000000000000000000000000000000000000000000000000e680000000000000000000000000000000000000000000000000000000000000e700000000000000000000000000000000000000000000000000000000000000e780000000000000000000000000000000000000000000000000000000000000e800000000000000000000000000000000000000000000000000000000000000e880000000000000000000000000000000000000000000000000000000000000e900000000000000000000000000000000000000000000000000000000000000e980000000000000000000000000000000000000000000000000000000000000ea00000000000000000000000000000000000000000000000000000000000000ea80000000000000000000000000000000000000000000000000000000000000eb00000000000000000000000000000000000000000000000000000000000000eb80000000000000000000000000000000000000000000000000000000000000ec00000000000000000000000000000000000000000000000000000000000000ec80000000000000000000000000000000000000000000000000000000000000ed00000000000000000000000000000000000000000000000000000000000000ed80000000000000000000000000000000000000000000000000000000000000ee00000000000000000000000000000000000000000000000000000000000000ee80000000000000000000000000000000000000000000000000000000000000ef00000000000000000000000000000000000000000000000000000000000000ef80000000000000000000000000000000000000000000000000000000000000f000000000000000000000000000000000000000000000000000000000000000f080000000000000000000000000000000000000000000000000000000000000f100000000000000000000000000000000000000000000000000000000000000f180000000000000000000000000000000000000000000000000000000000000f200000000000000000000000000000000000000000000000000000000000000f280000000000000000000000000000000000000000000000000000000000000f300000000000000000000000000000000000000000000000000000000000000f380000000000000000000000000000000000000000000000000000000000000f400000000000000000000000000000000000000000000000000000000000000f480000000000000000000000000000000000000000000000000000000000000f500000000000000000000000000000000000000000000000000000000000000f580000000000000000000000000000000000000000000000000000000000000f600000000000000000000000000000000000000000000000000000000000000f680000000000000000000000000000000000000000000000000000000000000f700000000000000000000000000000000000000000000000000000000000000f780000000000000000000000000000000000000000000000000000000000000f800000000000000000000000000000000000000000000000000000000000000f880000000000000000000000000000000000000000000000000000000000000f900000000000000000000000000000000000000000000000000000000000000f980000000000000000000000000000000000000000000000000000000000000fa00000000000000000000000000000000000000000000000000000000000000fa80000000000000000000000000000000000000000000000000000000000000fb00000000000000000000000000000000000000000000000000000000000000fb80000000000000000000000000000000000000000000000000000000000000fc00000000000000000000000000000000000000000000000000000000000000fc80000000000000000000000000000000000000000000000000000000000000fd00000000000000000000000000000000000000000000000000000000000000fd80000000000000000000000000000000000000000000000000000000000000fe00000000000000000000000000000000000000000000000000000000000000fe80000000000000000000000000000000000000000000000000000000000000ff00000000000000000000000000000000000000000000000000000000000000ff800000000000000000000000000000000000000000000000000000000"
           
            LUTSelectorStr = "LUTSelector" + "\t" + lut_sel
            LUTValueAllStr = "LUTValueAll" + "\t" + lut_value_all
            LUTEnableStr = "LUTEnable" + "\t" + str(lut_en)
                                
            lut_opts = [LUTSelectorStr, LUTEnableStr,
                        LUTSelectorStr, LUTSelectorStr,
                        LUTValueAllStr, LUTSelectorStr]  
            
            #############################################
            stream_channel = "StreamChannel0"
            
            frame_transm_delay = 0
            
            GevStreamChannelSelectorStr = "GevStreamChannelSelector" + "\t" + stream_channel
            GevSCPSPacketSizeStr = "GevSCPSPacketSize" + "\t" + str(packet_size)
            GevSCPDStr = "GevSCPD" + "\t" + str(inter_packet_delay)
            GevSCFTDStr = "GevSCFTD" + "\t" + str(frame_transm_delay)
            GevSCBWRStr = "GevSCBWR" + "\t" + str(bandwidth_resv)
            GevSCBWRAStr = "GevSCBWRA" + "\t" + str(bandwidth_resv_acc)
                
            getv_opt = [GevStreamChannelSelectorStr, GevSCPSPacketSizeStr,
                        GevStreamChannelSelectorStr, GevStreamChannelSelectorStr,
                        GevSCPDStr, GevStreamChannelSelectorStr,
                        GevStreamChannelSelectorStr, GevSCFTDStr,
                        GevStreamChannelSelectorStr, GevStreamChannelSelectorStr,
                        GevSCBWRStr, GevStreamChannelSelectorStr,
                        GevStreamChannelSelectorStr, GevSCBWRAStr,
                        GevStreamChannelSelectorStr];
    
    #############################################  

    if adv_written == False:           
        auto_target_value = 128
        gray_value_adjust_damp = 700
        bal_white_adjust_damp = 1000
        auto_gain_lower = 0
        auto_gain_upper = 63
        auto_exp_time_lower = 70
        auto_exp_time_upper = 350000
        auto_func_prof = "GainMinimum"
        auto_func_aoi_sel_one = "AOI1"
        auto_func_aoi_width = 1920
        auto_func_aoi_sel_two = "AOI2"
        auto_func_aoi_height = 1080
        auto_func_aoi_offsetx = 0
        auto_func_aoi_offsety = 0 
        user_def_value = 0
        user_def_value_sel_one = "Value1"    
        user_def_value_sel_sec = "Value2"
        user_def_value_sel_third = "Value3"
        user_def_value_sel_fourth = "Value4"
        user_def_value_sel_fifth = "Value5" 
        chunk_mode = 0
        first_ev = "ExposureEnd"
        sec_ev = "FrameStartOvertrigger"    
        third_ev = "AcquisitionStartOvertrigger"
        fourth_ev = "FrameStart"
        fifth_ev = "AcquisitionStart"
        sixth_ev = "EventOverrun"
        ev_notif = "Off"
        
        AutoTargetValueStr = "AutoTargetValue" + "\t" + str(auto_target_value)
        GrayValueAdjustmentDampingRawStr = "GrayValueAdjustmentDampingRaw" + "\t" + str(gray_value_adjust_damp)
        BalanceWhiteAdjustmentDampingRawStr = "BalanceWhiteAdjustmentDampingRaw" + "\t" + str(bal_white_adjust_damp)
        AutoGainRawLowerLimitStr = "AutoGainRawLowerLimit" + "\t" + str(auto_gain_lower)
        AutoGainRawUpperLimitStr = "AutoGainRawUpperLimit" + "\t" + str(auto_gain_upper)
        AutoExposureTimeAbsLowerLimitStr = "AutoExposureTimeAbsLowerLimit" + "\t" + str(auto_exp_time_lower)
        AutoExposureTimeAbsUpperLimitStr = "AutoExposureTimeAbsUpperLimit" + "\t" + str(auto_exp_time_upper)
        AutoFunctionProfileStr = "AutoFunctionProfile" + "\t" + auto_func_prof
        AutoFunctionAOISelectorStr = "AutoFunctionAOISelector" + "\t" + auto_func_aoi_sel_one
        AutoFunctionAOISelectorTwoStr = "AutoFunctionAOISelector" + "\t" + auto_func_aoi_sel_two
        AutoFunctionAOIWidthStr = "AutoFunctionAOIWidth" + "\t" + str(auto_func_aoi_width)
        AutoFunctionAOIHeightStr = "AutoFunctionAOIHeight" + "\t" + str(auto_func_aoi_height)
        AutoFunctionAOIOffsetXStr = "AutoFunctionAOIOffsetX" + "\t" + str(auto_func_aoi_offsetx)
        AutoFunctionAOIOffsetYStr = "AutoFunctionAOIOffsetY" + "\t" + str(auto_func_aoi_offsety)
        UserDefinedValueSelectorOneStr = "UserDefinedValueSelector" + "\t" + user_def_value_sel_one
        UserDefinedValueSelectorSecStr = "UserDefinedValueSelector" + "\t" + user_def_value_sel_sec
        UserDefinedValueSelectorThirdStr = "UserDefinedValueSelector" + "\t" + user_def_value_sel_third
        UserDefinedValueSelectorFourthStr = "UserDefinedValueSelector" + "\t" + user_def_value_sel_fourth
        UserDefinedValueSelectorFifthStr = "UserDefinedValueSelector" + "\t" + user_def_value_sel_fifth
        UserDefinedValueStr = "UserDefinedValue" + "\t" + str(user_def_value)
        ChunkModeActiveStr = "ChunkModeActive" + "\t" + str(chunk_mode)
        EventSelectorFirst = "EventSelector" + "\t" + first_ev
        EventSelectorSecond = "EventSelector" + "\t" + sec_ev
        EventSelectorThird = "EventSelector" + "\t" + third_ev
        EventSelectorFourth = "EventSelector" + "\t" + fourth_ev
        EventSelectorFifth = "EventSelector" + "\t" + fifth_ev
        EventSelectorSixth = "EventSelector" + "\t" + sixth_ev
        EventNotificationStr = "EventNotification" + "\t" + ev_notif 
    
        further_adjustments = [AutoTargetValueStr, GrayValueAdjustmentDampingRawStr,
                               BalanceWhiteAdjustmentDampingRawStr, AutoGainRawLowerLimitStr,
                               AutoGainRawUpperLimitStr, AutoExposureTimeAbsLowerLimitStr,
                               AutoExposureTimeAbsUpperLimitStr, AutoFunctionProfileStr,
                               AutoFunctionAOISelectorStr, AutoFunctionAOIWidthStr,
                               AutoFunctionAOISelectorTwoStr, AutoFunctionAOIWidthStr,
                               AutoFunctionAOISelectorStr, AutoFunctionAOISelectorStr,
                               AutoFunctionAOIHeightStr, AutoFunctionAOISelectorTwoStr,
                               AutoFunctionAOIHeightStr, AutoFunctionAOISelectorStr,
                               AutoFunctionAOISelectorStr, AutoFunctionAOIOffsetXStr,
                               AutoFunctionAOISelectorTwoStr, AutoFunctionAOIOffsetXStr,
                               AutoFunctionAOISelectorStr, AutoFunctionAOISelectorStr,
                               AutoFunctionAOIOffsetYStr, AutoFunctionAOISelectorTwoStr,
                               AutoFunctionAOIOffsetYStr, AutoFunctionAOISelectorStr,
                               UserDefinedValueSelectorOneStr, UserDefinedValueStr,
                               UserDefinedValueSelectorSecStr, UserDefinedValueStr,
                               UserDefinedValueSelectorThirdStr, UserDefinedValueStr,
                               UserDefinedValueSelectorFourthStr, UserDefinedValueStr,
                               UserDefinedValueSelectorFifthStr, UserDefinedValueStr,
                               UserDefinedValueSelectorOneStr, ChunkModeActiveStr,
                               EventSelectorFirst, EventNotificationStr,
                               EventSelectorSecond, EventNotificationStr,
                               EventSelectorThird, EventNotificationStr,
                               EventSelectorFourth, EventNotificationStr,
                               EventSelectorFifth, EventNotificationStr,
                               EventSelectorSixth, EventNotificationStr,
                               EventSelectorFirst]
                
    
    pfs_file_struct = []
    
    for x in header:
        pfs_file_struct.append(x)  
    
    for x in seq_espec:
        pfs_file_struct.append(x)  
        
    for x in gain_opts:
        pfs_file_struct.append(x)  
    
    for x in blackLevel_opts:
        pfs_file_struct.append(x)
        
    for x in gamma_opts: 
        pfs_file_struct.append(x)
    
    for x in other_opt: 
        pfs_file_struct.append(x)
        
    for x in img_dim: 
        pfs_file_struct.append(x)
        
    for x in ax_params: 
        pfs_file_struct.append(x)
        
    for x in binning: 
        pfs_file_struct.append(x)
    
    pfs_file_struct.append(AcquisitionFrameCountStr)
    
    for x in trigger_opts: 
        pfs_file_struct.append(x)   
    
    for x in exp_opts: 
        pfs_file_struct.append(x)
        
    pfs_file_struct.append(ShutterModeStr)
    
    pfs_file_struct.append(AcquisitionFrameRateEnableStr)
    
    pfs_file_struct.append(AcquisitionFrameRateAbsStr)    
  
    for x in line_opts: 
        pfs_file_struct.append(x)
   
    for x in couter_opts: 
        pfs_file_struct.append(x)
    
    for x in lut_opts: 
        pfs_file_struct.append(x)
    
    for x in getv_opt: 
        pfs_file_struct.append(x)
    
    for x in further_adjustments: 
        pfs_file_struct.append(x)   
  
    numberLines = 0
    pfs_ok = False    
    
    print("Lines for PFS file: " + str(len(pfs_file_struct)))
    
    if ".pfs" in dir_path:
    
        with open(dir_path, "w") as input: 
            
            for ind_line, line in enumerate(pfs_file_struct):
                input.write(line + "\n")
                
                if ind_line == len(pfs_file_struct)-1:
                    pfs_ok = True
                    print("Success !!! PFS file generated") 
    else:
        print("Check if filename has one of the following extensions: \n {.pfs}")
    
    return pfs_ok
            

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
    
            e.write(lines + "<br>\n")    ## <br>           
            
    webbrowser.open('list_features.html', new=2)   


def second_gui_show_results(firstLoaded):
    
    import PySimpleGUI as sg_py
    
     
    MAX_SIZE = (200, 200)
    
    print("Second Gui")
    
    image_viewer_second_graph = [
        [sg_py.Text("Distance to the centroid, for the first cluster")],
        [sg_py.Image(key = "-DIST_FIRST_CLUSTER-")],
        [sg_py.Button("Distance to the centroid, for the first cluster")]
    ]
     
    image_viewer_third_graph = [
        [sg_py.Text("Distance to the centroid, for the second cluster")],
        [sg_py.Image(key = "-DIST_SECOND_CLUSTER-")],
        [sg_py.Button("Distance to the centroid, for the second cluster")]
    ]
    
    layout = [ 
        [
            sg_py.Column(image_viewer_second_graph),
            sg_py.VSeparator(),
            sg_py.Column(image_viewer_third_graph) 
        ]
    ]
     
    window = sg_py.Window("Output Results", layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))   ## web_port=2219,
    
    thisDir = os.getcwd()
    dirResultsOutput = thisDir + '\\GraphsOutput\\'     
   
    secondLoaded = False 
    
    while(True):
        event, values = window.read()    ## timeout = 1000 * 10
        
        if event == 'Exit' or event == sg_py.WIN_CLOSED: 
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
               
                    img = cv2.imread(dirResultsOutput + "distances_firstCluster.png")
                    cv2.imshow('Distance to the centroid, for the first cluster', img)
                    cv2.waitKey(0)
                                         
                    firstLoaded = True
                    secondLoaded = True
                    print("One")
                    
    #        break   ## secondLoaded == True and
                
        if secondLoaded == True and event == 'Distance to the centroid, for the second cluster':
            if os.path.exists(dirResultsOutput):
                if os.path.isfile(dirResultsOutput + "distances_secondCluster.png"):
                    img = cv2.imread(dirResultsOutput + "distances_secondCluster.png")
                    cv2.imshow('Distance to the centroid, for the second cluster', img)
                    cv2.waitKey(0)
          #           image = Image.open(dirResultsOutput + "distances_secondCluster.png")
          #           image.thumbnail(MAX_SIZE)
                    
          #           bio = io.BytesIO() 
          # ##          bio = io           ## .save(bio, format="PNG") 
          
          #           image.save(bio, format="PNG")
          #           window['-DIST_SECOND_CLUSTER-'].update(data = bio.getvalue())    
    
          #           secondLoaded = False
                    
                    time.sleep(5)
                    
                    print("Two")
                    
                    # time.sleep(1)
                    
            break
        # else:
        #      print("Unknown event") 
        
##    window.close()

def show_images():
    import pypylon
    # Connect to the first available Basler camera
    camera = pypylon.factory.find_devices()[0]
    grab_result = pypylon.PylonImage()
    cam = pypylon.factory.create_device(camera)
    cam.open()

    # Start grabbing images
    cam.start_grabbing(pypylon.GrabStrategy_LatestImageOnly)

    while cam.is_grabbing():
        # Retrieve the next grabbed image
        grab_result = cam.retrieve_result(5000, pypylon.TimeoutHandling_ThrowException)

        # Convert the image to OpenCV format
        image = grab_result.array

        # Show the image with cv2.imshow
        cv2.imshow("Basler Image", image)
        cv2.waitKey(1)
    
    # Stop grabbing images
    cam.stop_grabbing()
    cam.close()
    cv2.destroyAllWindows()

def show_button():
    import PySimpleGUI as sg
    
    layout = [[sg.Button('Stop Showing Images')]]
    window = sg.Window('Control Panel', layout)

    while True:
        event, values = window.read()

        if event in (None, 'Stop Showing Images'):
            break

    window.close()    

### Reduzir o tamanho da janela e ver possiblidade de ajustar com o rato para ficar maior ou menor (autoajuste)
def gui_show_results(clusteringRes, execTime, numberImg, ind_data, data_to_save):
     
    import PySimpleGUI as sg_py
    import os
    from PIL import Image
    import io
    
    def listFeatures(list_features):
        
        import time    
        import webbrowser 
        
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
        
                e.write(lines + "<br>\n")    ## <br>           
                
        webbrowser.open('list_features.html', new=2) 
    
    def second_gui_show_results(firstLoaded):
        
        import PySimpleGUI as sg_py
        import cv2
        import time
        import os
        
        MAX_SIZE = (200, 200)
        
        print("Second Gui")
        
        image_viewer_second_graph = [
            [sg_py.Text("Distance to the centroid, for the first cluster")],
            [sg_py.Image(key = "-DIST_FIRST_CLUSTER-")],
            [sg_py.Button("Distance to the centroid, for the first cluster")]
        ]
         
        image_viewer_third_graph = [
            [sg_py.Text("Distance to the centroid, for the second cluster")],
            [sg_py.Image(key = "-DIST_SECOND_CLUSTER-")],
            [sg_py.Button("Distance to the centroid, for the second cluster")]
        ]
        
        layout = [ 
            [
                sg_py.Column(image_viewer_second_graph),
                sg_py.VSeparator(),
                sg_py.Column(image_viewer_third_graph) 
            ]
        ]
         
        window = sg_py.Window("Output Results", layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))   ## web_port=2219,
        
        thisDir = os.getcwd()
        dirResultsOutput = thisDir + '\\GraphsOutput\\'     
       
        secondLoaded = False 
        
        while(True):
            event, values = window.read()    ## timeout = 1000 * 10
            
            if event == 'Exit' or event == sg_py.WIN_CLOSED: 
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
                   
                        img = cv2.imread(dirResultsOutput + "distances_firstCluster.png")
                        cv2.imshow('Distance to the centroid, for the first cluster', img)
                        cv2.waitKey(0)
                                             
                        firstLoaded = True
                        secondLoaded = True
                        print("One")
                        
        #        break   ## secondLoaded == True and
                    
            if secondLoaded == True and event == 'Distance to the centroid, for the second cluster':
                if os.path.exists(dirResultsOutput):
                    if os.path.isfile(dirResultsOutput + "distances_secondCluster.png"):
                        img = cv2.imread(dirResultsOutput + "distances_secondCluster.png")
                        cv2.imshow('Distance to the centroid, for the second cluster', img)
                        cv2.waitKey(0)
              #           image = Image.open(dirResultsOutput + "distances_secondCluster.png")
              #           image.thumbnail(MAX_SIZE)
                        
              #           bio = io.BytesIO() 
              # ##          bio = io           ## .save(bio, format="PNG") 
              
              #           image.save(bio, format="PNG")
              #           window['-DIST_SECOND_CLUSTER-'].update(data = bio.getvalue())    
        
              #           secondLoaded = False
                        
                        time.sleep(5)
                        
                        print("Two")
                        
                        # time.sleep(1)
                        
                break
            
    
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
        [sg_py.Text("Number of images generated, for the selected test video: ")],
        [sg_py.Text("", size = (10,2), key='-TOT_IMG-')],
        [sg_py.Text("Execution time for the selected test video (sec): ")],
        [sg_py.Text("", size = (10,2), key='-EXEC_TIME-')],
        [sg_py.Text("Number of clusters: ")],
        [sg_py.Text("", size = (10,2), key='-NUMBER_CLUSTERS-')],
        [sg_py.Text("Number of recomended clusters: ")],
        [sg_py.Text("", size = (10,2), key='-NUMBER_REC_CLUSTERS-')],
        [sg_py.Text("Final list of features, for the current test video: ")],
   ##     [sg_py.Listbox(values=[], size = (10, 20), key='-LISTBOX_FEATURES-')],   ## no_scrollbar=True
        [sg_py.Button("Data Info"), sg_py.Button("Read more ..."), sg_py.Button("Exit")]
    
    ]    
    
    image_viewer_first_graph = [
        [sg_py.Text("PCA Results")],
        [sg_py.Image(key = "-PCA-", size=(200,200))],
        [sg_py.Button("PCA graph output for the first two features")],
        [sg_py.Button("Distances to the centroid")]
    ]    

    layout = [ 
        [
            sg_py.Column(layout_inf),
            sg_py.VSeparator(),
            sg_py.Column(image_viewer_first_graph)
      ##      sg_py.VSeparator(),
     ##       sg_py.Column(image_viewer_second_graph, image_viewer_third_graph)
      ##      sg_py.VSeparator(),
      ##      sg_py.Column(image_viewer_third_graph)         
         
        ]
    ]
    
    window = sg_py.Window("Output Results", layout, disable_close=True, resizable = True, finalize = True, margins=(0,0))   ## web_port=2219,
    
    thisDir = os.getcwd()
    dirResultsOutput = thisDir + '\\GraphsOutput\\'
    
    firstLoaded = False
    secondLoaded = False
##    thirdLoaded = False

    
    print("Loop Gui for results")
    
    while(True):
        event, values = window.read()    ## timeout = 1000 * 10
        
        print("Hear")
        
        if event == 'Exit' or event == sg_py.WIN_CLOSED: 
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
     #       againHere = True
            
           
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
        if (firstLoaded == True) and  event == 'Distances to the centroid': ##  or ind_data == len(data_to_save) - 1  ##  firstLoaded == True and 
             print("Distances to the centroid")
             second_gui_show_results(firstLoaded) 

             firstLoaded = False             
             
             time.sleep(3) 
              
             break

    
# def mouseHoverSlider():
    
#     layout = [  [sg.Text('Move mouse over me', key='-TEXT-')],
#             [sg.In(key='-IN-')],
#             [sg.Button('Right Click Me', key='-BUTTON-'), sg.Button('Exit')]  ]

#     window = sg.Window('Window Title', layout, finalize=True)
    
#     window.bind('<FocusOut>', '+FOCUS OUT+')
    
#     window['-BUTTON-'].bind('<Button-3>', '+RIGHT CLICK+')
#     window['-TEXT-'].bind('<Enter>', '+MOUSE OVER+')
#     window['-TEXT-'].bind('<Leave>', '+MOUSE AWAY+')
#     window['-IN-'].bind('<FocusIn>', '+INPUT FOCUS+')
    
#     while True:             # Event Loop
#         event, values = window.read()
#         print(event, values)
#         if event in (None, 'Exit'):
#             break
#     window.close()
 

# def setParametersToPypylon(values):
#     print("Setting parameters to Pylon ...")
#     gain = values['GAIN']
#     black_level = values['BLACK_THRESH']    
#     image_height = values['FIRST']                          
#     image_width = values['SEC']
#     exp_time_us = values['EXP_TIME']    
#     packet_size = values['PACKET_SIZE']    
#     inter_packet_delay = values['INTER_PACKET_DELAY']    
#     frame_rate = values['FRAME_RATE']    
#     bandwidth_reserv_acc = values['BANDWIDTH_RESV_ACC']
#     bandwidth_reserv = values['BANDWIDTH_RESV']  
     
#     print("gain: " + str(gain))
#     print("black_level: " + str(black_level))
#     print("image_height: " + str(image_height))
#     print("image_width: " + str(image_width))   
#     print("exp_time_us: " + str(exp_time_us))
#     print("packet_size: " + str(packet_size))
#     print("inter_packet_delay: " + str(inter_packet_delay))
#     print("frame_rate: " + str(frame_rate))
#     print("bandwidth_reserv_acc: " + str(bandwidth_reserv_acc))
#     print("bandwidth_reserv: " + str(bandwidth_reserv))     
    
#     camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
#     converter = pylon.ImageFormatConverter()   
    
#     camera.Open()
    
#     print(str(camera.Open()))
    
#     camera.Height.SetValue(image_height)
#     camera.Width.SetValue(image_width)
    
    
#     camera.CenterX=False 
#     camera.CenterY=False
    
#     camera.GainRaw = gain
#     camera.ExposureTimeRaw = exp_time_us
    
#     camera.AcquisitionFrameRateEnable.SetValue(True)
#     camera.AcquisitionFrameRateAbs.SetValue(frame_rate)
    
#     camera.GevSCPSPacketSize.SetValue(packet_size)
    
#     # Inter-Packet Delay            
#     camera.GevSCPD.SetValue(inter_packet_delay)
    
#     # Bandwidth Reserve 
#     camera.GevSCBWR.SetValue(bandwidth_reserv) 
    
#     # Bandwidth Reserve Accumulation
#     camera.GevSCBWRA.SetValue(bandwidth_reserv_acc)  
    
#     print("Configuration successful ... ")




# resTypes = ['Full HD',
#             'HD+',
#             'HD',
#             'qHD',
#             'nHD',
#             '960H',
#             'HVGA',
#             'VGA',
#             'SVGA',
#             'DVGA',
#             'QVGA',
#             'QQVGA',
#             'HQVGA'
#            ]

# res_dims = ['(1920x1080)',
#             '(1600x900)',
#             '(1280x720)',
#             '(960x540)',
#             '(640x360)',
#             '(960x480)',
#             '(480x320)',
#             '(640x480)',
#             '(800x600)',
#             '(960x640)',
#             '(320x240)',
#             '(160x120)',
#             '(240x160)'
#           ]   

# res_fullTypes = []

# for ind_r, r in enumerate(resTypes):
#     r += ' ' + res_dims[ind_r]
#     res_fullTypes.append(r)


# basler_check = False

# basler_check, model_name = confirm_basler() 




# def acq_task(queue, qcountx, eventt):
#     ta = threading.Thread(target=acq_image_camera, args=(queue, countx, True, eventt))
#  ##   t.setDaemon(True)
#     ta.start() 


# def button_task(queue, a, b):
#     tb = threading.Thread(target=stop_button_cam_imgs_layout, args=(queue, a, b))
#  ##   t.setDaemon(True)
#     tb.start() 
    
#     ta = threading.Thread(target=acq_image_camera, args=(queue, countx, True, eventt))
   
#     tb.join() 
#     ta.join()
    
def smap(f):
    return f()
    
 ##   t.join() 
    
  ##  return resp  
  
  
def worker_process(queue):
    """
    This function runs in a separate process and controls the camera.
    """
    
    print("Worker process")
    
    from pypylon import pylon as py
    
    # Initialize the camera
    camera = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
    converter = py.ImageFormatConverter() 
    camera.Open()
    
    print("Camera opened")

    while True:
        # Check the queue for commands
        if not queue.empty():
            command = queue.get()
            
            print(command)

            # Handle the command
            if command == "start":
                print("Start")
                camera.StartGrabbing(py.GrabStrategy_LatestImageOnly)
                 
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():   
                            print("Reading image")
                            image = converter.Convert(grabResult)
                            print("Converted")
                            img = image.GetArray()  
                     ##       showLiveImageGUI(img, counter)
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            cv2.imshow('Live camera image', img)
                ##            print("Shown ... ")
                            cv2.waitKey(1)  
                            
            elif command == "stop":
                print("Stop")
                camera.StopGrabbing() 
            elif command == "exit":
                break

    # Clean up
    camera.Close()

def gui_process(queue):
    """
    This function runs in the main process and creates the GUI.
    """
    
    print("GUI")
    
    import PySimpleGUI as sg 
    
    layout = [
        [sg.Text("Camera Control")],
        [sg.Button("Start"), sg.Button("Stop")],
        [sg.Exit()]
    ]

    window = sg.Window("Camera Control", layout)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Exit"):
            queue.put("exit")
            break
        elif event == "Start":
            queue.put("start")
            print("Start")
        elif event == "Stop":
            queue.put("stop")
            print("Stop")

    window.close()
    
def worker(q,func):
    result = func(q)
 

def run_in_parallel(funcs):
    q = multiprocessing.Queue()
    processes = []

    for func in funcs:       
            p = multiprocessing.Process(target=worker, args=(q, func))
            processes.append(p)
            p.start() 

    for p in processes:
        p.join()

    results = []
    while not q.empty():
        results.append(q.get())

    return results     


    
if True:
    
    countx = 0
    
    # eventt = Event()
    
    # acq_task(queue, countx, eventt)
    
    # button_task(queue, True, True) 
    
    print("A")

    
    videoRec = False
    
#    time.sleep(10)

    funct_list = control_gui()
    [acquire, process] = funct_list
    
    if acquire and process:
        
        resTypes = ['Full HD',
                    'HD+',
                    'HD',
                    'qHD',
                    'nHD',
                    '960H',
                    'HVGA',
                    'VGA',
                    'SVGA',
                    'DVGA',
                    'QVGA',
                    'QQVGA',
                    'HQVGA'
                   ]

        res_dims = ['(1920x1080)',
                    '(1600x900)',
                    '(1280x720)',
                    '(960x540)',
                    '(640x360)',
                    '(960x480)',
                    '(480x320)',
                    '(640x480)',
                    '(800x600)',
                    '(960x640)',
                    '(320x240)',
                    '(160x120)',
                    '(240x160)'
                  ]   

        res_fullTypes = []

        for ind_r, r in enumerate(resTypes):
            r += ' ' + res_dims[ind_r]
            res_fullTypes.append(r)


        basler_check = False

        basler_check, model_name = confirm_basler()
        
        count_overall = 0
        
        resp_demo = False
        
        while True: 
            
            here_demo = False
            
            if here_demo == False and count_overall > 0 and resp_demo == False:
                break

            if basler_check == True:
        
                videoRec = gui_video_rec()  
                  
                if videoRec == True:
                 
                
                    ok_resp = load_core_packages()
                    
                    while ok_resp == False:
                        print("Could not load required packages. Check for errors or updates !!!")    
                        ok_resp = load_core_packages()        
                    
                    
                        
                    
                    ## Show camera live
                    
                    def pop_up_pfs():
                        import PySimpleGUI as sg
                        
              ##          pfs_resp = sg.popup_get_text("Do you want to load a PFS file ?")
                        
                        repInit = True
                        
                        pfs_resp = -1
                        
                        layout = [
                            [sg.Text("Do you want to load a PFS file ?")],      ## Use a pre-recorded video    ## Perform a new image acquisition process
                            [sg.T("         "), sg.Checkbox('Yes', default=False, key="-IN1-")],
                            [sg.T("         "), sg.Checkbox('No', default=True, key="-IN2-")],
                            [sg.Button("Next")]  
                        ]           
                         
                        window = sg.Window('PFS File loading', layout)              
                          
                        while repInit:
                  
                           event, values = window.read()
                          
                           
                           if event == "Exit" or event == sg.WIN_CLOSED:
                               break 
                           
                           if event == "Next":
                               if values["-IN1-"] == True and values["-IN2-"] == False:
                                   pfs_resp = 1
                                   repInit = False
                                   break
                               elif values["-IN2-"] == True and values["-IN1-"] == False:
                                   pfs_resp = 0
                                   repInit = False
                                   break
                               else:
                                   if values["-IN1-"] == True and values["-IN2-"] == True:
                                       print("Only pick one option !!!")                   
                                       repInit = True           
                                       continue
                           
                        window.close()        
                        
                        return pfs_resp
                    
                    resp_pfs_opt = pop_up_pfs()  
                    
             #        resp_pfs_opt = pop_up_pfs() 
                        
             # ##       resp_pfs_opt = input("Do you want to load a PFS file ? ")
                    
             #        while not(('S' in resp_pfs_opt) or ('Y' in resp_pfs_opt) or ('s' in resp_pfs_opt) or ('y' in resp_pfs_opt) or ('N' in resp_pfs_opt) or ('n' in resp_pfs_opt)):
             #      ##      resp_pfs_opt = input("Do you want to load a PFS file ? {'S'; 's'; 'Y'; 'y'; 'N'; 'n'}")
             #              resp_pfs_opt = pop_up_pfs()
                        
               #     if ('N' in resp_pfs_opt) or ('n' in resp_pfs_opt):
                   
                    if resp_pfs_opt == 0:
                        
                        packet_size = 1500
                        inter_paket_delay = 5000
                        bw_resv_acc = 4
                        bw_resv = 10
                        
                        if True:
                     
                  
                            
                            repeat = True
                            
                            col2 = sg.Column([       
                                 
                                    # Information frame
                            [sg.Frame(layout=[[sg.Text('Gain:')],  
                                                      [sg.Input("50", size=(19, 1), key="GAIN")],
                                                      [sg.Text('Black Level:')],
                                                      [sg.Input("2", size=(19, 1), key="BLACK_THRESH")],    ## , sg.Button('Copy')
                                                      [sg.Text('Image Resolution:')],
                                                      [sg.InputCombo(res_fullTypes, size = (19, 1), key="RES_TYPE", change_submits=False)],
                                                      # [sg.Text('Image Height:')],
                                                      # [sg.InputCombo(itemHeight, size=(19, 1), key="FIRST", change_submits=False) ],
                                                      # [sg.Text('Image Width:')],
                                                      # [sg.InputCombo(itemHeight, size=(19, 1), key="SEC", change_submits=False)],
                                                      [sg.Text('Exposition Time:')], 
                                                      [sg.Input(default_text= "140", size=(19, 1), key="EXP_TIME")],     ## , sg.Button('Copy') 
                                                      [sg.Text('Recording Time:')],
                                                      [sg.Slider(default_value = 5, orientation ='horizontal', key='recTime', range=(1,100)),
                                                           sg.Text(size=(5,2), key='-SECONDS-'), sg.Text(" seconds")]
                                                      ], title='Information:')],])
                                    
                            col3 = sg.Column([     
                                 
                                    # Information frame
                                    [sg.Frame(layout=[
                                                      # [sg.Text('Packet Size:')], 
                                                      # [sg.Input(default_text= "1500", size=(19, 1), key="PACKET_SIZE")],     ## , sg.Button('Copy')
                                                      # [sg.Text('Inter-Packet Delay:')],
                                                      # [sg.Input(default_text= "5000", size=(19, 1), key="INTER_PACKET_DELAY")],
                                                      [sg.Text('Frame rate:')],
                                                      [sg.Input(default_text= "50", size=(19, 1), key="FRAME_RATE")],
                                                      [sg.T("         "), sg.Checkbox('Enable basic streaming panel', default=False, key="-STREAM_PANEL-")],
                                                      
                                                      [sg.Button("Ok")]
                                                      # [sg.Text('Bandwidth Reserve Accumulation:')],
                                                      # [sg.Input(default_text= "4", size=(19, 1), key="BANDWIDTH_RESV_ACC")],
                                                      # [sg.Text('Bandwidth Reserve:')],
                                                      # [sg.Input(default_text= "10", size=(19, 1), key="BANDWIDTH_RESV")]                     
                                                      ], title='Information 2:')],])
                             
                            col4 = sg.Column([ 
                                
                                [sg.Frame(layout=[[sg.Text('Decisor Level:')], 
                                                  [sg.Slider(default_value = 0, orientation ='horizontal', key='decLevel', range=(0,50))],
                                                  [sg.Text('', size=(5,2), key='Decisor Level:')],
                                                  [sg.Image('255_to_0' + '.png', size = (30,1))],
                                                  [sg.Text('Software Light Level')]
                                   
                            ], title='Advanced Parameters:')],])
                            
                            layout = [ [col2, col3, col4],     
                                    # Actions Frame 
                                    [sg.Frame(layout=[[sg.Button("Preview ...")],[sg.Button('Save'), sg.Button('Clear'), sg.Button('Delete')],
                                                      [sg.Button('Advanced Properties ...'), sg.Button('Extra Properties ...'),  sg.Button('Calibrate camera ...')]                 
                                        ], title='Actions:')]]       
                            window = sg.Window('GUI', layout, web_port=2219, disable_close=True, resizable = True, finalize = True, margins=(0,0))  
                                
                            valid = True  
                            setFolder = False   
                            adv_go = False
                            
                            params_changed = {}
                            only_adv_opt = {}
                            
                            code_ret = -1
                           
                            while repeat == True:             # Event Loop
                                    event, values = window.read()
                                    print(event, values)        
                                  
                                     
                                    if event == "Exit" or event == sg.WIN_CLOSED or code_ret == 0:
                                        break
                                    
                                    if event == 'Preview ...':
                                        
                                        gain = 50
                                        black_thresh = 2
                                        exp_time = 140    ## us
                                        
                                        format_image = values['RES_TYPE']
                                        frame_rate = 50                             
                                        
                                        if len(values["GAIN"]) == 0:
                                            print("Setting default gain = 50 ...")
                                        else:
                                            if int(values["GAIN"]) < 0 or int(values["GAIN"]) > 50:
                                                sg.popup_error('Gain must be within [0,50] interval') 
                                                valid = False 
                                                print("Failed to gain range")
                                            else:
                                                gain = int(values["GAIN"])
                                                
                                        if len(values["BLACK_THRESH"]) == 0:
                                            print("Setting default black threshold ...")
                                        else:
                                            if int(values["BLACK_THRESH"]) < 0 or int(values["BLACK_THRESH"]) > 255:
                                                sg.popup_error('Black level must be within [0,255] interval')    ## for Mono8 configuration 
                                                valid = False
                                                print("Failed to black level range")
                                            else:
                                                black_thresh = int(values["BLACK_THRESH"])
                                        
                                        if len(values["EXP_TIME"]) == 0:
                                            print("Setting default exposure time ...")
                                        else:                    
                                            if int(values["EXP_TIME"]) < 0 or int(values["EXP_TIME"]) > 35000: 
                                                sg.popup_error('Exposition time must be within [0,35000] interval')  
                                                valid = False
                                                print("Failed to exposure time range") 
                                            else:
                                                exp_time = int(values["EXP_TIME"])
                                                
                                        if len(values["FRAME_RATE"]) == 0:
                                             print("Setting default frame rate ...")
                                        else:       
                                             if int(values["FRAME_RATE"]) < 10 or int(values["FRAME_RATE"]) > 50: 
                                                 sg.popup_error('Frame rate (fps) must be within [10,50] interval') 
                                                 valid = False
                                                 print("Failed to frame rate range")
                                             else:
                                                 frame_rate = int(values["FRAME_RATE"]) 
                                                 
                                                 if frame_rate == 50:
                                                   
                                                     if format_image == resTypes[0]:
                                                        format_image = resTypes[1] 
                                                        print("Changing resolution from " + resTypes[0] + " to " + resTypes[1] + " due to the maximum frame rate limitations ...")
                                                    
                                        
                                        params_stream_basic = [gain, black_thresh, format_image, exp_time, packet_size, inter_paket_delay, bw_resv_acc, bw_resv, frame_rate]
                                        
                                        get_test_image(params_stream_basic)
                                  
                                        # from init_position_settings import show_images_and_button
                                        # show_images_and_button(params_stream_basic)
                                    
                                    if event == 'Advanced Properties ...':                         
                                             
                                            adv_go = True  
                                      
                                            params_changed = adv_params_gui()      
                                            
                                            
                                            ####################################################################### 
                                            ####################################################################### 
                                            #######################################################################     
                                    if event == 'Extra Properties ...':
                                         
                                        from pypylon import pylon
                                        
                                        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                                        converter = pylon.ImageFormatConverter() 
                                        
                                        camera.Open()
                                        
                                        resp_param = 0
                                        
                                        while resp_param == 0: 
                                        
                                            resp_param = extra_params_gui(camera)
                                            break
                                     
                                    
                                    if event == 'Calibrate camera ...':
                                        print("Clicked")
                                        code_ret = calib_camera(False, model_name)
                                        
                                        if code_ret == 1: 
                                            break 
                                         
                                             
                                                                    
                                    
                                    if event == 'decisorLevel':           
                                        
                                        windowDecisorLevelSlider = sg_py.Window('', [sg_py.Text("", size=(0, 1), key='OUTPUTX')]).read(close=True)
                                        
                                        windowDecisorLevelSlider.bind('<FocusOut>', '+FOCUS OUT+')
                            
                                        windowDecisorLevelSlider['-BUTTON-'].bind('<Button-3>', '+RIGHT CLICK+')
                                        windowDecisorLevelSlider['-TEXT-'].bind('<Enter>', '+MOUSE OVER+')
                                        windowDecisorLevelSlider['-TEXT-'].bind('<Leave>', '+MOUSE AWAY+')
                                        windowDecisorLevelSlider['-IN-'].bind('<FocusIn>', '+INPUT FOCUS+') 
                                        
                                        eventDecLevel, valSliderDecLevel = windowDecisorLevelSlider.read()
                                        if eventDecLevel == sg_py.WINDOW_CLOSED:
                                            break 
                                        else: 
                                            windowDecisorLevelSlider['OUTPUTX'].update(value=values['decLevel'])
                                        
                                        time.sleep(2)
                                         
                                        windowDecisorLevelSlider.close()
                                        
                                    
                                    if event == 'recTime': 
                                        
                                        print("Event up to slider triggered !!!")
                                        window_sliderVal = sg_py.Window('', [sg_py.Text("", size=(0, 1), key='OUTPUT')]).read(close=True)
                                        
                                        window_sliderVal.bind('<FocusOut>', '+FOCUS OUT+')
                            
                                        window_sliderVal['-BUTTON-'].bind('<Button-3>', '+RIGHT CLICK+')
                                        window_sliderVal['-TEXT-'].bind('<Enter>', '+MOUSE OVER+')
                                        window_sliderVal['-TEXT-'].bind('<Leave>', '+MOUSE AWAY+')
                                        window_sliderVal['-IN-'].bind('<FocusIn>', '+INPUT FOCUS+')
                                        
                                        eventSlider, valSlider = window_sliderVal.read()
                                        if eventSlider == sg_py.WINDOW_CLOSED:
                                            break
                                        else:
                                            window_sliderVal['OUTPUT'].update(value=values['recTime'])          
                                            
                                   
                                        time.sleep(2) 
                                         
                                        window_sliderVal.close()
                                    
                                    else:    
                                        
                                        if event == "Ok":
                                            ok_button = True
                                            
                                            if values['-STREAM_PANEL-'] == True:
                                                packet_size, inter_packet_delay, bw_resv_acc, bw_resv = get_basic_stream_params()
                                                
                                            
                                            
                                        if event == 'Save':
                                            
                                         ##   if values['-STREAM_PANEL-'] == True: 
                                             
                                            emptyVals = False
                                            
                                            if len(values["GAIN"]) == 0:
                                                window["GAIN"].update("50")
                                                emptyVals = True
                                            if len(values["BLACK_THRESH"]) == 0:
                                                window["BLACK_THRESH"].update("2")
                                                emptyVals = True
                                            if len(values["EXP_TIME"]) == 0:
                                                window["EXP_TIME"].update("140")
                                                emptyVals = True
                                            if len(values["FRAME_RATE"]) == 0:
                                                window["FRAME_RATE"].update("50")
                                                emptyVals = True
                                             
                                            if emptyVals:                                    
                                                time.sleep(5)                                    
                                            
                                            if adv_go == True:
                                                
                                                window['Decisor Level:'].update(value = values['decLevel'])                   
                                                window['-SECONDS-'].update(value = values['recTime'])
                                                
                                                time.sleep(2) 
                                                 
                                                only_adv_opt = params_changed.copy()
                                                
                                                                                     
                                                # if len(values["GAIN"]) == 0:
                                                #     sg.popup_error('Gain - Empty field') 
                                                #     valid = False
                                            #    else:
                                                
                                                if not emptyVals:
                                                    if int(values["GAIN"]) < 0 or int(values["GAIN"]) > 50:
                                                        sg.popup_error('Gain must be within [0,50] interval') 
                                                        valid = False 
                                                        print("Failed to gain range") 
                                                        
                                                # if len(values["BLACK_THRESH"]) == 0:
                                                #     sg.popup_error('Black Level - Empty field') 
                                                #     valid = False
                                                # else:
                                                if not emptyVals:
                                                    if int(values["BLACK_THRESH"]) < 0 or int(values["BLACK_THRESH"]) > 255:
                                                        sg.popup_error('Black level must be within [0,255] interval')    ## for Mono8 configuration 
                                                        valid = False
                                                        print("Failed to black level range")
                                                
                                                # if len(values["EXP_TIME"]) == 0:
                                                #     sg.popup_error('Exposure Time - Empty field')  
                                                #     valid = False
                                                # else:                                        
                                                if not emptyVals:
                                                    if int(values["EXP_TIME"]) < 0 or int(values["EXP_TIME"]) > 35000: 
                                                        sg.popup_error('Exposition time must be within [0,35000] interval')  
                                                        valid = False
                                                        print("Failed to exposure time range")
                                                
                                                # if len(values["PACKET_SIZE"]) == 0:
                                                #     sg.popup_error('Packet Size - Empty field')  
                                                #     valid = False 
                                                # else:
                                                #     if int(values["PACKET_SIZE"]) <= 0:
                                                #         sg.popup_error('Packet size must be positive')
                                                #         valid = False
                                                #         print("Failed to packet size range")
                                                
                                                # if len(values["INTER_PACKET_DELAY"]) == 0:
                                                #     sg.popup_error('Inter-Packet Delay - Empty field')  
                                                #     valid = False 
                                                # else:    
                                                #     if int(values["INTER_PACKET_DELAY"]) < 0 or int(values["INTER_PACKET_DELAY"]) > 10000:
                                                #         sg.popup_error('Inter-packet delay must be within [0,10000] interval') 
                                                #         valid = False  
                                                #         print("Failed to inter-packet delay range") 
                                                        
                                                # if len(values["FRAME_RATE"]) == 0:
                                                #     sg.popup_error('Frame Rate - Empty field')  
                                                #     valid = False 
                                                # else:
                                                if not emptyVals:
                                                    if int(values["FRAME_RATE"]) < 10 or int(values["FRAME_RATE"]) > 50: 
                                                        sg.popup_error('Frame rate (fps) must be within [10,50] interval') 
                                                        valid = False
                                                        print("Failed to frame rate range")
                                                    
                                                # if len(values["BANDWIDTH_RESV_ACC"]) == 0:
                                                #     sg.popup_error('Bandwidth Reserve Accumulation - Empty field')  
                                                #     valid = False
                                                # else:
                                                #     if int(values["BANDWIDTH_RESV_ACC"]) < 0 or int(values["BANDWIDTH_RESV_ACC"]) > 10:
                                                #         sg.popup_error('Bandwidth reserve accumulation must be within [0,100] interval')  
                                                #         valid = False
                                                #         print("Failed to bandwidth reserve accumalation range")
                                                        
                                                # if len(values["BANDWIDTH_RESV"]) == 0:
                                                #     sg.popup_error('Bandwidth Reserve - Empty field')  
                                                #     valid = False
                                                # else:                     
                                                #     if int(values["BANDWIDTH_RESV"]) < 0 or int(values["BANDWIDTH_RESV"]) > 100:
                                                #         sg.popup_error('Bandwidth reserve must be within [0,10] interval')   
                                                #         valid = False
                                                #         print("Failed to bandwidth reserve range") 
                                                        
                                                #### 
                                                                                  
                                                if 'Full HD' in values['RES_TYPE']:
                                                    width = 1920
                                                    height = 1080
                                                elif 'HD+' in values['RES_TYPE']:
                                                    width = 1600
                                                    height = 900
                                                elif 'HD' in values['RES_TYPE']:
                                                    width = 1280
                                                    height = 720
                                                elif 'qHD' in values['RES_TYPE']:
                                                    width = 960
                                                    height = 540
                                                elif 'nHD' in values['RES_TYPE']:
                                                    width = 640
                                                    height = 360
                                                elif '960H' in values['RES_TYPE']:
                                                    width = 960
                                                    height = 480
                                                elif 'HVGA' in values['RES_TYPE']:
                                                    width = 480
                                                    height = 320
                                                elif 'VGA' in values['RES_TYPE']:
                                                    width = 640
                                                    height = 480
                                                elif 'SVGA' in values['RES_TYPE']:
                                                    width = 800
                                                    height = 600
                                                elif 'DVGA' in values['RES_TYPE']:
                                                    width = 960
                                                    height = 640
                                                elif 'QVGA' in values['RES_TYPE']:
                                                    width = 320 
                                                    height = 240
                                                elif 'QQVGA' in values['RES_TYPE']:
                                                    width = 160
                                                    height = 120
                                                elif 'HQVGA' in values['RES_TYPE']:
                                                    width = 240
                                                    height = 160 
                                                    
                                                ######################################################################
                                                ######################################################################
                                                ## Add parameters from previous GUI, to params_changed dictionary ####
                                                
                                                if not emptyVals:
                                                
                                                    params_changed['GAIN'] = values["GAIN"]
                                                    params_changed['BLACK_THRESH'] = values["BLACK_THRESH"]
                                                    params_changed['EXP_TIME'] = values["EXP_TIME"]
                                               ##     params_changed['PACKET_SIZE'] = values["PACKET_SIZE"]
                                                    params_changed['PACKET_SIZE'] = str(packet_size)
                                            ##        params_changed['INTER_PACKET_DELAY'] = values["INTER_PACKET_DELAY"]
                                                    params_changed['INTER_PACKET_DELAY'] = str(inter_paket_delay)
                                                    params_changed['FRAME_RATE'] = values["FRAME_RATE"]
                                               ##     params_changed['BANDWIDTH_RESV_ACC'] = values["BANDWIDTH_RESV_ACC"]
                                                    params_changed['BANDWIDTH_RESV_ACC'] = str(bw_resv_acc)
                                               ##     params_changed['BANDWIDTH_RESV'] = values["BANDWIDTH_RESV"]
                                                    params_changed['BANDWIDTH_RESV'] = str(bw_resv)
                                                    params_changed['FIRST'] = str(width)
                                                    params_changed['SEC'] = str(height) 
                                                else:
                                                    params_changed['GAIN'] = "50"
                                                    params_changed['BLACK_THRESH'] = "2"
                                                    params_changed['EXP_TIME'] = "140"
                                               ##     params_changed['PACKET_SIZE'] = values["PACKET_SIZE"]
                                                    params_changed['PACKET_SIZE'] = "1500"
                                            ##        params_changed['INTER_PACKET_DELAY'] = values["INTER_PACKET_DELAY"]
                                                    params_changed['INTER_PACKET_DELAY'] = "5000"
                                                    params_changed['FRAME_RATE'] = "50"
                                               ##     params_changed['BANDWIDTH_RESV_ACC'] = values["BANDWIDTH_RESV_ACC"]
                                                    params_changed['BANDWIDTH_RESV_ACC'] = "4"
                                               ##     params_changed['BANDWIDTH_RESV'] = values["BANDWIDTH_RESV"]
                                                    params_changed['BANDWIDTH_RESV'] = "10"
                                                    params_changed['FIRST'] = "240"
                                                    params_changed['SEC'] = "160"
                                                 
                                                ###################################################################################
                                                ########## Function to order all the parameters inside params_changed dictionary ##
                                                
                                                params_changed = order_all_params_pfs_adv(params_changed) 
                                                
                                                print(" ------ All parameters:") 
                                            
                                                counter_missing = 0
                                                
                                                for ind_param, param in enumerate(params_changed):
                                                    if len(param) == 2:
                                                        print(str(param[0]) + ": " + str( param[1]))
                                                   # if len(param) != 2:
                                                   #      print("Missing parameter " + str(ind_param))
                                                   #      counter_missing += 1
                                                        
                                              #  print(str(counter_missing) + " missing parameters")                                         
                                                 
                                                ###################################################################################
                                                ###################################################################################
                                                
                                                pfs_ok, pfs_struct = write_to_pfs_adv_opt(params_changed)   
                                                
                                                windowx = sg_py.Window('Choose directory folder', [[sg_py.Text('Folder name')], [sg_py.Input(), sg_py.FolderBrowse()], [sg_py.OK(), sg_py.Cancel()] ]).read(close=True)
                                                (keyword, dict_dir) = windowx                
                                           
                                                dir_path = dict_dir['Browse'] 
                                                
                                                pfs_filename = "parameters_" + getDateTimeStrMarker() + "_full" + ".pfs"
                                                 
                                                dir_pfs_gen = dir_path + pfs_filename  
                                                
                                                pfs_gen = False
                                                 
                                                counter_tries_writing_to_pfs_file_adv = 0
                                                
                                                ##################################################################################
                                                
                                                list_keys = []
                                                list_values = []
                                                
                                                for key, value in only_adv_opt.items():
                                                    list_keys.append(key)
                                                    list_values.append(value)                                       
                                                    
                                                
                                                ##################################################################################
                                                
                                                while pfs_gen == False:
                                                    
                                                    print("Trying to write to full PFS file, for " + str(counter_tries_writing_to_pfs_file_adv+1) + " th time ...")
                                                 
                                                
                                                    if ".pfs" in dir_pfs_gen: 
                                                    
                                                        with open(dir_pfs_gen, "w") as input: 
                                                            
                                                            for ind_line, line in enumerate(pfs_struct):
                                                                input.write(line + "\n") 
                                                                
                                                                if ind_line == len(pfs_struct)-1:
                                                                    pfs_gen = True
                                                                    print("Success !!! PFS file generated with advanced options") 
                                                    else:
                                                        print("Check if filename has one of the following extensions: \n {.pfs}")
                                                        
                                                    counter_tries_writing_to_pfs_file_adv += 1 
                                                    
                                                    
                                                #############################################
                                                #############################################  
                                                
                                                print("Here")
                                            
                                                ## Another python file with also the remaining properties 
                                                
                                                # process = multiprocessing.Process(target=acq_task, args=(countx, eventt))
                                                
                                                # process.terminate()
                                                
                                                # eventt.set()
                                                
                                    #            t.join()      
                                    
                                    
                                                while True:
                                                    control_inf = exp_control()
                                                    
                                                    if control_inf[0] == True:
                                                        
                                                        numberTests = control_inf[1]
                                                        gap_bet_tests = control_inf[2]
                                                        
                                                        break
                                                    else: 
                                                        continue
                                                
                                                print("Number of tests: " + str(numberTests))
                                                    
                                                if numberTests > 1: 
                                                    print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
                                                else:
                                                    print("Going to perform just 1 test")
                                                    
                                                if True:
                                                    
                                                       
                                    
                                                    for curTest in range(0,numberTests): 
                                                        if not emptyVals:    
                                                        
                                                #            curTest = 0
                                             #               print("Here on test " + str(curTest) + " ...")                           
                                                        
                                                            data_to_save = whole_processing_software_adv(int(values['FRAME_RATE']), 
                                                                                      packet_size, 
                                                                                      inter_paket_delay, 
                                                                                      bw_resv, 
                                                                                      bw_resv_acc, 
                                                                                      int(values['GAIN']), 
                                                                                      int(values['EXP_TIME']), 
                                                                                      int(values['recTime']),
                                                                                      width,
                                                                                      height,
                                                                                      int(values['decLevel']),
                                                                                      dir_path,
                                                                                      curTest,
                                                                                      numberTests,
                                                                                      gap_bet_tests,
                                                                                      list_values,
                                                                                      list_keys)  
                                                        else:
                                                        
                                                     #   for curTest in range(0,numberTests): 
                                               
                                                 #           curTest = 0
                                                             
                                       #                     print("Here on test " + str(curTest) + " ...")    
                                                            
                                                            data_to_save = whole_processing_software_adv(50, 
                                                                                      1500, 
                                                                                      5000, 
                                                                                      4, 
                                                                                      10, 
                                                                                      50, 
                                                                                      140, 
                                                                                      5,
                                                                                      width,
                                                                                      height,
                                                                                      0,
                                                                                      dir_path,
                                                                                      curTest,
                                                                                      numberTests,
                                                                                      gap_bet_tests,
                                                                                      list_values,
                                                                                      list_keys)  
                                                      
                                                       
                                                        values = [values, dir_path]                     
                                                        print("Choosen directory: " + dir_path)  
                        
                                                        if len(data_to_save) > 0: 
                            
                                                            for ind_data, data in enumerate(data_to_save):
                                                            
                                                                clusteringRes, execTime, numberImg = data       
                                                                
                                                                gui_show_results(clusteringRes, execTime, numberImg, ind_data, data_to_save)
                                                                
                                                                print(" -- Showing GUI with results for test number " + str(ind_data+1))
                                                                
                                                                time.sleep(5)          
                                                            
                                                        else:
                                                            print("Output results not available !!!")                  
                            
                            
                                                        ###################
                            
                                                        executionTime = (time.time() - startTime)
                                                        print('Whole execution time in seconds: ' + str(executionTime))
                                                        
                                                        adv_go = False
                                                 
                                                ####################################################################### 
                                                ####################################################################### 
                                                #######################################################################   
                                                ####################################################################### 
                                                ####################################################################### 
                                                #######################################################################                              
                                                ####################################################################### 
                                                ####################################################################### 
                                                #######################################################################
                                            
                                            else:
                                            
                                                window['Decisor Level:'].update(value = values['decLevel'])                   
                                                window['-SECONDS-'].update(value = values['recTime'])
                                                
                                                time.sleep(2)
                                                 
                                                # if len(values["GAIN"]) == 0:
                                                #     sg.popup_error('Gain - Empty field')
                                                #     valid = False
                                                # else:
                                                    
                                                if not emptyVals:
                                                    if int(values["GAIN"]) < 0 or int(values["GAIN"]) > 50:
                                                        sg.popup_error('Gain must be within [0,50] interval') 
                                                        valid = False 
                                                        print("Failed to gain range") 
                                                        
                                                # if len(values["BLACK_THRESH"]) == 0:
                                                #     sg.popup_error('Black Level - Empty field') 
                                                #     valid = False
                                                # else:
                                                
                                                if not emptyVals:
                                                    if int(values["BLACK_THRESH"]) < 0 or int(values["BLACK_THRESH"]) > 255:
                                                        sg.popup_error('Black level must be within [0,255] interval')    ## for Mono8 configuration 
                                                        valid = False
                                                        print("Failed to black level range")
                                                
                                                # if len(values["EXP_TIME"]) == 0:
                                                #     sg.popup_error('Exposure Time - Empty field')  
                                                #     valid = False
                                                # else: 
                                                    
                                                if not emptyVals:
                                                    if int(values["EXP_TIME"]) < 0 or int(values["EXP_TIME"]) > 35000: 
                                                        sg.popup_error('Exposition time must be within [0,35000] interval')  
                                                        valid = False
                                                        print("Failed to exposure time range")
                                                
                                                # if len(values["PACKET_SIZE"]) == 0:
                                                #     sg.popup_error('Packet Size - Empty field')  
                                                #     valid = False 
                                                # else:
                                                #     if int(values["PACKET_SIZE"]) <= 0:
                                                #         sg.popup_error('Packet size must be positive')
                                                #         valid = False
                                                #         print("Failed to packet size range")
                                                
                                                # if len(values["INTER_PACKET_DELAY"]) == 0:
                                                #     sg.popup_error('Inter-Packet Delay - Empty field')  
                                                #     valid = False 
                                                # else:    
                                                #     if int(values["INTER_PACKET_DELAY"]) < 0 or int(values["INTER_PACKET_DELAY"]) > 10000:
                                                #         sg.popup_error('Inter-packet delay must be within [0,10000] interval') 
                                                #         valid = False  
                                                #         print("Failed to inter-packet delay range") 
                                                        
                                                # if len(values["FRAME_RATE"]) == 0:
                                                #     sg.popup_error('Frame Rate - Empty field')  
                                                #     valid = False 
                                                # else:
                                                
                                                if not emptyVals:
                                                    if int(values["FRAME_RATE"]) < 10 or int(values["FRAME_RATE"]) > 50: 
                                                        sg.popup_error('Frame rate (fps) must be within [10,50] interval') 
                                                        valid = False
                                                        print("Failed to frame rate range")
                                                    
                                                # if len(values["BANDWIDTH_RESV_ACC"]) == 0:
                                                #     sg.popup_error('Bandwidth Reserve Accumulation - Empty field')  
                                                #     valid = False
                                                # else:
                                                #     if int(values["BANDWIDTH_RESV_ACC"]) < 0 or int(values["BANDWIDTH_RESV_ACC"]) > 10:
                                                #         sg.popup_error('Bandwidth reserve accumulation must be within [0,100] interval')  
                                                #         valid = False
                                                #         print("Failed to bandwidth reserve accumalation range")
                                                        
                                                # if len(values["BANDWIDTH_RESV"]) == 0:
                                                #     sg.popup_error('Bandwidth Reserve - Empty field')  
                                                #     valid = False
                                                # else:                     
                                                #     if int(values["BANDWIDTH_RESV"]) < 0 or int(values["BANDWIDTH_RESV"]) > 100:
                                                #         sg.popup_error('Bandwidth reserve must be within [0,10] interval')   
                                                #         valid = False
                                                #         print("Failed to bandwidth reserve range") 
                                                        
                                                width = 1920
                                                height = 1080
                                                        
                                                if 'Full HD' in values['RES_TYPE']:
                                                     width = 1920
                                                     height = 1080
                                                elif 'HD+' in values['RES_TYPE']:
                                                     width = 1600
                                                     height = 900
                                                elif 'HD' in values['RES_TYPE']:
                                                     width = 1280
                                                     height = 720
                                                elif 'qHD' in values['RES_TYPE']:
                                                     width = 960
                                                     height = 540
                                                elif 'nHD' in values['RES_TYPE']:
                                                     width = 640
                                                     height = 360
                                                elif '960H' in values['RES_TYPE']:
                                                     width = 960
                                                     height = 480
                                                elif 'HVGA' in values['RES_TYPE']:
                                                     width = 480
                                                     height = 320 
                                                elif 'VGA' in values['RES_TYPE']:
                                                     width = 640
                                                     height = 480
                                                elif 'SVGA' in values['RES_TYPE']:
                                                     width = 800
                                                     height = 600
                                                elif 'DVGA' in values['RES_TYPE']:
                                                     width = 960
                                                     height = 640
                                                elif 'QVGA' in values['RES_TYPE']:
                                                     width = 320
                                                     height = 240
                                                elif 'QQVGA' in values['RES_TYPE']:
                                                     width = 160
                                                     height = 120
                                                elif 'HQVGA' in values['RES_TYPE']:
                                                     width = 240
                                                     height = 160
                                                
                                                a = width
                                                b = height                                    
                                                width = b
                                                height = a  
                                                       
                                                    
                                                if valid == True:  
                                                    print("Valid parameters")                
                                                    print("Next step")
                                                    
                                                    windowx = sg_py.Window('Choose directory folder', [[sg_py.Text('Folder name')], [sg_py.Input(), sg_py.FolderBrowse()], [sg_py.OK(), sg_py.Cancel()] ]).read(close=True)
                                                    (keyword, dict_dir) = windowx                
                                              
                                                    dir_path = dict_dir['Browse']               
                                                    
                                                    if len(dir_path) != 0:                       
                                                        repeat = False                      
                                                            
                                                        # print("Resulting dict: ")
                                                        # for value in values:  
                                                        #    print(value + " : " + values[value])   
                                                           
                                                    ##    setParametersToPypylon(values)   
                                                    
                                                        pfs_filename = "parameters_" + getDateTimeStrMarker() + ".pfs"
                                                        
                                                        dir_pfs_gen = dir_path + pfs_filename 
                                                        
                                                        pfs_gen = False
                                                        
                                                        counter_tries_writing_to_pfs_file = 0
                                                        
                                                        while pfs_gen == False:
                                                            
                                                            print("Trying to write to PFS file, for " + str(counter_tries_writing_to_pfs_file+1) + " th time ...")
                                                            
                                                            if emptyVals:
                                                                
                                                                pfs_gen =  save_to_pfs(dir_pfs_gen, 50, 2, 
                                                                             140, 50, packet_size, 
                                                                             inter_paket_delay, bw_resv, bw_resv_acc,
                                                                             width, height, False)
                                                            else:
                                                                                      
                                                        
                                                                pfs_gen =  save_to_pfs(dir_pfs_gen, int(values['GAIN']), int(values["BLACK_THRESH"]), 
                                                                             int(values['EXP_TIME']), int(values['FRAME_RATE']), packet_size, 
                                                                             inter_paket_delay, bw_resv, bw_resv_acc,
                                                                             width, height, True)
                                                            
                                                            counter_tries_writing_to_pfs_file += 1  
                                                        
                                                        print("Here")
                                                         
                                              ##          process = multiprocessing.Process(target=acq_task, args=(countx, eventt))
                                                        
                                                ##        cv2.destroyAllWindows()
                                                         
                                          ##              eventt.set()
                                                        
                                       #                 t.join()                                           
                                                       
                                                        
                                                ##        acq_image_camera(countx, False)
                                                
                                                        # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                                                        # converter = pylon.ImageFormatConverter() 
                                                        
                                                        # # grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                                                        # # grabResult.Release()
                                                        
                                                        # while camera.NumReadyBuffers.GetValue() > 0:
                                                        #        camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
                
                                                  ##      camera.Close()
                                                        
                                                  ##      time.sleep(5)
                                                  
                                              ##          time.sleep(20) 
            
                                                        # process = multiprocessing.current_process()        
                                                        # process.terminate()                                    
                                                        
                                                        # create the shared queue                                            
                                                        
                                             ##           process_but = Thread(target=button_task, args=(queue, True, True))
                                                ##        process_but.start()
                                                
                                                
                                                        
                                            ###            process = Thread(target=acq_task, args=(queue, countx, eventt))                                            
            
                                                        # process = multiprocessing.Process(target=acq_task, args=(queue, countx, eventt))
                                                        
                                                        # process_but =  multiprocessing.Process(target=button_task, args=(queue, True, True))
                                                        
                                                        # if resp == True:  
                                                        #     process.kill() 
             
                                                        # if process.is_alive():
                                                        #      print("Keep waiting ...")
                                                        # else:
                                                        #      print("Proceeding ...") 
                                                        #      break                                                
                                                            
                                                        
                                                        # while True:
                                                            
                                                        #     thread = CustomThread()
                                                        #     # start the thread  
                                                        #     thread.start()
                                                            
                                                        #     # wait for the thread to finish
                                                        # ##    thread.join()
                                                            
                                                        #     # get the value returned from the thread
                                                        #     data = thread.value                                              
                                                              
                                                        #     resp = data
                                                        
                                                        # print("Respx: " + str(respx))
                                                            
                                                        # if respx:
                                                        #      process_but.kill()
                                                        #      process.kill()
                                                            
                                                        # if process.is_alive():
                                                        #    print("Keep waiting ...")
                                                        # else:
                                                        #    print("Proceeding ...") 
                                                        #    break
                                                            
                                                            
                                              
                                                         
                                                        
                                                        # ok_list = get_ok()
                                                        
                                                        # if len(ok_list) > 0:
                                                        
                                                        #     print("Len Ok: (" + str(len(ok_list)) + " , " + str(len(ok_list[0])))
                                                        #     print(ok_list) 
                                                        
                                                        basler_check = True ## False
                                                        
                                                        # while not basler_check:                                            
                                                        #     basler_check, model_name = confirm_basler()
                                                        #     time.sleep(2)
                                                            
                                                        
                                                        while True:
                                                            control_inf = exp_control()
                                                            
                                                            if control_inf[0] == True:
                                                                
                                                                numberTests = control_inf[1]
                                                                gap_bet_tests = control_inf[2]
                                                                
                                                                break
                                                            else:
                                                                continue
                                                        
                                                        print("Number of tests: " + str(numberTests))
                                                            
                                                        if numberTests > 1: 
                                                            print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
                                                        else:
                                                            print("Going to perform just 1 test")
                                                            
                                                        curTest = 0
                                                        
                                                 ##       for curTest in range(0,numberTests): 
                                                           
                                                        if True:
                                                            if emptyVals:
                                                                
                                                                data_to_save = whole_processing_software(50, 
                                                                                          packet_size, 
                                                                                          inter_paket_delay, 
                                                                                          bw_resv, 
                                                                                          bw_resv_acc, 
                                                                                          50, 
                                                                                          140, 
                                                                                          5,
                                                                                          width,
                                                                                          height,
                                                                                          0,
                                                                                          dir_path,
                                                                                          curTest,
                                                                                          numberTests,
                                                                                          gap_bet_tests) 
                                                            else:
                                                        
                                                                data_to_save = whole_processing_software(int(values['FRAME_RATE']), 
                                                                                          packet_size, 
                                                                                          inter_paket_delay, 
                                                                                          bw_resv, 
                                                                                          bw_resv_acc, 
                                                                                          int(values['GAIN']), 
                                                                                          int(values['EXP_TIME']), 
                                                                                          int(values['recTime']),
                                                                                          width,
                                                                                          height, 
                                                                                          int(values['decLevel']),
                                                                                          dir_path,
                                                                                          curTest,
                                                                                          numberTests,
                                                                                          gap_bet_tests)   
                                                        
                                                        print(" ---- OK")
                                                        print(" ---- OK") 
                                                        print(" ---- OK")
                                                        print(" ---- OK")
                                                        print(" ---- OK") 
                                                           
                                                        values = [values, dir_path]                     
                                                        print("Choosen directory: " + dir_path)  
                            
                                                        if len(data_to_save) > 0: 
                            
                                                            for ind_data, data in enumerate(data_to_save):
                                                            
                                                                clusteringRes, execTime, numberImg = data       
                                                                
                                                                gui_show_results(clusteringRes, execTime, numberImg, ind_data, data_to_save) 
                                                                
                                                                print(" -- Showing GUI with results for test number " + str(ind_data+1))
                                                                
                                                                time.sleep(5)           
                                                                
                                                        else:
                                                            print("Output results not available !!!")                  
                            
                            
                                                        ###################
                            
                                                        executionTime = (time.time() - startTime)
                                                        print('Whole execution time in seconds: ' + str(executionTime))
                            
                                                                     
                                                        
                                                        break 
                                                
                                                    else:
                                                        sg.popup_error('Folder directory required !') 
                                                        repeat = True
                                                        del values, event  
                                                    
                                          ##      setParametersToPypylon(values)   
                                          ## or just call the whole python file related to image processing and so on               
                                                else:
                                                    repeat = True 
                                                    valid = True
                                                    print("Try again")
                                                    del values, event    
                                        else:
                                            if event == 'Clear':
                                                
                                     #           numberTests -= 1
                                                
                                                window["GAIN"].update(value = "50") 
                                                window["BLACK_THRESH"].update(value = "2") 
                                                window["EXP_TIME"].update(value = "140")                                                                                            
                                                window["RES_TYPE"].update(value = "qHD")
                                                
                                                
                                                repeat = True 
                                                valid = True
                                                
                                            elif event == 'Delete':                        
                                           ##     numberTests -= 1
                                           
                                               window["GAIN"].update(value = "") 
                                               window["BLACK_THRESH"].update(value = "") 
                                               window["EXP_TIME"].update(value = "")                                                                                             
                                               window["RES_TYPE"].update(value = "")
               #                                 break                          
                                
                        window.close()  
                    else:
                        
                        repeat = True  
                        
                        while True:
                            control_inf = exp_control()
                            
                            if control_inf[0] == True:
                                
                                numberTests = control_inf[1]
                                gap_bet_tests = control_inf[2]
                                
                                break
                            else:
                                continue
                        
                        print("Number of tests: " + str(numberTests))
                            
                        if numberTests > 1: 
                            print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
                        else:
                            print("Going to perform just 1 test")
                        
                        print("numTests init: " + str(numberTests))
                        
                        enab = optional_prop()
                         
                        enab_val_pfs = enab[0]
                        enab_timestamps = enab[1]
                        
                        read_pfs_file(gap_bet_tests, numberTests, repeat, startTime, enab_val_pfs, enab_timestamps)
                         
                else: 
                       print("Executing again ...")
                       resp_demo = demo_gui_menu()
                       here_demo = True
                       count_overall += 1
        
        resp_again = 2
        
        while resp_again == 2:
            from process_pfs_only import post_proc_pfs_only 
            resp_again = repeat_loop_proc()
     
            if resp_again == 2:
        
                resTypes = ['Full HD',
                            'HD+',
                            'HD',
                            'qHD',
                            'nHD',
                            '960H',
                            'HVGA',
                            'VGA',
                            'SVGA',
                            'DVGA',
                            'QVGA',
                            'QQVGA',
                            'HQVGA'
                           ]
        
                res_dims = ['(1920x1080)',
                            '(1600x900)',
                            '(1280x720)',
                            '(960x540)',
                            '(640x360)',
                            '(960x480)',
                            '(480x320)',
                            '(640x480)',
                            '(800x600)',
                            '(960x640)',
                            '(320x240)',
                            '(160x120)',
                            '(240x160)'
                          ]   
        
                res_fullTypes = []
        
                for ind_r, r in enumerate(resTypes):
                    r += ' ' + res_dims[ind_r]
                    res_fullTypes.append(r)
        
        
                basler_check = False
        
                basler_check, model_name = confirm_basler() 
        
        
                if basler_check == True:
            
                    videoRec = gui_video_rec()  
                      
                    if videoRec == True:
                    
                    
                        ok_resp = load_core_packages()
                        
                        while ok_resp == False:
                            print("Could not load required packages. Check for errors or updates !!!")    
                            ok_resp = load_core_packages()        
                        
                        
                            
                        
                        ## Show camera live
                        
                        def pop_up_pfs():
                            import PySimpleGUI as sg
                            
                  ##          pfs_resp = sg.popup_get_text("Do you want to load a PFS file ?")
                            
                            repInit = True
                            
                            pfs_resp = -1
                            
                            layout = [
                                [sg.Text("Do you want to load a PFS file ?")],      ## Use a pre-recorded video    ## Perform a new image acquisition process
                                [sg.T("         "), sg.Checkbox('Yes', default=False, key="-IN1-")],
                                [sg.T("         "), sg.Checkbox('No', default=True, key="-IN2-")],
                                [sg.Button("Next")]  
                            ]           
                             
                            window = sg.Window('PFS File loading', layout)              
                              
                            while repInit:
                      
                               event, values = window.read()
                              
                               
                               if event == "Exit" or event == sg.WIN_CLOSED:
                                   break 
                               
                               if event == "Next":
                                   if values["-IN1-"] == True and values["-IN2-"] == False:
                                       pfs_resp = 1
                                       repInit = False
                                       break
                                   elif values["-IN2-"] == True and values["-IN1-"] == False:
                                       pfs_resp = 0
                                       repInit = False
                                       break
                                   else:
                                       if values["-IN1-"] == True and values["-IN2-"] == True:
                                           print("Only pick one option !!!")                   
                                           repInit = True           
                                           continue
                               
                            window.close()        
                            
                            return pfs_resp
                        
                        resp_pfs_opt = pop_up_pfs()  
                        
                 #        resp_pfs_opt = pop_up_pfs() 
                            
                 # ##       resp_pfs_opt = input("Do you want to load a PFS file ? ")
                        
                 #        while not(('S' in resp_pfs_opt) or ('Y' in resp_pfs_opt) or ('s' in resp_pfs_opt) or ('y' in resp_pfs_opt) or ('N' in resp_pfs_opt) or ('n' in resp_pfs_opt)):
                 #      ##      resp_pfs_opt = input("Do you want to load a PFS file ? {'S'; 's'; 'Y'; 'y'; 'N'; 'n'}")
                 #              resp_pfs_opt = pop_up_pfs()
                            
                   #     if ('N' in resp_pfs_opt) or ('n' in resp_pfs_opt):
                       
                        if resp_pfs_opt == 0:
                            
                            packet_size = 1500
                            inter_paket_delay = 5000
                            bw_resv_acc = 4
                            bw_resv = 10
                            
                            if True:
                         
                      
                                
                                repeat = True
                                
                                col2 = sg.Column([       
                                     
                                        # Information frame
                                [sg.Frame(layout=[[sg.Text('Gain:')],  
                                                          [sg.Input("50", size=(19, 1), key="GAIN")],
                                                          [sg.Text('Black Level:')],
                                                          [sg.Input("2", size=(19, 1), key="BLACK_THRESH")],    ## , sg.Button('Copy')
                                                          [sg.Text('Image Resolution:')],
                                                          [sg.InputCombo(res_fullTypes, size = (19, 1), key="RES_TYPE", change_submits=False)],
                                                          # [sg.Text('Image Height:')],
                                                          # [sg.InputCombo(itemHeight, size=(19, 1), key="FIRST", change_submits=False) ],
                                                          # [sg.Text('Image Width:')],
                                                          # [sg.InputCombo(itemHeight, size=(19, 1), key="SEC", change_submits=False)],
                                                          [sg.Text('Exposition Time:')], 
                                                          [sg.Input(default_text= "140", size=(19, 1), key="EXP_TIME")],     ## , sg.Button('Copy') 
                                                          [sg.Text('Recording Time:')],
                                                          [sg.Slider(default_value = 5, orientation ='horizontal', key='recTime', range=(1,10800)),  ## 100
                                                               sg.Text(size=(5,2), key='-SECONDS-'), sg.Text(" seconds")]
                                                          ], title='Information:')],])
                                        
                                col3 = sg.Column([     
                                     
                                        # Information frame
                                        [sg.Frame(layout=[
                                                          # [sg.Text('Packet Size:')], 
                                                          # [sg.Input(default_text= "1500", size=(19, 1), key="PACKET_SIZE")],     ## , sg.Button('Copy')
                                                          # [sg.Text('Inter-Packet Delay:')],
                                                          # [sg.Input(default_text= "5000", size=(19, 1), key="INTER_PACKET_DELAY")],
                                                          [sg.Text('Frame rate:')],
                                                          [sg.Input(default_text= "50", size=(19, 1), key="FRAME_RATE")],
                                                          [sg.T("         "), sg.Checkbox('Enable basic streaming panel', default=False, key="-STREAM_PANEL-")],
                                                          
                                                          [sg.Button("Ok")]
                                                          # [sg.Text('Bandwidth Reserve Accumulation:')],
                                                          # [sg.Input(default_text= "4", size=(19, 1), key="BANDWIDTH_RESV_ACC")],
                                                          # [sg.Text('Bandwidth Reserve:')],
                                                          # [sg.Input(default_text= "10", size=(19, 1), key="BANDWIDTH_RESV")]                     
                                                          ], title='Information 2:')],])
                                 
                                col4 = sg.Column([ 
                                    
                                    [sg.Frame(layout=[[sg.Text('Decisor Level:')], 
                                                      [sg.Slider(default_value = 0, orientation ='horizontal', key='decLevel', range=(0,50))],
                                                      [sg.Text('', size=(5,2), key='Decisor Level:')],
                                                      [sg.Image('255_to_0' + '.png', size = (30,1))],
                                                      [sg.Text('Software Light Level')]
                                       
                                ], title='Advanced Parameters:')],])
                                
                                layout = [ [col2, col3, col4],     
                                        # Actions Frame 
                                        [sg.Frame(layout=[[sg.Button("Preview ...")],[sg.Button('Save'), sg.Button('Clear'), sg.Button('Delete')],
                                                          [sg.Button('Advanced Properties ...'), sg.Button('Extra Properties ...'),  sg.Button('Calibrate camera ...')]                 
                                            ], title='Actions:')]]       
                                window = sg.Window('GUI', layout, web_port=2219, disable_close=True, resizable = True, finalize = True, margins=(0,0))  
                                    
                                valid = True  
                                setFolder = False   
                                adv_go = False
                                
                                params_changed = {}
                                only_adv_opt = {}
                                
                                code_ret = -1
                               
                                while repeat == True:             # Event Loop
                                        event, values = window.read()
                                        print(event, values)        
                                      
                                         
                                        if event == "Exit" or event == sg.WIN_CLOSED or code_ret == 0:
                                            break
                                        
                                        if event == 'Preview ...':
                                            
                                            gain = 50
                                            black_thresh = 2
                                            esp_time = 140    ## us
                                            
                                            format_image = values['RES_TYPE']
                                            frame_rate = 50                             
                                            
                                            if len(values["GAIN"]) == 0:
                                                print("Setting default gain = 50 ...")
                                            else:
                                                if int(values["GAIN"]) < 0 or int(values["GAIN"]) > 50:
                                                    sg.popup_error('Gain must be within [0,50] interval') 
                                                    valid = False 
                                                    print("Failed to gain range")
                                                else:
                                                    gain = int(values["GAIN"])
                                                    
                                            if len(values["BLACK_THRESH"]) == 0:
                                                print("Setting default black threshold ...")
                                            else:
                                                if int(values["BLACK_THRESH"]) < 0 or int(values["BLACK_THRESH"]) > 255:
                                                    sg.popup_error('Black level must be within [0,255] interval')    ## for Mono8 configuration 
                                                    valid = False
                                                    print("Failed to black level range")
                                                else:
                                                    black_thresh = int(values["BLACK_THRESH"])
                                            
                                            if len(values["EXP_TIME"]) == 0:
                                                print("Setting default exposure time ...")
                                            else:                    
                                                if int(values["EXP_TIME"]) < 0 or int(values["EXP_TIME"]) > 35000: 
                                                    sg.popup_error('Exposition time must be within [0,35000] interval')  
                                                    valid = False
                                                    print("Failed to exposure time range") 
                                                else:
                                                    exp_time = int(values["EXP_TIME"])
                                                    
                                            if len(values["FRAME_RATE"]) == 0:
                                                 print("Setting default frame rate ...")
                                            else:       
                                                 if int(values["FRAME_RATE"]) < 10 or int(values["FRAME_RATE"]) > 50: 
                                                     sg.popup_error('Frame rate (fps) must be within [10,50] interval') 
                                                     valid = False
                                                     print("Failed to frame rate range")
                                                 else:
                                                     frame_rate = int(values["FRAME_RATE"])
                                                     
                                                     if frame_rate == 50:
                                                       
                                                         if format_image == resTypes[0]:
                                                            format_image = resTypes[1] 
                                                            print("Changing resolution from " + resTypes[0] + " to " + resTypes[1] + " due to the maximum frame rate limitations ...")
                                                        
                                            
                                            params_stream_basic = [gain, black_thresh, format_image, exp_time, packet_size, inter_paket_delay, bw_resv_acc, bw_resv, frame_rate]
                                            
                                            get_test_image(params_stream_basic)
                                     
                                            # from init_position_settings import show_images_and_button
                                            # show_images_and_button(params_stream_basic)
                                        
                                        if event == 'Advanced Properties ...':                         
                                                 
                                                adv_go = True  
                                          
                                                params_changed = adv_params_gui()      
                                                
                                                
                                                ####################################################################### 
                                                ####################################################################### 
                                                #######################################################################     
                                        if event == 'Extra Properties ...':
                                            
                                            from pypylon import pylon
                                            
                                            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                                            converter = pylon.ImageFormatConverter() 
                                            
                                            camera.Open()
                                            
                                            resp_param = 0
                                            
                                            while resp_param == 0: 
                                            
                                                resp_param = extra_params_gui(camera)
                                                break
                                         
                                        
                                        if event == 'Calibrate camera ...': 
                                            code_ret = calib_camera(False, model_name)
                                            
                                            if code_ret == 1: 
                                                break 
                                             
                                                 
                                                                        
                                        
                                        if event == 'decisorLevel':           
                                            
                                            windowDecisorLevelSlider = sg_py.Window('', [sg_py.Text("", size=(0, 1), key='OUTPUTX')]).read(close=True)
                                            
                                            windowDecisorLevelSlider.bind('<FocusOut>', '+FOCUS OUT+')
                                
                                            windowDecisorLevelSlider['-BUTTON-'].bind('<Button-3>', '+RIGHT CLICK+')
                                            windowDecisorLevelSlider['-TEXT-'].bind('<Enter>', '+MOUSE OVER+')
                                            windowDecisorLevelSlider['-TEXT-'].bind('<Leave>', '+MOUSE AWAY+')
                                            windowDecisorLevelSlider['-IN-'].bind('<FocusIn>', '+INPUT FOCUS+') 
                                            
                                            eventDecLevel, valSliderDecLevel = windowDecisorLevelSlider.read()
                                            if eventDecLevel == sg_py.WINDOW_CLOSED:
                                                break 
                                            else: 
                                                windowDecisorLevelSlider['OUTPUTX'].update(value=values['decLevel'])
                                            
                                            time.sleep(2)
                                             
                                            windowDecisorLevelSlider.close()
                                            
                                        
                                        if event == 'recTime': 
                                            
                                            print("Event up to slider triggered !!!")
                                            window_sliderVal = sg_py.Window('', [sg_py.Text("", size=(0, 1), key='OUTPUT')]).read(close=True)
                                            
                                            window_sliderVal.bind('<FocusOut>', '+FOCUS OUT+')
                                
                                            window_sliderVal['-BUTTON-'].bind('<Button-3>', '+RIGHT CLICK+')
                                            window_sliderVal['-TEXT-'].bind('<Enter>', '+MOUSE OVER+')
                                            window_sliderVal['-TEXT-'].bind('<Leave>', '+MOUSE AWAY+')
                                            window_sliderVal['-IN-'].bind('<FocusIn>', '+INPUT FOCUS+')
                                            
                                            eventSlider, valSlider = window_sliderVal.read()
                                            if eventSlider == sg_py.WINDOW_CLOSED:
                                                break
                                            else:
                                                window_sliderVal['OUTPUT'].update(value=values['recTime'])          
                                                
                                       
                                            time.sleep(2) 
                                             
                                            window_sliderVal.close()
                                        
                                        else:    
                                            
                                            if event == "Ok":
                                                ok_button = True
                                                
                                                if values['-STREAM_PANEL-'] == True:
                                                    packet_size, inter_packet_delay, bw_resv_acc, bw_resv = get_basic_stream_params()
                                                    
                                                
                                                
                                            if event == 'Save':
                                                
                                             ##   if values['-STREAM_PANEL-'] == True: 
                                                 
                                                emptyVals = False
                                                
                                                if len(values["GAIN"]) == 0:
                                                    window["GAIN"].update("50")
                                                    emptyVals = True
                                                if len(values["BLACK_THRESH"]) == 0:
                                                    window["BLACK_THRESH"].update("2")
                                                    emptyVals = True
                                                if len(values["EXP_TIME"]) == 0:
                                                    window["EXP_TIME"].update("140")
                                                    emptyVals = True
                                                if len(values["FRAME_RATE"]) == 0:
                                                    window["FRAME_RATE"].update("50")
                                                    emptyVals = True
                                                 
                                                if emptyVals:                                    
                                                    time.sleep(5)                                    
                                                
                                                if adv_go == True:
                                                    
                                                    window['Decisor Level:'].update(value = values['decLevel'])                   
                                                    window['-SECONDS-'].update(value = values['recTime'])
                                                    
                                                    time.sleep(2) 
                                                     
                                                    only_adv_opt = params_changed.copy()
                                                    
                                                                                         
                                                    # if len(values["GAIN"]) == 0:
                                                    #     sg.popup_error('Gain - Empty field') 
                                                    #     valid = False
                                                #    else:
                                                    
                                                    if not emptyVals:
                                                        if int(values["GAIN"]) < 0 or int(values["GAIN"]) > 50:
                                                            sg.popup_error('Gain must be within [0,50] interval') 
                                                            valid = False 
                                                            print("Failed to gain range") 
                                                            
                                                    # if len(values["BLACK_THRESH"]) == 0:
                                                    #     sg.popup_error('Black Level - Empty field') 
                                                    #     valid = False
                                                    # else:
                                                    if not emptyVals:
                                                        if int(values["BLACK_THRESH"]) < 0 or int(values["BLACK_THRESH"]) > 255:
                                                            sg.popup_error('Black level must be within [0,255] interval')    ## for Mono8 configuration 
                                                            valid = False
                                                            print("Failed to black level range")
                                                    
                                                    # if len(values["EXP_TIME"]) == 0:
                                                    #     sg.popup_error('Exposure Time - Empty field')  
                                                    #     valid = False
                                                    # else:                                        
                                                    if not emptyVals:
                                                        if int(values["EXP_TIME"]) < 0 or int(values["EXP_TIME"]) > 35000: 
                                                            sg.popup_error('Exposition time must be within [0,35000] interval')  
                                                            valid = False
                                                            print("Failed to exposure time range")
                                                    
                                                    # if len(values["PACKET_SIZE"]) == 0:
                                                    #     sg.popup_error('Packet Size - Empty field')  
                                                    #     valid = False 
                                                    # else:
                                                    #     if int(values["PACKET_SIZE"]) <= 0:
                                                    #         sg.popup_error('Packet size must be positive')
                                                    #         valid = False
                                                    #         print("Failed to packet size range")
                                                    
                                                    # if len(values["INTER_PACKET_DELAY"]) == 0:
                                                    #     sg.popup_error('Inter-Packet Delay - Empty field')  
                                                    #     valid = False 
                                                    # else:    
                                                    #     if int(values["INTER_PACKET_DELAY"]) < 0 or int(values["INTER_PACKET_DELAY"]) > 10000:
                                                    #         sg.popup_error('Inter-packet delay must be within [0,10000] interval') 
                                                    #         valid = False  
                                                    #         print("Failed to inter-packet delay range") 
                                                            
                                                    # if len(values["FRAME_RATE"]) == 0:
                                                    #     sg.popup_error('Frame Rate - Empty field')  
                                                    #     valid = False 
                                                    # else:
                                                    if not emptyVals:
                                                        if int(values["FRAME_RATE"]) < 10 or int(values["FRAME_RATE"]) > 50: 
                                                            sg.popup_error('Frame rate (fps) must be within [10,50] interval') 
                                                            valid = False
                                                            print("Failed to frame rate range")
                                                        
                                                    # if len(values["BANDWIDTH_RESV_ACC"]) == 0:
                                                    #     sg.popup_error('Bandwidth Reserve Accumulation - Empty field')  
                                                    #     valid = False
                                                    # else:
                                                    #     if int(values["BANDWIDTH_RESV_ACC"]) < 0 or int(values["BANDWIDTH_RESV_ACC"]) > 10:
                                                    #         sg.popup_error('Bandwidth reserve accumulation must be within [0,100] interval')  
                                                    #         valid = False
                                                    #         print("Failed to bandwidth reserve accumalation range")
                                                            
                                                    # if len(values["BANDWIDTH_RESV"]) == 0:
                                                    #     sg.popup_error('Bandwidth Reserve - Empty field')  
                                                    #     valid = False
                                                    # else:                     
                                                    #     if int(values["BANDWIDTH_RESV"]) < 0 or int(values["BANDWIDTH_RESV"]) > 100:
                                                    #         sg.popup_error('Bandwidth reserve must be within [0,10] interval')   
                                                    #         valid = False
                                                    #         print("Failed to bandwidth reserve range") 
                                                            
                                                    #### 
                                                                                      
                                                    if 'Full HD' in values['RES_TYPE']:
                                                        width = 1920
                                                        height = 1080
                                                    elif 'HD+' in values['RES_TYPE']:
                                                        width = 1600
                                                        height = 900
                                                    elif 'HD' in values['RES_TYPE']:
                                                        width = 1280
                                                        height = 720
                                                    elif 'qHD' in values['RES_TYPE']:
                                                        width = 960
                                                        height = 540
                                                    elif 'nHD' in values['RES_TYPE']:
                                                        width = 640
                                                        height = 360
                                                    elif '960H' in values['RES_TYPE']:
                                                        width = 960
                                                        height = 480
                                                    elif 'HVGA' in values['RES_TYPE']:
                                                        width = 480
                                                        height = 320
                                                    elif 'VGA' in values['RES_TYPE']:
                                                        width = 640
                                                        height = 480
                                                    elif 'SVGA' in values['RES_TYPE']:
                                                        width = 800
                                                        height = 600
                                                    elif 'DVGA' in values['RES_TYPE']:
                                                        width = 960
                                                        height = 640
                                                    elif 'QVGA' in values['RES_TYPE']:
                                                        width = 320 
                                                        height = 240
                                                    elif 'QQVGA' in values['RES_TYPE']:
                                                        width = 160
                                                        height = 120
                                                    elif 'HQVGA' in values['RES_TYPE']:
                                                        width = 240
                                                        height = 160 
                                                        
                                                    ######################################################################
                                                    ######################################################################
                                                    ## Add parameters from previous GUI, to params_changed dictionary ####
                                                    
                                                    if not emptyVals:
                                                    
                                                        params_changed['GAIN'] = values["GAIN"]
                                                        params_changed['BLACK_THRESH'] = values["BLACK_THRESH"]
                                                        params_changed['EXP_TIME'] = values["EXP_TIME"]
                                                   ##     params_changed['PACKET_SIZE'] = values["PACKET_SIZE"]
                                                        params_changed['PACKET_SIZE'] = str(packet_size)
                                                ##        params_changed['INTER_PACKET_DELAY'] = values["INTER_PACKET_DELAY"]
                                                        params_changed['INTER_PACKET_DELAY'] = str(inter_paket_delay)
                                                        params_changed['FRAME_RATE'] = values["FRAME_RATE"]
                                                   ##     params_changed['BANDWIDTH_RESV_ACC'] = values["BANDWIDTH_RESV_ACC"]
                                                        params_changed['BANDWIDTH_RESV_ACC'] = str(bw_resv_acc)
                                                   ##     params_changed['BANDWIDTH_RESV'] = values["BANDWIDTH_RESV"]
                                                        params_changed['BANDWIDTH_RESV'] = str(bw_resv)
                                                        params_changed['FIRST'] = str(width)
                                                        params_changed['SEC'] = str(height) 
                                                    else:
                                                        params_changed['GAIN'] = "50"
                                                        params_changed['BLACK_THRESH'] = "2"
                                                        params_changed['EXP_TIME'] = "140"
                                                   ##     params_changed['PACKET_SIZE'] = values["PACKET_SIZE"]
                                                        params_changed['PACKET_SIZE'] = "1500"
                                                ##        params_changed['INTER_PACKET_DELAY'] = values["INTER_PACKET_DELAY"]
                                                        params_changed['INTER_PACKET_DELAY'] = "5000"
                                                        params_changed['FRAME_RATE'] = "50"
                                                   ##     params_changed['BANDWIDTH_RESV_ACC'] = values["BANDWIDTH_RESV_ACC"]
                                                        params_changed['BANDWIDTH_RESV_ACC'] = "4"
                                                   ##     params_changed['BANDWIDTH_RESV'] = values["BANDWIDTH_RESV"]
                                                        params_changed['BANDWIDTH_RESV'] = "10"
                                                        params_changed['FIRST'] = "240"
                                                        params_changed['SEC'] = "160"
                                                     
                                                    ###################################################################################
                                                    ########## Function to order all the parameters inside params_changed dictionary ##
                                                    
                                                    params_changed = order_all_params_pfs_adv(params_changed) 
                                                    
                                                    print(" ------ All parameters:") 
                                                    
                                                    counter_missing = 0
                                                    
                                                    for ind_param, param in enumerate(params_changed):
                                                        if len(param) == 2:
                                                            print(str(param[0]) + ": " + str( param[1]))
                                                       # if len(param) != 2:
                                                       #      print("Missing parameter " + str(ind_param))
                                                       #      counter_missing += 1
                                                            
                                                  #  print(str(counter_missing) + " missing parameters")                                         
                                                     
                                                    ###################################################################################
                                                    ###################################################################################
                                                    
                                                    pfs_ok, pfs_struct = write_to_pfs_adv_opt(params_changed)   
                                                    
                                                    windowx = sg_py.Window('Choose directory folder', [[sg_py.Text('Folder name')], [sg_py.Input(), sg_py.FolderBrowse()], [sg_py.OK(), sg_py.Cancel()] ]).read(close=True)
                                                    (keyword, dict_dir) = windowx                
                                               
                                                    dir_path = dict_dir['Browse'] 
                                                    
                                                    pfs_filename = "parameters_" + getDateTimeStrMarker() + "_full" + ".pfs"
                                                     
                                                    dir_pfs_gen = dir_path + pfs_filename  
                                                    
                                                    pfs_gen = False
                                                     
                                                    counter_tries_writing_to_pfs_file_adv = 0
                                                    
                                                    ##################################################################################
                                                    
                                                    list_keys = []
                                                    list_values = []
                                                    
                                                    for key, value in only_adv_opt.items():
                                                        list_keys.append(key)
                                                        list_values.append(value)                                       
                                                        
                                                    
                                                    ##################################################################################
                                                    
                                                    while pfs_gen == False:
                                                        
                                                        print("Trying to write to full PFS file, for " + str(counter_tries_writing_to_pfs_file_adv+1) + " th time ...")
                                                     
                                                    
                                                        if ".pfs" in dir_pfs_gen: 
                                                        
                                                            with open(dir_pfs_gen, "w") as input: 
                                                                
                                                                for ind_line, line in enumerate(pfs_struct):
                                                                    input.write(line + "\n") 
                                                                    
                                                                    if ind_line == len(pfs_struct)-1:
                                                                        pfs_gen = True
                                                                        print("Success !!! PFS file generated with advanced options") 
                                                        else:
                                                            print("Check if filename has one of the following extensions: \n {.pfs}")
                                                            
                                                        counter_tries_writing_to_pfs_file_adv += 1 
                                                        
                                                        
                                                    #############################################
                                                    #############################################  
                                                    
                                                    print("Here")
                                                
                                                    ## Another python file with also the remaining properties 
                                                    
                                                    # process = multiprocessing.Process(target=acq_task, args=(countx, eventt))
                                                    
                                                    # process.terminate()
                                                    
                                                    # eventt.set()
                                                    
                                        #            t.join()      
                                        
                                        
                                                    while True:
                                                        control_inf = exp_control()
                                                        
                                                        if control_inf[0] == True:
                                                            
                                                            numberTests = control_inf[1]
                                                            gap_bet_tests = control_inf[2]
                                                            
                                                            break
                                                        else: 
                                                            continue
                                                    
                                                    print("Number of tests: " + str(numberTests))
                                                        
                                                    if numberTests > 1: 
                                                        print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
                                                    else:
                                                        print("Going to perform just 1 test")
                                                        
                                                    if True:
                                                        
                                                           
                                        
                                                        for curTest in range(0,numberTests): 
                                                            if not emptyVals:    
                                                            
                                                    #            curTest = 0
                                                 #               print("Here on test " + str(curTest) + " ...")                           
                                                            
                                                                data_to_save = whole_processing_software_adv(int(values['FRAME_RATE']), 
                                                                                          packet_size, 
                                                                                          inter_paket_delay, 
                                                                                          bw_resv, 
                                                                                          bw_resv_acc, 
                                                                                          int(values['GAIN']), 
                                                                                          int(values['EXP_TIME']), 
                                                                                          int(values['recTime']),
                                                                                          width,
                                                                                          height,
                                                                                          int(values['decLevel']),
                                                                                          dir_path,
                                                                                          curTest,
                                                                                          numberTests,
                                                                                          gap_bet_tests,
                                                                                          list_values,
                                                                                          list_keys)  
                                                            else:
                                                            
                                                         #   for curTest in range(0,numberTests): 
                                                   
                                                     #           curTest = 0
                                                                 
                                           #                     print("Here on test " + str(curTest) + " ...")    
                                                                
                                                                data_to_save = whole_processing_software_adv(50, 
                                                                                          1500, 
                                                                                          5000, 
                                                                                          4, 
                                                                                          10, 
                                                                                          50, 
                                                                                          140, 
                                                                                          5,
                                                                                          width,
                                                                                          height,
                                                                                          0,
                                                                                          dir_path,
                                                                                          curTest,
                                                                                          numberTests,
                                                                                          gap_bet_tests,
                                                                                          list_values,
                                                                                          list_keys)  
                                                          
                                                           
                                                            values = [values, dir_path]                     
                                                            print("Choosen directory: " + dir_path)  
                            
                                                            if len(data_to_save) > 0: 
                                
                                                                for ind_data, data in enumerate(data_to_save):
                                                                
                                                                    clusteringRes, execTime, numberImg = data       
                                                                    
                                                                    gui_show_results(clusteringRes, execTime, numberImg, ind_data, data_to_save)
                                                                    
                                                                    print(" -- Showing GUI with results for test number " + str(ind_data+1))
                                                                    
                                                                    time.sleep(5)          
                                                                
                                                            else:
                                                                print("Output results not available !!!")                  
                                
                                
                                                            ###################
                                
                                                            executionTime = (time.time() - startTime)
                                                            print('Whole execution time in seconds: ' + str(executionTime))
                                                            
                                                            adv_go = False
                                                     
                                                    ####################################################################### 
                                                    ####################################################################### 
                                                    #######################################################################   
                                                    ####################################################################### 
                                                    ####################################################################### 
                                                    #######################################################################                              
                                                    ####################################################################### 
                                                    ####################################################################### 
                                                    #######################################################################
                                                
                                                else:
                                                
                                                    window['Decisor Level:'].update(value = values['decLevel'])                   
                                                    window['-SECONDS-'].update(value = values['recTime'])
                                                    
                                                    time.sleep(2)
                                                     
                                                    # if len(values["GAIN"]) == 0:
                                                    #     sg.popup_error('Gain - Empty field')
                                                    #     valid = False
                                                    # else:
                                                        
                                                    if not emptyVals:
                                                        if int(values["GAIN"]) < 0 or int(values["GAIN"]) > 50:
                                                            sg.popup_error('Gain must be within [0,50] interval') 
                                                            valid = False 
                                                            print("Failed to gain range") 
                                                            
                                                    # if len(values["BLACK_THRESH"]) == 0:
                                                    #     sg.popup_error('Black Level - Empty field') 
                                                    #     valid = False
                                                    # else:
                                                    
                                                    if not emptyVals:
                                                        if int(values["BLACK_THRESH"]) < 0 or int(values["BLACK_THRESH"]) > 255:
                                                            sg.popup_error('Black level must be within [0,255] interval')    ## for Mono8 configuration 
                                                            valid = False
                                                            print("Failed to black level range")
                                                    
                                                    # if len(values["EXP_TIME"]) == 0:
                                                    #     sg.popup_error('Exposure Time - Empty field')  
                                                    #     valid = False
                                                    # else: 
                                                        
                                                    if not emptyVals:
                                                        if int(values["EXP_TIME"]) < 0 or int(values["EXP_TIME"]) > 35000: 
                                                            sg.popup_error('Exposition time must be within [0,35000] interval')  
                                                            valid = False
                                                            print("Failed to exposure time range")
                                                    
                                                    # if len(values["PACKET_SIZE"]) == 0:
                                                    #     sg.popup_error('Packet Size - Empty field')  
                                                    #     valid = False 
                                                    # else:
                                                    #     if int(values["PACKET_SIZE"]) <= 0:
                                                    #         sg.popup_error('Packet size must be positive')
                                                    #         valid = False
                                                    #         print("Failed to packet size range")
                                                    
                                                    # if len(values["INTER_PACKET_DELAY"]) == 0:
                                                    #     sg.popup_error('Inter-Packet Delay - Empty field')  
                                                    #     valid = False 
                                                    # else:    
                                                    #     if int(values["INTER_PACKET_DELAY"]) < 0 or int(values["INTER_PACKET_DELAY"]) > 10000:
                                                    #         sg.popup_error('Inter-packet delay must be within [0,10000] interval') 
                                                    #         valid = False  
                                                    #         print("Failed to inter-packet delay range") 
                                                            
                                                    # if len(values["FRAME_RATE"]) == 0:
                                                    #     sg.popup_error('Frame Rate - Empty field')  
                                                    #     valid = False 
                                                    # else:
                                                    
                                                    if not emptyVals:
                                                        if int(values["FRAME_RATE"]) < 10 or int(values["FRAME_RATE"]) > 50: 
                                                            sg.popup_error('Frame rate (fps) must be within [10,50] interval') 
                                                            valid = False
                                                            print("Failed to frame rate range")
                                                        
                                                    # if len(values["BANDWIDTH_RESV_ACC"]) == 0:
                                                    #     sg.popup_error('Bandwidth Reserve Accumulation - Empty field')  
                                                    #     valid = False
                                                    # else:
                                                    #     if int(values["BANDWIDTH_RESV_ACC"]) < 0 or int(values["BANDWIDTH_RESV_ACC"]) > 10:
                                                    #         sg.popup_error('Bandwidth reserve accumulation must be within [0,100] interval')  
                                                    #         valid = False
                                                    #         print("Failed to bandwidth reserve accumalation range")
                                                            
                                                    # if len(values["BANDWIDTH_RESV"]) == 0:
                                                    #     sg.popup_error('Bandwidth Reserve - Empty field')  
                                                    #     valid = False
                                                    # else:                     
                                                    #     if int(values["BANDWIDTH_RESV"]) < 0 or int(values["BANDWIDTH_RESV"]) > 100:
                                                    #         sg.popup_error('Bandwidth reserve must be within [0,10] interval')   
                                                    #         valid = False
                                                    #         print("Failed to bandwidth reserve range") 
                                                            
                                                    width = 1920
                                                    height = 1080
                                                            
                                                    if 'Full HD' in values['RES_TYPE']:
                                                         width = 1920
                                                         height = 1080
                                                    elif 'HD+' in values['RES_TYPE']:
                                                         width = 1600
                                                         height = 900
                                                    elif 'HD' in values['RES_TYPE']:
                                                         width = 1280
                                                         height = 720
                                                    elif 'qHD' in values['RES_TYPE']:
                                                         width = 960
                                                         height = 540
                                                    elif 'nHD' in values['RES_TYPE']:
                                                         width = 640
                                                         height = 360
                                                    elif '960H' in values['RES_TYPE']:
                                                         width = 960
                                                         height = 480
                                                    elif 'HVGA' in values['RES_TYPE']:
                                                         width = 480
                                                         height = 320 
                                                    elif 'VGA' in values['RES_TYPE']:
                                                         width = 640
                                                         height = 480
                                                    elif 'SVGA' in values['RES_TYPE']:
                                                         width = 800
                                                         height = 600
                                                    elif 'DVGA' in values['RES_TYPE']:
                                                         width = 960
                                                         height = 640
                                                    elif 'QVGA' in values['RES_TYPE']:
                                                         width = 320
                                                         height = 240
                                                    elif 'QQVGA' in values['RES_TYPE']:
                                                         width = 160
                                                         height = 120
                                                    elif 'HQVGA' in values['RES_TYPE']:
                                                         width = 240
                                                         height = 160
                                                    
                                                    a = width
                                                    b = height                                    
                                                    width = b
                                                    height = a  
                                                           
                                                        
                                                    if valid == True:  
                                                        print("Valid parameters")                
                                                        print("Next step")
                                                        
                                                        windowx = sg_py.Window('Choose directory folder', [[sg_py.Text('Folder name')], [sg_py.Input(), sg_py.FolderBrowse()], [sg_py.OK(), sg_py.Cancel()] ]).read(close=True)
                                                        (keyword, dict_dir) = windowx                
                                                  
                                                        dir_path = dict_dir['Browse']               
                                                        
                                                        if len(dir_path) != 0:                       
                                                            repeat = False                      
                                                                
                                                            # print("Resulting dict: ")
                                                            # for value in values:  
                                                            #    print(value + " : " + values[value])   
                                                               
                                                        ##    setParametersToPypylon(values)   
                                                        
                                                            pfs_filename = "parameters_" + getDateTimeStrMarker() + ".pfs"
                                                            
                                                            dir_pfs_gen = dir_path + pfs_filename 
                                                            
                                                            pfs_gen = False
                                                            
                                                            counter_tries_writing_to_pfs_file = 0
                                                            
                                                            while pfs_gen == False:
                                                                
                                                                print("Trying to write to PFS file, for " + str(counter_tries_writing_to_pfs_file+1) + " th time ...")
                                                                
                                                                if emptyVals:
                                                                    
                                                                    pfs_gen =  save_to_pfs(dir_pfs_gen, 50, 2, 
                                                                                 140, 50, packet_size, 
                                                                                 inter_paket_delay, bw_resv, bw_resv_acc,
                                                                                 width, height)
                                                                else:
                                                                                          
                                                            
                                                                    pfs_gen =  save_to_pfs(dir_pfs_gen, int(values['GAIN']), int(values["BLACK_THRESH"]), 
                                                                                 int(values['EXP_TIME']), int(values['FRAME_RATE']), packet_size, 
                                                                                 inter_paket_delay, bw_resv, bw_resv_acc,
                                                                                 width, height)
                                                                
                                                                counter_tries_writing_to_pfs_file += 1  
                                                            
                                                            print("Here")
                                                             
                                                  ##          process = multiprocessing.Process(target=acq_task, args=(countx, eventt))
                                                            
                                                    ##        cv2.destroyAllWindows()
                                                             
                                              ##              eventt.set()
                                                            
                                           #                 t.join()                                           
                                                           
                                                            
                                                    ##        acq_image_camera(countx, False)
                                                    
                                                            # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                                                            # converter = pylon.ImageFormatConverter() 
                                                            
                                                            # # grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                                                            # # grabResult.Release()
                                                            
                                                            # while camera.NumReadyBuffers.GetValue() > 0:
                                                            #        camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
                    
                                                      ##      camera.Close()
                                                            
                                                      ##      time.sleep(5)
                                                      
                                                  ##          time.sleep(20) 
                
                                                            # process = multiprocessing.current_process()        
                                                            # process.terminate()                                    
                                                            
                                                            # create the shared queue                                            
                                                            
                                                 ##           process_but = Thread(target=button_task, args=(queue, True, True))
                                                    ##        process_but.start()
                                                    
                                                    
                                                            
                                                ###            process = Thread(target=acq_task, args=(queue, countx, eventt))                                            
                
                                                            # process = multiprocessing.Process(target=acq_task, args=(queue, countx, eventt))
                                                            
                                                            # process_but =  multiprocessing.Process(target=button_task, args=(queue, True, True))
                                                            
                                                            # if resp == True:  
                                                            #     process.kill() 
                 
                                                            # if process.is_alive():
                                                            #      print("Keep waiting ...")
                                                            # else:
                                                            #      print("Proceeding ...") 
                                                            #      break                                                
                                                                
                                                            
                                                            # while True:
                                                                
                                                            #     thread = CustomThread()
                                                            #     # start the thread  
                                                            #     thread.start()
                                                                
                                                            #     # wait for the thread to finish
                                                            # ##    thread.join()
                                                                
                                                            #     # get the value returned from the thread
                                                            #     data = thread.value                                              
                                                                  
                                                            #     resp = data
                                                            
                                                            # print("Respx: " + str(respx))
                                                                
                                                            # if respx:
                                                            #      process_but.kill()
                                                            #      process.kill()
                                                                
                                                            # if process.is_alive():
                                                            #    print("Keep waiting ...")
                                                            # else:
                                                            #    print("Proceeding ...") 
                                                            #    break
                                                                
                                                                
                                                  
                                                             
                                                            
                                                            # ok_list = get_ok()
                                                            
                                                            # if len(ok_list) > 0:
                                                            
                                                            #     print("Len Ok: (" + str(len(ok_list)) + " , " + str(len(ok_list[0])))
                                                            #     print(ok_list) 
                                                            
                                                            basler_check = False
                                                            
                                                            while not basler_check:                                            
                                                                basler_check, model_name = confirm_basler()
                                                                time.sleep(2)
                                                                
                                                            
                                                            while True:
                                                                control_inf = exp_control()
                                                                
                                                                if control_inf[0] == True:
                                                                    
                                                                    numberTests = control_inf[1]
                                                                    gap_bet_tests = control_inf[2]
                                                                    
                                                                    break
                                                                else:
                                                                    continue
                                                            
                                                            print("Number of tests: " + str(numberTests))
                                                                
                                                            if numberTests > 1: 
                                                                print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
                                                            else:
                                                                print("Going to perform just 1 test")
                                                                
                                                            curTest = 0
                                                            
                                                     ##       for curTest in range(0,numberTests): 
                                                               
                                                            if True:
                                                                if emptyVals:
                                                                    
                                                                    data_to_save = whole_processing_software(50, 
                                                                                              packet_size, 
                                                                                              inter_paket_delay, 
                                                                                              bw_resv, 
                                                                                              bw_resv_acc, 
                                                                                              50, 
                                                                                              140, 
                                                                                              5,
                                                                                              width,
                                                                                              height,
                                                                                              0,
                                                                                              dir_path,
                                                                                              curTest,
                                                                                              numberTests,
                                                                                              gap_bet_tests) 
                                                                else:
                                                            
                                                                    data_to_save = whole_processing_software(int(values['FRAME_RATE']), 
                                                                                              packet_size, 
                                                                                              inter_paket_delay, 
                                                                                              bw_resv, 
                                                                                              bw_resv_acc, 
                                                                                              int(values['GAIN']), 
                                                                                              int(values['EXP_TIME']), 
                                                                                              int(values['recTime']),
                                                                                              width,
                                                                                              height, 
                                                                                              int(values['decLevel']),
                                                                                              dir_path,
                                                                                              curTest,
                                                                                              numberTests,
                                                                                              gap_bet_tests)   
                                                            
                                                            print(" ---- OK")
                                                            print(" ---- OK") 
                                                            print(" ---- OK")
                                                            print(" ---- OK")
                                                            print(" ---- OK") 
                                                               
                                                            values = [values, dir_path]                     
                                                            print("Choosen directory: " + dir_path)  
                                
                                                            if len(data_to_save) > 0: 
                                
                                                                for ind_data, data in enumerate(data_to_save):
                                                                
                                                                    clusteringRes, execTime, numberImg = data       
                                                                    
                                                                    gui_show_results(clusteringRes, execTime, numberImg, ind_data, data_to_save) 
                                                                    
                                                                    print(" -- Showing GUI with results for test number " + str(ind_data+1))
                                                                    
                                                                    time.sleep(5)           
                                                                    
                                                            else:
                                                                print("Output results not available !!!")                  
                                
                                
                                                            ###################
                                
                                                            executionTime = (time.time() - startTime)
                                                            print('Whole execution time in seconds: ' + str(executionTime))
                                
                                                                         
                                                            
                                                            break 
                                                    
                                                        else:
                                                            sg.popup_error('Folder directory required !') 
                                                            repeat = True
                                                            del values, event  
                                                        
                                              ##      setParametersToPypylon(values)   
                                              ## or just call the whole python file related to image processing and so on               
                                                    else:
                                                        repeat = True 
                                                        valid = True
                                                        print("Try again")
                                                        del values, event    
                                            else:
                                                if event == 'Clear':
                                                    
                                                    numberTests -= 1
                                                    
                                                    window['GAIN'].update(value = 10) 
                                                    window['BLACK_THRESH'].update(value = 2) 
                                                    window['EXP_TIME'].update(value = 140)
                                                    window['PACKET_SIZE'].update(value = 1500)
                                                    window['INTER_PACKET_DELAY'].update(value = 5000)
                                                    window['FRAME_RATE'].update(value = 50)
                                                    window['BANDWIDTH_RESV_ACC'].update(value = 4)
                                                    window['BANDWIDTH_RESV'].update(value = 10)
                                                    window['FIRST'].update(value = 1920)
                                                    window['SEC'].update(value = 1920)   
                                                    
                                                    repeat = True 
                                                    valid = True
                                                    
                                                elif event == 'Delete':                        
                                                    numberTests -= 1
                                                    break                          
                                        
                            window.close()  
                        else:
                            
                            repeat = True  
                            
                            while True:
                                control_inf = exp_control()
                                
                                if control_inf[0] == True:
                                    
                                    numberTests = control_inf[1]
                                    gap_bet_tests = control_inf[2]
                                    
                                    break
                                else:
                                    continue
                            
                            print("Number of tests: " + str(numberTests))
                                
                            if numberTests > 1: 
                                print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
                            else:
                                print("Going to perform just 1 test")
                            
                            print("numTests init: " + str(numberTests))
                            
                            enab = optional_prop()
                             
                            enab_val_pfs = enab[0]
                            enab_timestamps = enab[1]
                            
                            read_pfs_file(gap_bet_tests, numberTests, repeat, startTime, enab_val_pfs, enab_timestamps)
                             
                    else: 
                        demo_gui_menu() 
    else:
        if acquire and not process:
            
            resTypes = ['Full HD',
                        'HD+',
                        'HD',
                        'qHD',
                        'nHD',
                        '960H',
                        'HVGA',
                        'VGA',
                        'SVGA',
                        'DVGA',
                        'QVGA',
                        'QQVGA',
                        'HQVGA'
                       ]

            res_dims = ['(1920x1080)',
                        '(1600x900)',
                        '(1280x720)',
                        '(960x540)',
                        '(640x360)',
                        '(960x480)',
                        '(480x320)',
                        '(640x480)',
                        '(800x600)',
                        '(960x640)',
                        '(320x240)',
                        '(160x120)',
                        '(240x160)'
                      ]   

            res_fullTypes = []

            for ind_r, r in enumerate(resTypes):
                r += ' ' + res_dims[ind_r]
                res_fullTypes.append(r)


            basler_check = False

            basler_check, model_name = confirm_basler() 

            if basler_check == True:
                
                from pfs_input_acq_step_only import read_pfs_file
                
                repeat = True  
                
                while True:
                    control_inf = exp_control()
                    
                    if control_inf[0] == True:
                        
                        numberTests = control_inf[1]
                        gap_bet_tests = control_inf[2]
                        
                        break
                    else:
                        continue
                
                print("Number of tests: " + str(numberTests))
                    
                if numberTests > 1: 
                    print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
                else:
                    print("Going to perform just 1 test")
                
                print("numTests init: " + str(numberTests))
                
                enab = optional_prop()
                 
                enab_val_pfs = enab[0]
                enab_timestamps = enab[1]
                
                read_pfs_file(gap_bet_tests, numberTests, repeat, startTime, enab_val_pfs, enab_timestamps)
            
            resp_again = 2 
            
            while resp_again == 2:
                from process_pfs_only import post_proc_pfs_only 
                resp_again = repeat_loop_proc()
         
                if resp_again == 2:
                
                    resTypes = ['Full HD',
                                'HD+',
                                'HD',
                                'qHD',
                                'nHD',
                                '960H',
                                'HVGA',
                                'VGA',
                                'SVGA',
                                'DVGA',
                                'QVGA',
                                'QQVGA',
                                'HQVGA'
                               ]
    
                    res_dims = ['(1920x1080)',
                                '(1600x900)',
                                '(1280x720)',
                                '(960x540)',
                                '(640x360)',
                                '(960x480)',
                                '(480x320)',
                                '(640x480)',
                                '(800x600)',
                                '(960x640)',
                                '(320x240)',
                                '(160x120)',
                                '(240x160)'
                              ]   
    
                    res_fullTypes = []
    
                    for ind_r, r in enumerate(resTypes):
                        r += ' ' + res_dims[ind_r]
                        res_fullTypes.append(r)
    
    
                    basler_check = False
    
                    basler_check, model_name = confirm_basler() 
    
                    if basler_check == True:
                        
                        from pfs_input_acq_step_only import read_pfs_file
                        
                        repeat = True  
                        
                        while True:
                            control_inf = exp_control()
                            
                            if control_inf[0] == True:
                                
                                numberTests = control_inf[1]
                                gap_bet_tests = control_inf[2]
                                
                                break
                            else:
                                continue
                        
                        print("Number of tests: " + str(numberTests))
                            
                        if numberTests > 1: 
                            print("Going to perform for " + str(numberTests) + " tests with a " + str(gap_bet_tests) + " minutes gap")
                        else:
                            print("Going to perform just 1 test")
                        
                        print("numTests init: " + str(numberTests))
                        
                        enab = optional_prop()
                         
                        enab_val_pfs = enab[0]
                        enab_timestamps = enab[1]
                        
                        read_pfs_file(gap_bet_tests, numberTests, repeat, startTime, enab_val_pfs, enab_timestamps)
                                     
        elif process and not acquire:
           
            from process_pfs_only import post_proc_pfs_only 
            post_proc_pfs_only()
            
            resp_again = 2
            
            while resp_again == 2:
                from process_pfs_only import post_proc_pfs_only 
                resp_again = repeat_loop_proc()
                
                if resp_again == 2:
                    post_proc_pfs_only()
                
                
else: 
    print("Check if basler camera is connected via Ethernet")    
    respf = input('It is well connected ?')
    
    if respf == 's' or respf == 'S' or respf == 'y' or respf == 'Y':
        resps = input('It is a basler camera for sure ?')
        
        if resps == 's' or resps == 'S' or resps == 'y' or resps == 'Y':
            print("Check again ...")
        else:
            print("The software only allows basler cameras. \nTry again ...")
    else:
        print("Connect it and try again ...")

        
 

