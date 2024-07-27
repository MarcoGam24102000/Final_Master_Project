# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:05:21 2023

@author: Rui Pinto
"""

import PySimpleGUI as sg
import pandas as pd
import sys
from turn_new_feature_to_life import feature_life
import time

def feature_selection_gui():
    layout = [
        [sg.Text('Do you want to use any previously defined features?')],
        [sg.Checkbox('Yes', default=True, key='yes_checkbox'),
         sg.Checkbox('No', key='no_checkbox')],
        [sg.Button('OK'), sg.Button('Cancel')]
    ]

    window = sg.Window('Feature Selection', layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            window.close()
            return None
        elif event == 'OK':
            selected_option = values['yes_checkbox']
            window.close()
            return selected_option

        # If 'No' is selected, disable 'Yes'
        if values['no_checkbox']:
            window['yes_checkbox'].update(disabled=True)
        else:
            window['yes_checkbox'].update(disabled=False)

    window.close()

def get_valid_filename():
    layout = [
        [sg.Text("Enter an Excel filename (without .xlsx extension):")],
        [sg.InputText(key="filename")],
        [sg.Button("Submit")]
    ]

    window = sg.Window("Excel Filename Input", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "Submit":
            filename = values["filename"]

            # Check if the filename is not empty and has no spaces
            if filename.strip() != "" and " " not in filename:
                window.close()
                return filename
            else:
                sg.popup_error("Invalid filename. Please ensure it's not empty and has no spaces.")
    
    window.close()

def extract_directories(excel_file):
    try:
        # Read the Excel file
        df = pd.read_excel(f"{excel_file}")
        
        # Extract the directories and store them in a tuple
        directories = tuple(df[["ROI's first group", "ROI's second group"]].values.flatten())
        
        return directories
    except FileNotFoundError:
        sg.popup_error(f"The Excel file '{excel_file}' does not exist.")
        return ()

def just_clustering():

    import os 
    from txt_files_searcher import proc_txt_browsing
    from texts_info import read_from_txt_file_tests_info, read_configs_file
    
    again_x = True
    
    while again_x:
        try:
            from clustering_part import videoAnalysisClusteringPart, videoAnalysisMultipleVideosWithinOneClusteringPart
            again_x = False
            break
        except ModuleNotFoundError:
            again_x = True
        
    import PySimpleGUI as sg
    import pandas as pd
    import openpyxl
    
    def load_packages():
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
        
    def extract_feature_data():
        layout = [
            [sg.Text("Select an Excel file:")],
            [sg.InputText(key="file_path"), sg.FileBrowse(file_types=(("Excel Files", "*.xlsx"),))],
            [sg.Button("Ok"), sg.Button("Exit")],
        ]
    
        window = sg.Window("Excel File Reader", layout)
    
        data_list = []
    
        while True:
            event, values = window.read()
    
            if event == sg.WINDOW_CLOSED or event == "Exit":
                break
            elif event == "Ok":
                file_path = values["file_path"]
                if file_path.endswith(".xlsx"):
                    try:
                        
                        
                        wb = openpyxl.load_workbook("C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\today.xlsx")
                        sheet = wb.active
                        data_list = []
                        for row in sheet.iter_rows(values_only=True):
                                data_list.append(row)
                                
                        sg.popup("File successfully read and copied to a list!", title="Success")
                        break
                    except Exception as e:
                        sg.popup_error(f"Error reading the file: {e}", title="Error")
                else:
                    sg.popup_error("Please select a valid Excel file (.xlsx)", title="Error")
    
        window.close()
        return data_list
    
    def check_number_tests_init():
        
        x = 0
        
        layout = [
            [sg.Text("Was the experiment initially conducted with just one test or more?")],
            [sg.Radio("Just one test", "test_type", default=True, key="one_test")],
            [sg.Radio("More than one test", "test_type", key="more_tests")],
            [sg.Button("Submit"), sg.Button("Exit")]
        ]
    
        window = sg.Window("Experiment Test Type", layout)
    
        while True:
            event, values = window.read()
    
            if event == sg.WINDOW_CLOSED or event == "Exit":
                break
            elif event == "Submit":
                if values["one_test"]:
                    test_type = "Just one test"
                    x = 1
                elif values["more_tests"]:
                    test_type = "More than one test"
                    x = 2
                sg.popup(f"The experiment was initially conducted with: {test_type}", title="Result")
                break       
    
        window.close()
        
        return x
        
    def proc_just_clustering():
        
        this_dir = os.path.abspath(__file__)
        
        dParts = this_dir.split("\\")
    
        newDir = ""
        for indD, d in enumerate(dParts):
            if indD < len(dParts)-1:
                newDir += d + "\\"
        this_dir =  newDir      
        
        dir_txt_file, config_dirs = proc_txt_browsing(this_dir)
        
        print("pppppppppppppppppp")
        print(dir_txt_file)
        print(config_dirs)
        print("pppppppppppppppppp")
        
        if not isinstance(dir_txt_file, list):
        
            dir_tests_info_path = dir_txt_file
        else:
            dir_tests_info_path = dir_txt_file[0]
        
        data_tests = read_from_txt_file_tests_info(dir_tests_info_path)
         
        [numberTests, dur_test, time_bet_tests] = data_tests 
        
        load_packages()  
    
        dir_configs_path = config_dirs[0]
        
        # while again == True: 
        
        #     windowx = sg.Window('Choose path to configs file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
        #     (keyword, dict_dir) = windowx                
        
        #     dir_configs_path = dict_dir['Browse'] 
            
        #     if dir_configs_path is None:
        #         again = True
        #     else:
        #         if not ".txt" in dir_configs_path:
        #             again = True
        #         else:
        #             again = False
        #             break
        
        print(dir_configs_path)
         
        configs_list = read_configs_file(dir_configs_path)    
        
        print("\n\n\n configs list: ")
        print(configs_list)
        print("\n\n")
        
 #       again = True
        
        pathRoiStart = ""
        pathRoiEnd = ""
        
        directories_tuple = ("","")      
        
        again_dirsThis = True
        
        ok = False
        
        while again_dirsThis:
        
            layout = [
                [sg.Text("Select an Excel file containing ROI directories:")],
                [sg.InputText(key="file_path", enable_events=True, disabled=True), sg.FileBrowse(file_types=(("Excel Files", "*.xlsx"),))],
                [sg.Button("Read Directories"), sg.Button("Exit")],
                [sg.Multiline(size=(50, 10), key="output", disabled=True)]
            ]
        
            window = sg.Window("ROI Directory Reader", layout)
        
            while True:
                event, values = window.read()
        
                if event == sg.WINDOW_CLOSED or event == "Exit":
                    again_dirsThis = True
                    break
                elif event == "Read Directories":
                    again_dirsThis = False
                    file_path = values["file_path"]
                    if file_path.endswith(".xlsx"):
                        directories_tuple = extract_directories(file_path)
                        if directories_tuple:
                            output_text = "\n".join(directories_tuple)
                            window["output"].update(output_text)
                            break
                    else:
                        sg.popup_error("Please select a valid Excel file (.xlsx)", title="Error")
        
            window.close()
                    
        pathRoiStart, pathRoiEnd = directories_tuple
            
        
        # while again == True:
        
        #     windowx = sg.Window('Choose path for images before', [[sg.Text('Folder name')], [sg.Input(), sg.FolderBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
        #     (keyword, dict_dir) = windowx                
         
        #     dir_bef_path = dict_dir['Browse'] 
            
        #     if len(dir_bef_path) == 0:
        #         print("Asking again")
        #     else:
            
        #         if dir_bef_path is None:
        #             again = True 
        #         else:
        #             again = False
        #             pathRoiStart += dir_bef_path
        #             break 
      
        # again = True 
        
        # while again == True:
        
        #     windowx = sg.Window('Choose path for images after', [[sg.Text('Folder name')], [sg.Input(), sg.FolderBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
        #     (keyword, dict_dir) = windowx                
        
        #     dir_after_path = dict_dir['Browse'] 
            
        #     if len(dir_bef_path) == 0:
        #         print("Asking again")
        #     else:
            
        #         if dir_after_path is None:
        #             again = True
        #         else:
        #             again = False
        #             pathRoiEnd += dir_after_path
        #             break 
      
  #      print("Dir after path: " + dir_after_path)   
        
        [decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile] = configs_list
        
        infi = [decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile]
        
        base_roi_paths = infi[-3]
        print("\n Base roi path: ")
        print(base_roi_paths)
        print("\n\n")
        
        roi_set = (newRoiPath, pathRoiStart, pathRoiEnd)
        
        print("R0I path: " + str(roiPath))
        
        data_list = extract_feature_data() 
        
        print("Data from excel file: ")
        print(data_list)
        
        number_images = len(data_list)-1
             
        lenMax = number_images 
        count = number_images 
         
        legendFeat = data_list[0]
        
        while True:
        
            result = feature_selection_gui()
    
            if result is not None:
                print(f'Selected option: {"Yes" if result else "No"}')
                
                if result:
                    data_list_newAdd = feature_life(legendFeat, data_list[1:])
                else:
                    data_list_newAdd = data_list[1:]
                    print("Here")
                    print("data_list_newAdd: ")
                    print(data_list_newAdd)                    
                
                number_metrics = len(legendFeat)-1
                
                x = check_number_tests_init()
                
                dataList2 = data_list_newAdd
                
                print("A")
                
                newD = [] 
                for d in dataList2: 
                    dx = d[1:]
                    dx = list(dx)
                    newD.append(dx)
                
                secRoundNewListMetrics = newD  
                
                print("secRoundNewListMetrics: ")
                print(secRoundNewListMetrics)
                
                print("x: " + str(x))
                
                # import sys
                # sys.exit()
                
                print("\n\n\n Here \n\n\n")
                
                # import sys
                # sys.exit()
                
                if x == 1:
                    videoAnalysisMultipleVideosWithinOneClusteringPart(secRoundNewListMetrics, number_metrics,  lenMax, infi, roi_set, count)
                elif x == 2:
                    videoAnalysisClusteringPart(secRoundNewListMetrics, number_metrics,  lenMax, infi, roi_set, count)
                    
                print("\n\n  On main \n\n")
                
                break
                 
            else:
                print('Operation canceled.')
                
                time.sleep(5)        
        
        
    again = True
    
    while again:
        try:
            proc_just_clustering()
            again = False
            break
        except Exception:
            again = True
    