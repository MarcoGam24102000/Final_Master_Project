# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:13:51 2023

@author: Rui Pinto
"""

import PySimpleGUI as sg
from softwareImageProcessingForLaserSpeckleAnalysisFinal import videoAnalysis, videoAnalysisMultipleVideosWithinOne
from soft_single_test import videoAnalysis_single
from countdown_timer import countdown_timer_display 
from info_gui import theory_info
from txt_files_searcher import proc_txt_browsing
from interactive_approach__withinVideo import sepVideosFromWholeOne
from texts_info import read_from_txt_file_tests_info, read_configs_file
import time
import webbrowser  
import shlex
import subprocess
import os
import io   
import cv2
import numpy as np 
from PIL import Image

def read_number_tests_from_file():
    with open('temp_numberTests.txt', 'r') as file:
        numberTests = int(file.read())

    # Delete file after reading
    import os
    os.remove('temp_numberTests.txt')

    return numberTests 


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
        [sg.Button("Data Info"), sg.Button("Read more ..."), sg.Button("Exit")]
    
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
        if (firstLoaded == True) and  event == 'Distances to the centroid': ##  or ind_data == len(data_to_save) - 1  ##  firstLoaded == True and 
             print("Distances to the centroid")
             second_gui_show_results(firstLoaded) 

             firstLoaded = False             
             
             time.sleep(3) 
              
             break
         
def post_proc_pfs_only():
    
    curTest = 0    
    time_bet_tests = 0 
    completed = False
    
    this_dir = os.path.abspath(__file__)
    
    dParts = this_dir.split("\\")

    newDir = ""
    for indD, d in enumerate(dParts):
        if indD < len(dParts)-1:
            newDir += d + "\\"
    this_dir =  newDir      
    
    dir_txt_file, config_dirs = proc_txt_browsing(this_dir)
    
    infx = []
    
    if True:         
         
        again = True
        
        if curTest == 0: 
            
            
            dir_tests_info_path = dir_txt_file
            
            # while again == True:
            
            #     windowx = sg.Window('Choose path to tests info file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
            #     (keyword, dict_dir) = windowx                
            
            #     dir_tests_info_path = dict_dir['Browse'] 
                
            #     if dir_tests_info_path is None:
            #         again = True
            #     else:
            #         if not ".txt" in dir_tests_info_path:
            #             again = True
            #         else:
            #             again = False
            #             break
            
            data_tests = read_from_txt_file_tests_info(dir_tests_info_path)
            
            [numberTests, dur_test, time_bet_tests] = data_tests 
            
        print("Number of tests: " + str(numberTests))
        print("Duration of each test: " + str(dur_test))
        print("Time between tests: " + str(time_bet_tests))
        
        time_bet_tests = round(float(time_bet_tests/60),5)
    
        print("Test number " + str(curTest+1))   
        
        load_packages()
        
        if numberTests == 1:
              data_to_save = [] 
              
              if True:      
                  
                  again = True
                  
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
                  
                  again = True
                  
                  pathRoiStart = ""
                  pathRoiEnd = ""
                  
                  while again == True:
                  
                      windowx = sg.Window('Choose path for images before', [[sg.Text('Folder name')], [sg.Input(), sg.FolderBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
                      (keyword, dict_dir) = windowx                
                  
                      dir_bef_path = dict_dir['Browse'] 
                      
                      if dir_bef_path is None:
                          again = True
                      else:
                          again = False
                          pathRoiStart += dir_bef_path
                          break 
                
                  again = True
                  
                  while again == True:
                  
                      windowx = sg.Window('Choose path for images after', [[sg.Text('Folder name')], [sg.Input(), sg.FolderBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
                      (keyword, dict_dir) = windowx                
                  
                      dir_after_path = dict_dir['Browse'] 
                      
                      if dir_after_path is None:
                          again = True
                      else:
                          again = False
                          pathRoiEnd += dir_after_path
                          break 
                  
                  [decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile] = configs_list
                  
                  infi =[decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile]
                  
                  infin = []
                  
                  for indc, indf in enumerate(infi):
                      indfn = ""
             
                      for ifs in indf:
                          if (ifs is not " ") and (ifs is not "\t") and (ifs is not "\n") and (ifs is not "\r"):
                              indfn += ifs
                              
                      indfn = indfn[1:]
                      
                      if indc == 0 or indc == 2:
                              indfn = int(indfn)
                              
                      infin.append(indfn)
                      
                      print(str(indc) + " - " + str(indfn))
                             
                  
                  infi = infin  
             
                             
              layout = [[sg.ProgressBar(1000, orientation='h', size = (20,20), key = 'progressbar')]]
        
              window = sg.Window('Processing ...', layout)
            
              while True:
                 
                  event, values = window.read(timeout=1000)
                 
                  if event == sg.WIN_CLOSED: 
                      break
                 
                  if True:
                      data_from_tests = []
                      
                      x = 0
             
          ##            infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile)
                      if '/' in first_clustering_storing_output:
                        x = 1
                      elif "\\" in first_clustering_storing_output:
                        x = 2
                      
                      if os.path.exists(first_clustering_storing_output):
                        if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                            
                            first_clustering_storing_output = first_clustering_storing_output[:-1]
                            first_clustering_storing_output += '_' + "0"
                            
                            ind_newFolder = 1
                            
                            if x == 1:
                                while os.path.exists(first_clustering_storing_output + '/'):
                                   
                                    first_clustering_storing_output[-1] = str(ind_newFolder)
                                    ind_newFolder += 1
                                
                            elif x == 2:
                                while os.path.exists(first_clustering_storing_output + "\\"):
                                    first_clustering_storing_output[-1] = str(ind_newFolder)
                                    ind_newFolder += 1  

                                infi[-2] = first_clustering_storing_output
                       
     ##                 clusteringInfData, executionTime, totCountImages = videoAnalysis(curTest, numberTests, infi, data_from_tests)               
                      
                      pre_dir = mtsVideoPath  
                      
                      #%%
                      
                      ## IFVP = 'C:/Research/DataSequence_'
                      folder = IFVP + str(sequence_name) + '/'
                          
                      ##locationMP4_file = 'FilesFor_6_8_X1_'
                      
                      name_folder = locationMP4_file + str(sequence_name)
                      
                      pre_dirInit = "C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\"
                       
                      path_in = pre_dirInit + "test_" + "00" + str(sequence_name) + ".avi"    ## .mts 
                      
                      mp4_initDir = os.path.join(mp4VideoFile + name_folder + "\\") 
                      
                      new_mp = ""
                      
                      if " " in mp4_initDir or "\t" in mp4_initDir or "\n" in mp4_initDir or "\r" in mp4_initDir:
                      
                          for x in mp4_initDir:
                              if x != " " and x != "\t" and x != "\n" and x != "\r":
                                  new_mp += x
                          mp4_initDir = new_mp
                          
                          mp4_initDir = mp4_initDir[1:]
                          
                          mps = mp4_initDir.split('\\')
                          addFile = mps[-2]
                          mx = addFile.split(':')
                          
                          mx_n = ""
                          
                          for m in mx:
                              if len(m) != 0:
                                  mx_n += m            
                           
                          mps[-2] = mx_n
                          
                          mp4_initDir_new = ""
                          
                          for x in mps[:-1]:
                          
                              mp4_initDir_new += x + '\\'
                              
                          mp4_initDir = mp4_initDir_new
                      
                      i = 1
                  
                      while True:
                          if os.path.exists(mp4_initDir):
                              mp4_initDir_p = mp4_initDir[:-1]
                              mp4_initDir_p_x = mp4_initDir_p[:-1]
                              mp4_initDir_p = mp4_initDir_p_x + str(i)
                              mp4_initDir = mp4_initDir_p + "\\"
                              
                              i += 1
                          else:
                              break
                      
                      
                      os.mkdir(mp4_initDir) 
                      
                      seqnew = ""
                      
                      for seqn in str(sequence_name):
                          if seqn != " " and seqn != "\r" and seqn != "\n" and seqn != "\t" and seqn != ":":
                              seqnew += seqn
                              
                      sequence_name_str = seqnew
                      sequence_name = int(sequence_name_str) 
                      
                      path_out = mp4_initDir + "test_" + "00" + str(sequence_name) + '.mp4'
                      path_out_mp4 = pre_dirInit + "test_" + "00" + str(sequence_name) + ".mp4"    ## .mts     
                  
                      
                      def pairedNumber(n1, div):
                           
                          if not (n1%2 == 0): 
                              a = int(n1/2)
                               
                              if n1 > a*2: 
                                  n2 = a*2
                              else: 
                                  n2 = n1       
                              
                              return n2 
                          return n1    
                       
                      
                      def loadMP4_file(path_in, path_out):
                                  name_in = "test_" + "00" + str(sequence_name) + ".avi"
                                  name_out = "test_" + "00" + str(sequence_name) + ".mp4"
                                  cmd = 'ffmpeg -i  ""C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/' + name_in + "" + ' "" C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/' + name_out + ""
                                  print("ffmpeg command: " + cmd)
                                  
                                  cmd_arr = shlex.split(cmd)
                                  cmd_arr = np.array([cmd_arr])
                                  cmd_arr = np.delete(cmd_arr, 3)
                                  cmd = np.array([cmd_arr])[0].tolist() 
                                  
                                  subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=False)
                                  dir_out = path_out            
                                  time.sleep(5) 
                                  return dir_out 
                      
                      if os.path.isdir(pre_dir+name_folder): 
                          print("Directory already exists !!!")
                      else: 
                      
                          newPath = os.path.join(pre_dir, name_folder)   
                          
                          new_path_n = ""
                          
                          for n in newPath:
                              if n != " " and n != "\t" and n != "\r" and n != "\n" and n != "\\":
                                  new_path_n += n
                          
                          new_path_n = new_path_n[1:]
                          new_path_s = new_path_n.split('/')
                          
                          sp_n = ""
                          
                          for ind_sp, sp in enumerate(new_path_s):
                              if ind_sp < len(new_path_s)-1:
                                  sp_n += sp + '/'
                          
                          rem = new_path_s[-1]
                          
                          remx = rem[1:-3] + rem[-1]
                          sp_n += remx
                          newPath = sp_n 
                          
                          j = -1
                          cod = 0
                          
                          while True:
                              if '/' in newPath:
                                  cod = 1
                              elif "\\" in newPath:
                                  cod = 2
                              
                              if cod == 1:
                                  if os.path.exists(newPath + '/'):
                                      j += 1
                                      
                                      if newPath[-2] == '_':
                                          newPath = newPath[:-2] 
                                      if newPath[-3] == '_':
                                          newPath = newPath[:-3] 
                                          
                                      newPath = newPath + "_" + str(j)
                                      print("New Path: "+ newPath)
                                  else:
                                      break  
                              elif cod == 2:
                                  if os.path.exists(newPath + "\\"):
                                      j += 1
                                      
                                      if newPath[-2] == '_':
                                          newPath = newPath[:-2] 
                                      if newPath[-3] == '_':
                                          newPath = newPath[:-3] 
                                          
                                      newPath = newPath + "_" + str(j)
                                      print("New Path: "+ newPath)
                                  else:
                                      
                                      break            
                          
                          if newPath[0] != 'C':
                              newPath = "C" + newPath
                              
                          os.mkdir(newPath)  
                          
                          print("Directory created")
                       
                           
                          if os.path.isfile('path_out' + '/' + str(sequence_name) + '.mp4'):
                              print("MP4 File already exists inside directory") 
                          else:
                              print(path_in)
                              print(path_out)
                              src_dir = loadMP4_file(path_in, path_out_mp4) 
                              print("path_out_mp4: \t" + path_out_mp4)
                              print("path_out: \t" + path_out)
                              time.sleep(100)            
                              os.rename(path_out_mp4, path_out) 
                              while(os.path.exists(path_out) == False): 
                                  time.sleep(5)
                              print("MP4 file loaded")          
                          
                          time.sleep(100)    
                          
                          src_dir = path_out 
                          
                      output_video_filenames, frame_rate, n_images = sepVideosFromWholeOne(True, infi[5] + infi[7] + "\\" + "test_000.mp4")    ## (False, None)
                     
                      if output_video_filenames is None:
                          clusteringInfData, executionTime, totCountImages = videoAnalysis_single(infi)
                          datax = (clusteringInfData, executionTime, totCountImages)  
                          gui_show_results(clusteringInfData, executionTime, totCountImages) 
                         
                          data_to_save.append(datax)  
                          break  
                      else:
                          print("Going to execute a software for only a video, but analysing each core moment")
                          ok = False
                         ## (curTest, numberTests, tupleForProcessing, data_from_tests, direcPythonFile, filename_output_video):
                          
                          curTest = 0
                          limToCompare = 50
                          numberTests = len(output_video_filenames)  
                         
                          while True:
                              
                                         if os.path.isfile('temp_numberTests.txt'):
                                             numberTests = read_number_tests_from_file()
                              
                             ##             if ind_inf != len(infx) - 1: 
                                         if curTest < numberTests - 1:
                                             
                                             print("Inside")  
                                              
                                             
                                             if '/' in first_clustering_storing_output:
                                                 x = 1
                                             elif "\\" in first_clustering_storing_output:
                                                 x = 2
                                               
                                             if os.path.exists(first_clustering_storing_output):
                                                 if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                                                     
                                                     first_clustering_storing_output = first_clustering_storing_output[:-1]
                                                     first_clustering_storing_output += '_' + "0"
                                                     
                                                     ind_newFolder = 1  
                                                     
                                                     if x == 1:
                                                         if not os.path.exists(first_clustering_storing_output + '/'):
                                                             os.mkdir(first_clustering_storing_output + '/')
                                                         while os.path.exists(first_clustering_storing_output + '/'):                                               
                                                             first_clustering_storing_output[-1] = str(ind_newFolder)
                                                             ind_newFolder += 1
                                                         else:
                                                             os.mkdir(first_clustering_storing_output + '/')
                                                         
                                                     elif x == 2:
                                                         if not os.path.exists(first_clustering_storing_output + "\\"):
                                                             os.mkdir(first_clustering_storing_output + "\\")
                                                         while os.path.exists(first_clustering_storing_output + "\\"):
                                                             first_clustering_storing_output[-1] = str(ind_newFolder)
                                                             ind_newFolder += 1 
                                                         else:
                                                             os.mkdir(first_clustering_storing_output + "\\")

                                                 infi[-2] = first_clustering_storing_output
                                            
                                             print("Length for infi before: " + str(len(infi)))
                                             print(infi)
                                             print("------------------------")
                                             
                                             if len(infi) == 14:
                                             
                                                 infis = infi[:-4]
                                                 infis.append(infi[-2])
                                                 infis.append(infi[-1])
                                                 infi = infis
                                             
                                                 print("Infi: ")
                                                 print(infi)
                                                 print("Length for infi: " + str(len(infi)))                                          
                                           
                                             limToCompare = np.min(np.array([n_images]))
                                             print("Limite to compare: " + str(limToCompare))
                                             data_from_tests = videoAnalysisMultipleVideosWithinOne(curTest, numberTests, infi, data_from_tests, this_dir, output_video_filenames[curTest], frame_rate, limToCompare)
                                               
                                             print("Length for data_from_tests: " + str(len(data_from_tests)))
                                             
                                             if len(data_from_tests) > 0: 
                                                 for d in data_from_tests:
                                                     print("Data: ")
                                                     for x in d:  
                                                         print(str(x) + " \t")   
                                             
                                             curTest += 1  
                ##                             infx += 1 
                                             
                                             if curTest == numberTests-1:
                                                print("Turn")
                                         #       break
                                             
                                         else:
                                             
                                             print("Passed")
                                             
                                             print("----------------------------")
                                             print("----------------------------")
                                             print("----------------------------")
                                             print("----------------------------")
                                             print("----------------------------")
                                          
                                             if '/' in first_clustering_storing_output:
                                                 x = 1
                                             elif "\\" in first_clustering_storing_output:
                                                 x = 2
                                                  
                                             if os.path.exists(first_clustering_storing_output):
                                                 if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                                                     
                                                     first_clustering_storing_output = first_clustering_storing_output[:-1]
                                                     first_clustering_storing_output += '_' + "0"
                                                     
                                                     ind_newFolder = 1 
                                                       
                                                     if x == 1:
                                                         if not os.path.exists(first_clustering_storing_output + '/'):
                                                             os.mkdir(first_clustering_storing_output + '/')
                                                         while os.path.exists(first_clustering_storing_output + '/'):                                               
                                                             first_clustering_storing_output[-1] = str(ind_newFolder)
                                                             ind_newFolder += 1
                                                         else:
                                                             os.mkdir(first_clustering_storing_output + '/')
                                                          
                                                         
                                                     elif x == 2:
                                                         if not os.path.exists(first_clustering_storing_output + "\\"):
                                                             os.mkdir(first_clustering_storing_output + "\\")
                                                         while os.path.exists(first_clustering_storing_output + "\\"):
                                                             first_clustering_storing_output[-1] = str(ind_newFolder)
                                                             ind_newFolder += 1  
                                                         else:
                                                             os.mkdir(first_clustering_storing_output + "\\")

                                                 infi[-2] = first_clustering_storing_output
                                                 
                                       ##      print("Current test here: " + str(curTest))
                                       ##      print("Number of tests: " + str(numberTests))
                                             print("Length here: " + str(len(data_from_tests)))                                            
                                             
                                             if len(infi) == 14:
                                                 
                                                 infis = infi[:-4]
                                                 infis.append(infi[-2])
                                                 infis.append(infi[-1])
                                                 infi = infis
                                                 
                                                 for d in data_from_tests:
                                                     print(d)
                                            
                                             ok = True
                                                 
                                             clustering_output = videoAnalysisMultipleVideosWithinOne(curTest, numberTests, infi, data_from_tests, this_dir, output_video_filenames[curTest], frame_rate, limToCompare)
                                             
                                             print("A") 
                                               
                                             if clustering_output is not None:
                                                  # for ind_c, c in enumerate(clustering_output):  
                                                     
                                                  #     print("Iterating clustering output: " + str(int(ind_c)))
                                                  #     if ind_c == 0:
                                                  #         data_to_save = []   
                                                         
                                                  #     datax = c
                                                  if True:
                                                      
                                                      print("Clustering output: ")
                                                      print(clustering_output)
                                             
                                                      data_to_save.append(clustering_output)
                                                 
                                                  print("B") 
                                                 
                                                  completed = True
                                             else:
                                                  print("Error generating results. \n Clustering analysis finishes before reaches the end. \n Work around possible issues with for cycles")
                                             
                                             if completed:
                                                  completed = False 
                                                  break
                                         
                                         
                                         if ok:
                                             
                                             print("C")     
                                             break
                                     
                                 
                          print("D")
                                     
                          window.close() 
                              
                   #          counterTest += 1 
               ##              print("Waiting to see what happens ...")
             ##                cv2.waitKey(0)

                          print("Data:")
                          for d in data_to_save:
                                 for x in d:
                                     print(x)   
                                     
                          if not isinstance(data_to_save[0], int) and not isinstance(data_to_save[0], float):
                             
                              print("Length for data to save: " + str(len(data_to_save[0])))  
                          
                          
                              if len(data_to_save[0]) > 0: 
                                  
                                   if len(data_to_save) != 3: 
                                       print("Check length data ...")
                                   else:                                       
                         
                                   # for ind_data, data in enumerate(data_to_save):
                                       
                                   #         print(data)
                                   #         print("Length of Data after: " + str(len(data))) 
                                          
                                          
                                   #         if len(data) == 3:
                                                 
                                               clusteringRes, execTime, numberImg = data_to_save       
                                                         
                                               gui_show_results(clusteringRes, execTime, numberImg)
                                               
                                               # if ind_data != len(data_to_save)-1:
                                               #     print(" -- Showing GUI with results for test number " + str(ind_data+1))
                                               #     time.sleep(5)
                                               # else:
                                               #     print("Terminating ...")
                                         
                                              
                              else:
                                   print("Output results not available !!!")                  
                     
                     
                                     ###################
                  
                      
                  #           break 
                          
              window.close()
             
       #       break
              
        else:
            
                ind_inf = 0
        
        
        
      #       if numberTests == 1:
                  
      #           if curTest == 0:
      #               infx = []
      #               data_to_save = []
                     
      #    ##       decisorLevel = 0
                
      #           [decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile] = infi
                
                
      #           infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile))
                
      # ##          nextTest = countdown_timer_display(time_bet_tests)    ## in minutes         
      #           nextTest = True
      #           print("Ready for next test: " + str(nextTest))
                
 ##           else:  
    
                startTime = time.time()             
                
               
                data_from_tests = []
                data_to_save = []
               
          #      for ind_inf, infi in enumerate(infx):
              
                if True:
                    
                    layout = [[sg.ProgressBar(1000, orientation='h', size = (20,20), key = 'progressbar')]]
        
                    window = sg.Window('Processing ...', layout)
                   
                    while True: 
                        
                        event, values = window.read(timeout=1000)
                        
                        if event == sg.WIN_CLOSED:
                            break
                         
                        while True:
                            
                            if True:      
                                
                                again = True
                                
                                dir_configs_path = config_dirs[ind_inf]
                                
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
                                
                                [decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile] = configs_list
                                infi =[decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile]
                                
                                infin = [] 
                                
                                for indc, indf in enumerate(infi):
                                    indfn = ""
                           
                                    for ifs in indf:
                                        if (ifs is not " ") and (ifs is not "\t") and (ifs is not "\n") and (ifs is not "\r"):
                                            indfn += ifs
                                            
                                    indfn = indfn[1:]
                                    
                                    if indc == 0 or indc == 2:
                                            indfn = int(indfn)
                                            
                                    infin.append(indfn)
                                    
                                    print(str(indc) + " - " + str(indfn))
                                           
                                
                                infi = infin  
                            
                            infx.append((decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile))
                            infi = (decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile)
                            
                            print("Looking for info " + str(ind_inf) + " ...")
                            print("Length infx: " + str(len(infx)))
                            print("Current test: " + str(curTest))
                            print("Number tests: " + str(numberTests))
                              
                            time.sleep(10)
                              
                ##             if ind_inf != len(infx) - 1: 
                            if curTest < numberTests - 1:
                                
                                print("Inside")  
                                
                                
                                if '/' in first_clustering_storing_output:
                                    x = 1
                                elif "\\" in first_clustering_storing_output:
                                    x = 2
                                  
                                if os.path.exists(first_clustering_storing_output):
                                    if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                                        
                                        first_clustering_storing_output = first_clustering_storing_output[:-1]
                                        first_clustering_storing_output += '_' + "0"
                                        
                                        ind_newFolder = 1 
                                        
                                        if x == 1:
                                            if not os.path.exists(first_clustering_storing_output + '/'):
                                                os.mkdir(first_clustering_storing_output + '/')
                                            while os.path.exists(first_clustering_storing_output + '/'):                                               
                                                first_clustering_storing_output[-1] = str(ind_newFolder)
                                                ind_newFolder += 1
                                            else:
                                                os.mkdir(first_clustering_storing_output + '/')
                                            
                                        elif x == 2:
                                            if not os.path.exists(first_clustering_storing_output + "\\"):
                                                os.mkdir(first_clustering_storing_output + "\\")
                                            while os.path.exists(first_clustering_storing_output + "\\"):
                                                first_clustering_storing_output[-1] = str(ind_newFolder)
                                                ind_newFolder += 1 
                                            else:
                                                os.mkdir(first_clustering_storing_output + "\\")

                                    infi[-2] = first_clustering_storing_output
                    
                                data_from_tests = videoAnalysis(curTest, numberTests, infi, data_from_tests) 
                                  
                                print("Length for data_from_tests: " + str(len(data_from_tests)))
                                
                                if len(data_from_tests) > 0: 
                                    for d in data_from_tests:
                                        print("Data: ")
                                        for x in d:  
                                            print(str(x) + " \t")  
                                
                                curTest += 1  
   ##                             infx += 1 
                                
                                if curTest == numberTests-1:
                                   print("Turn")
                            #       break
                                
                            else:
                                
                                print("Passed")
                                
                                print("----------------------------")
                                print("----------------------------")
                                print("----------------------------")
                                print("----------------------------")
                                print("----------------------------")
                             
                                if '/' in first_clustering_storing_output:
                                    x = 1
                                elif "\\" in first_clustering_storing_output:
                                    x = 2
                                     
                                if os.path.exists(first_clustering_storing_output):
                                    if first_clustering_storing_output[-1] == '/' or first_clustering_storing_output[-1] == '//' or first_clustering_storing_output[-1] == "\\":
                                        
                                        first_clustering_storing_output = first_clustering_storing_output[:-1]
                                        first_clustering_storing_output += '_' + "0"
                                        
                                        ind_newFolder = 1 
                                          
                                        if x == 1:
                                            if not os.path.exists(first_clustering_storing_output + '/'):
                                                os.mkdir(first_clustering_storing_output + '/')
                                            while os.path.exists(first_clustering_storing_output + '/'):                                               
                                                first_clustering_storing_output[-1] = str(ind_newFolder)
                                                ind_newFolder += 1
                                            else:
                                                os.mkdir(first_clustering_storing_output + '/')
                                             
                                            
                                        elif x == 2:
                                            if not os.path.exists(first_clustering_storing_output + "\\"):
                                                os.mkdir(first_clustering_storing_output + "\\")
                                            while os.path.exists(first_clustering_storing_output + "\\"):
                                                first_clustering_storing_output[-1] = str(ind_newFolder)
                                                ind_newFolder += 1  
                                            else:
                                                os.mkdir(first_clustering_storing_output + "\\")

                                    infi[-2] = first_clustering_storing_output
                                    
                          ##      print("Current test here: " + str(curTest))
                          ##      print("Number of tests: " + str(numberTests))
                                print("Length here: " + str(len(data_from_tests)))
                                    
                                clustering_output = videoAnalysis(curTest, numberTests, infi, data_from_tests) 
                                
                                print("A")
                                
                                
                                 
                                if clustering_output is not None:
                                    if True:
                                        
                                        print("Clustering output: ")
                                        print(clustering_output)
                               
                                        data_to_save.append(clustering_output) 
                                    
                                    print("B")
                                    
                                    completed = True
                                else:
                                    print("Error generating results. \n Clustering analysis finishes before reaches the end. \n Work around possible issues with for cycles")
                                
                                break
                            
                        
                            print("C")     
                                
                                
                        
                            break
                        
                        if completed:
                            
                            break
                    
                    print("D")
                        
                    window.close() 
                 
      #          counterTest += 1 
  ##              print("Waiting to see what happens ...")
##                cv2.waitKey(0)

                print("Data:")
                for d in data_to_save:
                    for x in d:
                        print(x)                  
                
                print("Length for data to save: " + str(len(data_to_save)))  
                
                if not isinstance(data_to_save[0], int) and not isinstance(data_to_save[0], float):
                   
                    print("Length for data to save: " + str(len(data_to_save[0])))  
                
                
                    if len(data_to_save[0]) > 0: 
                        
                         if len(data_to_save) != 3: 
                             print("Check length data ...")
                         else:                                       
               
                         # for ind_data, data in enumerate(data_to_save):
                             
                         #         print(data)
                         #         print("Length of Data after: " + str(len(data))) 
                                
                                
                         #         if len(data) == 3:
                                       
                                     clusteringRes, execTime, numberImg = data_to_save       
                                               
                                     gui_show_results(clusteringRes, execTime, numberImg)
                                     
                                     # if ind_data != len(data_to_save)-1:
                                     #     print(" -- Showing GUI with results for test number " + str(ind_data+1))
                                     #     time.sleep(5)
                                     # else:
                                     #     print("Terminating ...")
                               
                                    
                    else:
                         print("Output results not available !!!")                  
           
        
                        ###################
        
                executionTime = (time.time() - startTime)
                print('Whole execution time in seconds: ' + str(executionTime))
                       
         
     #           break 
        
        
            
## post_proc_pfs_only()