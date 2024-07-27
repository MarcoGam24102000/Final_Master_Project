# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:04:55 2023

@author: Rui Pinto
"""

from common import splitfn
import glob
import cv2
import os
import numpy as np

import PySimpleGUI as sg_py
import PySimpleGUIWeb as sg
from getCurrentDateAndTime import getDateTimeStrMarker 

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

def gui_png_file_part(): 
     
     import PySimpleGUI as sg
     
     again = True
     
     while again == True:
     
         windowx = sg.Window('Choose path to png file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
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
         
     base_dir_png = ''
     
     for ind_d, d in enumerate(dir_png_parts):
          
         if ind_d < len(dir_png_parts) - 1: 
             base_dir_png += d + '/'
      
     png_info = [base_dir_png, png_filename]
      
     return dir_png_path
 

def write_data_to_csv_file_part(data_lab, metadata_csv_filename):
    
    with open(metadata_csv_filename, 'a', encoding='utf-8') as f:
        
        for data in data_lab: 
            line = ', '.join(data)
            f.write(line + '\n')
            
            print("One more line written to csv file ...") 
            
 
def write_to_csv_calib_params_database_part(calib_params, csv_database_filename, counter):
    
    if counter == 0:
        headers = ["Fx", "Fy", "PPx", "PPy"]
    
    ## Write calib_params line to csv file   
    
    with open(csv_database_filename, 'a', encoding='utf-8') as f:
        line = ', '.join(calib_params)
        f.write(line + '\n')
        
        print("One more set of calibration parameters written to csv file ...")       
        
 
def ask_user_metadata_filename_part():
    
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
    
    
def write_params_to_metadata_file_part(params_calib, metadata_csv_filename, model_name, counter_for_params, width, height): 
    
   
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
    
    write_data_to_csv_file_part(data_lab, metadata_csv_filename)

def calib_camera_part(continuous, model_name, dir_png_path):
    
    counter_for_params = 0 

    def exec_script(dir_png_path):  

        import sys 
        import getopt
        from glob import glob
        
        if continuous == False:
            dir_png_path = gui_png_file_part()   
        
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
            stop = False
            
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
                            
                            if j == max_window_j-1 and i == max_window_i-1:
                                stop = True
            
            if stop == False:
                
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
            else:
                return None

        threads_num = int(args.get('--threads'))
         
        print("Number of threads: " + str(threads_num))
        
    ##    if threads_num <= 1:
        if True:
            chessboards = [processImage(fn) for fn in img_names if processImage(fn) is not None]
        # else:
        #     print("Run with %d threads..." % threads_num)
        #     from multiprocessing.dummy import Pool as ThreadPool
        #     pool = ThreadPool(threads_num)
        #     chessboards = pool.map(processImage, img_names) 
        
        end = False
        
        for ches in chessboards:
            if ches is None:
                end = True
        
        if end == False:       
             
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
                    _path, name, _ext = splitfn(fn)                                      ## splitfn
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
        else:
            return None            
    

    tup = exec_script(dir_png_path)
    
    if tup is not None:

        newcameramtx, dir_png_path, width, height = tup 
        
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
                metadata_title_name = ask_user_metadata_filename_part() 
            
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
               
            code_ret = -1        
            
            if continuous == True: 
                
                # width_standard = 1920
                # height_standard = 1080
                
                width_standard = width
                height_standard = height
               
                write_params_to_metadata_file_part(params_calib, metadata_csv_filename, model_name, counter_for_params, width_standard, height_standard)
                
                csv_database_filename = "database_calib.csv" 
                
                write_to_csv_calib_params_database_part(params_calib, csv_database_filename, counter_for_params)
                
                counter_for_params += 1   
            else: 
                code_ret = gui_calib_params(fx_calib, fy_calib, ppx_calib, ppy_calib)  
            
        return code_ret   
    else:
        print("Error - Calibration not done. Moving forward to the next one")
        code_ret = -2
        return code_ret  