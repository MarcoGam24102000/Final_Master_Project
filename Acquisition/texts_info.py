# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:38:52 2023

@author: Rui Pinto
"""

import os
from getCurrentDateAndTime import getDateTimeStrMarker

base_name = "tests_info" 

import shutil

def copy_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"File '{source_path}' copied to '{destination_path}' successfully.")
    except FileNotFoundError:
        print("Error: Source file not found.")
    except PermissionError:
        print("Error: Permission denied. Make sure you have read access to the source file.")
    except Exception as e:
        print(f"An error occurred while copying the file: {e}")   

def write_to_txt_file_tests_info(base_path, inf_tests):
    
    filename = base_name + getDateTimeStrMarker()
    
    # filename_path = base_path + filename
     
    # filename_path = os.path.join(filename_path)
    # os.mkdir(filename_path)
    
    number_tests = inf_tests[0]
    time_test_secs = inf_tests[1]
    time_bet_tests_secs = inf_tests[2]
    
    time_bet_tests_secs*=60
    
    
    with open(filename + ".txt", "w") as f:
        f.write("Tests details")
        f.write("\n")
        f.write("\n")
        f.write("Number of tests: \t\t\t" + str(number_tests))
        f.write("\n")
        f.write("Duration of each tests (secs.): \t" + str(time_test_secs))
        f.write("\n")
        f.write("Time between tests (secs.): \t\t" + str(time_bet_tests_secs))
    
    b_ind = 0
    
    for ind_b, b in enumerate(base_path):
        if b == '/':
            b_ind = ind_b           
    
    print("Base Path: " + str(base_path))
    
    path_dest = base_path[:(b_ind+1)] + filename + '.txt'
    
    copy_file(filename + ".txt", path_dest)
 
def read_configs_file(full_path):
    
    check_one = False
    check_two = False
    check_three = False
    check_four = False
    check_five = False
    check_six = False
    check_seven = False
    check_eight = False
    check_nine = False
    check_ten = False
    check_a = False
    check_b = False
    check_c = False
    
    again = True
    
    decLev = ""
    mainPath = ""
    sequenceName = ""
    destPath = ""
    mtsVideoPath = ""
    mp4VideoFilename = ""
    storageFolder = ""
    mp4FileLoc = "" 
    roiPath = ""
    firstClusteringStorage = ""
    pythonFile = "" 
    
    nice = False
    
    while again:
    
        with open(full_path) as f:
            contents = f.readlines()
            
            if contents:
                again = False
                
                for ind_c, c in enumerate(contents):
                    if ind_c == 0:
                       if "Configs" in c:
                           check_one = True
                           print("1") 
                    
                    if len(c) > 0:
                        if ind_c == 2 and check_one:
                            if "Decisor Level" in c:
                                check_two = True
                                print("2")
                                decLevLine = c.split("Decisor Level")                            
                                for d in decLevLine:
                                    if len(d) != 0:
                                        decLev = d
                                        break                    
                                    
                    if ind_c == 3 and check_two:
                        if "Main Path" in c:
                            check_three = True
                            print("3")
                            mainPathLine = c.split("Main Path")
                            for d in mainPathLine:
                                if len(d) != 0:
                                    mainPath = d
                                    break
                                
                    if ind_c == 4 and check_three:
                        if "Sequence Name" in c:
                            check_four = True
                            print("4")
                            sequenceNameLine = c.split("Sequence Name")
                            for d in sequenceNameLine:
                                if len(d) != 0:
                                    sequenceName = d
                                    break
                                
                    if ind_c == 5 and check_four:
                        if "Destination Path" in c:
                            check_five = True
                            print("5")
                            destPathLine = c.split("Destination Path")
                            for d in destPathLine:
                                if len(d) != 0:
                                    destPath = d
                                    break
                                
                    if ind_c == 6 and check_five:
                        if "MTS video file path" in c:
                            check_six = True
                            print("6")
                            mtsVideoPathLine = c.split("MTS video file path")
                            for d in mtsVideoPathLine:
                                if len(d) != 0:
                                    mtsVideoPath = d
                                    break
                                
                    if ind_c == 7 and check_six:
                        if "MP4 Video Filename" in c:
                            check_seven = True
                            print("7")
                            mp4VideoFilenameLine = c.split("MP4 Video Filename")
                            for d in mp4VideoFilenameLine:
                                if len(d) != 0:
                                    mp4VideoFilename = d
                                    break
                                
                    if ind_c == 8 and check_seven:
                        if "Storage folder for this sequence" in c:
                            check_eight = True
                            print("8")
                            storageFolderLine = c.split("Storage folder for this sequence")
                            for d in storageFolderLine:
                                if len(d) != 0:
                                    storageFolder = d
                                    break                                
                                
                    if ind_c == 9 and check_eight:
                        if "MP4 File Location" in c:
                            check_nine = True
                            print("9")
                            mp4FileLocLine = c.split("MP4 File Location")
                            for d in mp4FileLocLine:
                                if len(d) != 0:
                                    mp4FileLoc = d
                                    break
                                
                    if ind_c == 10 and check_nine:
                        if "ROI Path - First Stage":
                            check_ten = True
                            print("10")
                            roiPathLine = c.split("ROI Path - First Stage")
                            for d in roiPathLine:
                                if len(d) != 0:
                                    roiPath = d
                                    break
                                
                    if ind_c == 11 and check_ten:
                        if "ROI Path - Second Stage":
                            check_a = True
                            print("11")
                            roiPathSecLine = c.split("ROI Path - Second Stage")
                            for d in roiPathSecLine:
                                if len(d) != 0:
                                    roiPathSec = d
                                    break
                                
                    if ind_c == 12 and check_a:
                        if "First Clustering Storage Output":
                            check_b = True
                            print("12")
                            firstClusteringStorageLine = c.split("First Clustering Storage Output")
                            for d in firstClusteringStorageLine:
                                if len(d) != 0:
                                    firstClusteringStorage = d
                                    break
                                
                    if ind_c == 13 and check_b:
                        if "Python File Path":
                            check_c = True 
                            print("13")
                            pythonFileLine = c.split("Python File Path")
                            for d in pythonFileLine:
                                if len(d) != 0:
                                    pythonFile = d
                                    break
                    
                    if check_c:
                        if decLev and mainPath and sequenceName and destPath and mtsVideoPath and mp4VideoFilename and storageFolder and mp4FileLoc and roiPath and firstClusteringStorage and pythonFile:
                               print("Going forward")
                               nice = True
                               break
                        else:
                            print("Please check configs file content ...")
                                   
                
            else:
                print("Failed on reading .txt file")

    configs_list = []
    
    if nice:
        configs_list = [decLev, mainPath, sequenceName, destPath, mtsVideoPath, mp4VideoFilename, storageFolder, mp4FileLoc, roiPath, roiPathSec, firstClusteringStorage, pythonFile]
        
    return configs_list 

def read_from_txt_file_tests_info(full_path):
    
    skip_one = False
    skip_two = False
    skip_three = False
    skip_four = False
    skip_five = False 
    
    numberTests = 0
    dur_test = 0
    time_bet_tests = 0      
    
    data_tests = []
    again = True
    
    while again:
    
            with open(full_path) as f:
                contents = f.readlines()
            
            # for x in contents:
            #     print(x)
            
        
            
            if contents:
            
                for ind_c, c in enumerate(contents):
                    
            ##        print(str(ind_c) + " - " + str(c))
                    if ind_c == 0:
                        if "Tests details" in c:
                            skip_one = True
              ##              print("One")
                    if len(c) > 10:
                    #    print("Here")
                        # if ind_c == 2:
                        #     print("Here")
                        # if ind_c == 2 and skip_one:
                        #     if len(c) == 0:
                        #         skip_two = True 
                        #         print("Two")
                        if ind_c == 2 and skip_one: 
                     ##       print("Here")
                            
                            if "Number of tests" in c:
                               
                                c_s = c.split('Number of tests: ')                   
                                
                                numberTests_str = "" 
                                
                                for d in c_s:
                                    if len(d) != 0:
                                        numberTests_str = d
                                        break 
                                
                        #        print("Number Tests Str.: " + str(numberTests_str))
                        #        if numberTests_str.isdigit():
                                if True:
                         #           print("Here")
                                    numberTests = int(numberTests_str)
                                    skip_three = True
                                    print("Three")
                                    
                        if ind_c == 3 and skip_three:
                            if "Duration of each tests (secs.)" in c:
                                c_s = c.split('Duration of each tests (secs.): ')                   
                                
                                durTests_str = ""
                                
                                for d in c_s:
                                    if len(d) != 0:
                                        durTests_str = d
                                        break
                                
##                                if durTests_str.isdigit():
                            if True:
                                    dur_test = int(durTests_str)
                                    skip_four = True 
                                    print("Four")
                                    
                        if ind_c == 4 and skip_four:
                            if "Time between tests (secs.)" in c:
                                c_s = c.split('Time between tests (secs.): ')                   
                                
                                time_bet_tests_str = ""
                                
                                for d in c_s:
                                    if len(d) != 0:
                                        time_bet_tests_str = d
                                        break
                                
                #                if time_bet_tests_str.isdigit():
                            if True:
                                    t_new = ""
                                    for t in time_bet_tests_str:
                                        if (t != " ") and (t != "\t") and (t != "\r") and (t != "\n"):
                                            t_new += t
                                    
                                    if '.' in t_new:
                                        x_new = ""
                                        for x in t_new:
                                            if x != '.':
                                                x_new += x
                                            else:
                                                break
                                            
                                        t_new = x_new                           
                                            
                                    time_bet_tests_str = t_new
                                    time_bet_tests = int(time_bet_tests_str)
                                    skip_five = True
                                    print("Five")
                        
                if skip_five:
                    data_tests = [numberTests, dur_test, time_bet_tests]            
                    again = False
            else:
                print("Failed on reading .txt file")
    
    return data_tests 
            

## write_to_txt_file_tests_info("", [3, 5, 10])
    
## data_tests = read_from_txt_file_tests_info("C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\tests_info_080323_162313_.txt")
    
