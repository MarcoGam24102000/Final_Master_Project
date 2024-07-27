# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 21:56:20 2022

@author: marco
"""

import PySimpleGUI as sg
from pip._internal.operations import freeze
import webbrowser
from PIL import Image
import io
import cv2


def theory_info():

    def list_requirements(): 
    
        name_packages = []
        version_packages = []
        dependences_list = []
    
        x = freeze.freeze()
        
       
        for dependency in x:         
        
               if '==' in dependency:                            
                  
                   dep = dependency.split('==')             
                   
                   dependences_list.append(dep)   
                   
                   name_packages.append(dep[0])
                   version_packages.append(dep[1])
                   
        df=open('requirements.txt','w')
        
        for depend in dependences_list:
            df.write(depend[0] + '\t\t\t\t\t' + depend[1])
            df.write('\n')
            
        df.close()
        
        contents = open("requirements.txt","r")
        with open("requirements.html", "w") as e:
            for lines in contents.readlines():
                e.write("<pre>" + lines + "</pre> <br>\n")   
    
        webbrowser.open('requirements.html', new=2)  # open in new tab   
         
    
    
    def main_info_page():
        ## clustering results - textual ones
        ## 
        counterTimesBack = 0
        
        colx = sg.Column([ 
        
               [sg.Frame(layout = [[sg.Button("PC info", key="-PC-")],
                                  [sg.Button("Software and related packages", key="-Software-")],
                                  [sg.Button("Results from last set of tests", key="-Results-")],
                                  [sg.Button("Aditional info", key="-AditionalInfo-")]], title='Info')] 
               
        ])             
                
         
        layout= [[colx], [sg.Frame(layout=[[sg.Button('Back')]], title='Actions:')]]     
              
        window = sg.Window("Info Menu", layout, disable_close = True, finalize = True)
        
        
        while True:  
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            
            if event == '-PC-':
                PC_info_page()
                
            elif event == '-Software-': 
                software_versions_page()
                
            elif event == '-Results-':
                output_results_lastSet()
                
            elif event == '-AditionalInfo-':
                additionalnfo_page()
                
            elif event == 'Back':
                print("Main menu")
                
                if counterTimesBack == 1:     
                    print("Breaking ... ")
                    counterTimesBack = 0  
                    break
                else:    
                    print("Increment 1 time")
                    counterTimesBack += 1       
                
            
        window.close()     
        
        
         
    def PC_info_page():
        print("Going to present all the features related to PC")
        
        colx = sg.Column([ 
        
               [sg.Frame(layout = [[sg.Text("Marca: "), sg.Text("ASUS Zenbook")],
                                  [sg.Text("Processador: "), sg.Text("11th Gen Intel(R) Core(TM) i7-1165G7 - 2.8 GHz")],
                                  [sg.Text("RAM: "), sg.Text("16 GB")],
                                  [sg.Text("Sistema Operativo: "), sg.Text("Windows 11 Home")],
                                  [sg.Text("Windows: "), sg.Text("")]], title='PC Info')]
               
        ])              
                
         
        layout= [[colx], [sg.Frame(layout=[[sg.Button('Back')]], title='Actions:')]]     
              
        window = sg.Window("Info Menu - PC Info", layout, disable_close = True, finalize = True)    
          
        while True:   
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "Back":
                main_info_page()
                break
            
        window.close()  
        
        
        
    def software_versions_page():
        print("Going to present all the softwares and versions associated to them, as well as all the required python packages")
        
        colx = sg.Column([ 
        
               [sg.Frame(layout = [[sg.Text("Python version"), sg.Text("")],
                                  [sg.Text("Spyder Version"), sg.Text("")],
                                  [sg.Text("Pylon Version"), sg.Text("")],
                                  [sg.Button("Required packages", key = "Get list of requirements")]], title='Software Info')]
               
        ])             
                 
         
        layout= [[colx], [sg.Frame(layout=[[sg.Button('Back')]], title='Actions:')]]     
              
        window = sg.Window("Info Menu - Software Info", layout, disable_close = True, finalize = True)    
          
        while True:  
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "Back":
                main_info_page()
                break
            elif event == "Get list of requirements": 
                list_requirements()
            
        window.close()
         
        
    def output_results_lastSet():
        print("Going to output all the relevant information regarding to clustering results from the last set of texts")
        
        colx = sg.Column([ 
        
               [sg.Frame(layout = [[sg.Text(""), sg.Text("")],
                                  [sg.Text(""), sg.Text("")],
                                  [sg.Text(""), sg.Text("")],
                                  [sg.Text(""), sg.Text("")]], title='Output Results')]
               
        ])             
                
         
        layout= [[colx], [sg.Frame(layout=[[sg.Button('Back')]], title='Actions:')]]     
              
        window = sg.Window("Info Menu - Output Results", layout, disable_close = True, finalize = True)    
          
        while True:  
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "Back":
                main_info_page()
                break
            
        window.close()
    
    
    def additionalnfo_page():      ## if it is the first time, increase image size, else just use it 
        print("Going to show aditional info, from a theoretical point of view")
        
        im_diagram = cv2.imread('C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/general_block_diagram.png')
       
      
        
        print(str(len(im_diagram[0]))) 
        print(str(len(im_diagram)))
        print(im_diagram.shape)
        ## (3*len(im_diagram), 3*len(im_diagram[0])) 
        
       
        image = Image.open('C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/general_block_diagram.png')
        image.thumbnail((400, 25000))    ## (100,5000)
        bio = io.BytesIO()    
        image.save(bio, format="PNG")    
        colx = sg.Column([ 
        ## Include an image with a general block diagram
        ## Insert a button for each block (below which one of them)
        ## For each button, a new event with a new window with the a list of the main theoretical approaches 
        
               [sg.Frame(layout = [[sg.Image(data= bio.getvalue(), key = "-GEN_BLOCK_DIAGRAM-", size=(500,100))],
                                  [sg.Button("Image Acquisition", key = "Image Acquisition"),   
                                   sg.Button("Image Processing", key = "Image Processing"), 
                                   sg.Button("Machine Learning Techniques", key = "Machine Learning Techniques"), 
                                   sg.Button("Results Analysis", key = "Results Analysis")],
                                 ], title='Additional Info')] 
               
        ])     
    
    #      image = Image.open(dirResultsOutput + "distances_firstCluster.png")
    #      image.thumbnail(MAX_SIZE)
         
    #      bio = io.BytesIO()
    # ##     bio = io       ## .save(bio, format="PNG")
    #      image.save(bio, format="PNG")
    #      window['-DIST_FIRST_CLUSTER-'].update(data = bio.getvalue())    
           
        layout= [[colx], [sg.Frame(layout=[[sg.Button('Back')]], title='Actions:')]]     
              
        window = sg.Window("Info Menu - Additional Info", layout, resizable = True, disable_close = True, finalize = True) 
          
        while True:  
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            
            if event == 'Image Acquisition':
                print("Going to describe briefly image acquisition procedure")
            elif event == 'Image Processing':           
                print("Going to describe briefly image processing procedure")
            elif event == 'Machine Learning Techniques':
                print("Going to describe briefly machine learning techniques used")
            elif event == 'Results analysis':
                print("Going to provide a brief analysis of the output results")        
                
            if event == "Back":
                main_info_page()
                break
            
        window.close()    
    
    additionalnfo_page()
    
    # main_info_page()





    