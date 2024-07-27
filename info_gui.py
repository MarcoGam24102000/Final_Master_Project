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
    
    back = False
    
    def results_analysis_gui():
        layout = [
            [sg.Text('Results Analysis', font=('Helvetica', 16), justification='center')],
            [sg.Text('By analyzing the graphs on the previous GUI, we can conclude about how many clusters were actually formed. \n'
                     'We can also assess whether the samples within a cluster are more or less separate from each other by \n'
                     'observing the distance between them and the cluster center. In a single graph, the samples of all the clusters \n'
                     'are represented by different colors. Additionally, alternative representation techniques could be used instead \n'
                     'of PCA, such as t-SNE or UMAP.', font=('Helvetica', 12))],
            [sg.Button('Back')]
        ]
    
        window = sg.Window('Results Analysis', layout)
    
        while True:
            event, values = window.read()
    
            if event == sg.WINDOW_CLOSED or event == 'Back':
                break
    
        window.close()
    
    def machine_learning_techniques_gui():
        k_means_description = (
            "k-Means Clustering is an unsupervised machine learning algorithm used for clustering data. \n"
            "It partitions the data into k clusters, where each data point belongs to the cluster with the \n"
            "nearest mean. The algorithm iteratively refines the clusters until convergence."
        )
    
        pca_description = (
            "Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform \n"
            "high-dimensional data into a lower-dimensional space. It identifies the principal components \n"
            "that capture the maximum variance in the data, allowing for a more concise representation."
        )
    
        layout = [
            [sg.Text('Machine Learning Techniques', font=('Helvetica', 16), justification='center')],
            [sg.Text('Many machine learning techniques were used, described next:', font=('Helvetica', 12))],
            [sg.Text('k-Means Clustering -', font=('Helvetica', 12, 'bold'))],
            [sg.Text(k_means_description, font=('Helvetica', 12))],
            [sg.Text('PCA -', font=('Helvetica', 12, 'bold'))],
            [sg.Text(pca_description, font=('Helvetica', 12))],
            [sg.Button('Back')]
        ]
    
        window = sg.Window('Machine Learning Techniques', layout)
    
        while True:
            event, values = window.read()
    
            if event == sg.WINDOW_CLOSED or event == 'Back':
                break
    
        window.close()
    
    def image_processing_gui():
        layout = [
            [sg.Text('Image Processing', font=('Helvetica', 16), justification='center')],
            [sg.Text('Image Processing can be done as one task, or separately. The first one to get the features dataset and the second one to get the clustering results.\n Image Processing has many steps. Here is just a few of them: ', font=('Helvetica', 12))],
            [sg.Listbox(values=["Shape detection of Petri dish",
                                "Division of bacterial matter, at the image level",
                                "Study of each one of the parts, by applying filters"],
                        size=(60, 5), key='topics_list', font=('Helvetica', 12))],
            [sg.Button('Back')]
        ]
    
        window = sg.Window('Image Processing', layout)
    
        while True:
            event, values = window.read()
    
            if event == sg.WINDOW_CLOSED or event == 'Back':
                break
    
        window.close()
    
    def image_acquisition_gui():
        layout = [
            [sg.Text('Image Acquisition', font=('Helvetica', 16), justification='center')],
            [sg.Text('Acquisition environment requires, at least, the following material:', font=('Helvetica', 12))],
            [sg.Listbox(values=["1 PC", "2 Ethernet cables", "1 Switch with at least 2 ports", "1 Camera",
                                "1 laser of 632 nm", "1 Petri dish", "Amount of bacterias for the experience"],
                        size=(40, 7), key='material_list', font=('Helvetica', 12))],
            [sg.Text('Make sure the laser is on the same plane as the camera, and the Petri dish with the bacterial matter is correctly placed', font=('Helvetica', 12))],
            [sg.Button('Back')]
        ]
    
        window = sg.Window('Image Acquisition', layout)
    
        while True:
            event, values = window.read()
    
            if event == sg.WINDOW_CLOSED or event == 'Back':
                break
    
        window.close()

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
        
        import os
        
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
                print(os.getcwd())
                
                # import sys
                # sys.exit()
                
                output_results_lastSet(os.getcwd() + '\GraphsOutput')
                
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
        
               [sg.Frame(layout = [[sg.Text("Marca: "), sg.Text("ASUS Vivobook")],
                                  [sg.Text("Processador: "), sg.Text("Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz, 1992 MHz")],
                                  [sg.Text("RAM: "), sg.Text("16 GB")],
                                  [sg.Text("Sistema Operativo: "), sg.Text("Windows 10 Pro")]], title='PC Info')]
               
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
        
               [sg.Frame(layout = [[sg.Text("Python version"), sg.Text("3.8.10")],
                                  [sg.Text("Spyder Version"), sg.Text("5.4.0")],
                                  [sg.Text("Pylon Version"), sg.Text("1.8.0")],
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
         
        
    def output_results_lastSet(folder_path):
        
        import os
        import PySimpleGUI as sg
        from PIL import Image
        import datetime
        
       # Get a list of all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
        # Create PySimpleGUI layout
        layout = [
            [sg.Text('Images in Folder:')],
            [sg.Listbox(values=image_files, size=(50, 10), key='-FILE LIST-', enable_events=True)],
            [sg.Button('Exit')]
        ]
    
        window = sg.Window('Info Menu - Output Results', layout, resizable=True)
    
        while True:
            event, values = window.read()
    
            if event in ('Exit', sg.WIN_CLOSED):
                break
            elif event == '-FILE LIST-':
                selected_filename = values['-FILE LIST-'][0]
                file_path = os.path.join(folder_path, selected_filename)
    
                # Create a new window to display the image
                image_layout = [
                    [sg.Text(f"Image: {selected_filename}")],
                    [sg.Image(filename=file_path)],
                    [sg.Button('Close')]
                ]
    
                image_window = sg.Window(f'Image Viewer - {selected_filename}', image_layout)
    
                while True:
                    image_event, _ = image_window.read()
    
                    if image_event in ('Close', sg.WIN_CLOSED):
                        break
    
                image_window.close()
    
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
                                   sg.Button("Results Analysis", key = "Results Analysis"),
                                   sg.Button("Others", key = "Others")]
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
          #      print("Going to describe briefly image acquisition procedure")
                image_acquisition_gui()
            elif event == 'Image Processing':           
        #        print("Going to describe briefly image processing procedure")
                image_processing_gui()
            elif event == 'Machine Learning Techniques':
   ##             print("Going to describe briefly machine learning techniques used")
               machine_learning_techniques_gui()
            elif event == 'Results Analysis':
          #      print("Going to provide a brief analysis of the output results")        
               results_analysis_gui()
            if event == "Others":
                main_info_page()
                break
            
            if event == "Back":
                back = True
                break
            
        window.close()  
    
    if not back:
    
        additionalnfo_page()
    
    # main_info_page()





    