# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:27:33 2023

@author: Rui Pinto
"""

import math
import os
import cv2 
import imageio
import shutil
import sys 
import time
import subprocess
import shlex 
import numpy as np  
import pandas as pd
import xlsxwriter
import configparser
import PySimpleGUI as sg
from PIL import Image
import io
from info_gui import theory_info
import webbrowser 

import matplotlib.pyplot as plt  
import skimage.morphology as morphology
import scipy.ndimage as ndi 
from skimage.color import label2rgb  
from skimage.segmentation import watershed 
## from _watershed import watershed
from sklearn.model_selection import train_test_split   
from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA  
from scipy.spatial.distance import pdist, squareform  
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import csv
from spyder_kernels.utils.iofuncs import load_dictionary  
from spyder_kernels.utils.iofuncs import save_dictionary 
import pickle
import tarfile
from sewar.full_ref import mse, rmse, _rmse_sw_single, rmse_sw, psnr, _uqi_single, uqi, ssim, ergas, scc, rase, sam, msssim, vifp, psnrb;
import matplotlib.pyplot as plt  
import skimage.morphology as morphology
import scipy.ndimage as ndi 
from skimage.color import label2rgb  
from skimage.segmentation import watershed 
## from _watershed import watershed
from sklearn.model_selection import train_test_split   
from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA  
from scipy.spatial.distance import pdist, squareform  
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import csv
from spyder_kernels.utils.iofuncs import load_dictionary  
from spyder_kernels.utils.iofuncs import save_dictionary 
import pickle
import tarfile
from sewar.full_ref import mse, rmse, _rmse_sw_single, rmse_sw, psnr, _uqi_single, uqi, ssim, ergas, scc, rase, sam, msssim, vifp, psnrb;


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




### Reduzir o tamanho da janela e ver possiblidade de ajustar com o rato para ficar maior ou menor (autoajuste)
def gui_show_results(clusteringRes, execTime, numberImg):
     
    execTime = int(execTime)
    
    print("Gui for results")
    
    clusteringResIn = clusteringRes[0][0]
    clusteringRes = clusteringResIn
    
    MAX_SIZE = (200, 200)
    
    print("Clustering res length: " + str(len(clusteringRes)))
    
    classAFurther = clusteringRes[0]
    print("A")
    classBFurther = clusteringRes[1]
    print("B")
    nclusters = clusteringRes[2]
    print("C")
    number_recommended_clusters = clusteringRes[3]
    print("D")
    remainingMetricsToClustering = clusteringRes[4]  
    print("E") 
    
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
         
def videoAnalysisMultipleVideosWithinOneClusteringPart(secRoundNewListMetrics, number_metrics,  lenMax, infi, roi_set, count, test_size = 0.2):
    
    startTime = time.time()  
    M = 0
    roi_path, roi_bef, roi_after = roi_set 
    
    decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile = infi
    
    parent_dir = dest_path
    
    # Suppress the deprecation warning
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    
    print("secRoundNewListMetrics: ")
    print(secRoundNewListMetrics)
    
   
    
    secRoundNewListMetricsArr = np.array([secRoundNewListMetrics])[0]
    
    clustering_output = []
    data_results = [] 
    
    print("\n\nData here on spot: \n")
    print(secRoundNewListMetricsArr)
    print("\n\n")
    
    # import sys
    # sys.exit()
    
 #   sys.exit()
 
    try:
        print("A")
        resT = np.array([ secRoundNewListMetricsArr[:, int(number_metrics)-1]]).T 
    except Exception as e:
        print(secRoundNewListMetricsArr)
        print(f"An error occurred: {e}")
        sys.exit()
        
    
    #%%
    
    train_data, test_data, labels_train_data, labels_test_data = train_test_split(secRoundNewListMetricsArr[:, 0:int(number_metrics)-1], resT, test_size =test_size, random_state = 42)
    treino_lenght = int((1-test_size)*lenMax)
    
    # Método Silhouette - análise para um nº variável de clusters   
    aux = 0 
    max_silhouette = 0
    silhouette_vector = [] 
    n_clusters3 = range(2, treino_lenght) 
    for j in n_clusters3:
        km =KMeans(n_clusters=j, max_iter=300, n_init=5).fit(train_data)
        labels3 = km.labels_
        silhouette_avg = silhouette_score(train_data, labels3)
        print("For n_clusters=", j, "The averegae silhouette_score is:", silhouette_avg)   
        aux = silhouette_avg 
        if aux > max_silhouette: 
            max_silhouette = aux
            number_recommended_clusters = j    
        silhouette_vector.append(silhouette_avg)  
        
    print("\n\nCom base no gráfico Elbow e no método Silhouette, é recomendável formar", number_recommended_clusters, "clusters!") 
    
    secRoundNewListMetrics = np.array([secRoundNewListMetrics])[0].T.tolist()
    trainListData = train_data.tolist() 
    nclusters = 6
    
    kmeans = KMeans(n_clusters=nclusters, max_iter=500, n_init=8).fit(trainListData) 
    Cluster_ID = kmeans.labels_ 
    centroides_A = kmeans.cluster_centers_   
    print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)
    
    Cluster_ID_transpose = np.array([np.array([Cluster_ID]).T[0:treino_lenght,0]])
    
    objetos_c1 = []
    objetos_c2 = []
    objetos_c3 = [] 
    objetos_c4 = [] 
    objetos_c5 = [] 
    objetos_c6 = [] 
    
    for i in range (0, len(Cluster_ID_transpose[0])):
        if Cluster_ID_transpose[0,i] == 0:
            objetos_c1.append(train_data[i, :])
            i_1 = i
        elif Cluster_ID_transpose[0,i] == 1:
            objetos_c2.append(train_data[i, :]) 
            i_2 = i
        elif Cluster_ID_transpose[0,i] == 2:
            objetos_c3.append(train_data[i, :]) 
            i_3 = i
        elif Cluster_ID_transpose[0,i] == 3:
            objetos_c4.append(train_data[i, :]) 
            i_4 = i
        elif Cluster_ID_transpose[0,i] == 4:
            objetos_c5.append(train_data[i, :])
            i_5 = i
        elif Cluster_ID_transpose[0,i] == 5:
            objetos_c6.append(train_data[i, :])  
            i_6 = i     
            
    list1 = list(zip(*objetos_c1)) 
    list2 = list(zip(*objetos_c2)) 
    list3 = list(zip(*objetos_c3)) 
    list4 = list(zip(*objetos_c4)) 
    list5 = list(zip(*objetos_c5)) 
    list6 = list(zip(*objetos_c6)) 
    
    print(" -- Lists of clusters generated")
    
    for l in list1:
        LenList_1 = len(l)
    for l in list2:
        LenList_2 = len(l)
    for l in list3:
        LenList_3 = len(l)
    for l in list4:
        LenList_4 = len(l)
    for l in list5:
        LenList_5 = len(l)
    for l in list6:
        LenList_6 = len(l)  
    
    print("Lists: ")
    print(list1)
    print(list2)
    print(list3)
    print(list4)
    print(list5)
    print(list6)  

#    sys.exit()                                   
    
    indForFolderClustering = []    
    
    list1FirstArr = np.array([np.array([list1])])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    print("\n\n ------- \nListArrToCompare: ")
    print(listArrToCompare)
    print("-------\n\n") 
    
    print("\n\n ------- \nList1FirstArray: ")
    print(list1FirstArr)
    print("-------\n\n")
    
    print("\n\n ------- \nsecRoundNewListMetricsArr: ")
    print(secRoundNewListMetricsArr)
    print("-------\n\n")
    
    print(len(list1FirstArr[0][0]))
    print("Iterating ...")
    
    singIndData = []
    
    for ind in range(0,len(list1FirstArr[0][0])):       ## 
        
        val = []
        
        for xi in list1FirstArr[0][0]:
            if ind < len(xi):
                val.append(xi[ind])
            
        print("First value: ")
        print(secRoundNewListMetricsArr[ind])
        
        print("Length:")
        print(len(secRoundNewListMetricsArr[ind]))
        
        new_strSecRound = ""
        secRoundList = []
        
        if len(secRoundNewListMetricsArr[ind]) == 1:
           print("String")
           new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
           print(new_strSecRound)
        else:
            for x in secRoundNewListMetricsArr[ind]:
                secRoundList.append(x)
            print("secRoundList: ")
            print(secRoundList)
            
      
        print("Value to compare to: ")
        print(val)
        print("list1 first array: ")
        print(list1FirstArr[0][0]) 
        
        if len(val) == len(secRoundList)-1:
            secRoundList = secRoundList[:-1]
        
   ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
        
        ind_data = []
        for x in range(len(val)):
            if val[x] == secRoundList[x]:
               ind_data.append(x) 
               
 ##       rub, ind_data = tupleIndiceImage
   #     ind_data = np.array([ind_data])
        
        print("Ind data for 1, for ind " + str(ind) + " : ")
        print(ind_data)
       
        # if len(ind_data[0]) == 1:
        #     print("Appending indice " + str(ind_data) + " to singIndData")
        #     singIndData.append(ind_data) 
        # else:
        #     if len(ind_data[0]) == 2:
        #         print("Not singular")
        #         singIndData.append(ind_data[0,0])
        #         singIndData.append(ind_data[0,1]) 
        
        print("Length ind data: " + str(len(ind_data)))
        
        if len(ind_data) > 0:
            
            for i in range(0,len(ind_data)):
            
                singIndData.append(ind_data[i])
    
            # ind_datax = ind_data[0]  
            # newDat = []
            
            # print("ind_data: ")
            # print(ind_data)
            
            # for i in ind_datax:
            #     newDat.append(i)
            
            # singIndData.append(newDat[0])
            # singIndData.append(newDat[1])
    
    newSingData = []
    
    print("Sing Ind Data:")
    print(singIndData)
    
    if len(singIndData) > 0:
    
        singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
    
        if True:
            for arrSin in singIndData:
                newSingData.append(int(arrSin))
        
            newSingIndArr = np.zeros((1,len(newSingData)))     
            newSingIndArr = np.array([newSingData]) 
            
            uniqueNewInd = np.array([np.unique(newSingIndArr)])
            
            if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
                print("All indices unique")
                 
                indForFolderClustering.append(uniqueNewInd)    
              
        else:
            print("Not equal at phase 2")  
            
   
    
    list2FirstArr = np.array([np.array([list2])])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,1]])
    
    print("\n\n ------- \nListArrToCompare: ")
    print(listArrToCompare)
    print("-------\n\n") 
    
    print("\n\n ------- \nList1FirstArray: ")
    print(list2FirstArr)
    print("-------\n\n")
    
    print("\n\n ------- \nsecRoundNewListMetricsArr: ")
    print(secRoundNewListMetricsArr)
    print("-------\n\n")
    
    print(len(list2FirstArr[0][0]))
    print("Iterating ...")
    
    singIndData = []
    
    for ind in range(0,len(list2FirstArr[0][0])):       ## 
        
        val = []
        
        for xi in list2FirstArr[0][0]:
            if ind < len(xi):
                val.append(xi[ind])
            
        print("First value: ")
        print(secRoundNewListMetricsArr[ind])
        
        print("Length:")
        print(len(secRoundNewListMetricsArr[ind]))
        
        new_strSecRound = ""
        secRoundList = []
        
        if len(secRoundNewListMetricsArr[ind]) == 1:
           print("String")
           new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
           print(new_strSecRound)
        else:
            for x in secRoundNewListMetricsArr[ind]:
                secRoundList.append(x)
            print("secRoundList: ")
            print(secRoundList)
            
      
        print("Value to compare to: ")
        print(val)
        print("list2 first array: ")
        print(list2FirstArr[0][0]) 
        
        if len(val) == len(secRoundList)-1:
            secRoundList = secRoundList[:-1]
        
   ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
        
        ind_data = []
        for x in range(len(val)):
            if val[x] == secRoundList[x]:
               ind_data.append(x) 
               
 ##       rub, ind_data = tupleIndiceImage
#        ind_data = np.array([ind_data])
        
        print("Ind data for 2, for ind " + str(ind) + " : ")
        print(ind_data)
       
        # if len(ind_data[0]) == 1:
        #     print("Appending indice " + str(ind_data) + " to singIndData")
        #     singIndData.append(ind_data) 
        # else:
        #     if len(ind_data[0]) == 2:
        #         print("Not singular")
        #         singIndData.append(ind_data[0,0])
        #         singIndData.append(ind_data[0,1])           
    
        print("Length ind data: " + str(len(ind_data)))
       
        if len(ind_data) > 0:
           
           for i in range(0,len(ind_data)):
           
               singIndData.append(ind_data[i])
    
    newSingData = []
    
    singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))     
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)    
          
    else:
        print("Not equal at phase 2")   
    
    list3FirstArr = np.array([np.array([list3])])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,2]])
    
    print("\n\n ------- \nListArrToCompare: ")
    print(listArrToCompare)
    print("-------\n\n") 
    
    print("\n\n ------- \nList3FirstArray: ")
    print(list3FirstArr)
    print("-------\n\n")
    
    print("\n\n ------- \nsecRoundNewListMetricsArr: ")
    print(secRoundNewListMetricsArr)
    print("-------\n\n")
    
    print(len(list3FirstArr[0][0]))
    print("Iterating ...")
    
    singIndData = []
    
    for ind in range(0,len(list3FirstArr[0][0])):       ## 
        
        val = []
        
        for xi in list3FirstArr[0][0]:
            if ind < len(xi):
                val.append(xi[ind])
            
        print("First value: ")
        print(secRoundNewListMetricsArr[ind])
        
        print("Length:")
        print(len(secRoundNewListMetricsArr[ind]))
        
        new_strSecRound = ""
        secRoundList = []
        
        if len(secRoundNewListMetricsArr[ind]) == 1:
           print("String")
           new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
           print(new_strSecRound)
        else:
            for x in secRoundNewListMetricsArr[ind]:
                secRoundList.append(x)
            print("secRoundList: ")
            print(secRoundList)
            
      
        print("Value to compare to: ")
        print(val)
        print("list3 first array: ")
        print(list3FirstArr[0][0]) 
        
        if len(val) == len(secRoundList)-1:
            secRoundList = secRoundList[:-1]
        
   ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
        
        ind_data = []
        for x in range(len(val)):
            if val[x] == secRoundList[x]:
               ind_data.append(x) 
               
 ##       rub, ind_data = tupleIndiceImage
  ##      ind_data = np.array([ind_data])
        
        print("Ind data for 3, for ind " + str(ind) + " : ")
        print(ind_data)
       
        # if len(ind_data[0]) == 1:
        #     print("Appending indice " + str(ind_data) + " to singIndData")
        #     singIndData.append(ind_data) 
        # else:
        #     if len(ind_data[0]) == 2:
        #         print("Not singular")
        #         singIndData.append(ind_data[0,0])
        #         singIndData.append(ind_data[0,1])           
    
        print("Length ind data: " + str(len(ind_data)))
       
        if len(ind_data) > 0:
           
           for i in range(0,len(ind_data)):
           
               singIndData.append(ind_data[i])
        
    newSingData = []
    
    singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))     
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)    
          
    else:
        print("Not equal at phase 2")    
    
    list4FirstArr = np.array([np.array([list4])])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,3]])
    
    print("\n\n ------- \nListArrToCompare: ")
    print(listArrToCompare)
    print("-------\n\n") 
    
    print("\n\n ------- \nList4FirstArray: ")
    print(list4FirstArr)
    print("-------\n\n")
    
    print("\n\n ------- \nsecRoundNewListMetricsArr: ")
    print(secRoundNewListMetricsArr)
    print("-------\n\n")
    
    print(len(list4FirstArr[0][0]))
    print("Iterating ...")
    
    singIndData = []
    
    for ind in range(0,len(list4FirstArr[0][0])):       ## 
        
        val = []
        
        for xi in list4FirstArr[0][0]:
            if ind < len(xi):
                val.append(xi[ind])
            
        print("First value: ")
        print(secRoundNewListMetricsArr[ind])
        
        print("Length:")
        print(len(secRoundNewListMetricsArr[ind]))
        
        new_strSecRound = ""
        secRoundList = []
        
        if len(secRoundNewListMetricsArr[ind]) == 1:
           print("String")
           new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
           print(new_strSecRound)
        else:
            for x in secRoundNewListMetricsArr[ind]:
                secRoundList.append(x)
            print("secRoundList: ")
            print(secRoundList)
            
      
        print("Value to compare to: ")
        print(val)
        print("list4 first array: ")
        print(list4FirstArr[0][0]) 
        
        if len(val) == len(secRoundList)-1:
            secRoundList = secRoundList[:-1]
        
   ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
        
        ind_data = []
        for x in range(len(val)):
            if val[x] == secRoundList[x]:
               ind_data.append(x) 
               
 ##       rub, ind_data = tupleIndiceImage
 ##       ind_data = np.array([ind_data])
        
        print("Ind data for 4, for ind " + str(ind) + " : ")
        print(ind_data)
       
        # if len(ind_data[0]) == 1:
        #     singIndData.append(ind_data)
        #     print("Appending indice " + str(ind_data) + " to singIndData")
        # else:
        #     if len(ind_data[0]) == 2:
        #         print("Not singular")
        #         singIndData.append(ind_data[0,0])
        #         singIndData.append(ind_data[0,1])           
    
        print("Length ind data: " + str(len(ind_data)))
       
        if len(ind_data) > 0:
           
           for i in range(0,len(ind_data)):
           
               singIndData.append(ind_data[i])
    
    newSingData = []
    
    singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))     
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)    
           
    else:
        print("Not equal at phase 2") 
    
    list5FirstArr = np.array([np.array([list5])])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,4]])
    
    print("\n\n ------- \nListArrToCompare: ")
    print(listArrToCompare)
    print("-------\n\n") 
    
    print("\n\n ------- \nList5FirstArray: ")
    print(list5FirstArr)
    print("-------\n\n")
    
    print("\n\n ------- \nsecRoundNewListMetricsArr: ")
    print(secRoundNewListMetricsArr)
    print("-------\n\n")
    
    print(len(list5FirstArr[0][0]))
    print("Iterating ...")
    
    singIndData = []
    
    for ind in range(0,len(list5FirstArr[0][0])):       ## 
        
        val = []
        
        for xi in list5FirstArr[0][0]:
            if ind < len(xi):
                val.append(xi[ind])
            
        print("First value: ")
        print(secRoundNewListMetricsArr[ind])
        
        print("Length:")
        print(len(secRoundNewListMetricsArr[ind]))
        
        new_strSecRound = ""
        secRoundList = []
        
        if len(secRoundNewListMetricsArr[ind]) == 1:
           print("String")
           new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
           print(new_strSecRound)
        else:
            for x in secRoundNewListMetricsArr[ind]:
                secRoundList.append(x)
            print("secRoundList: ")
            print(secRoundList)
            
      
        print("Value to compare to: ")
        print(val)
        print("list5 first array: ")
        print(list5FirstArr[0][0]) 
        
        if len(val) == len(secRoundList)-1:
            secRoundList = secRoundList[:-1]
        
   ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
        
        ind_data = []
        for x in range(len(val)):
            if val[x] == secRoundList[x]:
               ind_data.append(x) 
               
 ##       rub, ind_data = tupleIndiceImage
 ##       ind_data = np.array([ind_data])
        
        print("Ind data for 5, for ind " + str(ind) + " : ")
        print(ind_data)
        
        print("Length ind data: " + str(len(ind_data)))
       
        if len(ind_data) > 0:
           
           for i in range(0,len(ind_data)):
           
               singIndData.append(ind_data[i])
       
        # if len(ind_data[0]) == 1:
        #     print("Appending indice " + str(ind_data) + " to singIndData")
        #     singIndData.append(ind_data)  
        # else:
        #     if len(ind_data[0]) == 2:
        #         print("Not singular")
        #         singIndData.append(ind_data[0,0])
        #         singIndData.append(ind_data[0,1])           
    
    newSingData = []
    
    singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))     
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)    
          
    else:
        print("Not equal at phase 2") 
        
    list6FirstArr = np.array([np.array([list6])])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,5]])
    
    print("\n\n ------- \nListArrToCompare: ")
    print(listArrToCompare)
    print("-------\n\n") 
    
    print("\n\n ------- \nList6FirstArray: ")
    print(list6FirstArr)
    print("-------\n\n")
    
    print("\n\n ------- \nsecRoundNewListMetricsArr: ")
    print(secRoundNewListMetricsArr)
    print("-------\n\n")
    
    print(len(list6FirstArr[0][0]))
    print("Iterating ...")
    
    singIndData = []
    
    for ind in range(0,len(list6FirstArr[0][0])):       ## 
        
        val = []
        
        for xi in list5FirstArr[0][0]:
            if ind < len(xi):
                val.append(xi[ind])
            
        print("First value: ")
        print(secRoundNewListMetricsArr[ind])
        
        print("Length:")
        print(len(secRoundNewListMetricsArr[ind]))
        
        new_strSecRound = ""
        secRoundList = []
        
        if len(secRoundNewListMetricsArr[ind]) == 1:
           print("String")
           new_strSecRound = str(secRoundNewListMetricsArr[ind][0]) 
           print(new_strSecRound)
        else:
            for x in secRoundNewListMetricsArr[ind]:
                secRoundList.append(x)
            print("secRoundList: ")
            print(secRoundList)
            
      
        print("Value to compare to: ")
        print(val)
        print("list6 first array: ")
        print(list6FirstArr[0][0]) 
        
        if len(val) == len(secRoundList)-1:
            secRoundList = secRoundList[:-1]
        
   ##     tupleIndiceImage = np.where(secRoundList == val)    ## [0,ind]
        
        ind_data = []
        for x in range(len(val)):
            if val[x] == secRoundList[x]:
               ind_data.append(x) 
               
 ##       rub, ind_data = tupleIndiceImage
 ##       ind_data = np.array([ind_data])
        
        print("Ind data for 6, for ind " + str(ind) + " : ")
        print(ind_data)
        
        print("Length ind data: " + str(len(ind_data)))
       
        if len(ind_data) > 0:
           
           for i in range(0,len(ind_data)):
           
               singIndData.append(ind_data[i])
       
        # if len(ind_data[0]) == 1:
        #     print("Appending indice " + str(ind_data) + " to singIndData")
        #     singIndData.append(ind_data) 
        # else:
        #     if len(ind_data[0]) == 2:
        #         print("Not singular")
        #         singIndData.append(ind_data[0,0])
        #         singIndData.append(ind_data[0,1])           
    
    newSingData = []
    
    singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))     
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)    
          
    else:
        print("Not equal at phase 2") 
      
    print("\n\n-------------")
    print("Ind for folder clustering: ")
    print(indForFolderClustering)
    print("-------------\n\n")
        
    #%%
    
    
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################

    name_folder = first_clustering_storing_output + str(sequence_name) + "_7_8"
    newPath = os.path.join(parent_dir,name_folder)    
    
    
    if "\n" in newPath or "\t" in newPath or "\r" in newPath or " " in newPath:                                      
    
        n_path = ""
        
        for let in newPath:
            if let != "\n" and let != "\r" and l != "\t" and l != " ":
                n_path += let
        
        print("NewPath Bef: " + str(n_path))
        n_path = n_path[1:]
        newPath = n_path
        
        print("NewPath After: " + str(newPath))
        
        if '/' in newPath:
            x = 0
            while os.path.exists(newPath + '/'):
                newPath += '_' + str(x)
                x += 1
            nPath = newPath.split('/')
            pas = nPath[-1]
            rpas = nPath[:-1]
            pas = pas[2:]
            pasn = ""
            for p in pas:
                if p != "\t" and p != " ":
                    pasn += p
            pas = pasn
            rpas.append(pas)
            nnPath = ""
            for r in rpas:
                nnPath += r + '/'
            
            nnPath = nnPath[:-1]
            newPath = nnPath
            
        elif "\\" in newPath:
            if os.path.exists(newPath + "\\"):
                x = 0
                while os.path.exists(newPath + "\\"):
                    newPath += '_' + str(x)
                    x += 1
                nPath = newPath.split('/')
                pas = nPath[-1]
                rpas = nPath[:-1]
                pas = pas[2:]
                pasn = ""
                for p in pas:
                  if p != "\t" and p != " ":
                      pasn += p
                pas = pasn
                rpas.append(pas)
                nnPath = ""
                for r in rpas:
                    nnPath += r + '/'
                   
                nnPath = nnPath[:-1] 
                newPath = nnPath
        
        indxp = 0
        for indp, p in enumerate(newPath):
            if indp < 10:
                if p == "\n" or p == ' ' or p == "\t":
                    indxp = indp
                    
        newPath = newPath[indxp+1:]
        
        if os.path.exists(newPath + "\\"):
            x = 0
            while os.path.exists(newPath + "\\"):
                
                if newPath[-2] == '_':
                    newPath = newPath[:-2] 
                if newPath[-3] == '_':
                    newPath = newPath[:-3] 
                    
                newPath += '_' + str(x)
                x += 1
            else:
                os.mkdir(newPath)
        print("\n\n Got A \n\n")
    else:
        if not os.path.exists(newPath + "/"):
        
            os.mkdir(newPath) 
        else:
            indP = int(newPath[-1])
            
            indP += 1
            
            while True:
                
                newPathH = newPath[:-1] + str(indP)
                newPath = newPathH
                
                print("\n\n-------- newPath " + str(indP) + " : \n")
                print(newPath)
                 
                if not os.path.exists(newPath + "/"):
                    os.mkdir(newPath) 
                    break
                else:
                    indP += 1
                    
        print("\n\n Got B \n\n")
    
    print("newPath + /:")
    print(newPath + "/")
    print("\n\n")    
    
    indIntruso = 0
    detIntruso = False
    
    for indN, n in enumerate(newPath):
        if indN > 1 and n == ':':
            indIntruso = indN
            detIntruso = True
    
    if detIntruso:
        newNewPath = ""
        
        newNewPath = newPath[:indIntruso] + newPath[(indIntruso+1):]
                
        newPath = newNewPath
    
    print("newPath + /:")
    print(newPath + "/")
    print("\n\n")   
    
    if not os.path.exists(newPath + "/"):
        os.mkdir(newPath)
        print("\n\n Got C \n\n")
     
    print("First Directory created  \n\n\n")
    
    trainFolder = "Train_Results" 
     
    print("\n\n Path for training: ")
    print(newPath + "/" + trainFolder)
    print("\n\n")
    
    newPath = os.path.join(newPath + "/", trainFolder)   
        
    if not os.path.exists(newPath + "/"):
        os.mkdir(newPath) 
    
  #  newPath = newPath + "/" 
    
    print("Second Directory created")
     
    sub_name_folder1 = "Class_1"
    newPath_1 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_1 + "/"):
        os.mkdir(newPath_1)
     
    sub_name_folder1 = "Class_2"
    newPath_2 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_2 + "/"):        
        os.mkdir(newPath_2) 
         
    sub_name_folder1 = "Class_3"
    newPath_3 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_3 + "/"):        
        os.mkdir(newPath_3)   
     
    sub_name_folder1 = "Class_4" 
    newPath_4 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_4 + "/"): 
        os.mkdir(newPath_4)
     
    sub_name_folder1 = "Class_5"
    newPath_5 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_5 + "/"):    
        os.mkdir(newPath_5) 
         
    sub_name_folder1 = "Class_6"
    newPath_6 = os.path.join(newPath + "/",sub_name_folder1) 
    
    if not os.path.exists(newPath_6 + "/"):        
        os.mkdir(newPath_6)   
    
    counter_1 = 0
    counter_2 = 0 
    counter_3 = 0
    counter_4 = 0
    counter_5 = 0
    counter_6 = 0
    
    
    metricsIdTtrain1 = []
    metricsIdTtrain2 = []
    metricsIdTtrain3 = []
    metricsIdTtrain4 = []
    metricsIdTtrain5 = []
    metricsIdTtrain6 = []
    
    print("Here on A")
    print("newPath1: ")
    print(newPath_1)
    print("\n")
    
    print("indForFolderClustering: ")
    print(indForFolderClustering)
    
    # import sys
    # sys.exit()
    
 #   if not (os.listdir(newPath_1) or os.listdir(newPath_2) or os.listdir(newPath_3) or os.listdir(newPath_4) or os.listdir(newPath_5) or os.listdir(newPath_6)):
    
    if True:
        # for indCluster, cluster in enumerate(indForFolderClustering):
        #     print("Cluster:")
        #     print(cluster)
        #     print("\n\n\n") 
            
        #     cluster_list = cluster.tolist()
        #     print("\n---\n cluster list: ")
        #     print(cluster_list[0])
        #     print("The length: " + str(cluster_list[0]))
        #     for ind_imageInCluster in cluster_list[0]:
        
        #         image_counter = cv2.imread(roi_bef + "/roi_image" + str(ind_imageInCluster) + ".jpg")
        #         image_counter_2 = cv2.imread(roi_after + "/roi_image" + str(ind_imageInCluster) + ".jpg")       
                
        #         if indCluster == 0:   
        #             cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(0) + ".jpg", image_counter) 
        #             cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(1) + ".jpg", image_counter_2)
        #             metricsIdTtrain1.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
        #             counter_1 += 1
        #         if indCluster == 1:
        #             cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(0) + ".jpg", image_counter)
        #             cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(1) + ".jpg", image_counter_2)
        #             metricsIdTtrain2.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
        #             counter_2 += 1
        #         if indCluster == 2: 
        #             cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(0) + ".jpg", image_counter)
        #             cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(1) + ".jpg", image_counter_2)
        #             metricsIdTtrain3.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
        #             counter_3 += 1 
        #         if indCluster == 3:
        #             cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(0) + ".jpg", image_counter)
        #             cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(1) + ".jpg", image_counter_2)
        #             metricsIdTtrain4.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
        #             counter_4 += 1
        #         if indCluster == 4:
        #             cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(0) + ".jpg", image_counter)
        #             cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(1) + ".jpg", image_counter_2)
        #             metricsIdTtrain5.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
        #             counter_5 += 1
        #         if indCluster == 5:
        #             cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(0) + ".jpg", image_counter) 
        #             cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(1) + ".jpg", image_counter_2)
        #             metricsIdTtrain6.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
        #             counter_6 += 1
        
        print("Cluster_ID_transpose: ")
        print(Cluster_ID_transpose[0])
        
        print("roi_bef: ")
        print(roi_bef)
        
        print("roi_after: ")
        print(roi_after)   
        
        # import sys
        # sys.exit()
        
        for ind_imageInCluster, cluster in enumerate(Cluster_ID_transpose[0]):    ## indForFolderClustering
          ##  cluster_list = cluster.tolist()
         ##   for ind_imageInCluster in cluster_list[0]:
        
                image_counter = cv2.imread(roi_bef + "/roi_image" + str(ind_imageInCluster) + ".jpg")
                image_counter_2 = cv2.imread(roi_after + "/roi_image" + str(ind_imageInCluster) + ".jpg")       
                
                print("image_counter: ")
                print(image_counter)
                print("image_counter_2: ")
                print(image_counter_2)
                
                if cluster == 0:  
                    cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(0) + ".jpg", image_counter) 
                    cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain1.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_1 += 1
                if cluster == 1:
                    cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(0) + ".jpg", image_counter)
                    cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain2.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_2 += 1
                if cluster == 2: 
                    cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(0) + ".jpg", image_counter)
                    cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain3.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_3 += 1 
                if cluster == 3:
                    cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(0) + ".jpg", image_counter)
                    cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain4.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_4 += 1
                if cluster == 4:
                    cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(0) + ".jpg", image_counter)
                    cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain5.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_5 += 1
                if cluster == 5:
                    cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(0) + ".jpg", image_counter) 
                    cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain6.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_6 += 1
        
        LenListsClusters1= [LenList_1, LenList_2, LenList_3, LenList_4, LenList_5, LenList_6]
        
        ##########################################################################################################
        ##########################################################################################################
        ##########################################################################################################
        ##########################################################################################################
        
        m_1_f = []
        m_2_f = []
        m_3_f = []
        m_4_f = []
        m_5_f = []
        m_6_f = []
        
        
        metrics_afterClustering = []
        
        if len(metricsIdTtrain1) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain1 = np.array([metricsIdTtrain1]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain1) 
            
            for m_1 in metricsIdTrain1:
                m_1_f.append(np.mean(np.array([m_1])[0,:,0]))    
            
        if len(metricsIdTtrain2) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain2 = np.array([metricsIdTtrain2]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain2)
            
            for m_2 in metricsIdTrain2:
                m_2_f.append(np.mean(np.array([m_2])[0,:,0]))
            
        if len(metricsIdTtrain3) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain3 = np.array([metricsIdTtrain3]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain3)
            
            for m_3 in metricsIdTrain3:
                m_3_f.append(np.mean(np.array([m_3])[0,:,0]))
            
        if len(metricsIdTtrain4) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain4 = np.array([metricsIdTtrain4]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain4)
            
            for m_4 in metricsIdTrain4:
                m_4_f.append(np.mean(np.array([m_4])[0,:,0]))
            
        if len(metricsIdTtrain5) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain5 = np.array([metricsIdTtrain5]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain5)
            
            for m_5 in metricsIdTrain5:
                m_5_f.append(np.mean(np.array([m_5])[0,:,0]))
            
        if len(metricsIdTtrain6) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain6 = np.array([metricsIdTtrain6]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain6)
            
            for m_6 in metricsIdTrain6:
                m_6_f.append(np.mean(np.array([m_6])[0,:,0]))
                
                
        #################################################################################################################
        #################################################################################################################
        #%%
        
        labelsMetricsToScore = ['MSE', 'RMSE', 'RMSE_SINGLE', 'RMSE_SW', 'PSNR', 'UQI_SINGLE', 
                                'UQI', 'SSIMS', 'CSS', 'ERGAS', 'SCC',
                                'RASE', 'SAM', 'MSSSIM', 'VIFP', 'PSNRB',
                                'Mean', 'STD', 'Contrast', 'ASM', 'Max']
        
        stdListValues = []
        global_std = [] 
        
        secondMetricTable = np.array([metrics_afterClustering])[0].T.tolist() 
        
        listOfFlatten = []
        stdListValuesMetrics = []
        
        for indSec, secInd in enumerate(secondMetricTable):    
            flattenSecond = []
            flatten_list = [element for sublist in secInd for element in sublist]
            
            for fla in flatten_list:
                fla_one = np.array([fla])[0,0]
                flattenSecond.append(fla_one)
        
            listOfFlatten.append(flattenSecond)
        
        for indFlatten, metricFlatten in enumerate(listOfFlatten):
            stdValue = np.std(np.array([metricFlatten]))
            
            stdListValuesMetrics.append(stdValue)
        
        newstdlistValuesMetrics = []
        mainIndices = []
        
        sortedIndices = []
        listToPCA = []
        
        sorted_std_values = sorted(stdListValuesMetrics, reverse=True)
        
        for ind, stdValueSorted in enumerate(sorted_std_values):
            indStuff = np.where(np.array([stdListValuesMetrics]) == stdValueSorted)
            rub, indS = indStuff
            indS = np.array([indS])[0,0]
            indOfSorted = indS
            sortedIndices.append(indOfSorted)
        
        sortedIndicesToGo = sortedIndices[0:15]
        
        exlistToPCA = secRoundNewListMetricsArr.T.tolist()
        
        for ind_listPCA, listPCA in enumerate(exlistToPCA):
            if ind_listPCA in sortedIndicesToGo and ind_listPCA != 16: 
                listToPCA.append(listPCA) 
                
        filtered_metrics = []
        newstdlistValuesMetrics_sec = []
        metricsToPCA_analysis = [] 
        metricsToPCA_norm = []
        
        for ind_mat1, mat1 in enumerate(metrics_afterClustering):
            mat1_n = []
            for ind_mat2, mat2 in enumerate(mat1):
                mat2_n = [] 
                for ind_mat3, mat3 in enumerate(mat2):
                    mat3 = mat3[0]
                    mat2_n.append(mat3)
                mat1_n.append(mat2_n)  
            
            metricsToPCA_analysis.append(mat1_n)
         
        ## Standard normalization ########################################################################################################
        ##################################################################################################################################
        
        metricsToPCA_norm = []
        
        for met1 in listToPCA:     
            
                mean_value = np.mean(np.array([met1]))
                std_value = np.std(np.array([met1])) 
                metricPCA_norm1 = [] 
                
                for met2 in met1:
                    metricPCA_norm1.append((met2-mean_value)/std_value)        
                
                metricsToPCA_norm.append(metricPCA_norm1)
        
        print("Metrics to PCA norm")
        print(metricsToPCA_norm)
        
        ##################################################################################################################################
        #%%
        
        dfxi = pd.DataFrame(data=metricsToPCA_norm)  
        
        dfxi = dfxi.dropna()
        dfxi = dfxi.dropna(axis=1)
        
        print("Dataframe: ") 
        print(dfxi.to_string())   
        
        pcai = PCA(n_components=None)                             
        
        dfx_pcai = pcai.fit(dfxi)      ## Error
         
        X_pcai = pcai.transform(dfxi)   
        dfx_transi = pd.DataFrame(data=X_pcai)
        
        plt.scatter(dfx_transi[0], dfx_transi[1], c ="blue")
        plt.title("Correlation between first two PCA components")
        plt.xlabel("First PCA component")
        plt.ylabel("Second PCA component")
        plt.show()
        
        pca_coef_feat_first_comp = dfx_transi[0].tolist()
        abs_pca_coeff = np.array([abs(np.array([pca_coef_feat_first_comp])[0])])
        abs_pca_coeffList = abs_pca_coeff.tolist()
        
        ## sortedPCA_coeff = sorted(abs_pca_coeffList[0], reverse=True)
        sortedPCA_coeff = sorted(abs_pca_coeffList[0])
        
        ## ContrastAppending = sortedPCA_coeff[18] 
        
        sortedPCA_coeff = sortedPCA_coeff[0:4]
        ## sortedPCA_coeff = sortedPCA_coeff.append(ContrastAppending)
        
        sortedIndices2 = []
        
        for ind, pcaSorted in enumerate(sortedPCA_coeff):
            indStuff = np.where(np.array([abs_pca_coeffList]) == pcaSorted)
            rub, rub2, indS = indStuff  
            indOfSorted = indS 
            sortedIndices2.append(indOfSorted)    
             
        doubleSortedElements = []
        
        indicesSortedFromPCA = [] 
            
        for sorted_ind in sortedIndices2:
            if len(sorted_ind) == 2:
                doubleSortedElements.append(sorted_ind)
                list_aux = sorted_ind.tolist()
                if list_aux[0] in indicesSortedFromPCA:
                    indicesSortedFromPCA.append(list_aux[1])
                else:
                    indicesSortedFromPCA.append(list_aux[0])
            else:
                if len(sorted_ind) == 1:
                    indicesSortedFromPCA.append(np.array([sorted_ind])[0,0])
                    
        indicesSortedFromPCA.append(18)            
                    
        remainingMetricsToClustering = [] 
        trainListData = []
        
        for ind in indicesSortedFromPCA: 
            if ind <= 21 and ind < len(exlistToPCA):
                remainingMetricsToClustering.append(labelsMetricsToScore[ind])   
                trainListData.append(exlistToPCA[ind]) 
                 
        nclusters = 2
          
        trainListData = np.array([trainListData])[0].T.tolist()
        
        train_data = np.array([trainListData])[0].T
        treino_lenght = 370
        
        kmeans = KMeans(n_clusters=nclusters, max_iter=500, n_init=8).fit(trainListData) 
        Cluster_ID = kmeans.labels_ 
        centroides_A = kmeans.cluster_centers_   
        print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)
         
        
        
        Cluster_ID_transpose = np.array([np.array([Cluster_ID]).T[0:treino_lenght,0]])
        
        objetos_c1 = []
        objetos_c2 = []
        ind_first = []
        ind_second = []
        
        for i in range (0, len(Cluster_ID_transpose[0])):
            if i<296:
                if Cluster_ID_transpose[0,i] == 0:
                    objetos_c1.append(train_data[:,i])             
                    i_1 = i
                    ind_first.append(i)
                elif Cluster_ID_transpose[0,i] == 1:
                    objetos_c2.append(train_data[:,i]) 
                    i_2 = i
                    ind_second.append(i)
                    
        trainDataFurther = train_data.tolist()
        classClustering = []
        secFurther = []
        
        for tF in trainDataFurther:
            tF = tF[0:296]
            secFurther.append(tF)
        
        for indHere in range(0,296):
            if indHere in ind_first:
                classClustering.insert(indHere, 'A')        
            else:
                if indHere in ind_second:
                    classClustering.insert(indHere, 'B')
        
        secFurther.insert(0, classClustering)
        classCl = secFurther[0]
        classAFurther = []
        classBFurther = []
        secFurther = np.array([secFurther])[0].T.tolist()
        
        for sInd, sF in enumerate(secFurther):
            if classCl[sInd] == 'A':
                classAFurther.append(sF[1:])
            else:
                if classCl[sInd] == 'B':
                    classBFurther.append(sF[1:])
                    
        clustering_inf_data = [classAFurther, classBFurther, nclusters, number_recommended_clusters, remainingMetricsToClustering]
        
        print("Clustering inf data: ")
        print(clustering_inf_data)
        
        #%%
                    
        dfxi1 = pd.DataFrame(data=classAFurther) 
        pcai1 = PCA(n_components=None) 
        dfx_pcai1 = pcai1.fit(dfxi1)   
        
        print("dfxi1:")
        print(dfxi1)
         
        X_pcai1 = pcai1.transform(dfxi1)   
        dfx_transi1 = pd.DataFrame(data=X_pcai1)
        X_pcai1T = X_pcai1.T.tolist()
        
        print("X_pcai1:")
        print(X_pcai1)
        
        print("dfx_transi1:")
        print(dfx_transi1)
        
        print("X_pcai1T:")
        print(X_pcai1T)
        
        dfxi2 = pd.DataFrame(data=classBFurther)  
        
        print("dfxi2:")
        print(dfxi1)
        
        pcai2 = PCA(n_components=None) 
        dfx_pcai2 = pcai2.fit(dfxi2)    
        X_pcai2 = pcai2.transform(dfxi2)   
        dfx_transi2 = pd.DataFrame(data=X_pcai2)
        
        print("dfx_pcai2:")
        print(dfx_pcai2)
        
        print("X_pcai2:")
        print(X_pcai2)
        
        print("dfx_transi2:")
        print(dfx_transi2)
        
        X_pcai2T = X_pcai2.T.tolist()
        
        print("X_pcai2T:")
        print(X_pcai2T)
        
        
        centroid_pca_A = []
        centroid_pca_B = []
        
        for indA in range(0,5):
            if indA < len(X_pcai1T):
                mean_value = np.mean(np.array([X_pcai1T[indA]]))
                centroid_pca_A.append(mean_value)    
            
        for indB in range(0,5):
            if indB < len(X_pcai2T):
                mean_value = np.mean(np.array([X_pcai2T[indB]]))
                centroid_pca_B.append(mean_value)
        
        if len(centroid_pca_A) < len(centroid_pca_B):
            centroid_pca_B = centroid_pca_B[:len(centroid_pca_A)]
        elif len(centroid_pca_A) > len(centroid_pca_B):
            centroid_pca_A = centroid_pca_A[:len(centroid_pca_B)]
            
            
        distCentroidsPCA = np.linalg.norm(np.array([centroid_pca_A])[0]-np.array([centroid_pca_B])[0])
        
        #####
        thisDir = os.getcwd()
        dirResultsOutput = thisDir + '\\GraphsOutput\\'
        
        if os.path.isdir(dirResultsOutput) == False:  
        
            dirResults = os.path.join(dirResultsOutput) 
            os.mkdir(dirResults)    
        
        #####
        
        if len(dfx_transi1[0]) > 0 and len(dfx_transi1[1]) > 0:    
            plt1 = plt.scatter(dfx_transi1[0], dfx_transi1[1], c ="blue")
        if len(dfx_transi2[0]) > 0 and len(dfx_transi2[1]) > 0:
            plt2 = plt.scatter(dfx_transi2[0], dfx_transi2[1], c ="red")
            
        plt.legend((plt1, plt2),
                    ('Class A', 'Class B'))
        plt.title("Correlation between first two PCA components")
        plt.xlabel("First PCA component")
        plt.ylabel("Second PCA component")    
        
        plt.savefig(dirResultsOutput + "pca_graph_" + str(M) + ".png")   ## _numnerComb
        
        plt.show() 
        
        if not(len(dfx_transi1[0]) > 1 and len(dfx_transi2[0]) > 1):
            print("\n\n ---------- \n Just one PCA element \n\n Finishing earlier  \n ----------- \n\n")
         
        else:
        
            data_results.append(clustering_inf_data)
            
            data_results.append(dirResultsOutput + "pca_graph_" + str(M) + ".png")
            
            
            list1 = list(zip(*objetos_c1)) 
            list2 = list(zip(*objetos_c2)) 
            
            #%%
            
            list1ToDist = np.array([list1])[0].T
            list2ToDist = np.array([list2])[0].T 
            
            dists1 = []
            dists2 = []
            cent_1 = np.array([centroides_A[0,:]])
            cent_2 = np.array([centroides_A[1,:]])
            
            for i in range(0,len(list1ToDist)):
                
                a = list1ToDist[i,:]
                b = cent_1[0,:]    
                dist = np.linalg.norm(a-b)    
                dists1.append(dist) 
                
            for i in range(0,len(list2ToDist)):
                
                a = list2ToDist[i,:]
                b = cent_2[0,:]
                dist = np.linalg.norm(a-b)    
                dists2.append(dist)
            
            xDist1 = []
            xDist2 = []
            indD_1 = 0
            indD_2 = 0
            
            for ind in range(0, len(dists1)):
                indD_1 += 1
                xDist1.append(indD_1)
            
            for ind in range(0, len(dists2)):
                indD_2 += 1
                xDist2.append(indD_2)
                
            plt.scatter(xDist1, dists1, c ="blue")
            plt.title("Distance to centroid A")
            plt.xlabel("Number of point")
            plt.ylabel("Distance of points of first cluster to its centroid")
            plt.savefig(dirResultsOutput + "_" + str(M) + "distances_firstCluster.png")
            plt.show()
            
            data_results.append(dirResultsOutput + "distances_firstCluster_" + str(M) + ".png")
            
            plt.scatter(xDist2, dists2, c ="blue")
            plt.title("Distance to centroid B")
            plt.xlabel("Number of point")
            plt.ylabel("Distance of points of second cluster to its centroid")
            plt.savefig(dirResultsOutput + "_" + str(M) + "distances_secondCluster.png")  
            plt.show() 
            
            data_results.append(dirResultsOutput + "distances_secondCluster_" + str(M) + ".png")
            
                
            print(" -- Lists of clusters generated")
            
            for l in list1:
                LenList_1 = len(l)
            for l in list2:
                LenList_2 = len(l)
            
            #%%
                
            trainListData = np.array([trainListData])[0].T.tolist()   
                
            Y_euclidean = pdist(trainListData, metric='euclidean')
            Y_euclidean_square = squareform(Y_euclidean)
            Y_cityblock = pdist(trainListData, metric='cityblock')
            Y_euclidean_square = squareform(Y_cityblock)
            
            Z_euclidean_average = linkage(trainListData, method='average', metric='euclidean')
            Z_euclidean_ward = linkage(trainListData, method='ward', metric='euclidean')
            
            Z_cityblock_average = linkage(trainListData, method='average', metric='cityblock') 
            
            distances_from_euclidean_average = Z_euclidean_average[:,2].tolist()
            
            clustersList = Z_euclidean_average[:,0].tolist() + Z_euclidean_average[:,1].tolist()
            
            totNumbClusters_firstExp = np.max(np.array([clustersList])[0])
            
            numberObservationsForEachCluster_first = Z_euclidean_average[:,3].tolist()
            totObservationsFirst = np.sum(np.array([numberObservationsForEachCluster_first]))
            
            
            distances_from_euclidean_ward = Z_euclidean_ward[:,2].tolist()
            
            clustersList = Z_euclidean_ward[:,0].tolist() + Z_euclidean_average[:,1].tolist()
            
            totNumbClusters_secExp = np.max(np.array([clustersList])[0])
            
            numberObservationsForEachCluster_second = Z_euclidean_ward[:,3].tolist()
            totObservationsSec = np.sum(np.array([numberObservationsForEachCluster_second]))
            
            
            
            distances_from_cityblock_average = Z_cityblock_average[:,2].tolist()
            
            clustersList = Z_cityblock_average[:,0].tolist() + Z_euclidean_average[:,1].tolist()
            
            totNumbClusters_thirdExp = np.max(np.array([clustersList])[0])
            
            numberObservationsForEachCluster_third = Z_cityblock_average[:,3].tolist()
            totObservationsThird = np.sum(np.array([numberObservationsForEachCluster_third]))
            
            distancesMeanVar = [np.mean(np.array([distances_from_euclidean_average])), np.mean(np.array([distances_from_euclidean_ward])), np.mean(np.array([distances_from_cityblock_average]))]
            labelsMeasuresDistances = ['Euclidean Average', 'Euclidean Ward', 'Cityblock Average']
            
            maxMeanDistance = 0
            
            for indDist, distMeanValue in enumerate(distancesMeanVar):
                if distMeanValue > maxMeanDistance:
                    maxMeanDistance = distMeanValue
                    indMaxMeasureDistance = indDist 
            
            print("Selected Measure for distance between clusters: " + labelsMeasuresDistances[indMaxMeasureDistance])
            
            distancesBetClustersBestMeasure = []
            
            if indMaxMeasureDistance == 0:
                distancesBetClustersBestMeasure = Z_euclidean_average.tolist()
                numberClustersFromDist = totNumbClusters_firstExp
            else:
                if indMaxMeasureDistance == 1:
                    distancesBetClustersBestMeasure = Z_euclidean_ward.tolist()
                    numberClustersFromDist = totNumbClusters_secExp
                else:
                    if indMaxMeasureDistance == 2:
                        distancesBetClustersBestMeasure = Z_cityblock_average.tolist()
                        numberClustersFromDist = totNumbClusters_thirdExp
                        
            distance_output = []
                        
            for dist1 in distancesBetClustersBestMeasure:
                distance_output2 = []
                for ind_dist2, dist2 in enumerate(dist1):
                    if ind_dist2 != 2:
                        distance_output2.append(int(dist2))
                    else:
                        distance_output2.append(dist2)
                distance_output.append(distance_output2)
                
            #### Comparison of number of clusters between distances approach and the clustering one:
            if  number_recommended_clusters == numberClustersFromDist:
                print("The above methods provide the same number of clusters")
            else:
                if number_recommended_clusters > numberClustersFromDist:
                    print("The number of recommended clusters (from clustering approach) is higher than the number of clusters computed from dist-linkage method.")
                else:
                    if number_recommended_clusters < numberClustersFromDist:
                        print("The number of clusters computed from dist-linkage method is higher than the number of recommended clusters (from clustering approach).")
            
            
            #%%     
            
            def convolve2D(image, kernel, padding=0, strides=1):
                # Cross Correlation
                kernel = np.flipud(np.fliplr(kernel))
            
                # Gather Shapes of Kernel + Image + Padding
                xKernShape = kernel.shape[0]
                yKernShape = kernel.shape[1]
                xImgShape = image.shape[0]
                yImgShape = image.shape[1]
            
                # Shape of Output Convolution
                xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
                yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
                output = np.zeros((xOutput, yOutput))
            
                # Apply Equal Padding to All Sides
                if padding != 0:
                    imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
                    imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
                    print(imagePadded)
                else:
                    imagePadded = image
            
                # Iterate through image
                for y in range(image.shape[1]):
                    # Exit Convolution
                    if y > image.shape[1] - yKernShape:
                        break
                    # Only Convolve if y has gone down by the specified Strides
                    if y % strides == 0:
                        for x in range(image.shape[0]):
                            # Go to next row once kernel is out of bounds
                            if x > image.shape[0] - xKernShape:
                                  break
                            try:
                                # Only Convolve if x has moved by the specified Strides
                                if x % strides == 0:
                                    output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                            except:
                                break
            
                return output
            
            #%%
            
            k1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            k2 = [[1, 0, -1],  [2, 0, -2], [1, 0, -1]]
            
            k1_arr = np.array([k1])[0]
            k2_arr = np.array([k2])[0]
            
            M_c = []
             
            #%%
            
            ind_slash = 0
            
            for ind_x, x in enumerate(IFVP):
                if x == '/':
                    ind_slash = ind_x
                    break
            
            IFVP_p = IFVP[ind_slash:]
            IFVP = 'C:' + IFVP_p
                    
            
            # if IFVP[0] != 'C' and IFVP[0] == ':':
            #     IFVP = 'C' + IFVP
            # elif IFVP[0] != 'C' and  IFVP[0] != ':':
            #     IFVP = IFVP[1:]
            
            print("\n IFVP: " + IFVP)
            print("\n")
            
            seqNew = ""
            
            for i in sequence_name:
                if i.isdigit():
                    seqNew += i
            
            sequence_name = seqNew
            
            print(IFVP + str(sequence_name) + "_2" + "/video_image")
            
            for video_image in range(0,count):           #### 321   ## from 361   ## to count = 637
                
                    print("Analysing for image " + str(video_image) + " th")
                
                    image = cv2.imread(IFVP + str(sequence_name) + "_2" + "/video_image%d.jpg" % video_image)      
                    print("\n")
                    print(image)
                    print("\n")
                    
                    if image is not None:
                        imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
                           
                       
                         
                        M1 = convolve2D(imx, k1)
                        M2 = convolve2D(imx, k2)      
                        
                        
                        print("Generating output for image " + str(video_image))
                        
                        comp_x = np.power(np.array([M1])[0], 2)      
                        comp_y = np.power(np.array([M2])[0], 2) 
                        
                        sum_comps = comp_x + comp_y 
                        
                        gen_output = np.power(sum_comps, 1/2).astype(int)
                        
                        cv2.imwrite(roi_path + "/gen_output_%d.jpg" % video_image, gen_output)
                    else:
                        count = video_image
            #######################################
            #######################################
            #######################################
            #######################################
            #######################################  
             
                    executionTime = (time.time() - startTime)
                    print('Execution time in seconds: ' + str(executionTime))
            print("Here")
            clustering_output.append((clustering_inf_data, executionTime, count))
            print("Inter")
            gui_show_results(clustering_output, executionTime, count)
    else:
        print("\n\n Clustering processed before \n Ending ... \n\n\n")

def videoAnalysisClusteringPart(secRoundNewListMetrics, number_metrics,  lenMax, infi, roi_set, count, test_size = 0.2):
    
   
    startTime = time.time()  
    
    roi_path, roi_bef, roi_after = roi_set
    
    decisorLevel, mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, first_clustering_storing_output, pathPythonFile = infi
    
    parent_dir = dest_path
    
    secRoundNewListMetricsArr = np.array([secRoundNewListMetrics])[0]
    
    clustering_output = []
    data_results = [] 
    
    resT = np.array([ secRoundNewListMetricsArr[:, int(number_metrics)-1]]).T 
    
    train_data, test_data, labels_train_data, labels_test_data = train_test_split(secRoundNewListMetricsArr[:, 0:int(number_metrics)-1], resT, test_size =test_size, random_state = 42)
    treino_lenght = int((1-test_size)*lenMax)
    
    # Método Silhouette - análise para um nº variável de clusters   
    aux = 0 
    max_silhouette = 0
    silhouette_vector = [] 
    n_clusters3 = range(2, treino_lenght) 
    for j in n_clusters3:
        km =KMeans(n_clusters=j, max_iter=300, n_init=5).fit(train_data)
        labels3 = km.labels_
        silhouette_avg = silhouette_score(train_data, labels3)
        print("For n_clusters=", j, "The averegae silhouette_score is:", silhouette_avg)   
        aux = silhouette_avg 
        if aux > max_silhouette: 
            max_silhouette = aux
            number_recommended_clusters = j    
        silhouette_vector.append(silhouette_avg)  
        
    print("\n\nCom base no gráfico Elbow e no método Silhouette, é recomendável formar", number_recommended_clusters, "clusters!") 
    
    secRoundNewListMetrics = np.array([secRoundNewListMetrics])[0].T.tolist()
    trainListData = train_data.tolist() 
    nclusters = 6
    
    kmeans = KMeans(n_clusters=nclusters, max_iter=500, n_init=8).fit(trainListData) 
    Cluster_ID = kmeans.labels_ 
    centroides_A = kmeans.cluster_centers_   
    print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)
    
    Cluster_ID_transpose = np.array([np.array([Cluster_ID]).T[0:treino_lenght,0]])
    
    objetos_c1 = []
    objetos_c2 = []
    objetos_c3 = [] 
    objetos_c4 = [] 
    objetos_c5 = [] 
    objetos_c6 = [] 
    
    for i in range (0, len(Cluster_ID_transpose[0])):
        if Cluster_ID_transpose[0,i] == 0:
            objetos_c1.append(train_data[i, :])
            i_1 = i
        elif Cluster_ID_transpose[0,i] == 1:
            objetos_c2.append(train_data[i, :]) 
            i_2 = i
        elif Cluster_ID_transpose[0,i] == 2:
            objetos_c3.append(train_data[i, :]) 
            i_3 = i
        elif Cluster_ID_transpose[0,i] == 3:
            objetos_c4.append(train_data[i, :]) 
            i_4 = i
        elif Cluster_ID_transpose[0,i] == 4:
            objetos_c5.append(train_data[i, :])
            i_5 = i
        elif Cluster_ID_transpose[0,i] == 5:
            objetos_c6.append(train_data[i, :])  
            i_6 = i     
            
    list1 = list(zip(*objetos_c1)) 
    list2 = list(zip(*objetos_c2)) 
    list3 = list(zip(*objetos_c3)) 
    list4 = list(zip(*objetos_c4)) 
    list5 = list(zip(*objetos_c5)) 
    list6 = list(zip(*objetos_c6)) 
    
    print(" -- Lists of clusters generated")
    
    for l in list1:
        LenList_1 = len(l)
    for l in list2:
        LenList_2 = len(l)
    for l in list3:
        LenList_3 = len(l)
    for l in list4:
        LenList_4 = len(l)
    for l in list5:
        LenList_5 = len(l)
    for l in list6:
        LenList_6 = len(l) 
        
    
    print("Lists: ")
    print(list1)
    print(list2)
    print(list3)
    print(list4)
    print(list5)
    print(list6)  

  ##  sys.exit()  
    
    indForFolderClustering = []    
    
    list1FirstArr = np.array([np.array([list1[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list1FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list1FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           if len(ind_data[0]) == 2:
               print("Not singular")
               singIndData.append(ind_data[0,0])
               singIndData.append(ind_data[0,1])           
    
    newSingData = []
    
    singIndData = np.array([np.unique(np.array([singIndData]))]).tolist()[0]
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))     
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)    
          
    else:
        print("Not equal at phase 2") 
        
    
  ##  sys.exit()
    
    list2FirstArr = np.array([np.array([list2[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list2FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list2FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1])    
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd) 
           
    else:
        print("Not equal at phase 2")   
    
    list3FirstArr = np.array([np.array([list3[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list3FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list3FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1])  
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd)     
          
    else:
        print("Not equal at phase 2")   
    
    list4FirstArr = np.array([np.array([list4[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list4FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list4FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1])  
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
            
            indForFolderClustering.append(uniqueNewInd) 
            
    else:
        print("Not equal at phase 2")
    
    list5FirstArr = np.array([np.array([list5[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list5FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list5FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1]) 
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
             
            indForFolderClustering.append(uniqueNewInd) 
            
    else:
        print("Not equal at phase 2")
        
    list6FirstArr = np.array([np.array([list6[0]])[0]])    
    listArrToCompare = np.array([secRoundNewListMetricsArr[:,0]])
    
    singIndData = []
    
    for ind in range(0,len(list6FirstArr[0])):
       tupleIndiceImage = np.where(listArrToCompare == list6FirstArr[0,ind])
       rub, ind_data = tupleIndiceImage
       ind_data = np.array([ind_data])
       
       if len(ind_data[0]) == 1:
           singIndData.append(ind_data)
       else:
           print("Not singular")
           singIndData.append(ind_data[0,0])
           singIndData.append(ind_data[0,1]) 
    
    newSingData = []
    
    if True:
        for arrSin in singIndData:
            newSingData.append(int(arrSin))
    
        newSingIndArr = np.zeros((1,len(newSingData)))    
        newSingIndArr = np.array([newSingData]) 
        
        uniqueNewInd = np.array([np.unique(newSingIndArr)])
        
        if len(uniqueNewInd[0]) == len(newSingIndArr[0]):
            print("All indices unique")
             
            indForFolderClustering.append(uniqueNewInd)        
           
    else:
        print("Not equal at phase 2")
        
    #%%
    
    
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################
    ###############################################################################################################################################

    name_folder = first_clustering_storing_output + str(sequence_name) + "_7_8"
    newPath = os.path.join(parent_dir,name_folder)    
    
    
    if "\n" in newPath or "\t" in newPath or "\r" in newPath or " " in newPath:                                      
    
        n_path = ""
        
        for let in newPath:
            if let != "\n" and let != "\r" and l != "\t" and l != " ":
                n_path += let
        
        print("NewPath Bef: " + str(n_path))
        n_path = n_path[1:]
        newPath = n_path
        
        print("NewPath After: " + str(newPath))
        
        if '/' in newPath:
            print("\n\n A \n\n")
            x = 0
            while os.path.exists(newPath + '/'):
                newPath += '_' + str(x)
                x += 1
            nPath = newPath.split('/')
            pas = nPath[-1]
            rpas = nPath[:-1]
            pas = pas[2:]
            pasn = ""
            for p in pas:
                if p != "\t" and p != " ":
                    pasn += p
            pas = pasn
            rpas.append(pas)
            nnPath = ""
            for r in rpas:
                nnPath += r + '/'
            
            nnPath = nnPath[:-1]
            newPath = nnPath
            
        elif "\\" in newPath:
            print("\n\n B \n\n")
            if os.path.exists(newPath + "\\"):
                x = 0
                while os.path.exists(newPath + "\\"):
                    newPath += '_' + str(x)
                    x += 1
                nPath = newPath.split('/')
                pas = nPath[-1]
                rpas = nPath[:-1]
                pas = pas[2:]
                pasn = ""
                for p in pas:
                  if p != "\t" and p != " ":
                     pasn += p
                pas = pasn
                rpas.append(pas)
                nnPath = ""
                for r in rpas:
                   nnPath += r + '/'
                   
                nnPath = nnPath[:-1] 
                newPath = nnPath
        
        indxp = 0
        for indp, p in enumerate(newPath):
            if indp < 10:
                if p == "\n" or p == ' ' or p == "\t":
                    indxp = indp
                    
        newPath = newPath[indxp+1:]     
        
        print("newPath_ " + str(newPath))
        
        indIntruse = 0
        
        for indN, n in enumerate(newPath):
            if indN > 1 and n == ':':
                indIntruse = indN
        
        newPathN = newPath[:indIntruse] + newPath[(indIntruse+1):]
        
        newPath = newPathN
        
        if os.path.exists(newPath + "\\"):
            x = 0
            while os.path.exists(newPath + "\\"):
                
                if newPath[-2] == '_':
                    newPath = newPath[:-2] 
                if newPath[-3] == '_':
                    newPath = newPath[:-3] 
                    
                newPath += '_' + str(x)
                x += 1
            else:
                os.mkdir(newPath) 
    else:
        print("\n\n C \n\n")
        os.mkdir(newPath) 
        
    print("\n\n newPath + /")
    print(newPath + "/")
    print("\n\n")
    
    if not os.path.exists(newPath + "/"):
        os.mkdir(newPath) 
     
    print("First Directory created")
    
    trainFolder = "Train_Results" 
     
    newPath = os.path.join(newPath + "/", trainFolder)   
    os.mkdir(newPath) 
    
 #   newPath = newPath + "/" 
    
    print("Second Directory created")
     
    sub_name_folder1 = "Class_1"
    newPath_1 = os.path.join(newPath + "/",sub_name_folder1)
    
    print("\n\n newPath1: ")
    print(newPath_1)
    print("\n\n")
    
    if not os.path.exists(newPath_1 + "/"):
        os.mkdir(newPath_1)
     
    sub_name_folder1 = "Class_2"
    newPath_2 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_2 + "/"):        
        os.mkdir(newPath_2) 
         
    sub_name_folder1 = "Class_3"
    newPath_3 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_3 + "/"):        
        os.mkdir(newPath_3)   
     
    sub_name_folder1 = "Class_4" 
    newPath_4 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_4 + "/"): 
        os.mkdir(newPath_4)
     
    sub_name_folder1 = "Class_5"
    newPath_5 = os.path.join(newPath + "/",sub_name_folder1)
    
    if not os.path.exists(newPath_5 + "/"):    
        os.mkdir(newPath_5) 
         
    sub_name_folder1 = "Class_6"
    newPath_6 = os.path.join(newPath + "/",sub_name_folder1) 
    
    if not os.path.exists(newPath_6 + "/"):        
        os.mkdir(newPath_6)      
    
    counter_1 = 0
    counter_2 = 0 
    counter_3 = 0
    counter_4 = 0
    counter_5 = 0
    counter_6 = 0
    
    print("\n\n Roi set:")
    print(roi_set)   
    
    
    metricsIdTtrain1 = []
    metricsIdTtrain2 = []
    metricsIdTtrain3 = []
    metricsIdTtrain4 = []
    metricsIdTtrain5 = []
    metricsIdTtrain6 = []
    
    print("\nLength for indForFolderClustering: " + str(len(indForFolderClustering)))
    print("\n\n")
    
    if not (os.listdir(newPath_1) or os.listdir(newPath_2) or os.listdir(newPath_3) or os.listdir(newPath_4) or os.listdir(newPath_5) or os.listdir(newPath_6)):
        print("\n\n Inside \n\n")
        for indCluster, cluster in enumerate(indForFolderClustering):
            cluster_list = cluster.tolist()
            print("Cluster number " + str(indCluster))
            print("Cluster list:")
            print(cluster_list)
            for ind_imageInCluster in cluster_list[0]: 
                print("Image number " + str(ind_imageInCluster))
                image_counter = cv2.imread(roi_bef + "/roi_image" + str(ind_imageInCluster) + ".jpg")
                image_counter_2 = cv2.imread(roi_after + "/roi_image" + str(ind_imageInCluster) + ".jpg")       
                
                print(roi_bef + "/roi_image" + str(ind_imageInCluster) + ".jpg")
                print("Here") 
                print("image_counter: ")
                print(image_counter)
                
                print("\n\n")
                
                print(roi_after + "/roi_image" + str(ind_imageInCluster) + ".jpg")
                print("Here 2") 
                print("image_counter_2: ")
                print(image_counter_2)
                
                if indCluster == 0:  
                    print("\n Put inside folder for cluster 0 \n")
                    if image_counter is not None:
                        cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(0) + ".jpg", image_counter)
                    if image_counter_2 is not None:
                        cv2.imwrite(newPath_1 + "/" + "image_" + str(counter_1) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain1.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_1 += 1  
                if indCluster == 1: 
                    print("\n Put inside folder for cluster 1 \n")
                    if image_counter is not None:
                        cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(0) + ".jpg", image_counter)
                    if image_counter_2 is not None:
                        cv2.imwrite(newPath_2 + "/" + "image_" + str(counter_2) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain2.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_2 += 1
                if indCluster == 2: 
                    print("\n Put inside folder for cluster 2 \n")
                    if image_counter is not None:
                        cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(0) + ".jpg", image_counter)
                    if image_counter_2 is not None:
                        cv2.imwrite(newPath_3 + "/" + "image_" + str(counter_3) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain3.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_3 += 1 
                if indCluster == 3:
                    print("\n Put inside folder for cluster 3 \n")
                    if image_counter is not None:
                        cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(0) + ".jpg", image_counter)
                    if image_counter_2 is not None:
                        cv2.imwrite(newPath_4 + "/" + "image_" + str(counter_4) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain4.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_4 += 1
                if indCluster == 4:
                    print("\n Put inside folder for cluster 4 \n")
                    if image_counter is not None:
                        cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(0) + ".jpg", image_counter)
                    if image_counter_2 is not None:
                        cv2.imwrite(newPath_5 + "/" + "image_" + str(counter_5) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain5.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_5 += 1
                if indCluster == 5:
                    print("\n Put inside folder for cluster 5 \n")
                    if image_counter is not None:
                        cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(0) + ".jpg", image_counter) 
                    if image_counter_2 is not None:
                        cv2.imwrite(newPath_6 + "/" + "image_" + str(counter_6) + "_" + str(1) + ".jpg", image_counter_2)
                    metricsIdTtrain6.append(secRoundNewListMetricsArr[ind_imageInCluster, :])
                    counter_6 += 1
         
        LenListsClusters1= [LenList_1, LenList_2, LenList_3, LenList_4, LenList_5, LenList_6]
        
        ##########################################################################################################
        ##########################################################################################################
        ##########################################################################################################
        ##########################################################################################################
        
        m_1_f = []
        m_2_f = []
        m_3_f = []
        m_4_f = []
        m_5_f = []
        m_6_f = []
        
        
        metrics_afterClustering = []
        
        if len(metricsIdTtrain1) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain1 = np.array([metricsIdTtrain1]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain1) 
            
            for m_1 in metricsIdTrain1:
                m_1_f.append(np.mean(np.array([m_1])[0,:,0]))    
            
        if len(metricsIdTtrain2) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain2 = np.array([metricsIdTtrain2]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain2)
            
            for m_2 in metricsIdTrain2:
                m_2_f.append(np.mean(np.array([m_2])[0,:,0]))
            
        if len(metricsIdTtrain3) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain3 = np.array([metricsIdTtrain3]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain3)
            
            for m_3 in metricsIdTrain3:
                m_3_f.append(np.mean(np.array([m_3])[0,:,0]))
            
        if len(metricsIdTtrain4) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain4 = np.array([metricsIdTtrain4]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain4)
            
            for m_4 in metricsIdTrain4:
                m_4_f.append(np.mean(np.array([m_4])[0,:,0]))
            
        if len(metricsIdTtrain5) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain5 = np.array([metricsIdTtrain5]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain5)
            
            for m_5 in metricsIdTrain5:
                m_5_f.append(np.mean(np.array([m_5])[0,:,0]))
            
        if len(metricsIdTtrain6) == 0: 
            print("Cluster not formed. Discarding this one")
        else:
            metricsIdTrain6 = np.array([metricsIdTtrain6]).T.tolist()
            metrics_afterClustering.append(metricsIdTrain6)
            
            for m_6 in metricsIdTrain6:
                m_6_f.append(np.mean(np.array([m_6])[0,:,0]))
                
                
        #################################################################################################################
        #################################################################################################################
        #%%
        
        labelsMetricsToScore = ['MSE', 'RMSE', 'RMSE_SINGLE', 'RMSE_SW', 'PSNR', 'UQI_SINGLE', 
                                'UQI', 'SSIMS', 'CSS', 'ERGAS', 'SCC',
                                'RASE', 'SAM', 'MSSSIM', 'VIFP', 'PSNRB',
                                'Mean', 'STD', 'Contrast', 'ASM', 'Max']
        
        stdListValues = []
        global_std = [] 
        
        secondMetricTable = np.array([metrics_afterClustering])[0].T.tolist() 
        
        listOfFlatten = []
        stdListValuesMetrics = []
        
        for indSec, secInd in enumerate(secondMetricTable):    
            flattenSecond = []
            flatten_list = [element for sublist in secInd for element in sublist]
            
            for fla in flatten_list:
                fla_one = np.array([fla])[0,0]
                flattenSecond.append(fla_one)
        
            listOfFlatten.append(flattenSecond)
        
        for indFlatten, metricFlatten in enumerate(listOfFlatten):
            stdValue = np.std(np.array([metricFlatten]))
            
            stdListValuesMetrics.append(stdValue)
        
        newstdlistValuesMetrics = []
        mainIndices = []
        
        sortedIndices = []
        listToPCA = []
        
        sorted_std_values = sorted(stdListValuesMetrics, reverse=True)
        
        for ind, stdValueSorted in enumerate(sorted_std_values):
            indStuff = np.where(np.array([stdListValuesMetrics]) == stdValueSorted)
            rub, indS = indStuff
            indS = np.array([indS])[0,0]
            indOfSorted = indS
            sortedIndices.append(indOfSorted)
        
        sortedIndicesToGo = sortedIndices[0:15]
        
        exlistToPCA = secRoundNewListMetricsArr.T.tolist()
        
        for ind_listPCA, listPCA in enumerate(exlistToPCA):
            if ind_listPCA in sortedIndicesToGo and ind_listPCA != 16: 
                listToPCA.append(listPCA) 
                
        filtered_metrics = []
        newstdlistValuesMetrics_sec = []
        metricsToPCA_analysis = [] 
        metricsToPCA_norm = []
        
        for ind_mat1, mat1 in enumerate(metrics_afterClustering):
            mat1_n = []
            for ind_mat2, mat2 in enumerate(mat1):
                mat2_n = [] 
                for ind_mat3, mat3 in enumerate(mat2):
                    mat3 = mat3[0]
                    mat2_n.append(mat3)
                mat1_n.append(mat2_n)  
            
            metricsToPCA_analysis.append(mat1_n)
         
        ## Standard normalization ########################################################################################################
        ##################################################################################################################################
        
        metricsToPCA_norm = []
        
        for met1 in listToPCA:     
            
                mean_value = np.mean(np.array([met1]))
                std_value = np.std(np.array([met1])) 
                metricPCA_norm1 = [] 
                
                for met2 in met1:
                    metricPCA_norm1.append((met2-mean_value)/std_value)        
                
                metricsToPCA_norm.append(metricPCA_norm1)
        
        ##################################################################################################################################
        #%%
        
        dfxi = pd.DataFrame(data=metricsToPCA_norm)  
        
        dfxi = dfxi.dropna()
        dfxi = dfxi.dropna(axis=1)
        
        print("Dataframe: ") 
        print(dfxi.to_string())   
        
        pcai = PCA(n_components=None)                             
        
        dfx_pcai = pcai.fit(dfxi)      ## Error
         
        X_pcai = pcai.transform(dfxi)   
        dfx_transi = pd.DataFrame(data=X_pcai)
        
        plt.scatter(dfx_transi[0], dfx_transi[1], c ="blue")
        plt.title("Correlation between first two PCA components")
        plt.xlabel("First PCA component")
        plt.ylabel("Second PCA component")
        plt.show()
        
        pca_coef_feat_first_comp = dfx_transi[0].tolist()
        abs_pca_coeff = np.array([abs(np.array([pca_coef_feat_first_comp])[0])])
        abs_pca_coeffList = abs_pca_coeff.tolist()
        
        ## sortedPCA_coeff = sorted(abs_pca_coeffList[0], reverse=True)
        sortedPCA_coeff = sorted(abs_pca_coeffList[0])
        
        ## ContrastAppending = sortedPCA_coeff[18] 
        
        sortedPCA_coeff = sortedPCA_coeff[0:4]
        ## sortedPCA_coeff = sortedPCA_coeff.append(ContrastAppending)
        
        sortedIndices2 = []
        
        for ind, pcaSorted in enumerate(sortedPCA_coeff):
            indStuff = np.where(np.array([abs_pca_coeffList]) == pcaSorted)
            rub, rub2, indS = indStuff  
            indOfSorted = indS 
            sortedIndices2.append(indOfSorted)    
             
        doubleSortedElements = []
        
        indicesSortedFromPCA = [] 
            
        for sorted_ind in sortedIndices2:
            if len(sorted_ind) == 2:
                doubleSortedElements.append(sorted_ind)
                list_aux = sorted_ind.tolist()
                if list_aux[0] in indicesSortedFromPCA:
                    indicesSortedFromPCA.append(list_aux[1])
                else:
                    indicesSortedFromPCA.append(list_aux[0])
            else:
                if len(sorted_ind) == 1:
                    indicesSortedFromPCA.append(np.array([sorted_ind])[0,0])
                    
        indicesSortedFromPCA.append(18)            
                    
        remainingMetricsToClustering = [] 
        trainListData = []
        
        for ind in indicesSortedFromPCA: 
            if ind <= 21 and ind < len(exlistToPCA):
                remainingMetricsToClustering.append(labelsMetricsToScore[ind])   
                trainListData.append(exlistToPCA[ind]) 
                 
        nclusters = 2
          
        trainListData = np.array([trainListData])[0].T.tolist()
        
        train_data = np.array([trainListData])[0].T
        treino_lenght = 370
        
        kmeans = KMeans(n_clusters=nclusters, max_iter=500, n_init=8).fit(trainListData) 
        Cluster_ID = kmeans.labels_ 
        centroides_A = kmeans.cluster_centers_   
        print("Centroides dos ",number_recommended_clusters, " clusters recomendados:\n", centroides_A)
         
        
        
        Cluster_ID_transpose = np.array([np.array([Cluster_ID]).T[0:treino_lenght,0]])
        
        objetos_c1 = []
        objetos_c2 = []
        ind_first = []
        ind_second = []
        
        for i in range (0, len(Cluster_ID_transpose[0])):
            if i<296:
                if Cluster_ID_transpose[0,i] == 0:
                    objetos_c1.append(train_data[:,i])             
                    i_1 = i
                    ind_first.append(i)
                elif Cluster_ID_transpose[0,i] == 1:
                    objetos_c2.append(train_data[:,i]) 
                    i_2 = i
                    ind_second.append(i)
                    
        trainDataFurther = train_data.tolist()
        classClustering = []
        secFurther = []
        
        for tF in trainDataFurther:
            tF = tF[0:296]
            secFurther.append(tF)
        
        for indHere in range(0,296):
            if indHere in ind_first:
                classClustering.insert(indHere, 'A')        
            else:
                if indHere in ind_second:
                    classClustering.insert(indHere, 'B')
        
        secFurther.insert(0, classClustering)
        classCl = secFurther[0]
        classAFurther = []
        classBFurther = []
        secFurther = np.array([secFurther])[0].T.tolist()
        
        for sInd, sF in enumerate(secFurther):
            if classCl[sInd] == 'A':
                classAFurther.append(sF[1:])
            else:
                if classCl[sInd] == 'B':
                    classBFurther.append(sF[1:])
                    
        clustering_inf_data = [classAFurther, classBFurther, nclusters, number_recommended_clusters, remainingMetricsToClustering]
        
        #%%
                    
        dfxi1 = pd.DataFrame(data=classAFurther) 
        pcai1 = PCA(n_components=None) 
        dfx_pcai1 = pcai1.fit(dfxi1)   
         
        X_pcai1 = pcai1.transform(dfxi1)   
        dfx_transi1 = pd.DataFrame(data=X_pcai1)
        X_pcai1T = X_pcai1.T.tolist()
        
        dfxi2 = pd.DataFrame(data=classBFurther)  
        
        pcai2 = PCA(n_components=None) 
        dfx_pcai2 = pcai2.fit(dfxi2)    
        X_pcai2 = pcai2.transform(dfxi2)   
        dfx_transi2 = pd.DataFrame(data=X_pcai2)
        
        X_pcai2T = X_pcai2.T.tolist()
        
        centroid_pca_A = []
        centroid_pca_B = []
        
        for indA in range(0,5):
            if indA < len(X_pcai1T):
                mean_value = np.mean(np.array([X_pcai1T[indA]]))
                centroid_pca_A.append(mean_value)    
            
        for indB in range(0,5):
            if indB < len(X_pcai2T):
                mean_value = np.mean(np.array([X_pcai2T[indB]]))
                centroid_pca_B.append(mean_value)
        
        if len(centroid_pca_A) < len(centroid_pca_B):
            centroid_pca_B = centroid_pca_B[:len(centroid_pca_A)]
        elif len(centroid_pca_A) > len(centroid_pca_B):
            centroid_pca_A = centroid_pca_A[:len(centroid_pca_B)]
            
            
        distCentroidsPCA = np.linalg.norm(np.array([centroid_pca_A])[0]-np.array([centroid_pca_B])[0])
        
        #####
        thisDir = os.getcwd()
        dirResultsOutput = thisDir + '\\GraphsOutput\\'
        
        if os.path.isdir(dirResultsOutput) == False:  
        
            dirResults = os.path.join(dirResultsOutput) 
            os.mkdir(dirResults)    
        
        #####
        
        if len(dfx_transi1[0]) > 0 and len(dfx_transi1[1]) > 0:    
            plt1 = plt.scatter(dfx_transi1[0], dfx_transi1[1], c ="blue")
        if len(dfx_transi2[0]) > 0 and len(dfx_transi2[1]) > 0:
            plt2 = plt.scatter(dfx_transi2[0], dfx_transi2[1], c ="red")
            
        plt.legend((plt1, plt2),
                   ('Class A', 'Class B'))
        plt.title("Correlation between first two PCA components")
        plt.xlabel("First PCA component")
        plt.ylabel("Second PCA component")    
        
        plt.savefig(dirResultsOutput + "pca_graph.png")
        
        plt.show() 
        
        data_results.append(clustering_inf_data)
        
        data_results.append(dirResultsOutput + "pca_graph.png")
        
        
        list1 = list(zip(*objetos_c1)) 
        list2 = list(zip(*objetos_c2)) 
        
        #%%
        
        list1ToDist = np.array([list1])[0].T
        list2ToDist = np.array([list2])[0].T 
        
        dists1 = []
        dists2 = []
        cent_1 = np.array([centroides_A[0,:]])
        cent_2 = np.array([centroides_A[1,:]])
        
        for i in range(0,len(list1ToDist)):
            
            a = list1ToDist[i,:]
            b = cent_1[0,:]    
            dist = np.linalg.norm(a-b)    
            dists1.append(dist) 
            
        for i in range(0,len(list2ToDist)):
            
            a = list2ToDist[i,:]
            b = cent_2[0,:]
            dist = np.linalg.norm(a-b)    
            dists2.append(dist)
        
        xDist1 = []
        xDist2 = []
        indD_1 = 0
        indD_2 = 0
        
        for ind in range(0, len(dists1)):
            indD_1 += 1
            xDist1.append(indD_1)
        
        for ind in range(0, len(dists2)):
            indD_2 += 1
            xDist2.append(indD_2)
            
        plt.scatter(xDist1, dists1, c ="blue")
        plt.title("Distance to centroid A")
        plt.xlabel("Number of point")
        plt.ylabel("Distance of points of first cluster to its centroid")
        plt.savefig(dirResultsOutput + "distances_firstCluster.png")
        plt.show()
        
        data_results.append(dirResultsOutput + "distances_firstCluster.png")
        
        plt.scatter(xDist2, dists2, c ="blue")
        plt.title("Distance to centroid B")
        plt.xlabel("Number of point")
        plt.ylabel("Distance of points of second cluster to its centroid")
        plt.savefig(dirResultsOutput + "distances_secondCluster.png")  
        plt.show() 
        
        data_results.append(dirResultsOutput + "distances_secondCluster.png")
        
            
        print(" -- Lists of clusters generated")
        
        for l in list1:
            LenList_1 = len(l)
        for l in list2:
            LenList_2 = len(l)
        
        #%%
            
        trainListData = np.array([trainListData])[0].T.tolist()   
            
        Y_euclidean = pdist(trainListData, metric='euclidean')
        Y_euclidean_square = squareform(Y_euclidean)
        Y_cityblock = pdist(trainListData, metric='cityblock')
        Y_euclidean_square = squareform(Y_cityblock)
        
        Z_euclidean_average = linkage(trainListData, method='average', metric='euclidean')
        Z_euclidean_ward = linkage(trainListData, method='ward', metric='euclidean')
        
        Z_cityblock_average = linkage(trainListData, method='average', metric='cityblock') 
        
        distances_from_euclidean_average = Z_euclidean_average[:,2].tolist()
        
        clustersList = Z_euclidean_average[:,0].tolist() + Z_euclidean_average[:,1].tolist()
        
        totNumbClusters_firstExp = np.max(np.array([clustersList])[0])
        
        numberObservationsForEachCluster_first = Z_euclidean_average[:,3].tolist()
        totObservationsFirst = np.sum(np.array([numberObservationsForEachCluster_first]))
        
        
        distances_from_euclidean_ward = Z_euclidean_ward[:,2].tolist()
        
        clustersList = Z_euclidean_ward[:,0].tolist() + Z_euclidean_average[:,1].tolist()
        
        totNumbClusters_secExp = np.max(np.array([clustersList])[0])
        
        numberObservationsForEachCluster_second = Z_euclidean_ward[:,3].tolist()
        totObservationsSec = np.sum(np.array([numberObservationsForEachCluster_second]))
        
        
        
        distances_from_cityblock_average = Z_cityblock_average[:,2].tolist()
        
        clustersList = Z_cityblock_average[:,0].tolist() + Z_euclidean_average[:,1].tolist()
        
        totNumbClusters_thirdExp = np.max(np.array([clustersList])[0])
        
        numberObservationsForEachCluster_third = Z_cityblock_average[:,3].tolist()
        totObservationsThird = np.sum(np.array([numberObservationsForEachCluster_third]))
        
        distancesMeanVar = [np.mean(np.array([distances_from_euclidean_average])), np.mean(np.array([distances_from_euclidean_ward])), np.mean(np.array([distances_from_cityblock_average]))]
        labelsMeasuresDistances = ['Euclidean Average', 'Euclidean Ward', 'Cityblock Average']
        
        maxMeanDistance = 0
        
        for indDist, distMeanValue in enumerate(distancesMeanVar):
            if distMeanValue > maxMeanDistance:
                maxMeanDistance = distMeanValue
                indMaxMeasureDistance = indDist 
        
        print("Selected Measure for distance between clusters: " + labelsMeasuresDistances[indMaxMeasureDistance])
        
        distancesBetClustersBestMeasure = []
        
        if indMaxMeasureDistance == 0:
            distancesBetClustersBestMeasure = Z_euclidean_average.tolist()
            numberClustersFromDist = totNumbClusters_firstExp
        else:
            if indMaxMeasureDistance == 1:
                distancesBetClustersBestMeasure = Z_euclidean_ward.tolist()
                numberClustersFromDist = totNumbClusters_secExp
            else:
                if indMaxMeasureDistance == 2:
                    distancesBetClustersBestMeasure = Z_cityblock_average.tolist()
                    numberClustersFromDist = totNumbClusters_thirdExp
                    
        distance_output = []
                    
        for dist1 in distancesBetClustersBestMeasure:
            distance_output2 = []
            for ind_dist2, dist2 in enumerate(dist1):
                if ind_dist2 != 2:
                    distance_output2.append(int(dist2))
                else:
                    distance_output2.append(dist2)
            distance_output.append(distance_output2)
            
        #### Comparison of number of clusters between distances approach and the clustering one:
        if  number_recommended_clusters == numberClustersFromDist:
            print("The above methods provide the same number of clusters")
        else:
            if number_recommended_clusters > numberClustersFromDist:
                print("The number of recommended clusters (from clustering approach) is higher than the number of clusters computed from dist-linkage method.")
            else:
                if number_recommended_clusters < numberClustersFromDist:
                    print("The number of clusters computed from dist-linkage method is higher than the number of recommended clusters (from clustering approach).")
        
        
        #%%     
        
        def convolve2D(image, kernel, padding=0, strides=1):
            # Cross Correlation
            kernel = np.flipud(np.fliplr(kernel))
        
            # Gather Shapes of Kernel + Image + Padding
            xKernShape = kernel.shape[0]
            yKernShape = kernel.shape[1]
            xImgShape = image.shape[0]
            yImgShape = image.shape[1]
        
            # Shape of Output Convolution
            xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
            yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
            output = np.zeros((xOutput, yOutput))
        
            # Apply Equal Padding to All Sides
            if padding != 0:
                imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
                imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
                print(imagePadded)
            else:
                imagePadded = image
        
            # Iterate through image
            for y in range(image.shape[1]):
                # Exit Convolution
                if y > image.shape[1] - yKernShape:
                    break
                # Only Convolve if y has gone down by the specified Strides
                if y % strides == 0:
                    for x in range(image.shape[0]):
                        # Go to next row once kernel is out of bounds
                        if x > image.shape[0] - xKernShape:
                            break
                        try:
                            # Only Convolve if x has moved by the specified Strides
                            if x % strides == 0:
                                output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                        except:
                            break
        
            return output
        
        #%%
        
        k1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        k2 = [[1, 0, -1],  [2, 0, -2], [1, 0, -1]]
        
        k1_arr = np.array([k1])[0]
        k2_arr = np.array([k2])[0]
        
        M_c = []
         
        #%%
        
        ind_slash = 0
        
        for ind_x, x in enumerate(IFVP):
            if x == '/':
                ind_slash = ind_x
                break
        
        IFVP_p = IFVP[ind_slash:]
        IFVP = 'C:' + IFVP_p
                
        
        # if IFVP[0] != 'C' and IFVP[0] == ':':
        #     IFVP = 'C' + IFVP
        # elif IFVP[0] != 'C' and  IFVP[0] != ':':
        #     IFVP = IFVP[1:]
        
        print("\n IFVP: " + IFVP)
        print("\n") 
        
        seqNew = ""
        
        for i in sequence_name:
            if i.isdigit():
                seqNew += i
        
        sequence_name = seqNew
        
        print(IFVP + str(sequence_name) + "_2" + "/video_image")
        
        indPx = 0
        
        for indP, p in enumerate(IFVP):
            if p == ' ' or p == '\n':
          #      print("\n Space here \n")
          #      print(indP)
              indPx = indP
        
        IFVPo = IFVP[:indPx]
        IFVP = IFVPo
        
      #   for video_image in range(0,count):           #### 321   ## from 361   ## to count = 637
            
      #           print("Analysing for image " + str(video_image) + " th")
                
      # #          print("")
                
      #           x = IFVP + str(sequence_name) + "_2" + "/video_image%d.jpg" % video_image
      #           print(x)
      #           print("\n") 
                
      #           toCompare = 'C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/out16823_1/Image_Processing/DataSequence__160823_164429_0_2/video_image' + str(video_image) + '.jpg'
                
      #           if x == toCompare:
      #               print("Equal")
      #           else:
      #               for indI, i in x:
      #                   if i != toCompare[indI]:
      #                       print("Different at index" + str(indI))
      #                       print(x[indI-2:indI+2])
      #                       print(toCompare[indI-2:indI+2])
                
      #           # xi = ""
                 
      #           # for i in x:
      #           #     if i == '/' or i == "/":
      #           #         i == "\\"
                        
      #           #     xi.append(i)                
            
      #           image = cv2.imread(x)        
      #    #       image = cv2.imread('C:/Users/Other/files_python/py_scripts/ffmpeg-5.0.1-full_build/bin/GUI/out16823_1/Image_Processing/DataSequence__160823_164429_0_2/video_image0.jpg')
      #           print("\n")
      #           print(image)
      #           print("\n")
                 
      #           if image is not None:
      #               imx = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
                   
               
                 
      #               M1 = convolve2D(imx, k1)
      #               M2 = convolve2D(imx, k2)      
                    
                    
      #               print("Generating output for image " + str(video_image))
                    
      #               comp_x = np.power(np.array([M1])[0], 2)      
      #               comp_y = np.power(np.array([M2])[0], 2) 
                    
      #               sum_comps = comp_x + comp_y 
                    
      #               gen_output = np.power(sum_comps, 1/2).astype(int)
                    
      #               cv2.imwrite(roi_path + "/gen_output_%d.jpg" % video_image, gen_output)
                    
      #       #######################################
      #       #######################################
      #       #######################################
      #       #######################################
      #       #######################################  
             
      #               executionTime = (time.time() - startTime)
      #               print('Execution time in seconds: ' + str(executionTime))
      #           else:
      #               break
                
        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime))
        
        print("Here")
        clustering_output.append((clustering_inf_data, executionTime, count))
        print("Inter")
        gui_show_results(clustering_output, executionTime, count)
        
    else:
        print("\n\n Clustering processed before \n Ending ... \n\n\n")    