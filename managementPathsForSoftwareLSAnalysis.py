# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:00:22 2022

@author: marco
"""

from softwareImageProcessingForLaserSpeckleAnalysis import videoAnalysis
from getCurrentDateAndTime import getDateTimeStrMarker

import os

## Paths Definition  

dateTimeMarker = getDateTimeStrMarker()

sequence_name = 315
dest_path = 'C:/Research/Image_Processing/'
mainPathVideoData = dest_path + 'VideoData_' + dateTimeMarker + '/'
mainPathVideo = os.path.join(mainPathVideoData)
os.mkdir(mainPathVideo) 
mtsVideoPath = dest_path + 'VideosAlmostLaserSpeckle/' 
mp4VideoFile = dest_path + 'VideosAlmostLaserSpeckle' + dateTimeMarker + '/' 
mp4VideoFilePath = os.path.join(mp4VideoFile)
os.mkdir(mp4VideoFilePath)  
# mtsVideoPathP = os.path.join(mtsVideoPath)
# os.mkdir(mtsVideoPathP) 
IFVP = dest_path + 'DataSequence_'
locationMP4_file = 'FilesFor_'
roiPath = 'Approach'
newRoiPath = 'Approach_new'
pathRoiStart = dest_path + 'modRoisFirstMom_'
pathRoiEnd = dest_path + 'modRoisSecMom_'
first_clustering_storing_output = "Quality_kMeans_Clustering_real_"
pathPythonFile = dest_path + "SpeckleTraining/ffmpeg-5.0.1-full_build/bin"

## Adding date and time to which one of the paths shown above
IFVP = IFVP + dateTimeMarker
locationMP4_file = locationMP4_file + dateTimeMarker 
roiPath = roiPath + dateTimeMarker
newRoiPath = newRoiPath + dateTimeMarker 
pathRoiStart = pathRoiStart + dateTimeMarker 
pathRoiEnd = pathRoiEnd + dateTimeMarker 
first_clustering_storing_output = first_clustering_storing_output + dateTimeMarker


## Run Laser Speckle Video Analyser for the input data above
percIntervalForK, executionTime, totCountImages = videoAnalysis(mainPathVideoData, sequence_name, dest_path, mtsVideoPath, mp4VideoFile, IFVP, locationMP4_file, roiPath, newRoiPath, pathRoiStart, pathRoiEnd, first_clustering_storing_output, pathPythonFile)

   

