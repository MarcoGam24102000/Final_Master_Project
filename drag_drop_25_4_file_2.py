# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:21:13 2023

@author: marco
"""

import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import PySimpleGUI as sg
import cv2


global str_paths

def create_video(image_paths, video_path, fps=25):
    # Get the first image to use as a template for the video dimensions
    
    print("Creating video ...")
    
    first_image = cv2.imread(image_paths[0])
    print("First image: ")
    print(first_image)
    height, width, channels = first_image.shape

    # Create a VideoWriter object to write the video to disk
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Loop over the image paths, read in each image, and write it to the video
    for image_path in image_paths:
        image = cv2.imread(image_path)
        out.write(image)

    # Release the VideoWriter object to save the video
    out.release()
    
 
class DragDropImage(QtWidgets.QWidget):
    global str_paths
    def __init__(self):
        super().__init__()

        # Create a QLabel to display the path(s) of the dropped image(s)
        self.path_label = QtWidgets.QLabel()
        self.path_label.setAlignment(QtCore.Qt.AlignTop)
        self.path_label.setMinimumSize(200, 200)
        self.path_label.setWordWrap(True)

        # Create a QVBoxLayout to hold the QLabel
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.path_label)
        self.setLayout(self.layout)

        # Set up the window to accept drag and drop events
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        # Set the background color to indicate that the window can accept the drop
        self.setStyleSheet("background-color: #e0e0e0;")
        event.accept()

    def dragLeaveEvent(self, event):
        # Restore the background color
        self.setStyleSheet("")
        event.accept()

    def dropEvent(self, event):
        # Get the path to the dropped file
        path = event.mimeData().urls()[0].toLocalFile()

        # Append the path to the QLabel
        self.path_label.setText(self.path_label.text() + "\n" + path)
        
  ##      print(str(len(self.path_label.text())))
  
        global str_paths
  
        size_path = 50
        
        layout = [
            [sg.Button("Next", key = "-DROP_TO_VID-")]
        ]
        
        window = sg.Window("Drop to Video", layout)
        
        while True:
            
            if len(self.path_label.text()) > size_path*25:
                event, values = window.read()
                
                if event == sg.WIN_CLOSED or event == "Exit":
                    break
                elif event == "Next":
                    str_paths = self.path_label.text()
                    
                    break
            else:
                break
        
        
        # if len(self.path_label.text()) > size_path*5:
        
        #     resp = input("One more image: ")
            
        #     if not(resp == 'y' or resp == 'Y' or resp == 's' or resp == 'S'):
        #         print("Here")        
        #         str_paths = self.path_label.text()
        #         print("Here 2") 
        #         window.close()
        # #        sys.exit(app.exec_())
        #         print("Here 3") 
            

        # Restore the background color
        self.setStyleSheet("") 
        
        print("Here 4") 
        
        str_pathsList = str_paths.split('\n')
        
        for indS, s in enumerate(str_pathsList):
            print(str(indS) + ": " + s)

        
        video_path = "output_video.mp4"
        create_video(str_pathsList, video_path)
        
        
 #       return str_paths

## if __name__ == "__main__":

while True:
 ##   if __name__ == "__main__":
        app = QtWidgets.QApplication(sys.argv)
        window = DragDropImage()
        window.show()
        sys.exit(app.exec_())
    
    
    