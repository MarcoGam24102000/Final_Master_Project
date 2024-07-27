import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import time
import os


global str_paths

global n_images


def write_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for element in lst:
            file.write(str(element) + '\n')

def read_list_from_file(filename):
    lst = []
    
    with open(filename, 'r') as file:
        for line in file:
            lst.append(line.strip())
    
    return lst


def create_video(image_paths, video_path, fps=25):
    # Get the first image to use as a template for the video dimensions
    print("Creating video ...")
    first_image = cv2.imread(image_paths[0])
    height, width, channels = first_image.shape
    
    print("A")

    # Create a VideoWriter object to write the video to disk
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print("B")

    # Loop over the image paths, read in each image, and write it to the video
    for image_path in image_paths:
        image = cv2.imread(image_path)
        out.write(image)
        print("Written")  

    print("Before release")
    
    # Release the VideoWriter object to save the video
    out.release()
    
    print("Ending here ...")  


class DragDropImage(QtWidgets.QWidget):
    global str_paths
    
    global n_images
    
    print("B")

    def __init__(self, video_path):
        super().__init__()

        # Create a QListWidget to display the dropped image paths
        self.path_list_widget = QtWidgets.QListWidget()
        self.path_list_widget.setMinimumSize(200, 200)

        # Create a QVBoxLayout to hold the QListWidget
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.path_list_widget)
        self.setLayout(self.layout)

        # Set up the window to accept drag and drop events
        self.setAcceptDrops(True)

        self.video_path = video_path  # Store the video path as an instance variable

    def dragEnterEvent(self, event):
        # Set the background color to indicate that the window can accept the drop
        self.setStyleSheet("background-color: #e0e0e0;")
        event.accept()

    def dragLeaveEvent(self, event):
        # Restore the background color
        self.setStyleSheet("")
        event.accept()

    def dropEvent(self, event):
        # Get the list of dropped file URLs
        urls = event.mimeData().urls()

        # Iterate over the URLs and add the file paths to the QListWidget
        for url in urls:
            path = url.toLocalFile()
            self.path_list_widget.addItem(path)

        # Check if the maximum number of images has been reached
        if self.path_list_widget.count() >= 5:
            resp = QtWidgets.QMessageBox.question(self, "Add more images",
                                                  "Do you want to add more images?")
            if resp == QtWidgets.QMessageBox.No:
                # If the user chooses not to add more images, collect the paths and create the video
                str_paths = [self.path_list_widget.item(i).text() for i in
                             range(self.path_list_widget.count())]
                create_video(str_paths, self.video_path)  # Use the stored video path
                write_list_to_file(str_paths, 'imagesNotes.txt')
                
                self.close() 

        # Restore the background color
        self.setStyleSheet("") 
        
def functionProt(video_path):
    print("B")
    app = QtWidgets.QApplication(sys.argv)
    video_path += '.mp4'
    window = DragDropImage(video_path)  # Pass the video path as an argument
    window.show()
    print("C")
 
def drag_drop_activity(video_path): 
    print("A")
 #   if __name__ == "__main__": 
    if True:
#        print("A")

        print("B")
        app = QtWidgets.QApplication(sys.argv)
        video_path += '.mp4'
        window = DragDropImage(video_path)  # Pass the video path as an argument
        window.show()
        print("C")
        
        app.exec_()
        
     #   return 
        
#        sys.exit(app.exec_())
        
        

        
 ##       functionProt(video_path)
 
        # cv2.waitKey(0)
        
        # print("D")
        
        # again = True
        
        # while again == True:
        #     if os.path.isfile(video_path):
        #         again = False
        #         break
        #     else:
        #         again = True

        # print("Here")
        
        import cv2
        
        imgs = []
        
        str_paths = read_list_from_file('imagesNotes.txt')
        
        print("\n\nStr Paths: \n\n")
        print(str_paths)
        
        for s in str_paths:
            img = cv2.imread(s)
            imgs.append(img)
            
        
        return imgs  
        
    
        # import PySimpleGUI as sg
        
        # layout = [
        #     [sg.Button("Next")]
        # ]
        
        # window = sg.Window("", layout)
        
        # again = True
        
        # while again == True:
        #     event, values = window.read()
            
        #     if event == sg.WIN_CLOSED or event == "Exit" or event == "Next":
        #         again = False
        #         break
        #     else:
        #         again = True
            
    
  ##      sys.exit(app.exec_())   
       
    
      

## drag_drop_activity("output")