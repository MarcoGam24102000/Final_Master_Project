import os
import cv2
import PySimpleGUI as sg
from pypylon import pylon
import matplotlib.pyplot as plt

# Define some sample camera parameters
# width = 1280
# height = 1024
# exposure_time = 10000
# gain = 0

def single_image_acq(width, height, exposure_time, gain, flag, camera):
    
    import time
     
    print("Here - A")

    # Create a camera object and set the parameters
        
    again = False
    
    print("Here - B")
    print(camera)
    
    number_images_test = 50
    t_screen_time = 50
    
    # Acquire a single image and save it to a user-specified folder
 ##   folder_path = sg.popup_get_folder('Select folder to save image')
##    if folder_path:
    if True:
  ##      file_path = os.path.join(folder_path, 'image.png')
  
    ##        for i in range(0,number_images_test):
            if flag: 
                
                for i in range(0,100):  
                
                    with camera.GrabOne(1000) as grab_result:
                        print("Grabbing")
                        print(grab_result)
                        image = grab_result.Array
                        
                        print(image)
                  ##      cv2.imwrite(file_path, image)
                        
                   ##     image = cv2.imread(file_path)
                   
                 ##       print("(" + str(len(image)) + ", " + str(len(image[0])) + ")")   
                   
                        if image is not None:
                            
                            print("(" + str(len(image)) + ", " + str(len(image[0])) + ")")  
                        
                            image = cv2.resize(image,(960,540))
                        
                        # if i == number_images_test-1:
                        #     cv2.destroyWindow('image')
                        
                            cv2.imshow('image', image)
                            cv2.waitKey(t_screen_time)  
                        else:
                            print("Issue with camera ...") 
      
                # plt.clf()
                # plt.imshow(image)
                # plt.show()
                
      
  ##          cv2.waitKey(0)
        
    
            # Display the image using cv2.imshow()
            # image = cv2.imread(file_path)
            
            # image = cv2.resize(image,(960,540))
            
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)
        
            # Show a PySimpleGUI window with three buttons
            layout = [
                [sg.Button('Adjust')],
                [sg.Button('Take another')],
                [sg.Button('Next')]
            ]
            window = sg.Window('Image Acquisition', layout)
            
            # Loop to wait for button presses
            
            # cv2.destroyWindow('image')
            # time.sleep(5)            
            
            while True:
                event, values = window.read()
                if event == sg.WINDOW_CLOSED or event == 'Next':
                    # cv2.destroyWindow('image')
                    # time.sleep(5)
                    break
                elif event == 'Adjust':
                    # Execute the "Adjust" function here
                    sg.popup('Adjust function executed')
                elif event == 'Take another':
                    again = True
                    # cv2.destroyWindow('image')
                    # time.sleep(5)
                    window.close()
                    break
         
            print("Here - C \n Close")
            
            # Close the PySimpleGUI window and the camera connection
            window.close()
            camera.Close()
        
    return again  
    
