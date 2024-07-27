import os
import cv2
import PySimpleGUI as sg
from pypylon import pylon
import matplotlib.pyplot as plt
import time
import numpy as np

# Define some sample camera parameters
# width = 1280
# height = 1024
# exposure_time = 10000
# gain = 0


def show_message(message):
    # print(message)
    # time.sleep(5)
    
    sg.popup_timed(message, title='Message', auto_close_duration=5)

def draw_circle_and_measure_distance(image):
    # Convert the image to grayscale
##   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = image

   # Apply GaussianBlur to reduce noise and help circle detection
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)

   # Use Hough Circle Transform to detect circles
   circles = cv2.HoughCircles(
       blurred,
       cv2.HOUGH_GRADIENT,
       dp=1,
       minDist=20,
       param1=50,
       param2=30,
       minRadius=10,
       maxRadius=50
   )

   if circles is not None:
       circles = np.uint16(np.around(circles))

       # Draw the first detected circle
       circle = circles[0, 0]
       center_x, center_y, radius = circle
       cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)

       # Measure distance from circle center to image center
       distance_to_center = np.sqrt((center_x - image.shape[1] // 2)**2 + (center_y - image.shape[0] // 2)**2)

       # Show the image with the circle
       cv2.imshow("Image with Circle", image)

       # Show message based on the measured distance
       if center_x < image.shape[1] // 2 and distance_to_center > 50:
           show_message("Move to the left")
       elif center_x > image.shape[1] // 2 and distance_to_center > 50:
           show_message("Move to the right")
       elif center_y < image.shape[0] // 2 and distance_to_center > 50:
           show_message("Move up")
       elif center_y > image.shape[0] // 2 and distance_to_center > 50:
           show_message("Move down")
       else:
           show_message("You are centered!")

       # Wait for a moment before closing the window
       cv2.waitKey(5000)
       cv2.destroyAllWindows()
   else:
       print("No circle detected.")

def single_image_acq(width, height, exposure_time, gain, flag, camera):
    
    import time
     
    print("Here - A")

    # Create a camera object and set the parameters
        
    again = False
    
    print("Here - B")
    
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
                    
                    try:
                
                        with camera.GrabOne(1000) as grab_result:
                            image = grab_result.Array
                      ##      cv2.imwrite(file_path, image)
                    except:
                        
                        if camera.IsGrabbing():
                            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                            if grab_result.GrabSucceeded():
                                image = grab_result.Array
                    print(image)
                        
                   ##     image = cv2.imread(file_path)
                        
                    image = cv2.resize(image,(960,540))
                        
                        # if i == number_images_test-1:
                        #     cv2.destroyWindow('image')
                        
                    cv2.imshow('image', image) 
                    cv2.waitKey(t_screen_time)      
      
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
               
                if event == sg.WINDOW_CLOSED:
                    import sys
                    sys.exit()
                    
                elif event == 'Next':
                    # cv2.destroyWindow('image')
                    # time.sleep(5)
                    break
                elif event == 'Adjust':
                    # Execute the "Adjust" function here
           #         sg.popup('Adjust function executed')
                   draw_circle_and_measure_distance(image)
                   again = True
                   window.close()
                   break
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
    
