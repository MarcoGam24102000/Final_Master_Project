# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:07:23 2023

@author: marco
"""

# Python program to open the
# camera in Tkinter
# Import the libraries,
# tkinter, cv2

from preview_opt_for_pfs_like_cam import single_image_acq

def dynamic_video_repr(width, height, exposure_time, gain):
  
    import tkinter as tk
    from pypylon import pylon
    import cv2
    
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()    
      
    camera.Width.SetValue(width)
    camera.Height.SetValue(height)
    camera.ExposureTimeRaw = exposure_time
    camera.GainRaw = gain 
    
#    video_filename += '.mp4'
      
#    print("Video filename: " + video_filename)
    
    # Create a GUI
    root = tk.Tk()
    
    running = False
      
    # Define a video capture object
    # vid = cv2.VideoCapture(video_filename)
      
    # # Declare the width and height in variables
    # width, height = 960, 540
      
    # # Set the width and height
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
    
      
    def play_video():
        
        global running 
        
        print("Running: " + str(running))
        
        running = True
      	
        if running:
            
            again = single_image_acq(width, height, exposure_time, gain, running, camera)
            
            if not again:
                print("Quitting here ...")
                on_exit()
                # cv2.destroyWindow("Display")
                running = False
                # root.quit()         
    
    		# # Capture the video frame by frame
      #       ret, frame = vid.read()
            
      #       print("Ret: " + str(ret))
            
      #       vid.release()
            
      #       if ret:
    
      #           # Show the image
      #           cv2.imshow("Display",frame)
      #           print("Showing image")
      #       else:
      #           # Release the video capture and close the display window
      #           vid.release()
      #           cv2.destroyAllWindows()
      #           print("Can´t read video ...")
      
        if again:        
            root.after(1, play_video) 
    
    # Define a function to start the video
    def on_start():
       global running
       print("Start")
       running = True
       play_video()
    
    # Define a function to stop the video
    def on_stop(): 
       global running
       print("Stop")
       running = False
    
    
    # Define a function to exit 
    def on_exit():
  #      vid.release()
  ##      cv2.destroyWindow("Display")
        root.quit()
        cv2.destroyAllWindows()
        root.destroy()
        
        # r = input("Do you want to exit ?")
        
        # if 'y' in r or 'Y' in r:
        #     import sys
        #     sys.exit()
     
    
    #frame for buttons
    button_frame = tk.Frame(root) 
    button_frame.grid(column=3, row=3, columnspan=2)
    
    #Start button
    load_button = tk.Button(master=button_frame, text='Start',command=lambda: on_start())
    load_button.grid(column=1, row=0, sticky='ew')
    #Stop button
    stop_button = tk.Button(master=button_frame, text='Stop',command=lambda: on_stop())
    stop_button.grid(column=1, row=1, sticky='ew')
    #quit button
    quit_button = tk.Button(master=button_frame, text='Quit',command=lambda: on_exit())
    quit_button.grid(column=1, row=2, sticky='ew')
    
    root.mainloop()
    
### # video_filename = input("Which input video filename (have to be in the same folder than current program): ")

# print(video_filename)
    
## dynamic_video_repr(video_filename)