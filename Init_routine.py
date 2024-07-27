def process_image_button(params_streaming):


    import cv2
    import pypylon
    from pypylon import pylon
    import PySimpleGUI as sg
    import threading
    
    from live_camera_image import showLiveImageGUI
    
    def show_images(cam, converter, params_streaming):
        # Connect to the first available Basler camera
  #      camera = pypylon.factory.find_devices()[0]
  
          gain = int(params_streaming[0])
          black_lavel = int(params_streaming[1])
          format_image = params_streaming[2]
          expos_time = int(params_streaming[3])   
          
          packet_size = int(params_streaming[4])
          inter_packet_delay = int(params_streaming[5])
          bw_reserv_acc = int(params_streaming[6])
          bw_reserv = int(params_streaming[7])
          
          frame_rate = int(params_streaming[8]) 
          
          
          ####################################################################
          
          width = 1920
          height = 1080
          
          if format_image == 'Full HD':
              width = 1920
              height = 1080
          elif format_image == 'HD+':
              width = 1600
              height = 900
          elif format_image == 'HD':
              width = 1280
              height = 720    
          elif format_image == 'qHD':
              width = 960
              height = 540
          elif format_image == 'nHD':
              width = 640
              height = 360        
          elif format_image == '960H':
              width = 960
              height = 480 
          elif format_image == 'HVGA':
              width = 480
              height = 320          
          elif format_image == 'VGA':
              width = 640
              height = 480    
          elif format_image == 'SVGA':
              width = 800
              height = 600  
          elif format_image == 'DVGA':
              width = 960
              height = 640 
          elif format_image == 'QVGA':
              width = 320
              height = 240
          elif format_image == 'QQVGA':
              width = 160
              height = 120
          elif format_image == 'HQVGA':
              width = 240
              height = 160  
          
          #################################################################### 
          
          basler_check = False
          camera_opened = True
          
          # while not basler_check:                                            
          #     basler_check, model_name = confirm_basler()
          #     time.sleep(2)    
            
              
          while camera_opened == True:
              try:
                  camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                  converter = pylon.ImageFormatConverter()  
                  camera.Open()
                  camera_opened = False
              except:
                  camera.Close()
                  camera_opened = True 
          
          
          camera.CenterX=False
          camera.CenterY=False
                 
                
                 
          # Set the upper limit of the camera's frame rate to 30 fps
          camera.AcquisitionFrameRateEnable.SetValue(True)
          camera.AcquisitionFrameRateAbs.SetValue(frame_rate) 
                 
          camera.GevSCPSPacketSize.SetValue(packet_size)
                 
          # Inter-Packet Delay            
          camera.GevSCPD.SetValue(inter_packet_delay)
                 
          # Bandwidth Reserve 
          camera.GevSCBWR.SetValue(bw_reserv)
                 
          # Bandwidth Reserve Accumulation
          camera.GevSCBWRA.SetValue(bw_reserv_acc)    
                 
          ## Save feature data to .pfs file
          ##  pylon.FeaturePersistence.Save(nodeFile, camera.GetNodeMap())            
             
          # demonstrate some feature access
          new_width = camera.Width.GetValue() - camera.Width.GetInc()
          if new_width >= camera.Width.GetMin():
                camera.Width.SetValue(new_width)
                     
          camera.Width.SetValue(width) 
          camera.Height.SetValue(height)
          
          camera.StartGrabbing()           
          
          camera.Open()
                 
          camera.GainRaw=gain
          camera.ExposureTimeRaw=expos_time
  
          while True:
       
                        print("Showing image ...")
                        
                        counter = 0 
    ##    cam.start_grabbing(pypylon.GrabStrategy_LatestImageOnly)
    
    #    if cam.IsGrabbing :
            # Retrieve the next grabbed image
        ##    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
                        grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        if grabResult.GrabSucceeded():   

               
                 
             #       if counter > 0:                      
                        
                            print("Image ")
                            image = converter.Convert(grabResult)
                            img = image.GetArray()  
                            cv2.imshow('Live camera image', img)
                     #       showLiveImageGUI(img, counter)
                    ##        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    ##        cv2.imwrite("example_image.jpg", img)
                    ##        img = cv2.imread("example_image.jpg")
                    #        cv2.imshow('Live camera image', img)
                            cv2.waitKey(1)
                            print("Shown ... ")
                            print(img)
                            # import time
                            # time.sleep(2)
                        # else:
                        #     print("No success on grabbing activity ...")
                            
                         
            #        counter += 1
        
        # Stop grabbing images 
        
    
    def show_button(window):        
    
       #     while True:
           
                print("Button")
                event, values = window.read()
        
                if event in (None, 'Stop Showing Images'):
                    
                    if event == 'Stop Showing Images':
                   #     image_thread.join()
                        print("Stop")
             #            self.value = None
                         
                
    
                # import time
                # time.sleep(10)
                    
                window.close() 
                
    
        
    
    def main_process(params_streaming):
        # Start the image showing and button threads
        
        ok = False
        
        layout = [[sg.Button('Stop Showing Images')]]
        window = sg.Window('Control Panel', layout)
        
        cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        converter = pylon.ImageFormatConverter()  
        
    ##    grab_result = pylon.PylonImage()
 #       cam = pypylon.factory.create_device(camera)
        
        going_now = True
        
        while going_now: 
 
            try:
                cam.Open()
                going_now = False
            except: 
                cam.Close()
                going_now = True
            
        
        print("Camera opening")
        
        counter = 0   
        
        cam.StartGrabbing()
        
        print("Camera grabbing")
        
        cam.Open()
        
        print("Going to threads ...")
        
        image_thread = threading.Thread(target=show_images, args=(cam, converter, params_streaming))
        image_thread.daemon = True
        button_thread = threading.Thread(target=show_button, args=(window, ))
        image_thread.start()
        button_thread.start()
        
        button_thread.join()
        
        print("inter ...")
        
        
    
        # Do the bigger process here
    
        # Wait for the threads to finish
        
        if not image_thread.is_alive():
            
            print("Closing") 
            
            cam.StopGrabbing()
            cam.Close()
            cv2.destroyAllWindows()
            
            button_thread.join()
            
            ok = True
    
    # if __name__ == "__main__":
    #     main_process()
        
        return ok
    
    ok = main_process(params_streaming)
    import multiprocessing
    
    return ok
        
## ok = process_image_button()