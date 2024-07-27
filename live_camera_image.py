# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:34:24 2023

@author: Rui Pinto
"""

from pypylon import pylon
import cv2
import time
import PySimpleGUI as sg
from check_centered_image import help_camera_center
import threading 
from threading import Thread 
import keyboard
## import system 

def stop_button_cam_imgs_layout():
    import PySimpleGUI as sg
    
    resp = False
    
    layout = [
        [sg.Button("Stop live camera images routine")]
    ]
    
    window = sg.Window('Live camera images routine control', layout) 

    start_time = time.time()

    while True:
        
  ##      if abs(time.time()-start_time) < 1:
            event, values = window.read()
            print(event, values)        
          
             
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            
            if event == 'Live camera images routine control':
                resp = True
                break
            break 
        # else:
        #     break
        
##    window.close()
    
    return resp

class CustomThread(Thread):
    # constructor
    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.value = None
 
    # function executed in a new thread
    def run(self):
        # block for a moment
   ##     time.sleep(1)
        # store data in an instance variable
        
        resp = stop_button_cam_imgs_layout()
        
        self.value = str(resp)
        
        import multiprocessing
        
        process = multiprocessing.current_process()        
        process.terminate()
        

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)




def further_reduction(image, init_dim):
##    print("Further reduction")
    
    if len(image) % 2 == 0 and len(image[0]) % 2 == 0:
        small_dim = (len(image)/2, len(image[0])/2)
    else:
        if len(image) % 2 != 0 and len(image[0]) % 2 == 0:
            small_dim = ((len(image)-1)/2, len(image[0])/2)
        else:
            if len(image) % 2 == 0 and len(image[0]) % 2 != 0:
                small_dim = ((len(image))/2, (len(image[0])-1)/2)
            else:
                small_dim = (((len(image))-1)/2, (len(image[0])-1)/2) 
                
    small_dim = tuple(int(item) for item in small_dim)  
    
    resized_img = cv2.resize(image, small_dim, interpolation = cv2.INTER_AREA)    
    
    return small_dim, resized_img   
    

def showLiveImageGUI(image, numberImageSeqVideo):
    
    next = False 
    ok = False
    
##    print("Rendering " + str(numberImageSeqVideo) + " th image ...")
    
    cv2.imwrite("video_image" + str(numberImageSeqVideo) + ".png", image)
    
    if len(image) % 2 == 0 and len(image[0]) % 2 == 0:
        new_dim = (len(image)/2, len(image[0])/2) 
    else:
        if len(image) % 2 != 0 and len(image[0]) % 2 == 0:
            new_dim = ((len(image)-1)/2, len(image[0])/2)
        else:
            if len(image) % 2 == 0 and len(image[0]) % 2 != 0:
                new_dim = ((len(image))/2, (len(image[0])-1)/2)
            else:
                new_dim = (((len(image))-1)/2, (len(image[0])-1)/2) 

    new_dim = tuple(int(item) for item in new_dim)               
                
    image = cv2.imread("video_image" + str(numberImageSeqVideo) + ".png")
    
    resized_img = cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)
    
    small_dim, resized_img = further_reduction(resized_img, new_dim)
    
    cv2.imwrite("micro_image" + str(numberImageSeqVideo) + ".png", resized_img)
    
    size_img = (len(resized_img), len(resized_img[0]))
    
    print("A")
    print(resized_img.shape)
    
   
    layout = [

        [sg.Image('micro_image' + str(numberImageSeqVideo) + '.png', size = size_img)],
        [sg.Button('Next'), sg.Button('Exit')]   ## 
    ]    
    
    
    window = sg.Window('Live Camera Image', layout, resizable = True, finalize = True, margins=(0,0)) 
    
    button,values = window.read(timeout=200)    ## 1000  
    
    
     
    window.close()   


    
def process_img(image):

    import time
    start_time = time.time()
    
    if len(image.shape) == 3:    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    ok = help_camera_center(image)
    
    ok_new = []

    for okv in ok:
        ok_new.append(str(okv))
   
    ok = ok_new 
    
  #  process = threading.Thread()
    
#    process._Thread_stop()
    
#    system.exit()
    
    
    print("CSV Bef") 

  ##  with lock:
    if True:
        with open('C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\' + 'centering.csv','a', encoding='utf-8') as file:
               new_str = ', '.join(ok)
               file.write(new_str + '\n')
           
               print("Another line written to csv file -- estimating the best camera position")    
        
    print("CSV After")
     
    print("B")
    
 ##   process.run()
 
 ##   process.run()
    
 ##   process._Thread_start()
 
    end_time = time.time()
    
    time_proc_for_centralizing = abs(end_time-start_time)
    
    print("Prcessing time for this live image: " + str(round(time_proc_for_centralizing, 3)) + " seconds")
    
    return time_proc_for_centralizing
 
 
def stop_button_task(countx, trigger):
    t = ThreadWithResult(target=stop_button_cam_imgs_layout, args=(countx, trigger))
    t.start() 
    
    t.join() 
    
##    resp = stop_button_cam_imgs_layout()    

    
    
def process_task(var_bool, img):

    if var_bool:
        time_proc_live = process_img(img)
        var_bool = False


def acq_image_camera(countx, camera, converter, real): 

    print("acq_image_camera")    
    
    if True:       ## trig == True   
           
     
         #   camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
         #   converter = pylon.ImageFormatConverter()  
           
         #   camera.Open()
           
         #   # while run_again == True:

         #   #     try:               
         #   #         camera.Open()
         #   #         run_again = False
         #   #     except RuntimeError:
         #   #         print(" -- Please check the wire connection !!!")
         #   #         print("Trying again in 10 seconds ...")
                   
         #   #         time.sleep(10)
                   
         #   #         run_again = True                        
                   
               
         #   camera.CenterX=False
         #   camera.CenterY=False
           
          
           
         #   # Set the upper limit of the camera's frame rate to 30 fps
         #   camera.AcquisitionFrameRateEnable.SetValue(True)
         #   camera.AcquisitionFrameRateAbs.SetValue(50)
           
         #   camera.GevSCPSPacketSize.SetValue(1500)
           
         #   # Inter-Packet Delay            
         #   camera.GevSCPD.SetValue(5000)
           
         #   # Bandwidth Reserve 
         #   camera.GevSCBWR.SetValue(10)
           
         #   # Bandwidth Reserve Accumulation
         #   camera.GevSCBWRA.SetValue(4)    
           
         #   ## Save feature data to .pfs file
         # ##  pylon.FeaturePersistence.Save(nodeFile, camera.GetNodeMap())            
       
         #   # demonstrate some feature access
         #   new_width = camera.Width.GetValue() - camera.Width.GetInc()
         #   if new_width >= camera.Width.GetMin():
         #       camera.Width.SetValue(new_width)
               
         #   camera.Width.SetValue(1920) 
         #   camera.Height.SetValue(1080)           
          
         #   camera.StartGrabbing()
           
         #   run_again = True
         #   camera.Open()
           
           counter = 0
           
         #   camera.GainRaw=50
         #   camera.ExposureTimeRaw=140
           
       ##    ans_but = stop_button_cam_imgs_layout()
       
     ##      ans_but = False
           
           # layout = [
           #     [sg.Button("Stop live camera images routine")]
           # ]
           
           # window = sg.Window('Live camera images routine control', layout, resizable = True, finalize = True, margins=(0,0)) 
           
           
           
##           list_actions_stop_but = ['s', 'S', 'y', 'Y']          
           
           # for act in list_actions_stop_but:
           #     if act in resp:
           #         ans_but = True
           #         break    
           
           while camera.IsGrabbing() and real: 
               
                   # item = queue.get()
                  
                   # ans_but = bool(item[1])
                   
          ##         if not ans_but:
                   
                       # event, values = window.read()
                       # print(event, values)                                
                     
                        
                       # # if event == "Exit" or event == sg.WIN_CLOSED:
                       # #     break
                       
                       # if event == 'Live camera images routine control':
                       #     ans_but = True
                       #     break
                   
                               # for letter in list_actions_stop_but:
                               #     if keyboard.is_pressed(letter):
                               #         ans_but = True
                               #         break
                                   
                               # if ans_but == True:
                               #     break
                               # else: 
                               #     continue
                       
                               if counter < 100000:
                           ##        print("Grabbing to initial show ... ")
                                   grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                                   if grabResult.GrabSucceeded():   
                
                                       # if counter == 0:
                                       
                                       #     print("Image " + str(counter))
                                       #     image = converter.Convert(grabResult)
                                       #     img = image.GetArray()
                                           
                                       #     filename = base_path + "/image" + "_" + "test" + ".tiff"
                                           
                                       #     cv2.imwrite(filename, img)                           
                
                                       #     code_img = type_img(filename, base_path, 1)
                                           
                                       #     if code_img == 0:
                                       #         print("Error classifying image !!!")
                                       #     elif code_img == 1:
                                       #         print("Laser Speckle Image detected !!!")
                                       #     elif code_img == 2:
                                       #         print("Chess Image detected !!!")
                                       
                                  ##         print("Grabbing activity succeeding ... ")
                                       
                                           if counter > 0:                     
                                               
                            ##                   print("Image " + str(counter))
                                               image = converter.Convert(grabResult)
                                               img = image.GetArray()  
                                        ##       showLiveImageGUI(img, counter)
                                               img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                                               cv2.imshow('Live camera image', img)
                                   ##            print("Shown ... ")
                                               cv2.waitKey(1)  
                                   ##            process_img(img)
                                   
                                            #    if counter%200 == 0:
                                                   
                                            #        thread = CustomThread() 
                                            #        # start the thread 
                                            #        thread.start()
                                                   
                                            #        # wait for the thread to finish
                                            #    ##    thread.join()
                                                   
                                            #        # get the value returned from the thread
                                            #        data = thread.value                                              
                                                     
                                            #        resp = data 
                                                                                   
                                            # ##       resp = stop_button_task(counter, True)                                        
                                                   
                                            #        if resp:
                                            #            ans_but = True
                                               
                                               # var_bool = True                                   
                                               # thread = Thread(target=process_task, args=(var_bool, img))
                                               # thread.start()  
                                                                 
                                       
                                            
                                      
                                       
                                           counter += 1  
                               else:                        
                                   break
                     
            ##       cv2.waitKey(1) 
     
            
            
##           image_graph_dice_coeff = dice_coeff(np.array([list_img])[0])
            
    #           print("End of image acquisition 1")
               
           grabResult.Release() 
           camera.Close()     
               
#           print("End of image acquisition 2")


