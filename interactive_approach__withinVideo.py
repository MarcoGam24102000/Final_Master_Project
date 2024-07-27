# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:14:06 2023

@author: marco
"""

from gui_singleVideo_SevMomentsOptions import select_option
from drag_drop_multiple_v2 import drag_drop_activity


def getOutputVideoFilename(numberPart):
    
    import PySimpleGUI as sg
     
  ##  video_filename = ""
    fps = 50
    
    layout = [
        [sg.Text("Enter a name for the output video: "), sg.Input(key="-OUTPUT_VIDEO_FILE-")],
        [sg.Button("Exit"), sg.Button("Next")]        
    ] 
    
     
    window = sg.Window("Output video filename, for part " + str(numberPart), layout)
    
    
    while True:
        event, values = window.read()
        
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "Next":
            video_filename = values["-OUTPUT_VIDEO_FILE-"]
            break
    
    window.close()
    
    return video_filename

def getInputVideoFilename():
    
    import PySimpleGUI as sg
    
    video_filename = ""
    
    layout = [
        [sg.Text("Specify the video filename:"), sg.Input(key="-MAIN_VIDEO_FILE-")],
        [sg.Button("Exit"), sg.Button("Next")]        
    ]
    
     
    window = sg.Window("Input video filename", layout)
    
    
    while True:
        event, values = window.read()
        
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "Next":
            video_filename = values["-MAIN_VIDEO_FILE-"]
            break
    
    window.close()
    
    return video_filename

def sepVideosFromWholeOne(inputVideoFlag, inputVideo):
    
    output_video_filenames = []
    bufferVideosX = []
    inputSlices = []
    
    newInStr = ''
    
    if '/' in inputVideo: 
        inputSlices = inputVideo.split('/')         
        
        for indN, n in enumerate(inputSlices):
            if indN < 11:
                newInStr += n + '/'
        
        spec = inputSlices[11]
        
        newInStr += spec[:-1] + '_0' + '/'
        inputVideo = newInStr + inputSlices[12]
            
    elif "\\" in inputVideo:
        inputSlices = inputVideo.split("\\")      
        
        for indN, n in enumerate(inputSlices):
            if indN < 11:
                newInStr += n + "\\"
        
        spec = inputSlices[11]
        
        newInStr += spec[:-1] + '_0' + "\\"
        inputVideo = newInStr + inputSlices[12]
      
    print("Input video: " + inputVideo)

    import PySimpleGUI as sg
    import os
    import cv2
    
    number_max_parts = 10
    standard_fr = 50
    
    video_filaname = ""
    
    
    def get_buffer_from_mp4Video(video_filename):
         
        import cv2
    
        # Open the video file
        video_capture = cv2.VideoCapture(video_filename)
        
        # Set the frame rate to 50 FPS
        fps = 50
        video_capture.set(cv2.CAP_PROP_FPS, fps)
        
        # Initialize a counter to keep track of the frame number
        frame_num = 0
        
        framesBuffer = []
        
        # Loop through the video frames
        while True:
            # Read the next frame from the video
            ret, frame = video_capture.read()
        
            # If there are no more frames, break out of the loop
            if not ret:
                break
        
            # Save the frame as an image file
      #      filename = f'frame{frame_num}.jpg'
     #       cv2.imwrite(filename, frame)
     
            framesBuffer.append(frame)
        
            # Increment the frame counter
            frame_num += 1
        
        return framesBuffer
    
    
    def get_partial_videos(lims, video_total_time, durTotal, buffer_imgs):
        
        width = 1920
        height = 1080
        fps = 25
        fps_bef = 25
        
        fr_true = True   
        resp_fr_true = True
        
        code_resp = 0
        
        while resp_fr_true:
            
            resp_fr_true = False
            
            
            import PySimpleGUI as sg

            layout = [
                [sg.Text("New frame rate?")],
                [sg.Checkbox("Yes", key="-YES-"), sg.Checkbox("No", key="-NO-")],
                [sg.Button("Next"), sg.Button("Exit")]
            ]
            
            window = sg.Window("Frame Rate Adaptable", layout)
            
            while True:
                
                if code_resp == 1 or code_resp == 2:
                   print("Here")
                   break
                 
                event, values = window.read()
                if event in (sg.WIN_CLOSED, "Exit"):
            #        break
                    print("Again")
                elif event == "Next":
                    if values['-YES-'] or values['-NO-']:
                        resp_fr = 'YES' if values['-YES-'] else "NO"
                        resp_fr_true = True
                        print(f"New frame rate: {resp_fr}")
                        if resp_fr == 'YES': 
                            code_resp = 1
                        else: 
                            code_resp = 2
                        
                        print("Out")
                        
                        break
                    else:
                        sg.popup("Please select an option.", title="Error")
                    
                
                    
            window.close()
            
            # resp_fr = input("New frame rate ? ")
            
            # if 'S' in resp_fr or 's' in resp_fr or 'y' in resp_fr or 'Y' in resp_fr:
            #     code_resp = 1
            #     print("New frame rate ...")
            #     break
            # elif 'n' in resp_fr or 'N' in resp_fr:
            #     code_resp = 2
            #     print("Same frame rate than before ...")
            #     break
            # else:
            #     resp_fr_true = True
        
        if code_resp == 1:
            
            fps_new = 0
            
            frs_list = []
            
            for x in range(1,11):
            
                frs_list.append(int(x*5))
            
            layout = [
                [sg.Text("New frame rate: ")],
                [sg.Combo(frs_list, key = "-FRAME_RATES-")],
                [sg.Button("Next"), sg.Button("Exit")]
            ]
            
            window = sg.Window("Frame rate adaptation", layout)
            
            while True:
                event, values = window.read()
                
                if event == sg.WINDOW_CLOSED or event == "Exit":
            #        break
                    print("Again")
                elif event == "Next":
                    fps_new = int(values["-FRAME_RATES-"])
                    break
            
            fps = fps_new
                
            window.close()               
            
        elif code_resp == 2:
            print("No change ...")
            
            
        
        output_video_filenames = []
        n_imgs = []
        
        print("Length of buffer: " + str(len(buffer_imgs)))
        
        import time 
        
        buffer_imgsx = []
        
        for ind_l, l in enumerate(lims):
            
            not_created = True
            
            while not_created:
                start_time, end_time = l 
                
                start_number_img = int((start_time/video_total_time)*10*video_total_time)
                end_number_img = int((end_time/video_total_time)*10*video_total_time) 
                
                print(start_number_img)
                print(end_number_img) 
                print("durTotal: " + str(durTotal))
                
                 
                partVideoFilename = buffer_imgs[start_number_img:end_number_img+1]
                
                if len(partVideoFilename) == 0:
                    import sys
                    sys.exit() 
                
                buffer_imgsx.append(partVideoFilename)
                # Initialize the output video writer
       ##         output_video_name = input("Enter a name for the output video: ")
                output_video_name = getOutputVideoFilename(ind_l)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_video = cv2.VideoWriter(f'{output_video_name}.mp4', fourcc, fps, (width, height))
                
                n_img = (fps_bef/fps)*(end_number_img-start_number_img)
                
                n_imgs.append(n_img) 
                
                for ind_frame, frame in enumerate(partVideoFilename):
                    print("Image " + str(ind_frame) + " written to video number " + str(ind_frame) + " ...")
                    output_video.write(frame)
                    
                output_video.release()
                    
                time.sleep(100)
                
                if os.path.isfile(f'{output_video_name}.mp4'):
                    print("Output video file exists")
                    output_video_filenames.append(output_video_name) 
                    not_created = False
                else:
                    print("Output video file not exist")
                    
               
            
        
        return output_video_filenames, fps, n_imgs, buffer_imgsx
    
    
    def get_video_total_time(video_filaname, std_frame_rate):
        print("Finding duration for the specified video filename")
        
        from moviepy.editor import VideoFileClip 
    
        # Replace 'video.mp4' with the path to your MP4 file
        clip = VideoFileClip(video_filaname)
        
        # Get the duration of the video in seconds 
        duration = clip.duration
        
        # Calculate the total number of frames in the video
        frame_count = duration * std_frame_rate  
        
        return duration, frame_count 
    
    
    def range_parts_gui(video_filaname):   
        
        video_total_time, durTotal = get_video_total_time(video_filaname, standard_fr)
        
        layout = [
            [sg.Text("Video total time: "), sg.Text(str(video_total_time) + " seconds")],
            [sg.Text("Start: "), sg.Input(key="-START-")],
            [sg.Text("End: "), sg.Input(key="-END-")],
            [sg.Button("Next"), sg.Button("Exit")]
        ]
        
        
        lims = (0,0)      
        
        ok1 = False
        ok2 = False
        
        window = sg.Window("Please Specify the range for this part of the video ...", layout)
        
        
        while True:
            event, values = window.read()
            
            if event == sg.WINDOW_CLOSED or event == "Exit":
                break
            elif event == "Next":
                
                init_str = values["-START-"]
                
                number_digits = 0
                
                start = 0          
                
                for d in init_str:
                    if d.isdigit():
                        number_digits += 1            
                
                if number_digits > 0 and number_digits < len(init_str):
                    
                    count_commas = 0
                    
                    for c in values["-START-"]: 
                        if c == ',':
                            count_commas += 1
                    if count_commas > 1:
                        print("Please insert a valid number (just one comma if needed)")
                    else:
                        
                        if count_commas == 1:                    
                    
                            full_str = values["-START-"]
                            
                            print(full_str)
                            
                            full_splitted = full_str.split(',')
                            
                            print(full_splitted) 
                            
                            rec_str = ""                    
                            for p in full_splitted:                        
                                rec_str += p + '.'
                            
                            print(rec_str)
                                
                            rec_str = rec_str[:-1]
                            
                            print(rec_str)
                            
                            start  = float(rec_str)
                            
                        else:
                            count_dots = 0
                            
                            for c in values["-START-"]: 
                                if c == '.':
                                    count_dots += 1
                            
                            if count_dots == 1:
                                    start = float(values["-START-"])
                            else:
                                print("Please insert a valid number (just one dot if needed)")
                                
                                  
                   
                    
                    if start < 0 or start >= video_total_time:
                        print("Wrong start time ...")
                    else:
                        print("One nice")
                        ok1 = True
                else:
                    if number_digits == 0:
                        print("Please provide just numbers, for the start time ...")
                        
                    elif number_digits == len(init_str):
                       start = int(values["-START-"]) 
                       ok1 = True
                  #####
                  
                  
                init_str = values["-END-"]
                
                number_digits = 0
                
                end = 0
                
                for d in init_str:
                    if d.isdigit():
                        number_digits += 1            
                
                if number_digits > 0 and number_digits < len(init_str):
                     
                    count_commas = 0
                    
                    for c in values["-END-"]: 
                        if c == ',':
                            count_commas += 1
                    if count_commas > 1:
                        print("Please insert a valid number (just one comma if needed)")
                    else:
                        
                        if count_commas == 1:                    
                    
                            full_str = values["-END-"]
                            
                            print(full_str)
                            
                            full_splitted = full_str.split(',')
                            
                            print(full_splitted) 
                            
                            rec_str = ""                    
                            for p in full_splitted:                        
                                rec_str += p + '.'
                            
                            print(rec_str)
                                
                            rec_str = rec_str[:-1]
                            
                            print(rec_str)
                            
                            end  = float(rec_str)
                            
                        else:
                            count_dots = 0
                            
                            for c in values["-END-"]: 
                                if c == '.':
                                    count_dots += 1
                            
                            if count_dots == 1:
                                    end = float(values["-END-"])
                            else:
                                print("Please insert a valid number (just one dot if needed)")
                                
                                  
                   
                    
                        
                    
                    if end < 0 or end > video_total_time:
                        print("Wrong ending time ...")
                    else:     
                        print("Two nice")
                        ok2 = True
                else:            
                    
                    if number_digits == 0:
                        print("Please provide just numbers, for the ending time ...")
                
                    elif number_digits == len(init_str):
                       end = int(values["-END-"]) 
                       ok2 = True
                    
                if ok1 and ok2:    
                     lims = (start,end)
                     break          
                
    
        window.close()   
        
         
        return lims, video_total_time, durTotal
    
    def selectNumberParts_video_splitted(number_max_parts):   
        
        parts_list = []
        
        number_parts = 0
        
        for i in range(2,number_max_parts+1):
            parts_list.append(str(i))
            
        layout = [
            [sg.Text("In how many parts you want to divide the video ?")],
            [sg.Combo(parts_list, key = "-PART_TEXT-")],
            [sg.Button("Next"), sg.Button("Exit")]
        ]
        
        window = sg.Window("Analysis intra-video guide", layout)
        
        again = True
        
        while again == True: 
            event, values = window.read()
            
            if event == sg.WINDOW_CLOSED or event == "Exit":
                again = True
           #     break
            elif event == "Next":
                if values["-PART_TEXT-"]:
                    number_parts = int(values["-PART_TEXT-"])
                    again = False
                    break
                else:
                    print("Asking again")              
        
        window.close()
        
        return number_parts
    
    n_parts = selectNumberParts_video_splitted(number_max_parts)
    
    video_total_time = 0
    durTotal = 0 
    
    neither = False
    
    
    if n_parts == 2:
        print("Executing standard software ...")
        neither = True
        output_video_filenames = None
    else:
        if select_option() == 1:
            
            if n_parts > 2 and n_parts <= number_max_parts:
                
                output_video_filenames = []
                n_images = []
                
                for indPart in range(0,n_parts):
            
                    output_video_name = getOutputVideoFilename(indPart)
            
                    print("Here 1A")
                    
                    buffer_imgs = drag_drop_activity(output_video_name) 
                    print("-------- \n\n ---- Buffer Imgs opt: ")
                    print(buffer_imgs)
                    bufferVideosX.append(buffer_imgs)
                    print("Here 1B")
                     
                    fps = 25 
                    
                    video = cv2.VideoCapture(output_video_name)
                    numberImages = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    output_video_filenames.append(output_video_name)
                    n_images.append(numberImages)
                    
            
        elif select_option() == 2:
        
            if n_parts > 2 and n_parts <= number_max_parts:
                lims = []
                
                go = True
                
                this_dir = "" 
                
                if inputVideoFlag:
                    video_filaname = inputVideo 
            #    else:
                
                    while go:
                    
                ##        video_filaname = getInputVideoFilename()
                 ##       video_filaname = input("Specify the video filename: ")
                         
                        if len(video_filaname) > 0:
                     #       if os.path.isfile(video_filaname):
                                go = False
                     #           break
                            # else:
                            #     print("File doesn´t exist !!!")
                        else:
                            print("Video filename empty ... \n Try again ...") 
                            
                buffer_imgs = get_buffer_from_mp4Video(video_filaname) 
                
                # for b in buffer_imgs:
                #     print(b)  
                        
                for n in range(0, n_parts):
                    lims_part, video_total_time, durTotal = range_parts_gui(video_filaname) 
                    lims.append(lims_part)
                    
                output_video_filenames, fps, n_images, bufferVideosX = get_partial_videos(lims, video_total_time, durTotal, buffer_imgs)
                       
            else:
                print("Executing standard software ...")
                
                output_video_filenames = None
    
    print("Buffer Video Images:")        
    print(bufferVideosX)
    
    
    print(neither)
    
    if not neither:
    
        if fps not in globals():
            fps = 25 
    else:
        print(inputVideo)
        
        
        
   #     import sys
   #     sys.exit()
        
        fps = 25
        duration, frame_count = get_video_total_time(inputVideo, fps)
        n_images = frame_count
    
    return output_video_filenames, fps, n_images, bufferVideosX, n_parts
    
## sepVideosFromWholeOne()


