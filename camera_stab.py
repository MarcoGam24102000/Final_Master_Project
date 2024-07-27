# Apply image stabilization to the current frame


import numpy as np
from pypylon import pylon
import cv2

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
converter = pylon.ImageFormatConverter() 


prev_frame = None
M = np.eye(3)

count = 0

camera.Open()

camera.CenterX=False
camera.CenterY=False
       
      
       
# Set the upper limit of the camera's frame rate to 30 fps
camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRateAbs.SetValue(50) 
       
camera.GevSCPSPacketSize.SetValue(1500)
       
# Inter-Packet Delay            
camera.GevSCPD.SetValue(5000)
       
# Bandwidth Reserve 
camera.GevSCBWR.SetValue(4)
       
# Bandwidth Reserve Accumulation
camera.GevSCBWRA.SetValue(10)    
       
## Save feature data to .pfs file
##  pylon.FeaturePersistence.Save(nodeFile, camera.GetNodeMap())            
   
# demonstrate some feature access
new_width = camera.Width.GetValue() - camera.Width.GetInc()
if new_width >= camera.Width.GetMin():
      camera.Width.SetValue(new_width)
           
camera.Width.SetValue(1920) 
camera.Height.SetValue(1080)

camera.StartGrabbing()           
 
camera.Open()
       
camera.GainRaw=50
camera.ExposureTimeRaw=140

while camera.IsGrabbing():
    
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        
        img_conv = converter.Convert(grabResult)
        
        img = img_conv.GetArray()
        
        if prev_frame is not None:
            
            print("Prev Frame")
            
            # Calculate the optical flow between the previous and current frames
            if len(prev_frame.shape) == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else: 
                prev_gray = prev_frame 
                 
            if len(img.shape) == 3: 
                curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = img 
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
            # Update the transformation matrix
            
            M = np.vstack([M, np.array([[1,0,np.median(flow[:,:,0])],[0,1,np.median(flow[:,:,1])], [0,0,1]])])
              
     ##       M *= np.vstack([flow[:,:3,0], [0, 0, 1]]) 
    
    		# Apply the transformation matrix to the current frame
            
            img = cv2.warpAffine(img, M[:2], (img.shape[1], img.shape[0]))    ## M[:2]
  
            print("Warping ...")
     
		# Find and track the object of interest in the stabilized frame
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        else:
            gray = img 
             
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Draw the bounding box around the object of interest
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cx, cy = x + w/2, y + h/2

		# Center the object in the frame
        dx = img.shape[1]/2 - cx 
        dy = img.shape[0]/2 - cy
        M[:2, 2] += [dx, dy]

		# Display the stabilized frame with the centered object
        cv2.imshow("Stabilized Frame", img)
        
        
        print("Showing " + str(count+1) + " th image ...")

		# Update the previous frame variable 
        prev_frame = img 
        
        count += 1

        grabResult.Release()
        
        if cv2.waitKey(1) == ord('q'):
            break 

camera.StopGrabbing()
camera.Close()



        
        