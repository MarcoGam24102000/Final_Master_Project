# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:09:19 2023

@author: Rui Pinto
"""

import pypylon.pylon as py
import numpy as np
import matplotlib.pyplot as plt

# handle exception trace for debugging 
# background loop
import traceback

import time
import random
cam = py.InstantCamera(py.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()

# to get consistant results it is always good to start from "power-on" state
cam.UserSetSelector = "Default"
cam.UserSetLoad.Execute()

cam.ExposureTime.setValue(60)

def ForegroundLoopSample():
    # fetch some images with foreground loop
    img_sum = np.zeros((cam.Height.Value, cam.Width.Value), dtype=np.uint16)
    cam.StartGrabbingMax(100)
    while cam.IsGrabbing():
        with cam.RetrieveResult(1000) as res:
            if res.GrabSucceeded():
                img = res.Array
                img_sum += img
            else:
                raise RuntimeError("Grab failed")
    cam.StopGrabbing()
    return img_sum

ForegroundLoopSample()