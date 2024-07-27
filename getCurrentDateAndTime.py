# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 22:36:15 2022

@author: marco
"""

def getDateTimeStrMarker(): 

    import datetime
    
    e = datetime.datetime.now()
    
    print ("Current date and time = %s" % e)
    
    print ("Today's date:  = %s/%s/%s" % (e.day, e.month, e.year))
    
    print ("The time is now: = %s:%s:%s" % (e.hour, e.minute, e.second))
    
    if e.day < 10:
        daystr = '0' + str(e.day)
    else:
        daystr = str(e.day)
    
    if e.month < 10:
        monthstr = '0' + str(e.month)
    else:
        monthstr = str(e.month)
        
    year_list = list(str(e.year))
    yearList = year_list[2:]
    yearstr = str(yearList[0]) + str(yearList[1])    
        
    
    if e.hour < 10:
        hourstr = '0' + str(e.hour)
    else:
        hourstr = str(e.hour)
    
    if e.minute < 10:
        minutestr = '0' + str(e.minute)
    else:
        minutestr = str(e.minute)
        
    if e.second < 10:
        secondstr = '0' + str(e.second)
    else:
        secondstr = str(e.second)    
        
        
    adderToPath = '_' + daystr + monthstr + yearstr + '_' + hourstr + minutestr + secondstr + '_'
    
    return adderToPath