

def control_gui():
    import PySimpleGUI as sg
    
    layout = [
        [sg.Text("Please select the desired funcionalities: ")],
        [sg.T("         "), sg.Checkbox('Acquisition Step', default=False, key="-IN1-")],
        [sg.T("         "), sg.Checkbox('Processing Step', default=True, key="-IN2-")],
        [sg.Button("Next")]
    ]
    
    window = sg.Window("Control GUI", layout)
    
    first_true = False
    second_true = False
    
    repeat = True
    
    while repeat:
        event, values = window.read()
        
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Next":
        
            if values['-IN1-'] == True:
                first_true = True
                repeat = False
            if values['-IN2-'] == True:
                second_true = True
                repeat = False
            if values['-IN1-'] == False and values['-IN2-'] == False:
                repeat = True
                
            if first_true or second_true:
                break
    
    window.close()
    
    controls = [first_true, second_true]
    
    return controls
        
## controls = control_gui()
    
    
    
    