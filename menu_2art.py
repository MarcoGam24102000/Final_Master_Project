# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:04:09 2023

@author: marco
"""

import PySimpleGUI as sg

def main():
    sg.theme('DarkBlue')  # Set the PySimpleGUI theme to DarkBlue for a blue background

    import os 
    print(os.getcwd()) 
     
    
    # Define the layout for the GUI
    layout = [
        [sg.Text("Select an option:")],
        [sg.Column([
            [sg.Button("All-In-One", key="All-In-One", button_color=("black", "blue"), size=(20, 1))],
            [sg.Button("Aquisition", key="Aquisition", button_color=("black", "blue"), size=(20, 1))],
            [sg.Button("Processing All-In-Ones", key="Processing All-In-Ones", button_color=("black", "blue"), size=(20, 1))],
            [sg.Button("Processing-till-features characterization", key="Processing-till-features characterization", button_color=("black", "blue"), size=(20, 1))],
            [sg.Button("Clustering step", key="Clustering step", button_color=("black", "blue"), size=(20, 1))]
        ], element_justification="c")],
        [sg.Button("Exit"), sg.Button("Next")],
    ]

    # Create the window
    window = sg.Window("Button Selection", layout, finalize=True)
 
    # Initialize the selected button variable
    
    selected_button = None
 
    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Exit"):
            break
        elif event == "Next":
            if selected_button is not None:
                window.close()
                execute_selected_function(selected_button)
                selected_button = None                
        elif event in ("All-In-One", "Aquisition", "Processing All-In-Ones",
                       "Processing-till-features characterization", "Clustering step"):
            if selected_button:
                window[selected_button].update(button_color=("black", "blue"))
            selected_button = event
            window[event].update(button_color=("black", "green"))

    window.close()

def execute_selected_function(button_name):
    # You can implement the specific functionality for each button here
    if button_name == "All-In-One":
        print("Executing All-In-One function")
        script_name = 'gui_html_v2.py'
    elif button_name == "Aquisition":
        print("Executing Aquisition function")
        script_name = 'Acquisition/acquisition.py'
    elif button_name == "Processing All-In-Ones":
        print("Executing Processing All-In-Ones function")
        script_name = 'processing_step.py'
    elif button_name == "Processing-till-features characterization":
        print("Executing Processing-till-features characterization function")
        script_name = 'processing_step_fixed_metrics.py'
    elif button_name == "Clustering step":
        print("Executing Clustering step function")
        script_name = 'clustering_part_top.py'
        
    with open(script_name, 'r') as file:
        script_contents = file.read()

    # Execute the script using exec() 
    exec(script_contents)

if __name__ == "__main__":
    main()