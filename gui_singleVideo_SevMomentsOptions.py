# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:14:39 2023

@author: marco
"""

import PySimpleGUI as sg

def select_option():
    # Define the layout of the GUI
    layout = [
        [sg.Text("Select an option:")],
        [sg.Checkbox("Image selection", key="image_selection")],
        [sg.Checkbox("Manual specification of time marks", key="manual_specification")],
        [sg.Button("OK")]
    ]

    # Create the GUI window
    window = sg.Window("Select an option", layout)

    while True:
        # Read events from the GUI window 
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            # If the user closed the window, exit the function
            return None

        elif event == "OK":
            # Check which options are selected
            image_selection = values["image_selection"]
            manual_specification = values["manual_specification"]

            # If none or both options are selected, show an error message and continue the loop
            if not image_selection and not manual_specification:
                sg.popup("Please select one option", title="Error")
                continue
            elif image_selection and manual_specification:
                sg.popup("Please select only one option", title="Error")
                continue

            # Determine the selected option and return the corresponding number
            if image_selection:
                return 1
            elif manual_specification:
                return 2

    # Close the GUI window
    window.close()

## option_sel = select_option()