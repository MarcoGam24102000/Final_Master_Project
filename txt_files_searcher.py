import shutil
import os

def proc_txt_browsing(folder_to_search_in):

    def fetch_to_gui(dates_list):
    
        import PySimpleGUI as sg
        
        date_sel = ""
        
        ## Export to table with a checkbox at the end of each line
        
        sg.set_options(font=("Arial Bold", 14))
        
        rows = []
        
        row_sel = "0"
        
        for ind_d, d in enumerate(dates_list):
            # print(ind_d+1)
            # print(type(d[0]) + str(d[0]))
            # print(type(d[1]) + str(d[1]))
            
            if len(d[0]) == 6:            
                rows.append([ind_d+1, d[0], d[1]])
        
        toprow = ['File index', 'Date', 'Time'] 
            
     #   toprow = ['S.No.', 'Name', 'Age', 'Marks']
     #   rows = [[1, 'Rajeev', 23, 78], 
     #           [2, 'Rajani', 21, 66],
     #           [3, 'Rahul', 22, 60],
     #           [4, 'Robin', 20, 75]]
        tbl1 = sg.Table(values=rows, headings=toprow,
           auto_size_columns=True,
           display_row_numbers=False,
           justification='center', key='-TABLE-',
           selected_row_colors='red on yellow',
           enable_events=True,      
           expand_x=True,
           expand_y=True,
         enable_click_events=True)
        layout = [[tbl1]]
        window = sg.Window("Table Demo", layout, size=(715, 200), resizable=True)
        while True:
           event, values = window.read()
           print("event:", event, "values:", values)
           if event == sg.WIN_CLOSED:
              break
           if '+CLICKED+' in event:
              sg.popup("You clicked row:{} Column: {}".format(event[2][0], event[2][1]))
              row_sel = event[2][0]
              break
        window.close()
        
        row_sel = int(row_sel)
    #    print(row_sel)
        
    #    print(type(rows))
        
        row_selected = rows[row_sel]
        print("Here A - rows")
        print(str(row_selected))
        
        date_sel = [row_selected[1], row_selected[2]] 
        
        print("Here B - rows")
        print(date_sel)
        
        return date_sel  
    
    def list_dates(folder_to_search_in):
         
        import os
        
        dates = []
        
        files = os.listdir(folder_to_search_in)   ## without ending slash ('/')
    
        for file in files:
            if '.txt' in file and ('test' in file or 'config' in file):
                print(file)
                file_sepx = []
                fullDirFlag = False
                if '/' in file:
                    fullDirFlag = True
                    file_sepx = file.split('/')
                elif "\\" in file:
                    fullDirFlag = True
                    file_sepx = file.split("\\")
                    
                if fullDirFlag:                
                    print(file_sepx)        
                    filename_only = file_sepx[-1]
                else:
                    filename_only = file
    
                filename_sec = filename_only.split('_')
                
                dates_this = []
                
                date_disp = False
                
                for d in filename_sec:
                    dig = False
                    count_dig = 0
                    for x in d:
                        if x.isdigit():
                            dig = True
                            count_dig += 1
                    if dig and count_dig == len(d):
                        dates_this.append(d)
                        date_disp = True
                
                if date_disp:
                    dates.append(dates_this)
    
        return dates  
    
    new_folder = ""
    
    if '/' in folder_to_search_in:
        if folder_to_search_in[-1] == '/':                
            new_folder += folder_to_search_in + 'Acquisition' + '/'
        else:
            new_folder += folder_to_search_in + '/' + 'Acquisition' 
            
    elif "\\" in folder_to_search_in:
        if folder_to_search_in[-1] == "\\":                
            new_folder += folder_to_search_in + 'Acquisition' + "\\"
        else:
            new_folder += folder_to_search_in + "\\" + "Acquisition"
    
    files_inside_acqFolder = os.listdir(new_folder)
    
    print(folder_to_search_in)
    print(os.path.exists(folder_to_search_in))
    
    print(new_folder)
    print(os.path.exists(new_folder))
    
    for file in files_inside_acqFolder:
        if '.txt' in file: 
            print(file) 
             
            if True:
                shutil.copy2(os.path.join(new_folder,str(file)), os.path.join(folder_to_search_in,str(file)))
                print(str(file) + "copied from " + new_folder + " to " + folder_to_search_in)
    
    
    
    dates_updated = list_dates(folder_to_search_in) 
    
    date_sel = fetch_to_gui(dates_updated)   
    dateFormatStr = '_' + date_sel[0] + '_' + date_sel[1] + '_'                
    
    
    def search_config_files_by_date(dateFormatStr, folder_to_search_in, key_filename, data):
        
        import os
         
        txt_dir_configs = []
        
        print("dateFormatStr: ")
        print(dateFormatStr)
        
        splitD = dateFormatStr.split('_')
        
        remDFS = []
        
        for d in splitD:
            if len(d) > 0:
                remDFS.append(d)  
        
        onlyDate = remDFS[0]
        remDate = remDFS[1]
        
        
        
        files = os.listdir(folder_to_search_in)   ## without ending slash ('/')
        
        ok = False
        
        print("Only date: " + onlyDate)
        print("Files here: ")
        
        for file in files:
            if '.txt' in file:
       ##         print(file)
       
             ##   if key_filename == 'tests_info' and remDate[0:3] in file or key_filename != 'tests_info':
                
                    if onlyDate in file and key_filename in file:   ## and remDate[0:3] in file
                        if data is not None:
                            print("File1: " + file[-18:-8])
                            print("Data1: " + data[1:11])
                            
                            if file[-18:-8] == data[1:11]:
                                ok = True
                        else:
                            ok = True
                        
                        if ok:
                            txt_dir_configs.append(file)
                            print("File extracted: " + file)
             
        return txt_dir_configs 
        
    
    ## dateFormatStr = '_150323_174850_'    ## Example  
    
    
        
    key_filename = 'config_dirs'  
 #   config_dirs = []
        
    txt_config_files = search_config_files_by_date(dateFormatStr, folder_to_search_in, key_filename, data=None)
    print(txt_config_files)
    
    key_filename = 'tests_info'  

    txtOne = txt_config_files[0]

    data = txtOne[-19:-4]       
            
    txt_tests_file = search_config_files_by_date(dateFormatStr, folder_to_search_in, key_filename, data)
    
    dir_txt_file = "" 
    
    if len(txt_tests_file) > 1:
        dir_txt_file = txt_tests_file[0]
    else:
        dir_txt_file = txt_tests_file[0]
    
    # if len(txt_config_files) > 2:
    #    config_dirs = txt_config_files[:2]
    # if len(txt_config_files) == 1:
    #    config_dirs = txt_config_files[0]
         
    # config_dir_one = config_dirs[0]
    # config_dir_two = config_dirs[1]  
    
    print(dir_txt_file)   
    print(txt_config_files)  
 

    return dir_txt_file, txt_config_files


 



