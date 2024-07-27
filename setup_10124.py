# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:41:22 2024

@author: Rui Pinto
"""
import ast
import os
import importlib

def extract_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)

    return list(imports)

direct = "C:\\Users\\Other\\files_python\\py_scripts\\ffmpeg-5.0.1-full_build\\bin\\GUI\\"
script_path = "gui_html_v2.py"
included_imports = extract_imports(direct + script_path)

print("included_imports:")
print(included_imports)

ex = None
additional_dir = "C:\\Users\\Rui Pinto\\AppData\\Local\\Programs\\Spyder\\pkgs\\"

while True:
    try:
        # Attempt to run setup
        from cx_Freeze import setup, Executable

        if ex:
            print(f"Importing the library from {additional_dir}")
                     
            os.chdir(additional_dir)
            
            import sys
            sys.exit()
            
            module_name = str(ex).split()[-1]
            importlib.import_module(module_name)
            
            included_imports = extract_imports(direct + script_path)
            os.chdir(direct)

        setup(
            name="allinones",
            version="1.0",
            description="Aquisição e Processamento",
            executables=[Executable(script_path, base="Console")],
            options={
                'build_exe': {
                    'includes': included_imports,
                }
            }
        )
        break  # Break the loop if setup is successful
    except ImportError as e:
        print(f"Error importing the following library: {e}")
        ex = e
        module_name = str(e).split()[-1]
        included_imports.append(module_name)
