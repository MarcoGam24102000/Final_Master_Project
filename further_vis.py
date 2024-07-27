import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import os

def show_plot_and_back_button(plt, title, X, y):
    temp_file_path = "temp_plot.png"
    plt.savefig(temp_file_path, format='png')
    plt.close()

    # Create the layout with the image and back button
    layout = [
        [sg.Image(filename=temp_file_path)],
        [sg.Button("Back")]
    ]

    # Create the window 
    window = sg.Window(title, layout) 

    # Event loop to handle button clicks 
    while True:
        event, values = window.read(timeout=100)  # Check every 100 milliseconds

        if event == sg.WIN_CLOSED or event == "Back":
            break

    # Close the window  
    window.close()

    # Remove the temporary file
    os.remove(temp_file_path)
    
def show_saved_plot_and_back_button(image_path, title, X, y):
    # Create a new window with the saved image
    layout = [
        [sg.Image(filename=image_path)],
        [sg.Button("Back")]
    ]

    window = sg.Window(title, layout)

    # Event loop to handle button clicks
    while True:
        event, values = window.read(timeout=100)  # Check every 100 milliseconds

        if event == sg.WIN_CLOSED or event == "Back":
            break

    # Close the window
    window.close()

def load_data_with_gui():
    # PySimpleGUI file selection GUI
    
    again = True
    
    while again:
        layout = [
            [sg.Text("Select Excel File")],
            [sg.Input(key="-FILE-", enable_events=True), sg.FileBrowse()],
            [sg.Button("OK")]
        ]
    
        window = sg.Window("Excel File Selector", layout)
    
        while True:
            event, values = window.read()
    
            if event == sg.WIN_CLOSED:
       ##         break
                print("Asking again ...")
                again = True
                window.close()
                break
            elif event == "OK":
                again = False
                file_path = values["-FILE-"]
                break

    window.close()

    # Load data from Excel file
    if file_path:
        df = pd.read_excel(file_path, header=None)

        print("Dataframe: ")
        print(df)

        return df, file_path

    return None, None


def further_visualization(features, classes, file_path):
    load = False 

    import numpy as np
    import matplotlib.pyplot as plt
    import PySimpleGUI as sg
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, MDS, Isomap
    from umap import UMAP
    from sklearn.preprocessing import StandardScaler
    from yellowbrick.features.manifold import Manifold
    from yellowbrick.contrib.scatter import ScatterVisualizer
    from yellowbrick.features.radviz import RadViz
    from yellowbrick.features.pcoords import ParallelCoordinates
    from yellowbrick.features.rankd import Rank2D

    def scatter_visualization(X, y):
        X = X.drop(columns=['x'])
        # visualizer = ScatterVisualizer(x='Contrast', y='RMSE', classes=classes, size=(800, 600))
        # visualizer.fit(X, y)
        # visualizer.transform(X)
      #  visualizer.poof()
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X['Contrast'], X['RMSE'], c=y, cmap='viridis', edgecolor='k')
        plt.title('Scatter Visualization')
        plt.xlabel('Contrast')
        plt.ylabel('RMSE')
        plt.colorbar() 
        
        plt.tight_layout()
        
        show_plot_and_back_button(plt, 'Scatter Visualization', X, y) 


    def radviz_visualization(X, y):
       
            feat = features[:-1]

            print("length of features: ")
            print(len(feat))

            feat = feat[1:]
            X = X.drop(columns=['x'])

            print("Length of X:")
            print(len(X.columns))

            unique_classes = y.unique()
            classes = unique_classes

            visualizer = RadViz(classes=classes, features=feat, size=(800, 600))

            visualizer.fit(X, y)
            visualizer.transform(X)            
            
            image_path = "radvizimage.png"
            
            visualizer.poof(outpath=image_path, clear_figure=False)
            
            show_saved_plot_and_back_button(image_path, 'RadViz Visualization', X, y)
           
       
    def parallel_coordinates_visualization(X, y):
        feat = features[:-1]
        unique_classes = y.unique()

        feat = feat[1:]
        X = X.drop(columns=['x'])
        
        classes = unique_classes
        
        class_colors = {classes[0]: 'green', classes[1]: 'blue'}

        visualizer = ParallelCoordinates(
            classes=classes,
            features=feat,
            normalize='standard',
            sample=0.1,
            size=(800, 600),
            colors=[class_colors[cls] for cls in classes]
        )
        visualizer.fit(X, y)
        visualizer.transform(X)
  #      visualizer.poof()   

         # Set distinct colors for each class       
        
        image_path = "pc_visualization.png"
        
        visualizer.poof(outpath=image_path, clear_figure=False)
        
        show_saved_plot_and_back_button(image_path, 'Parallel Coordinates Visualization', X, y)
          
    def rank2d_covariance(X, y):
        feat = features[:-1]
        unique_classes = y.unique()
        classes = unique_classes

        feat = feat[1:]
        X = X.drop(columns=['x'])

        print("feat: ")
        print(feat)
        print("x: ")
        print(X)
        print("y: ")
        print(y)
        visualizer = Rank2D(features=feat, algorithm='covariance')
        visualizer.fit(X, y)
        visualizer.transform(X)
        image_path = "rank2d_covariance.png"
        
        visualizer.poof(outpath=image_path, clear_figure=False)
        
        # plt.figure(figsize=(8, 6))
        # plt.title('Rank2D (Pearson) Visualization')
        # pd.plotting.scatter_matrix(X, c=y, figsize=(12, 10), marker='o', hist_kwds={'bins': 20}, s=30,
        #                            alpha=0.8, cmap='viridis')
        # plt.tight_layout()

        # Display plot and back button
        
        # Get the current figure
    #    fig = plt.gcf()
    
        # Save the figure
        
   #     fig.savefig(image_path)
    
        # Show the saved figure and back button
        show_saved_plot_and_back_button(image_path, 'Rank2D (Covariance) Visualization', X, y)

    def rank2d_pearson(X, y):
        feat = features[:-1]
        unique_classes = y.unique()
        classes = unique_classes

        feat = feat[1:]
        X = X.drop(columns=['x'])

        visualizer = Rank2D(features=feat, algorithm='pearson')
        visualizer.fit(X, y)
        visualizer.transform(X)
        
   #     plt.figure(figsize=(8, 6))
   
        image_path = "rank2d_pearson.png"
        
        visualizer.poof(outpath=image_path, clear_figure=False)
        
        # plt.figure(figsize=(8, 6))
        # plt.title('Rank2D (Pearson) Visualization')
        # pd.plotting.scatter_matrix(X, c=y, figsize=(12, 10), marker='o', hist_kwds={'bins': 20}, s=30,
        #                            alpha=0.8, cmap='viridis')
        # plt.tight_layout()

        # Display plot and back button
        
        # Get the current figure
    #    fig = plt.gcf()
    
        # Save the figure
        
   #     fig.savefig(image_path)
    
        # Show the saved figure and back button
        show_saved_plot_and_back_button(image_path, 'Rank2D (Pearson) Visualization', X, y)

    def lle_visualization(X, y):
        feat = features[:-1]
        feat = feat[1:]
        X = X.drop(columns=['x'])
    
        # Standardize the data
        X_std = StandardScaler().fit_transform(X)
    
        # Apply Locally Linear Embedding
        lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, random_state=42)
        X_reduced = lle.fit_transform(X_std)
    
        # Plot the reduced data
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k')
        plt.title('Locally Linear Embedding (LLE) Visualization')
        plt.xlabel(f'{feat[0]}')
        plt.ylabel(f'{feat[1]}')
        plt.colorbar()
 ##       plt.show() 
        plt.tight_layout()
    
        show_plot_and_back_button(plt, 'Locally Linear Embedding (LLE) Visualization', X, y)
        
    def manifold_visualization(X, y, method='pca'):
        feat = features[:-1]
        feat = feat[1:]
        X = X.drop(columns=['x'])
        # Standardize the data
        X_std = StandardScaler().fit_transform(X)

        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'umap':
            reducer = UMAP(n_components=2, random_state=42)
        elif method == 'mds':
            reducer = MDS(n_components=2)
        elif method == 'isomap':
            reducer = Isomap(n_neighbors=30, n_components=2)
        else:
            raise ValueError("Invalid method. Use 'pca', 'tsne', 'umap', 'mds', 'isomap' or 'yellowbrick'.")

        X_reduced = reducer.fit_transform(X_std)

        # Plot the reduced data 
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k')
        plt.title(f'{method.upper()} Visualization')
        plt.xlabel(f'{feat[0]}')
        plt.ylabel(f'{feat[1]}')
        plt.colorbar()
  #      plt.show() 
        
        plt.tight_layout()
        
        show_plot_and_back_button(plt, f'{method.upper()} Visualization', X_reduced, y)

    from sklearn.manifold import LocallyLinearEmbedding
    
    # Define the layout of the GUI 
    layout = [
        [sg.Button("Load Dataset")],
        [sg.Button("Scatter Visualization")],
        [sg.Button("RadViz Visualization")],
        [sg.Button("Parallel Coordinates Visualization")],
        [sg.Button("Rank2D (Covariance) Visualization")],
        [sg.Button("Rank2D (Pearson) Visualization")],
        [sg.Button("LLE Manifold Visualization")],
        [sg.Button("Manifold Visualization (PCA)")],
        [sg.Button("Manifold Visualization (t-SNE)")],
        [sg.Button("Manifold Visualization (UMAP)")],
        [sg.Button("Manifold Visualization (MDS)")],
        [sg.Button("Manifold Visualization (Isomap)")],
        [sg.Button("Back")],
        [sg.Button("Exit")]
    ] 
 
    # Create the window
    window = sg.Window("Data Analysis and Visualization", layout)

    # Event loop to handle button clicks
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break
        elif event == "Load Dataset":
            load = True
            
            while True:
                file_path = sg.popup_get_file("Select a CSV file", file_types=(("CSV Files", "*.csv"),))
                if file_path:
                    data = pd.read_csv(file_path)
                    X = pd.read_csv(file_path, usecols=lambda col: col != 'Classes')
    
                    y = data['Classes']  # replace 'occupancy' with your actual target variable
    
                    print("\n\n -------------- \n Loaded ...")
                    break
                else:
                    print("Asking again ...")
                
        elif event == "Back": 
            load = False
            break
        elif event == "Scatter Visualization":
            if load:
                scatter_visualization(X, y)
               
        elif event == "RadViz Visualization":
            if load:
                radviz_visualization(X, y)
        
        elif event == "Parallel Coordinates Visualization":
            if load:
                parallel_coordinates_visualization(X, y)
            
        elif event == "Rank2D (Covariance) Visualization":
            if load:
                rank2d_covariance(X, y)
         
        elif event == "Rank2D (Pearson) Visualization":
            if load:
                rank2d_pearson(X, y)
         
        elif event == "LLE Manifold Visualization":
            if load: 
                lle_visualization(X, y)
             
        elif event == "Manifold Visualization (PCA)":
            if load:
                manifold_visualization(X, y, method='pca')
             
        elif event == "Manifold Visualization (t-SNE)":
            if load:
                manifold_visualization(X, y, method='tsne')
            
        elif event == "Manifold Visualization (UMAP)":
            if load:
                manifold_visualization(X, y, method='umap')
            
        elif event == "Manifold Visualization (MDS)":
            if load:
                manifold_visualization(X, y, method='mds')
          
        elif event == "Manifold Visualization (Isomap)":
            if load:
                manifold_visualization(X, y, method='isomap')
          

    # Close the window
    window.close()

def apply_kmeans_clustering(features_dataset, num_clusters, file_path):
    # Convert the list of lists to a DataFrame
    
     
    df = pd.DataFrame(features_dataset[1:], columns=features_dataset[0])

    print("Dataframe: ")
    print(df)
    
    # import sys
    # sys.exit()
    
    
    # Extract numerical features
    numeric_features = df.select_dtypes(include=[np.number])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(numeric_features)

    # Add a new column to the DataFrame with cluster labels
    df['Classes'] = cluster_labels
    
    # Save the modified dataset to a CSV file with the same filename
    csv_filename = os.path.splitext(file_path)[0] + "_modified.csv"
    df.to_csv(csv_filename, index=False)

    # Convert the DataFrame back to a list of lists
    result_dataset = [df.columns.tolist()] + df.values.tolist()

    # Extract feature names and cluster labels
    feature_names = df.columns.tolist()
    cluster_list = cluster_labels.tolist()

    return result_dataset, feature_names, cluster_list


# Main part of the script
your_dataset_df, file_path = load_data_with_gui()

if your_dataset_df is not None:

    num_clusters = 2

    result_dataset_list, feature_names, cluster_list = apply_kmeans_clustering(
        your_dataset_df.values.tolist(),
        num_clusters,
        file_path)

    further_visualization(feature_names, cluster_list, file_path)

    # Print the modified dataset
    for row in result_dataset_list:
        print(row)

    # Print the feature names
    print("Feature Names:", feature_names)

    # Print the cluster labels
    print("Cluster Labels:", cluster_list)
else:
    print("No file selected.")
