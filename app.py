import time
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import LabelEncoder

st.title("Data Mining and Analysis")


input_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])


def check_table_structure(data):
    num_columns = len(data.columns)
    if num_columns < 2:
        return False, "Table must have at least one feature and one label column."
    if data.isnull().values.any():
        return False, "Data contains missing values."
    return True, "Data is structured correctly."


if input_file is not None:
    try:
        # Handle CSV, Excel, TSV files
        if input_file.name.endswith(".csv"):
            data = pd.read_csv(input_file)
        elif input_file.name.endswith(".xlsx"):
            data = pd.read_excel(input_file)
        elif input_file.name.endswith(".tsv"):
            data = pd.read_csv(input_file, sep="\t")
        
        # Validate table structure
        is_valid, message = check_table_structure(data)
        if is_valid:
            st.success("Data loaded successfully!")
            st.write(data)
            
            # Display structure details
            st.write(f"Number of Samples (S): {data.shape[0]}")
            st.write(f"Number of Features (F): {data.shape[1] - 1}")
            st.write(f"Output Label Column: {data.columns[-1]}")
            
            

            # Data preparation for visualization (all except label)
            features = data.iloc[:, :-1]  
            labels = data.iloc[:, -1]     

            # Convert categorical labels to numeric values
            label_encoder = LabelEncoder()
            class_labels = label_encoder.fit_transform(labels)

            # Visualization Tab
            st.subheader("Visualization")
            
            # 2D PCA Visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(features)
            
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=class_labels, cmap='viridis')
            plt.title("2D PCA Visualization")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            st.pyplot(plt.gcf())

            # 3D PCA Visualization
            pca_3d = PCA(n_components=3)
            pca_result_3d = pca_3d.fit_transform(features)
            fig = plt.figure()
            ax_3d = fig.add_subplot(111, projection='3d')
            scatter_3d = ax_3d.scatter(pca_result_3d[:, 0], pca_result_3d[:, 1], pca_result_3d[:, 2], 
                                       c=class_labels, cmap='viridis')
            ax_3d.set_title("3D PCA Visualization")
            ax_3d.set_xlabel("PCA Component 1")
            ax_3d.set_ylabel("PCA Component 2")
            ax_3d.set_zlabel("PCA Component 3")
            fig.colorbar(scatter_3d)
            st.pyplot(fig)

            # 2D UMAP Visualization
            reducer = umap.UMAP(n_components=2)
            umap_result = reducer.fit_transform(features)
            plt.figure()
            scatter_umap = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=class_labels, cmap='viridis')
            plt.title("2D UMAP Visualization")
            plt.xlabel("UMAP Component 1")
            plt.ylabel("UMAP Component 2")
            plt.colorbar(scatter_umap)
            st.pyplot(plt.gcf())


            # 3D UMAP Visualization
            reducer_3d = umap.UMAP(n_components=3)
            umap_result_3d = reducer_3d.fit_transform(features)
            plt.figure()
            scatter_umap_3d = plt.scatter(umap_result_3d[:, 0], umap_result_3d[:, 1], c=class_labels, cmap='viridis', s=50)
            plt.title("3D UMAP Visualization")
            plt.xlabel("UMAP Component 1")
            plt.ylabel("UMAP Component 2")
            plt.colorbar(scatter_umap_3d)
            st.pyplot(plt.gcf())

            # Exploratory Data Analysis (EDA)
            st.subheader("Exploratory Data Analysis (EDA)")

            # Correlation heatmap
            fig = plt.figure()
            sns.heatmap(features.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap")
            st.pyplot(fig)

            # Pairplot for visualizing pairwise relationships
            pairplot_fig = sns.pairplot(data, hue=data.columns[-1])
            plt.suptitle("Pairwise Relationships", y=1.02)   
            fig = plt.gcf()
            st.pyplot(fig)
    except:
        print("error")
