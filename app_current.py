import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Initialize the app
st.set_page_config(page_title="Data Analysis and Machine Learning", layout="centered")

# Handle file upload
input_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])

def check_table_structure(data):
    num_columns = len(data.columns)
    if num_columns < 2:
        return 0, "Table must have at least one feature and one label column."
    if data.isnull().values.any():
        return 2, "Data contains missing values."
    return 1, "Data is structured correctly."

if input_file is not None:
    try:
        # Handle CSV, Excel, TSV files
        if input_file.name.endswith(".csv"):
            data = pd.read_csv(input_file)
        elif input_file.name.endswith(".xlsx"):
            data = pd.read_excel(input_file)
        elif input_file.name.endswith(".tsv"):
            data = pd.read_csv(input_file, delimiter='\t')
        
        # Create tabs
        tabs = st.tabs(["Visualization", "Machine Learning", "Classification", "Information"])

        with tabs[0]:
            # Validate table structure
            is_valid, message = check_table_structure(data)
            
            if is_valid==1 or is_valid==2 :
                if is_valid==2:
                    st,write(message)
                    st.write("Now remvoing missing values")
                num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                for column in data.columns:
                    if column in num_cols:
                        data[column].fillna(data[column].mean(), inplace=True)
                    else:
                        data[column].fillna(data[column].mode()[0], inplace=True)    
                st.success("Data loaded successfully!")
                for col in data.select_dtypes(include=['object']).columns:
                    data[col] = data[col].astype('category').cat.codes
                features = data.iloc[:, :-1]  # Features
                labels = data.iloc[:, -1]     # Labels

                label_encoder = LabelEncoder()
                class_labels = label_encoder.fit_transform(labels)
                
                st.write("### Data Preview")
                st.write(data)
                st.write(f"Number of Samples (S): {data.shape[0]}")
                st.write(f"Number of Features (F): {data.shape[1] - 1}")
                st.write(f"Output Label Column: {data.columns[-1]}")
                
                
            
                # Visualization Tab
                st.subheader("Visualization")
                
                # 2D PCA Visualization
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(features)
                
                plt.figure()
                scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=class_labels, cmap='viridis')
                plt.title("2D PCA Visualization")
                plt.xlabel("PCA Component 1")
                plt.ylabel("PCA Component 2")
                plt.colorbar(scatter)
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
                fig = plt.figure()
                ax_3d_umap = fig.add_subplot(111, projection='3d')
                scatter_3d_umap = ax_3d_umap.scatter(umap_result_3d[:, 0], umap_result_3d[:, 1], umap_result_3d[:, 2], 
                                                     c=class_labels, cmap='viridis')
                ax_3d_umap.set_title("3D UMAP Visualization")
                ax_3d_umap.set_xlabel("UMAP Component 1")
                ax_3d_umap.set_ylabel("UMAP Component 2")
                ax_3d_umap.set_zlabel("UMAP Component 3")
                fig.colorbar(scatter_3d_umap)
                st.pyplot(fig)

                # Exploratory Data Analysis (EDA)
                st.subheader("Exploratory Data Analysis (EDA)")

                # Correlation heatmap (using first 5 features)
                features_small = features.iloc[:, :5]  # Select the first 5 features
                fig = plt.figure()
                sns.heatmap(features_small.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                plt.title("Correlation Heatmap")
                st.pyplot(fig)

                # Pairplot for visualizing pairwise relationships (using first 5 features)
                subset_features = features.iloc[:, :5]  # Select the first 5 features
                subset_data = subset_features.copy()
                subset_data[data.columns[-1]] = labels  # Add label column for hue
                
                try:
                    pairplot_fig = sns.pairplot(subset_data, hue=data.columns[-1], palette='viridis')
                    plt.suptitle("Pairwise Relationships", y=1.02)
                    st.pyplot(pairplot_fig)
                except Exception as e:
                    st.error(f"Error generating pairplot: {e}")

            with tabs[1]:
                # Feature Selection using PCA
                st.subheader("Feature Selection using PCA")
                m,n = features.shape
                # Slider for PCA components, show 1slider at all fetaures initially
                n_comp = st.slider("Select the number of PCA components", min_value=1, 
                                            max_value=n, value=n)

                # PCA transformation
                pca = PCA(n_components=n_comp)
                reduced_features = pca.fit_transform(features)

                st.write("Explained Variance Ratio: "+str(pca.explained_variance_ratio_))
                cols=['PCA'+str(i) for i in range(n_comp)]
                # Display reduced dataset
                df_new = pd.DataFrame(reduced_features, columns=cols)
                df_new['label'] = labels
                st.dataframe(df_new)

            # Inside the Classification tab
            with tabs[2]:
                # Prompt user to choose 'k' for K-Nearest Neighbors
                k = st.slider("Choose 'k' for K-Nearest Neighbors", min_value=1, max_value=20, value=None)

                if k is None:
                    st.write("Please choose the value of 'k' for KNN before proceeding.")
                    # Optionally clear any previous plots or tables
                    st.empty()
                else:
                    st.write("### Results for K-Nearest Neighbors and Random Forest")

                    # Split data into train-test sets
                    rnd = 42
                    test_size = 0.2
                    
                    X_train, X_test, y_train, y_test = train_test_split(features, class_labels, test_size=test_size, random_state=rnd,stratify=class_labels)
                    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(reduced_features, class_labels, test_size=test_size, random_state=rnd,stratify=class_labels)

                    # Function to train and evaluate the model
                    def train_and_evaluate(algo, X_train, X_test, y_train, y_test, k=None):
                        if algo == "K-Nearest Neighbors":
                            model = KNeighborsClassifier(n_neighbors=k)
                        elif algo == "Random Forest":
                            model = RandomForestClassifier()

                        model.fit(X_train, y_train)
                        ypred_tr = model.predict(X_train)
                        ypred_tst = model.predict(X_test)

                        tr_accuracy = accuracy_score(y_train, ypred_tr)
                        tst_accuracy = accuracy_score(y_test, ypred_tst)
                        tr_f1 = f1_score(y_train, ypred_tr, average='weighted')
                        tst_f1 = f1_score(y_test, ypred_tst, average='weighted')

                        # ROC AUC calculation
                        try:
                            y_train_proba = model.predict_proba(X_train)
                            y_test_proba = model.predict_proba(X_test)
                            tr_roc_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr')
                            tst_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
                        except AttributeError:
                            tr_roc_auc = None
                            tst_roc_auc = None

                        return tr_accuracy, tst_accuracy, tr_f1, tst_f1, tr_roc_auc, tst_roc_auc

                    
                    # Train and evaluate models on original and PCA features
                    results=[]
                    classifiers=["K-Nearest Neighbors","Random Forest"]
                    for clf in classifiers:
                        results.append(train_and_evaluate("K-Nearest Neighbors", X_train, X_test, y_train, y_test, k=k))
                    for clf in classifiers:
                        results.append(train_and_evaluate("K-Nearest Neighbors", X_train_pca, X_test_pca, y_train_pca, y_test_pca, k=k))
                        
                   
                    results =[np.array(res) for res in results]
                    results = np.array(results)
                    
                    # Metrics Before PCA
                    st.write("### Metrics Before Feature Reduction")
                    perf_metrics = ["Train Accuracy", "Test Accuracy", "Train F1", "Test F1", "Train ROC AUC", "Test ROC AUC"]
                    results_df = pd.DataFrame({
                        "Metric": perf_metrics,
                        "KNN": results[0],
                        "Random Forest": results[1]
                    })
                    st.write(results_df)
                    
                    # Metrics After PCA
                    st.write("### Metrics After Feature Reduction (PCA)")
                    results_df_pca = pd.DataFrame({
                        "Metric": perf_metrics,
                        "KNN (PCA)": results[2],
                        "Random Forest (PCA)": results[3]
                    })
                    st.write(results_df_pca)

                    # Function to plot metrics for Train and Test in separate figures
                    def plot_metrics(knn_train_before,knn_train_after,rf_train_before,rf_train_after,dist="train"):
                        metrics =["Accuracy","F1", "ROC_AUC"]
                        for j,met in enumerate(metrics):
                        
                            bar_width = 0.1
                            index = range(len(labels))
                            # creating the dataset
                            data = {'KNN Train':knn_train_before[j],'KNN(PCA) Train':knn_train_after[j] ,'RF Train':rf_train_before[j],'RF(PCA) Train':rf_train_after[j]}
                            keys = list(data.keys())
                            vals = list(data.values())
                             
                          
                            colors = ['red', 'green', 'blue', 'orange']
                            fig = plt.figure()
                            # creating the bar plot
                            plt.bar(keys, vals, color =colors, width = 0.4)
                            plt.ylim(0, 1)
                            plt.ylabel(metrics[j])
                            plt.xlabel("Clasification Algorithms")
                            plt.title("Comaprison of "+ metrics[j]+" for "+ dist+" data")
                            st.pyplot(fig)

                        
                    # Prepare data for plotting (separate train and test)
                    knn_train_before = results[0,[0,2,4]]  # Train Accuracy, Train F1, Train ROC AUC
                    knn_train_after = results[2,[0,2,4]]   # After PCA (Train)
                    rf_train_before = results[1,[0,2,4]]   # Random Forest Train Metrics
                    rf_train_after = results[3,[0,2,4]]   # After PCA (Train)
                    
                    knn_test_before = results[0,[1,3,5]]   # Test Accuracy, Test F1, Test ROC AUC
                    knn_test_after = results[2,[1,3,5]] # After PCA (Test)
                    rf_test_before = results[1,[1,3,5]]   # Random Forest Test Metrics
                    rf_test_after = results[3,[1,3,5]]  # After PCA (Test)

                    # Plot separate Train and Test metrics in separate figures
                    plot_metrics(knn_train_before,knn_train_after,rf_train_before,rf_train_after,"train")
                    plot_metrics(knn_test_before,knn_test_after,rf_test_before,rf_test_after,"test")
                    
            with tabs[3]:
                st.write("### Information")
                
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.write("No file uploaded yet")