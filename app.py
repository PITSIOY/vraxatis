import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, silhouette_score

def load_data(file_path):
    data = None
    try:
        # Try loading data from CSV file
        data = pd.read_csv(file_path)
    except Exception as e:
        # If failed, try loading data from Excel file
        try:
            data = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            st.error(f"Error loading data: {e}")
    return data

def validate_data(df):
    if df is None:
        return False, "Failed to load data."

    # Check if the dataframe has at least 2 columns (F features + 1 label)
    if df.shape[1] < 2:
        return False, "The dataframe must have at least two columns (F features + 1 label)."

    # Check if the last column contains the label
    if not pd.api.types.is_numeric_dtype(df.iloc[:, -1]) and not pd.api.types.is_string_dtype(df.iloc[:, -1]):
        return False, "The last column should contain the labels of the samples."

    return True, "Data is valid."

def visualize_2d(df):
    st.subheader("2D Visualization")

    # Perform dimensionality reduction
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)

    pca_result = pca.fit_transform(df.iloc[:, :-1])
    tsne_result = tsne.fit_transform(df.iloc[:, :-1])

    # Convert labels to numeric values
    label_encoder = LabelEncoder()
    df.iloc[:, -1] = label_encoder.fit_transform(df.iloc[:, -1])

    # Plot PCA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=df.iloc[:, -1], cmap='viridis')
    ax1.set_title('PCA')

    # Plot t-SNE
    ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=df.iloc[:, -1], cmap='viridis')
    ax2.set_title('t-SNE')

    st.pyplot(fig)

def classification_tab():
    st.subheader("Classification Algorithms")
    st.write("Please select parameters for classification algorithms.")

    # Add UI elements for classification algorithms
    algorithm = st.selectbox("Select Classification Algorithm", ["Random Forest"])
    if algorithm == "Random Forest":
        n_estimators = st.slider("Number of Estimators", min_value=1, max_value=100, value=10)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)

        # Create and train Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        # Perform classification and display results
        # ...

def clustering_tab():
    st.subheader("Clustering Algorithms")
    st.write("Please select parameters for clustering algorithms.")

    # Add UI elements for clustering algorithms
    algorithm = st.selectbox("Select Clustering Algorithm", ["KMeans"])
    if algorithm == "KMeans":
        n_clusters = st.slider("Number of Clusters", min_value=1, max_value=10, value=3)

        # Create and fit KMeans clustering model
        kmeans = KMeans(n_clusters=n_clusters)
        # Perform clustering and display results
        # ...

def evaluate_classification_algorithm(df, algorithm, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=test_size, random_state=42)
    
    if algorithm == "Random Forest":
        n_estimators = st.slider("Number of Estimators", min_value=1, max_value=100, value=10)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)

        # Create and train Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        classifier.fit(X_train, y_train)
        
        # Predict
        y_pred = classifier.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

def evaluate_clustering_algorithm(df, algorithm):
    if algorithm == "KMeans":
        n_clusters = st.slider("Number of Clusters", min_value=1, max_value=10, value=3)

        # Create and fit KMeans clustering model
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(df.iloc[:, :-1])
        
        # Predict
        labels = kmeans.labels_

        # Evaluate
        silhouette = silhouette_score(df.iloc[:, :-1], labels)
        return silhouette

def main():
    st.title("Data Analysis App")

    uploaded_file = st.file_uploader("Select a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        is_valid, message = validate_data(df)
        if is_valid:
            st.write("Data loaded successfully:")
            st.write(df.head())

            visualize_2d(df)
        else:
            st.error(message)

if __name__ == "__main__":
    main()
    classification_tab()
    clustering_tab()