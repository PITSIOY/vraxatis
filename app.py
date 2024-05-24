import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

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
