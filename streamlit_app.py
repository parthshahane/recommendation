import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------
# LOAD DATA
# ------------------------
@st.cache_data
def load_data():
    file_path = 'ratings (1).csv'
    df = pd.read_csv(file_path)

    df.rename(columns={
        'userid': 'user_id',
        'productid': 'product_id'
    }, inplace=True)

    df = df[['user_id', 'product_id', 'rating']]
    return df

# ------------------------
# ITEM SIMILARITY USING Nearest Neighbors
# ------------------------
def build_similarity(df, min_user_ratings=10, min_product_ratings=10, n_neighbors=10):
    user_counts = df['user_id'].value_counts()
    product_counts = df['product_id'].value_counts()
    df_filtered = df[
        df['user_id'].isin(user_counts[user_counts >= min_user_ratings].index) &
        df['product_id'].isin(product_counts[product_counts >= min_product_ratings].index)
    ]

    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    df_filtered['user_idx'] = user_encoder.fit_transform(df_filtered['user_id'])
    df_filtered['product_idx'] = product_encoder.fit_transform(df_filtered['product_id'])

    user_product_sparse = csr_matrix(
        (df_filtered['rating'], (df_filtered['user_idx'], df_filtered['product_idx']))
    )

    item_matrix = user_product_sparse.T
    nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine')
    nn_model.fit(item_matrix)

    product_ids = product_encoder.inverse_transform(np.arange(item_matrix.shape[0]))
    return nn_model, item_matrix, product_encoder, product_ids

# ------------------------
# GET RECOMMENDATIONS
# ------------------------
def get_recommendations(product_id, nn_model, item_matrix, product_encoder, product_ids, top_n=5):
    if product_id not in product_ids:
        return []

    idx = np.where(product_ids == product_id)[0][0]
    distances, indices = nn_model.kneighbors(item_matrix[idx], n_neighbors=top_n + 1)

    recommendations = [product_ids[i] for i in indices.flatten()[1:]]
    return recommendations

# ------------------------
# CLUSTERING FUNCTION (optional)
# ------------------------
def run_clustering(item_matrix, model_type='kmeans', num_clusters=5):
    dense_matrix = item_matrix.toarray()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(dense_matrix)

    if model_type == 'kmeans':
        model = KMeans(n_clusters=num_clusters, random_state=42)
        labels = model.fit_predict(reduced)
    elif model_type == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(reduced)
    else:
        raise ValueError("Unsupported model type")

    return reduced, labels

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="🛍️ Product Recommendation System", layout="wide")
st.title("🛒 E-Commerce Product Recommendation Dashboard")

df = load_data()
nn_model, item_matrix, product_encoder, product_ids = build_similarity(df)

tab1, tab2 = st.tabs(["🔍 Recommender System", "📊 Clustering Analysis"])

# -------------------- TAB 1 --------------------
with tab1:
    st.header("📦 Product Recommendations")
    selected_product = st.selectbox("Select a product ID:", df['product_id'].unique())
    if st.button("Get Recommendations"):
        recs = get_recommendations(selected_product, nn_model, item_matrix, product_encoder, product_ids)
        if recs:
            st.success(f"Top recommendations for product `{selected_product}`:")
            for i, rec in enumerate(recs, 1):
                st.markdown(f"**{i}.** Product ID: `{rec}`")
        else:
            st.warning("No similar products found.")

# -------------------- TAB 2 --------------------
with tab2:
    st.header("🧬 Clustering Products Based on Ratings")
    st.markdown("Use PCA to visualize and compare clustering algorithms.")

    cluster_method = st.selectbox("Choose clustering method:", ['KMeans', 'DBSCAN'])

    if cluster_method == 'KMeans':
        k = st.slider("Number of clusters (K)", 2, 10, 5)
    else:
        k = None

    if st.button("Run Clustering"):
        try:
            reduced, labels = run_clustering(item_matrix, model_type=cluster_method.lower(), num_clusters=k if k else 5)
            plot_df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
            plot_df['Cluster'] = labels

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built for the live Product Recommendation System project (P535).")
