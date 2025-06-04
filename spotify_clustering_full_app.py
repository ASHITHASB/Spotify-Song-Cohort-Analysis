
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="🎵 Rolling Stones Spotify Cohort Analysis", layout="wide")

st.title("🎸 Rolling Stones Song Clustering & Recommendation App")

# Upload section
uploaded_file = st.file_uploader("Upload your Rolling Stones Spotify CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Initial preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Data Cleaning
    st.subheader("🧹 Data Cleaning & Info")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    st.write("Remaining rows after cleaning:", df.shape[0])

    # Feature selection for numerical analysis
    numeric_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                    'popularity', 'duration_ms']
    song_features = df[numeric_cols]

    # EDA Section
    st.subheader("📊 Exploratory Data Analysis")

    with st.expander("Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(song_features.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(fig)

    with st.expander("Popularity Distribution"):
        fig, ax = plt.subplots()
        sns.histplot(df['popularity'], bins=20, kde=True)
        st.pyplot(fig)

    with st.expander("Top Albums with Most Popular Songs (Popularity > 70)"):
        popular_songs = df[df['popularity'] > 70]
        top_albums = popular_songs['album'].value_counts().head(5)
        st.bar_chart(top_albums)

    with st.expander("Popularity vs Audio Features"):
        feature = st.selectbox("Choose feature to compare with popularity:", numeric_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature, y='popularity')
        st.pyplot(fig)

    # Clustering
    st.subheader("📈 Clustering Analysis with KMeans")

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(song_features)

    # Elbow Method
    inertia = []
    K = range(2, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    sns.lineplot(x=K, y=inertia, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    st.pyplot(fig)

    # PCA for visualization
    st.subheader("🎨 PCA + Clustering Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    optimal_k = 4
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans_final.fit_predict(X_scaled)

    df['cluster'] = clusters
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10')
    plt.title("Cluster Visualization in PCA Space")
    st.pyplot(fig)

    # Cluster interpretation
    st.subheader("🔍 Cluster Summary Stats")
    st.dataframe(df.groupby("cluster")[numeric_cols].mean().round(2))

    # Recommendation Insight
    st.subheader("💡 Recommended Albums Based on Popular Songs")
    top_albums = df[df['popularity'] > 70]['album'].value_counts().head(2)
    for album in top_albums.index:
        st.markdown(f"**🎧 Recommended Album:** {album}")

else:
    st.info("📂 Please upload your CSV file to begin.")
