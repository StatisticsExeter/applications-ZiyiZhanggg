from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from course.utils import find_project_root

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def hcluster_analysis():
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    outpath = base_dir / VIGNETTE_DIR / 'dendrogram.html'
    fig = _plot_dendrogram(df)
    fig.write_html(outpath)


def hierarchical_groups(height):
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    linked = _fit_dendrogram(df)
    clusters = _cutree(linked, height)  # adjust this value based on dendrogram scale
    df_plot = _pca(df)
    df_plot['cluster'] = clusters.astype(str)  # convert to string for color grouping
    outpath = base_dir / VIGNETTE_DIR / 'hscatter.html'
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)


def _fit_dendrogram(df):
    X = df.values
    X_scaled = StandardScaler().fit_transform(X)
    tree = linkage(X_scaled, method="ward")
    return tree


def _plot_dendrogram(df):
    X = df.values
    X_scaled = StandardScaler().fit_transform(X)

    labels = df.index.astype(str).tolist()
    fig = ff.create_dendrogram(
        X_scaled,
        labels=labels,
        linkagefun=lambda x: linkage(x, method="ward"),
    )
    fig.update_layout(title="Interactive Hierarchical Clustering Dendrogram")
    return fig


def _cutree(tree, height):
    clusters = fcluster(tree, t=height, criterion="distance")
    return pd.DataFrame({"cluster": clusters})


def _pca(df):
    X = df.values if hasattr(df, "values") else df
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_scaled)
    return pd.DataFrame(Z, columns=["PC1", "PC2"])


def _scatter_clusters(df):
    plot_df = df.copy()
    plot_df["cluster"] = plot_df["cluster"].astype(str)

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="PCA Scatter Plot Colored by Cluster Labels",
    )
    return fig
