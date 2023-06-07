import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import tsfel
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.correlation_filter import correlation_filter

from config import ASSETS, OUTPUTS

PART = 'Part 7'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

file = f'{assets}/food_sales.csv'

# reading the data set
data = pd.read_csv(file, index_col='Date')

categories = pd.Series([x[1] for x in data.columns.str.split('_')])
# len(categories.unique())

# Feature extraction

# get configuration
cfg = tsfel.get_features_by_domain()

# extract features for each food subcategory
features = {col: tsfel.time_series_features_extractor(cfg, data[col])
            for col in data}

features_df = pd.concat(features, axis=0)

# normalizing the features
features_norm_df = pd.DataFrame(MinMaxScaler().fit_transform(features_df),
                                columns=features_df.columns)

# removing features with 0 variance
min_var = VarianceThreshold(threshold=0)
min_var.fit(features_norm_df)
features_norm_df = pd.DataFrame(min_var.transform(features_norm_df),
                                columns=min_var.get_feature_names_out())

# removing correlated features
features_norm_df = correlation_filter(features_norm_df, 0.9)
features_norm_df.index = data.columns

# Time series clustering with K-means
kmeans_parameters = {
    'init': 'k-means++',
    'n_init': 100,
    'max_iter': 50,
}

n_clusters = range(2, 25)
silhouette_coef = []
for k in n_clusters:
    kmeans = KMeans(n_clusters=k, **kmeans_parameters)
    kmeans.fit(features_norm_df)

    score = silhouette_score(features_norm_df, kmeans.labels_)

    silhouette_coef.append(score)

silhouette_coef = pd.Series(silhouette_coef, index=n_clusters)

silhouette_df = silhouette_coef.reset_index()
silhouette_df.columns = ['No. clusters', 'Silhouette']

plt.style.use('fivethirtyeight')
sns.set_style(rc={'axes.facecolor': 'white',
                  'figure.facecolor': 'white'})
plt.figure(figsize=(10, 5))
sns.barplot(data=silhouette_df,
            x="No. clusters",
            y="Silhouette",
            color='#fdd128',
            ).set(title='Silhouette scores')

# Time series clustering with Hierarchical clustering

clustering = shc.linkage(features_norm_df, method='ward')

fig, ax = plt.subplots()
dend = shc.dendrogram(clustering,
                      labels=categories.values,
                      orientation='right',
                      leaf_font_size=9)
fig.subplots_adjust(left=0.5)

# Visualization with PCA


n_clusters = 5

pca = PCA(n_components=2)
model = KMeans(n_clusters=n_clusters, **kmeans_parameters)

feats_pca = pca.fit_transform(features_norm_df)
model.fit(feats_pca)

pca_df = pd.DataFrame(
    feats_pca,
    columns=['PC1', 'PC2'],
)

pca_df['Predicted'] = model.labels_

plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 8))
scat = sns.scatterplot(
    x="PC1",
    y="PC2",
    s=70,
    data=pca_df,
    hue="Predicted",
    # style="Actual",
    # palette="Accent",
    palette="Dark2",
)

#
features_norm_df_small = correlation_filter(features_norm_df, 0.3)

kmeans = KMeans(n_clusters=5, **kmeans_parameters)
kmeans.fit(features_norm_df_small)

features_norm_df_small['Cluster'] = kmeans.labels_

plt.style.use('fivethirtyeight')
sns.set(style="ticks", rc={'lines.linewidth': 0.7})
ax = sns.pointplot(data=features_norm_df_small.melt('Cluster'),
                   x="variable",
                   errorbar=None,
                   y="value",
                   hue="Cluster",
                   dodge=True)
ax.set(xlabel='', ylabel='Feature value')
