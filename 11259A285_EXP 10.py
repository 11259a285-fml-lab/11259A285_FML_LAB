from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
print("Dataset shape",X.shape)
print("Number of samples:",X.shape[0])
print("Number of features:",X.shape[1])
print("features")
print("features_names")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
linked = linkage(
    X_scaled,method="ward")
plt.figure(figsize = (10,6))
dendrogram(
    linked,
    truncate_mode = 'lastp',
    p = 200
)
plt.title("Hierarchical Clustering")
plt.xlabel("clusters")
plt.ylabel("target")
plt.show()

model = AgglomerativeClustering(
    n_clusters = 3,
    linkage = "ward"
)
labels = model.fit_predict(X_scaled)
print("Cluster labels:",labels)
model.fit(X_scaled)
print("cluster labels")
print("labels")
silhouette = silhouette_score(X_scaled, model.labels_)
print("silhouette",round(silhouette,2))
print("Number of samples in each cluster")
for i in range(3):
    print(
        "cluster",i,
        ":",
      sum(labels == i),
        "samples"
    )
    plt.figure(figsize = (10,6))
    plt.scatter(
        X_scaled[:, 2],
        X_scaled[:, 3],
        c = labels

    )
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.title("Hierarchical Clustering")

plt.show()




