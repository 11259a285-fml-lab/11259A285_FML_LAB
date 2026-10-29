from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
print("Dataset shape",X.shape)
print("Number of samples",X.shape[0])
print("Number of features ",X.shape[1])
print("features")
print(iris.feature_names)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(
    n_clusters = 3,
    random_state = 42,
    n_init = 10
)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print("cluster labels")
print(labels)
print("centroids centres")
print(centroids)
silhouette = silhouette_score(X_scaled,labels)

print("KMeans model evaluation")
print("--------------")
print("silhouette score:",round(silhouette,2))
print("Number of samples in each cluster count")
for i in range(3):
    print(
        "cluster",i,
        ":",
        sum(labels == i),
        "samples"
    )


plt.figure(figsize =(8,6))
plt.scatter(
    X_scaled[:,2],
    X_scaled[:,3],
    c = labels,
)
plt.scatter(
    centroids[:,2],
    centroids[:,3],
    marker = 'X',
    s= 200
)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.title("Silhouette Score")
plt.show()





