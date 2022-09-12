## Utilitary libraries

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing
import matplotlib.pyplot as plt



from itertools import product
from statistics import mean

from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold , ShuffleSplit , GridSearchCV , StratifiedKFold ,RandomizedSearchCV

## Dimensionality reduction PCA
from sklearn.decomposition import PCA

## Classifiers

from sklearn.cluster import KMeans , DBSCAN
from sklearn.neighbors import NearestNeighbors


from All_counts_data_processed_for_classification  import *
outpath = '../../ML_results/'


#It's imporant to scale the data 
scaler = StandardScaler()
scaler.fit(X)
scaled_X =  scaler.transform(X)



#Project the data onto the first two PCs
model = PCA(n_components=4)
X_reduced = model.fit_transform(scaled_X)

plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            c = y , edgecolor='none', alpha=0.8  )
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.savefig(path.join(outpath,"PC_1_2.png"))

plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 2],
            c = y , edgecolor='none', alpha=0.8  )
plt.xlabel('PC 1')
plt.ylabel('PC 3')
plt.savefig(path.join(outpath,"PC_1_3.png"))

plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 3],
            c = y , edgecolor='none', alpha=0.8  )
plt.xlabel('PC 1')
plt.ylabel('PC 4')
plt.savefig(path.join(outpath,"PC_1_4.png"))

plt.figure()
plt.scatter(X_reduced[:, 1], X_reduced[:, 2],
            c = y , edgecolor='none', alpha=0.8  )
plt.xlabel('PC 2')
plt.ylabel('PC 3')
plt.savefig(path.join(outpath,"PC_2_3.png"))

plt.figure()
plt.scatter(X_reduced[:, 1], X_reduced[:, 3],
            c = y , edgecolor='none', alpha=0.8  )
plt.xlabel('PC 2')
plt.ylabel('PC 4')
plt.savefig(path.join(outpath,"PC_2_4.png"))

plt.figure()
plt.scatter(X_reduced[:, 2], X_reduced[:, 3],
            c = y , edgecolor='none', alpha=0.8  )
plt.xlabel('PC 3')
plt.ylabel('PC 4')
plt.savefig(path.join(outpath,"PC_3_4.png"))

#Identify how many PCs do we need to explain up to 95% of the variance 

model = PCA(n_components=65)
X_reduced = model.fit_transform(scaled_X)



plt.figure()
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.axhline(y=0.95, c="red", linewidth=1)
plt.axvline(x=50, c="red", linewidth=1)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Cumulative variance plot")
plt.savefig(path.join(outpath,"Cumulative variance plot.png"))

min_PCs = np.argmax(np.cumsum(model.explained_variance_ratio_) > 0.95)
print(np.cumsum(model.explained_variance_ratio_))
print( "we need " , min_PCs , "PCs to explain up to 95% of the data variance" )
print(np.cumsum(model.explained_variance_ratio_))


PC_values = np.arange(model.n_components_) + 1
print(PC_values)
plt.plot(PC_values, model.explained_variance_ratio_, 'o-', linewidth=1, color='blue')
plt.xticks(PC_values)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

K_means_model=KMeans(n_clusters = 2)
K_means_model.fit( X_reduced)
K_means_model.predict(X_reduced)

plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=K_means_model.predict( X_reduced))
plt.scatter(K_means_model.cluster_centers_[:, 0], K_means_model.cluster_centers_[:, 1], c="r")
plt.title("K_means on the data projected into PC 1 and PC2")
plt.savefig(path.join(outpath,"K_means_onto_PCs.png"))


K_means_model=KMeans(n_clusters = 2 )
K_means_model.fit(scaled_X)
#{K_means_model.predict(scaled_X)
print(K_means_model.cluster_centers_)
plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=K_means_model.predict(scaled_X))
plt.title("K_means on the scaled unprojected data ")
plt.savefig(path.join(outpath,"K_means_onto_scaled_data.png"))


inertia =[]
k_range = range(1,20)
for k in k_range:
    K_means_model= KMeans(n_clusters = k ).fit(scaled_X)
    inertia.append(K_means_model.inertia_)
    
plt.figure()
plt.plot(k_range , inertia)
plt.xlabel('number of clusters ')
plt.axvline(x=4, c="red", linewidth=1)
plt.xticks(np.arange(0, 20, step=1)) 
plt.ylabel('Model cost ')
plt.grid()
plt.title("number of clusters cost tradeoff")
plt.savefig(path.join(outpath,"K_means_cost_accuracy.png"))


K_means_model=KMeans(n_clusters = 4 )
K_means_model.fit( X_reduced)
K_means_model.predict(X_reduced)

plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=K_means_model.predict( X_reduced))
plt.scatter(K_means_model.cluster_centers_[:, 0], K_means_model.cluster_centers_[:, 1], c="r")
plt.title("K_means with the optimal K")
plt.savefig(path.join(outpath,"K_means_opt_k.png"))

#Essayer d'identifier combien de points on etait misclassified 
#Essayer d'éliminer les samples qui sont considéres comme des outliers pour voir si la missclassification diminue

def show_clusters(X,cluster):
    df =DataFrame(dict(x=X_reduced[:, 0],
                       y=X_reduced[:, 1] ,
                       label = cluster ) )
    colors = {outliers :'red' , 
              cluster_1:'blue' , 
              cluster_2:'green' }
    fig , ax = plt.subplot(figsize(8,8))
    grouped = df.groupby('label')
    for key,group in grouped :
        group.plot(ax=ax , kind ='scatter' , x='x' , y='y' , label=key , color = colors[key])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    
    
    
#DBSCAN

#FROM MY READINGS optimal n_neighbors = 2* data dimension

# DBSCAN Parameter Estimation 
neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure()
plt.plot(distances)
plt.axvline(x=56, c="red", linewidth=1)
plt.axhline(y=1.0e6, c="red", linewidth=1)


# Compute DBSCAN
db = DBSCAN(eps=1.0e6, min_samples=4).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(y, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(y, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# Plot result

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X_reduced[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X_reduced[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_ , ' by DBSCAN ')
plt.show()
plt.savefig(path.join(outpath,"DBSCAN_clustering.png"))



