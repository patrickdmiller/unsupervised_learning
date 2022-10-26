from include.Framingham_data import *
# from include.Switrs_data import *
from include.Spam_data import *
from include.Fire_data import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# fhc = Framingham()
fire = Fire(verbose = True)
# spam = Spam(verbose=True)
# spam.describe()
# print(spam.X_train[0:10])
def silhouette_analysis2(n_clusters, seed, X):
    labels = KMeans(n_clusters=n_clusters, init="k-means++", random_state=seed).fit(X).labels_
    print(n_clusters, " = ", silhouette_score(X, labels, metric='euclidean'))
    
for i in range(2,40):
    silhouette_analysis2(i, 42, fire.X_train)
