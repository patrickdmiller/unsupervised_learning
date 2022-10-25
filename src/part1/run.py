from include.Framingham_data import *
# from include.Switrs_data import *
from include.Spam_data import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# switrs = Switrs(verbose=True, test_size=0.75)
# fhc = Framingham(verbose=True, oversample=True)
# br = Bankruptcy(verbose = True)
# br.describe()

fhc = Framingham()

spam = Spam(verbose=True)
spam.describe()
def silhouette_analysis2(n_clusters, seed, X):
    labels = KMeans(n_clusters=n_clusters, init="k-means++", random_state=seed).fit(X).labels_
    print(n_clusters, " = ", silhouette_score(X, labels, metric='euclidean'))
    
    