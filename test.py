import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1, 1], [-2, -1, 2], [-3, -2, 3], [1, 1, 4], [2, 1, 5], [3, 2, 6]])
pca = PCA(0.95)
pca.fit(X)
print(pca.explained_variance_ratio_)

print(pca.singular_values_)
