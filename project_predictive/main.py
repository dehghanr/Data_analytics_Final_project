import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

''' Load data '''
data = load_breast_cancer()
# print(data)
# data = a.target_names

target_names = data.target_names
# print(target_names)

target = data.target
# print(target)

filename = data.filename
# print(filename)

DESCR = data.DESCR
# print(DESCR)

data = data.data
print(data.shape)

'''Change dataset to data frame'''
class_names_min = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                   'compactness', 'concavity', 'concave points',
                   'symmetry', 'fractal dimension']
class_names_mean = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                    'smoothness_mean',
                    'compactness_mean', 'concavity_mean', 'concave points_mean',
                    'symmetry', 'fractal dimension']
class_names_max = ['radius_max', 'texture_max', 'perimeter_max', 'area_max',
                   'smoothness_max',
                   'compactness_max', 'concavity_max', 'concave points_max',
                   'symmetry_max', 'fractal dimension_max']
class_names = class_names_min + class_names_mean + class_names_max
# print(class_names)
df = pd.DataFrame(data)
df['targets'] = target
print(df)
groups = df.groupby('targets')

''' Visualize data '''
# scatter
fig, ax = plt.subplots()
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    if name is 0:
        color = 'red'
    else:
        color = 'blue'
    feature_1 = 4
    feature_2 = 8
    ax.plot(group[feature_1], group[feature_2], marker='o',
            linestyle='', ms=5, label=name, color=color)
    plt.xlabel(class_names[feature_1])
    plt.ylabel(class_names[feature_2])
    plt.title('feature_{} VS feature_{}'.format(class_names[feature_1], class_names[feature_2]))
ax.legend(['Malignant', 'Benign'])
plt.show()

# Bar plot
target_names = ['Benign', 'Malignant']
a = df['targets'].value_counts().to_numpy(dtype=np.float)
dict_bar = dict(zip(target_names, a))
plt.bar(range(len(dict_bar)), list(dict_bar.values()), align='center')
plt.xticks(range(len(dict_bar)), list(dict_bar.keys()))
plt.title('Malignant and Benign cases number')
plt.ylabel('Cases number')
plt.show()

# ratio bar plot
total = sum(a)
for i in range(len(a)):
    a[i] = a[i] / total
plt.show()
dict_bar = dict(zip(target_names, a))
plt.bar(range(len(dict_bar)), list(dict_bar.values()), align='center')
plt.xticks(range(len(dict_bar)), list(dict_bar.keys()))
plt.title('Malignant and Benign cases ratio')
plt.ylabel('ratio')
plt.show()

from sklearn.decomposition import PCA

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(0.99)
principalComponents = pca.fit_transform(data)
# PCA(n_components=2)
print(pca.explained_variance_ratio_)

''' Visualize the data '''

principalDf = pd.DataFrame(data=principalComponents
                           , columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['targets']]], axis=1)
print(finalDf)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = [0, 1]
colors = ['b', 'r']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['targets'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
ax.legend(['Benign', 'Malignant'])
plt.show()
