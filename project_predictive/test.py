from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

''' Load data '''
data = load_breast_cancer()
y = data.target
x = data.data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
names_list = ['Random forest', 'SVM', 'Naive Bayes', 'MLP', 'K-Nearest neighbor', 'Linear regression',
              'Logistic regression']
results = []

''' Random forest '''
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
# print(clf.score(x_test, y_test))
print("Accuracy Random forest: {:.2f}".format(rfc.score(x_test, y_test)))
results.append(rfc.score(x_test, y_test))

'''  SVM '''
# from sklearn import svm
#
# clf = svm.SVC(kernel='linear')
# clf.fit(x_train, y_train)
# # print(clf.score(x_test, y_test))
# print("Accuracy SVM: {:.2f}".format(clf.score(x_test, y_test)))
# results.append(clf.score(x_test, y_test))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC

clf = LinearSVC()
clf = CalibratedClassifierCV(clf)
clf.fit(x_train, y_train)
print("Accuracy SVM: {:.2f}".format(clf.score(x_test, y_test)))
results.append(clf.score(x_test, y_test))

''' Naive Bayes'''
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)
print("Accuracy Naive Bayes: {:.2f}".format(gnb.score(x_test, y_test)))
results.append(gnb.score(x_test, y_test))

''' MLP '''
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(500, 10), activation='relu')
mlp.fit(x_train, y_train)
print("Accuracy MLP: {:.2f}".format(mlp.score(x_test, y_test)))
results.append(mlp.score(x_test, y_test))

''' K-Nearest neighbor'''
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print("Accuracy KNN: {:.2f}".format(knn.score(x_test, y_test)))
results.append(knn.score(x_test, y_test))

''' Linear regression '''
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
# y_pred_grd = regr.predict_proba(x_test)[:, 1]
print("Accuracy Linear regression: {:.2f}".format(regr.score(x_test, y_test)))
results.append(regr.score(x_test, y_test))

''' Logistic regression '''
log = linear_model.LogisticRegression(C=1e5)
log.fit(x_train, y_train)
print("Accuracy Linear regression: {:.2f}".format(log.score(x_test, y_test)))
results.append(log.score(x_test, y_test))

''' Sort results '''
final_file = zip(names_list, results)
final_file = sorted(final_file, key=lambda x: x[1], reverse=True)
a = list(zip(*final_file))
print(a)
plt.bar(a[0], a[1])
plt.xticks(a[0], rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Classifiers')
plt.title('Comparison of classifiers')
plt.show()

''' ROC '''
from sklearn.metrics import roc_curve

roc_result = []
y_pred_grd = rfc.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='Random_forest')

y_pred_grd = clf.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='SVM')

y_pred_grd = gnb.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='Gaussian NB')

y_pred_grd = mlp.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='MLP')

y_pred_grd = knn.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='KNN')

y_pred_grd = regr.predict(x_test)
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='Linear Regression')

y_pred_grd = log.predict(x_test)
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='Logistic regression')

plt.legend()
plt.title("ROC curve")
plt.xlabel('Specificity (FPR)')
plt.ylabel('Sensitivity (TPR)')

# plt.xlim([0, 0.3])
# plt.ylim([0, 1])
plt.show()


y_pred_grd = rfc.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='Random_forest')

y_pred_grd = clf.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='SVM')

y_pred_grd = gnb.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='Gaussian NB')

y_pred_grd = mlp.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='MLP')

y_pred_grd = knn.predict_proba(x_test)[:, 1]
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='KNN')

y_pred_grd = regr.predict(x_test)
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='Linear Regression')

y_pred_grd = log.predict(x_test)
roc_result.append(roc_curve(y_test, y_pred_grd))
plt.plot(roc_curve(y_test, y_pred_grd)[0], roc_curve(y_test, y_pred_grd)[1],
         label='Logistic regression')

plt.legend()
plt.title("ROC curve")
plt.xlabel('Specificity (FPR)')
plt.ylabel('Sensitivity (TPR)')

plt.xlim([0, 0.3])
plt.ylim([0, 1])
plt.show()
