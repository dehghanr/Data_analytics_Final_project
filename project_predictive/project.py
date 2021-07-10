import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

y = data.target
x = data.data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

''' Deep learning '''
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)
print(y_test)

''' Build your keras model '''
from keras.models import Sequential
from keras.layers import Dense

input_shape = x_train[0].shape
print(input_shape)
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=input_shape))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc', 'mse'])  # it could be MSE, MAE, ...

''' Summary your model'''
print(model.summary())

''' Train your model '''
history = model.fit(x_train, y_train, epochs=100, verbose=0)
print('Training done! ')

''' Prediction '''
pred_train = model.predict(x_train)
scores = model.evaluate(x_train, y_train, verbose=0)
print('\nAccuracy on training data: {}% \nError on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test = model.predict(x_test)
scores2 = model.evaluate(x_test, y_test, verbose=0)
print('\nAccuracy on test data: {}% \nError on test data: {}\n\n'.format(scores2[1], 1 - scores2[1]))

''' See your results '''
import matplotlib.pyplot as plt

abstract = pd.DataFrame(history.history)
print(abstract)

epoch_plot = np.arange(len(abstract['acc']))
plt.plot(epoch_plot, abstract['mse'], label='MSE')
plt.plot(epoch_plot, abstract['acc'], label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Error rate per epoch')
plt.legend()
plt.grid(True)
# plt.ylim(0, 1)
plt.show()

''' Precision and recall '''
print('Precision and recall')
from sklearn.metrics import classification_report

print(y_test)
y_pred = model.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(y_pred)

# remove one hot encoding
y_test = np.argmax(y_test, axis=1)
print(y_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

print(classification_report(y_test, y_pred_bool))

''' Confusion matrix '''
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

''' A useful function for you to see confusion matrix '''
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


class_names = np.array(['Benign', 'Malignant'])
# np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()




