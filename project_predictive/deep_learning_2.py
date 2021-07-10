import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

''' Load dataset '''
dataset = pd.read_csv('milk.data', header=None)
features = list(np.arange(1, 14))

y = dataset[0].to_numpy()
y = y - 1
x = dataset[features]
x = x.values

feature_1 = x.T[3]
# # print(feature_1)

'''plot feature 1'''
density = stats.gaussian_kde(feature_1)
n, a, _ = plt.hist(feature_1, bins=30, histtype=u'step',
                   density=True)
plt.plot(a, density(a))
plt.legend(['Density', 'Histogram'])
plt.xlabel('feature 1 distribution')
plt.ylabel('feature 1 frequency')
plt.title('Feature 1 distribution')
plt.show()

'''Dataset normalization'''
# from sklearn.preprocessing import normalize
# x = normalize(x, norm='l2', axis=0)
# feature_1 = x.T[3]


'''plot feature 1'''
# density = stats.gaussian_kde(feature_1)
# n, a, _ = plt.hist(feature_1, bins=30, histtype=u'step', density=True)
# plt.plot(a, density(a))
# plt.legend(['Density', 'Histogram'])
# plt.xlabel('feature 1 distribution')
# plt.ylabel('feature 1 frequency')
# plt.title('Feature 1 distribution')
# plt.show()

''' Dataset scale '''  # Scaled dataset variance=1, mean=0
from sklearn.preprocessing import scale

x = scale(x, axis=0)
feature_1 = x.T[3]
print(x.std(axis=0))
# print(x.mean(axis=0))

''' plot feature 1 '''
density = stats.gaussian_kde(feature_1)
n, a, _ = plt.hist(feature_1, bins=30, histtype=u'step', density=True)
plt.plot(a, density(a))
plt.legend(['Density', 'Histogram'])
plt.xlabel('feature 1 distribution')
plt.ylabel('feature 1 frequency')
plt.title('Feature 1 distribution')
plt.show()

''' split your dataset'''
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=42,
                                                    test_size=0.1)

print(len(x_test))
print(y_test)

''' One hot encoder '''
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)
print(y_test)

''' Build your keras model '''
from keras.models import Sequential
from keras.layers import Dense


def build_model(optimizer):
    input_shape = x_train[0].shape
    # print(input_shape)
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=input_shape))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc', 'mse'])  # it could be MSE, MAE, ...
    return model


''' Now lets try different optimizers '''
optimizers_list = ['RMSprop', 'Adadelta', 'adam', 'sgd', 'Adagrad', 'Nadam', 'Adamax']
for i in range(len(optimizers_list)):
    model = build_model(optimizer=optimizers_list[i])
    ''' Summary your model'''
    # # print(model.summary())

    ''' Train your model '''
    history = model.fit(x_train, y_train, epochs=100, verbose=0)
    print('Training done!, optimizer: {}'.format(optimizers_list[i]))

    ''' Prediction '''
    pred_train = model.predict(x_train)
    scores = model.evaluate(x_train, y_train, verbose=0)
    # print('\nAccuracy on training data: {}% \nError on training data: {}'.format(scores[1], 1 - scores[1]))

    pred_test = model.predict(x_test)
    scores2 = model.evaluate(x_test, y_test, verbose=0)
    # print('\nAccuracy on test data: {}% \nError on test data: {}\n\n'.format(scores2[1], 1 - scores2[1]))

    ''' See your results '''
    import matplotlib.pyplot as plt

    abstract = pd.DataFrame(history.history)
    # print(abstract)

    epoch_plot = np.arange(len(abstract['acc']))
    plt.plot(epoch_plot, abstract['mean_squared_error'], label='MSE')
    plt.plot(epoch_plot, abstract['acc'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Error rate per epoch for "{}"'.format(optimizers_list[i]))
    plt.legend()
    plt.grid(True)
    # plt.ylim(0, 1)
    plt.show()
