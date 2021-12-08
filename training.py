import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


def pre_process(degree):
    os.chdir('D:/new371H/newECE371git/ECE371H')
    data = []
    labels = []
    # We have 43 Classes
    classes = 43
    cur_path = os.getcwd()

    for i in range(classes):
        if degree == 90:
            path = os.path.join(cur_path,'Train',str(i))
        else:
            path = os.path.join(cur_path,'newTrain',str(i))

        images = os.listdir(path)
        for a in images:
            try:
                image = Image.open(path + '\\'+ a)
                image = image.resize((30,30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(e)

    data = np.array(data)
    labels = np.array(labels)
    if os.path.isdir('training'):
        pass
    else:
        os.mkdir('training')

    np.save('./training/data',data)
    np.save('./training/target',labels)
    return


def build_model(epochs, degree =90, plot = 0):
    pre_process(degree)
    data=np.load('./training/data.npy')
    labels=np.load('./training/target.npy')

    print(data.shape, labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)

    # building model
    if degree == 90:
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        # We have 43 classes that's why we have defined 43 in the dense
        model.add(Dense(43, activation='softmax'))
    else:
        model = load_model("./training/TSR_" + str(epochs) + ".h5")

    #Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    if degree != 90:
        degree = "_" + str(degree)
        model.save("./training/TSR_" + str(epochs) + degree + ".h5")
    else:
        model.save("./training/TSR_" + str(epochs) + ".h5")

    if plot:
        plot_accuracy(history, epochs, degree)
        plot_loss(history, epochs, degree)

    return


def plot_accuracy(history, epochs, degree):
    #accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig("Accuracy_" + str(epochs) + "_" + str(degree) + ".png")
    plt.clf()
    return


def plot_loss(history, epochs, degree):
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("Loss_" + str(epochs) + "_" + str(degree) + ".png")
    plt.clf()
    return


plot = 0
epoch_test = [75,100]

for epoch in epoch_test:
    build_model(epoch, 45, 1)