

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
os.chdir('D:/new371H/ECE371H')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# Classes of trafic signs
class_list = {0: 'Speed limit (20km/h)',
              1: 'Speed limit (30km/h)',
              2: 'Speed limit (50km/h)',
              3: 'Speed limit (60km/h)',
              4: 'Speed limit (70km/h)',
              5: 'Speed limit (80km/h)',
              6: 'End of speed limit (80km/h)',
              7: 'Speed limit (100km/h)',
              8: 'Speed limit (120km/h)',
              9: 'No passing',
              10: 'No passing veh over 3.5 tons',
              11: 'Right-of-way at intersection',
              12: 'Priority road',
              13: 'Yield',
              14: 'Stop',
              15: 'No vehicles',
              16: 'Veh > 3.5 tons prohibited',
              17: 'No entry',
              18: 'General caution',
              19: 'Dangerous curve left',
              20: 'Dangerous curve right',
              21: 'Double curve',
              22: 'Bumpy road',
              23: 'Slippery road',
              24: 'Road narrows on the right',
              25: 'Road work',
              26: 'Traffic signals',
              27: 'Pedestrians',
              28: 'Children crossing',
              29: 'Bicycles crossing',
              30: 'Beware of ice/snow',
              31: 'Wild animals crossing',
              32: 'End speed + passing limits',
              33: 'Turn right ahead',
              34: 'Turn left ahead',
              35: 'Ahead only',
              36: 'Go straight or right',
              37: 'Go straight or left',
              38: 'Keep right',
              39: 'Keep left',
              40: 'Roundabout mandatory',
              41: 'End of no passing',
              42: 'End no passing veh > 3.5 tons'
              }


def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data = []
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30, 30))
        data.append(np.array(image))
    X_test = np.array(data)
    return X_test, label


def test_on_img(img):
    data = []
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))
    X_test = np.array(data)
    predict_x = model.predict(X_test)
    Y_pred = np.argmax(predict_x, axis=1)
    return image, Y_pred

data = []
labels = []
# We have 43 Classes
classes = 43
cur_path = os.getcwd()

for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
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

# os.mkdir('training')

np.save('./training/data',data)
np.save('./training/target',labels)

data=np.load('./training/data.npy')
labels=np.load('./training/target.npy')

print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# building model
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


#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# Loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
'''
X_test, label = testing('Test.csv')

#Y_pred = model.predict_classes(X_test) # deprecated?
predict_x= model.predict(X_test)
Y_pred = np.argmax(predict_x,axis=1)

print(accuracy_score(label, Y_pred))

model.save("./training/TSR.h5")

os.chdir(r'D:/new371H/ECE371H')

model = load_model('./training/TSR.h5')

plot,prediction = test_on_img(r'D:/new371H/ECE371H/Test/00500.png')
s = [str(i) for i in prediction]
a = int("".join(s))
print("Predicted traffic sign is: ", class_list[a])
plt.imshow(plot)
plt.show()
