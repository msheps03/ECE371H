

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model

os.chdir('D:/new371H/newECE371git/ECE371H')
# Classes of trafic signs
class_list = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' 
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

model = load_model('./training/TSR.h5')
X_test, label = testing('Test.csv')

predict_x= model.predict(X_test)
Y_pred = np.argmax(predict_x,axis=1)

print("Model Accuracy: ", accuracy_score(label, Y_pred))

for i in range(10):
    #current_file = './Test/' + f"{1:02d}" + '.png'
    plot,prediction = test_on_img(r'./Test/' + f"{i:05d}" + '.png')
    s = [str(i) for i in prediction]
    a = int("".join(s))
    print("Predicted traffic sign is: ", class_list[a], "Actual Traffic Sign: ", label)
    #print("actual traffic sign is: ", class_list[])
    plt.imshow(plot)
    plt.show()