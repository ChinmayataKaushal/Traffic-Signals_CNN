#PROJECT - Traffic Signals using CNN

## Importing all libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model

## Setting paths 
path = '/home/chinmayata/Documents/Projects/CNN-TrafficSignals/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'
path1 = '/home/chinmayata/Documents/Projects/CNN-TrafficSignals/GTSRB-Training_fixed/GTSRB/Training'
path2 = '/home/chinmayata/Documents/Projects/CNN-TrafficSignals/GTSRB_Final_Test_GT/GT-final_test.csv'

## Creating the CNN model

def model():
    ## Convulation and max pooling
    mod= Sequential()
    # 1st layer
    mod.add(Conv2D(32, (5, 5), input_shape = (32, 32, 3), activation = 'relu'))
    mod.add(MaxPooling2D(pool_size = (2, 2)))
    # 2nd layer 
    mod.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), activation = 'relu'))
    mod.add(MaxPooling2D(pool_size = (2, 2)))

    ## Flatten 
    mod.add(Flatten())

    ## Full connection
    mod.add(Dense(units = 128, activation = 'relu'))
    mod.add(Dropout(0.2))
    mod.add(Dense(units = 43, activation = 'softmax'))
    ## Compile the model
    mod.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return mod

Mymodel= model()

## Pre-process an image while importing

def pre_process(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)     #convert to gray
    image = image/255                                #normalize in 0 to 1
    images = np.dstack([image, image, image])        #stack 3 images together to make id 3 dimensional
    return images

train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        preprocessing_function=pre_process)

train_generator = train_datagen.flow_from_directory(
        path1,
        subset = 'training',
        target_size=(32, 32),
        batch_size=50)

validation_generator = train_datagen.flow_from_directory(
        path1,
        subset = 'validation',
        target_size=(32, 32),
        batch_size=50)

modX = Mymodel.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=11,
        validation_data=validation_generator) 


## Plot
plt.figure(1)
plt.plot(modX.history['loss'])
plt.plot(modX.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(modX.history['accuracy'])
plt.plot(modX.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

## Save model 
Mymodel.save('CNNtrafficSigns.h5')

## Load model
#Mymodel = load_model('/home/chinmayata/Documents/Projects/CNN-TrafficSignals/CNNtrafficSigns.h5')

## Load test images

Img = []
Name = []
Lst = os.listdir(path)
NoOfClasses = len(Lst)
print("No of classes detected are - ", NoOfClasses)

for x in Lst:
    try:
        img_ = cv2.imread(path+"/" + x)
        img_ = cv2.resize(img_ , (32, 32))
        Name.append(x)
        image = pre_process(img_ )
        Img.append(image)
    except Exception as e:
        print(str(e))
    
    
Img = np.array(Img)

## Find answer labels

ans = []
data_f = pd.read_csv(path2, sep=';')

for n in Name:
    data_frame = data_f[data_f['Filename']==n]
    ans.append(int(data_frame['ClassId'].values))



## Make prections
pred = Mymodel.predict_classes(Img)



## Confusion matrix
matrix = confusion_matrix(ans, pred)
clf_report = classification_report(ans, pred)


## Plot graphs
sns.clustermap(matrix)
