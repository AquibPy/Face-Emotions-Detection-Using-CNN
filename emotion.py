import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPool2D,BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

df = pd.read_csv('fer2013.csv')
# print(df.head())

X_train,Y_train,X_test,Y_test = [],[],[],[]

for index,row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val,'float32'))
            Y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val,'float32'))
            Y_test.append(row['emotion'])
    except:
        print(f"Error Occured at index :{index} and row :{row}")


num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48


X_train = np.array(X_train,'float32')
X_test = np.array(X_test,'float32')
Y_train = np.array(Y_train,'float32')
Y_test = np.array(Y_test,'float32')

Y_train = np_utils.to_categorical(Y_train,num_classes=num_labels)
Y_test = np_utils.to_categorical(Y_test,num_classes=num_labels)

# Noarmalizing Data between 0 and 1

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)



# Reshaping Our Data
X_train = X_train.reshape(X_train.shape[0],width,height,1)
X_test = X_test.reshape(X_test.shape[0],width,height,1)

# Initialising a CNN

model = Sequential()

# 1st Layer

model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

# 2nd Layer

model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

# 3rd Layer

model.add(Conv2D(128,kernel_size= (3, 3), activation='relu'))
model.add(Conv2D(128,kernel_size= (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

# Fully Connected Neural Networks
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels,activation='softmax'))

# Compile the Model

model.compile(optimizer=Adam(),loss=categorical_crossentropy,metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=64,epochs=50,verbose=1,validation_data=(X_test,Y_test),shuffle=True)

# Saving Our Model
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")