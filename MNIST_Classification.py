import pandas as pd

train_labels = pd.read_csv("train.csv")

train_labels['label']= train_labels['label'].map(str)
train_labels.info()

from sklearn.model_selection import train_test_split
train,validate =train_test_split(train_labels, test_size=0.15, random_state=0)



train_labels.groupby(['label']).count().plot(kind='bar')

image_size = (28,28)



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory='D:\\PracticeWS\\Analytic Vidya\\MNIST\\Train_UQcUa52\\Images\\train',
        x_col='filename',
        y_col='label',
        target_size=image_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
        )

validation_generator = test_datagen.flow_from_dataframe(
        dataframe=validate,
        directory='D:\\PracticeWS\\Analytic Vidya\\MNIST\\Train_UQcUa52\\Images\\train',
        x_col='filename',
        y_col='label',
        target_size=image_size,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
        )

from keras.models import Input,Model
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,BatchNormalization


################################################################

input = Input(shape=(28,28,1))
conv1_1 = Conv2D(filters=16, kernel_size=(3,3), strides=2, activation='relu', kernel_initializer='he_uniform')(input)
conv1_2 = Conv2D(filters=16, kernel_size=(3,3), strides=1, activation='relu', kernel_initializer='he_uniform')(conv1_1)
max_pool1 = MaxPool2D(pool_size=(2,2), strides=1)(conv1_2)



conv2_1 = Conv2D(filters=32, kernel_size=(3,3), strides=2, activation='relu', kernel_initializer='he_uniform')(max_pool1)
max_pool2 = MaxPool2D(pool_size=(2,2), strides=1)(conv2_1)


flatten = Flatten()(max_pool2)

max_pool2.shape

dense1 = Dense(32,activation='relu', kernel_initializer='he_uniform')(flatten)

batch_nor1= BatchNormalization()(dense1)

dense2 = Dense(64,activation='relu', kernel_initializer='he_uniform')(batch_nor1)

output_layer=Dense(10, activation='softmax')(dense2)


model = Model(inputs=input, outputs=output_layer)

model.predict()



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics =['accuracy'])

model.summary()

history = model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        steps_per_epoch=len(train),
        validation_steps=len(validate),
        verbose=1)

history.history

################################################################

#####################  Dumping the Model  ###########################################
from sklearn.externals import joblib
joblib.dump(model,'MNIST_CNN_12_june.pkl')

####################   Loading the Model   ###########################################

model = joblib.load('MNIST_CNN_12_june.pkl')

test_labels = pd.read_csv('D:\\PracticeWS\\Analytic Vidya\\MNIST\\Test_fCbTej3_0j1gHmj.csv')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_labels,
        directory='D:\\PracticeWS\\Analytic Vidya\\MNIST\\Train_UQcUa52\\Images\\test',
        x_col='filename',
        y_col=None,
        target_size=image_size,
        color_mode='grayscale',
        class_mode=None,
        batch_size=70,
        shuffle=False,
        seed=0
        )

y_pred = model.predict_generator(test_generator)


import numpy as np

y_pred = np.argmax(y_pred,axis=1)

submission_df = pd.DataFrame({
        'filename' : test_generator.filenames,
        'label' : y_pred
        })

submission_df.to_csv('submission.csv', index=False)

