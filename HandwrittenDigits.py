import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
import random

(X_train, y_train),(X_test, y_test) = mnist.load_data()

plt.figure(figsize=(5,5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig('OriginalDigits.png',bbox_inches = 'tight')
    
model = Sequential(
    [
        Flatten(input_shape = (28,28)),
        Dense(25,activation = 'relu'),
        Dense(5,activation = 'sigmoid'),
        Dense(10,activation = 'softmax')
    ]
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs = 10,
    validation_data=(X_test,y_test)
)

model.summary()

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis = 1)

plt.figure(figsize=(5,5))
plt.subplots_adjust(top=2, right = 3)
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f'P:{y_pred[i]}| A:{y_test[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('PredictedDigits.png',bbox_inches = 'tight')
    
