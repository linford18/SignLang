import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
# model reconstruction from JSON:
#from keras.models import model_from_json

batch_size = 128
num_classes = 25
epochs = 5

#784 pixels-28x28
signs = np.genfromtxt("sign_mnist_train.csv", delimiter=",", skip_header=1)
signs=signs.astype(int)
labels=signs[...,0]
y_train=labels
data=signs[...,1:]
x_train=np.array([np.reshape(data[i],(28,28)) for i in range(data.shape[0])])
#data.shape=(27445,784)


#784 pixels-28x28
test_signs = np.genfromtxt("sign_mnist_test.csv", delimiter=",", skip_header=1)
test_signs=test_signs.astype(int)
t_labels=test_signs[...,0]
y_test=t_labels
#labels.size=7172
t_data=test_signs[...,1:]
#data.shape=(7172,784)
x_test=np.array([np.reshape(t_data[i],(28,28)) for i in range(t_data.shape[0])])

# input image dimensions
img_x, img_y = 28, 28


# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save_weights('my_model_weights.h5',overwrite=True)
print('Model saved')

