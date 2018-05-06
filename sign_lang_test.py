import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np
import PIL
from PIL import Image
import string
import cv2
import imutils
#resize
basewidth = 28
img = Image.open('american_sign_c.png').convert('L')
orig = img.copy()
img = img.resize((28, 28), PIL.Image.ANTIALIAS)
img.save('r2_grey2.png')
data=np.asarray(img)

num_classes=25

# input image dimensions
img_x, img_y = 28, 28

x_train=np.array(data)
x_train = x_train.reshape(1, img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

x_train = x_train.astype('float32')
x_train /= 255

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


model.load_weights('my_model_weights.h5')

#create a dict to predict classes
classes_dict=dict(zip(range(0,26),string.ascii_uppercase))


classes = model.predict_classes(x_train)

print(classes)
print('The sign is letter {}'.format(classes_dict[classes[0]]))
label = "{}".format(classes_dict[classes[0]])

output = data
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
cv2.imshow('output',output)
#press esc to close window
cv2.waitKey(0)
cv2.destroyAllWindows()
