import os
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from PIL import Image
from skimage.io import imread
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


# Constants
training = False
model_dir = './data/models/alexnet/'
base_data_dir = './data/out/robot--4209387126734636757/world-5546682508642403194/'
image_size = (128, 128)
input_shape = (*image_size, 1)
epochs = 5
batch_size = 256
test_split = 0.1

# Rangle data
x = []
y = []
for filename in tqdm(list(os.listdir(base_data_dir))):
    _, extension = os.path.splitext(filename)
    if extension != '.bmp':
        continue
    image = imread(base_data_dir + filename , as_gray=True)
    if filename == 'regions.bmp':
        y.append(image)
    else:
        x.append(image)
y *= len(x)
x = np.array(x)
y = np.array(y)
x = x.astype('float32') / 255
y = y.astype('float32') / 255
x = np.reshape(x, (len(x), *input_shape))
y = np.reshape(y, (len(y), *input_shape))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)

# Build fully convolutional network
fcn = Sequential()
fcn.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
fcn.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
fcn.add(MaxPooling2D((2, 2), strides=(2, 2)))
fcn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
fcn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
fcn.add(MaxPooling2D((2, 2), strides=(2, 2)))
fcn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
fcn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
fcn.add(UpSampling2D((2, 2)))
fcn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
fcn.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
fcn.add(UpSampling2D((2, 2)))
fcn.add(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))

if training:
    # Compile and fit
    fcn.compile(optimizer='adadelta', loss='binary_crossentropy')
    fcn.summary()
    fcn.fit(x_train, y_train,
            verbose=1,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, y_test))
    fcn.save(model_dir + "final_weights.hdf5")
else:
    fcn.load_weights(model_dir + "final_weights.hdf5")

# Visually evaluate results
predictions = fcn.predict(x_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(1, n + 1):
    # display original
    ax = plt.subplot(3, n, i + 0 * n)
    plt.imshow(x_test[i].reshape(*image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display truth
    ax = plt.subplot(3, n, i + 1 * n)
    plt.imshow(y_test[i].reshape(*image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display prediction
    ax = plt.subplot(3, n, i + 2 * n)
    plt.imshow(predictions[i].reshape(*image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
