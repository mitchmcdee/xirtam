import os
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras.datasets import mnist
from PIL import Image
from skimage.io import imread
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
base_data_dir = './data/generated_output/world-5546682508642403194/robot--4209387126734636757/'
image_size = (100, 100)
input_shape = (*image_size, 1)
epochs = 2
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

# Build autoencoder
autoencoder = Sequential()

# Encoder
autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Decoder
autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
autoencoder.add(UpSampling2D(size=(2, 2)))
autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
autoencoder.add(UpSampling2D(size=(2, 2)))
autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
autoencoder.add(UpSampling2D(size=(2, 2)))
autoencoder.add(Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid'))

# Compile and fit
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))

# Visually evaluate results
decoded_imgs = autoencoder.predict(x_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(*image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(*image_size))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
