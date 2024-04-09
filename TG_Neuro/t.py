# librarie
import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, UpSampling2D
from keras.optimizers import Adam
from keras import layers
import cv2
import numpy as np
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
def prep():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype('float32') / 255.
    test_images = test_images.astype('float32') / 255.

    train_images = train_images.reshape(-1, 32, 32, 3)
    test_images = test_images.reshape(-1, 32, 32, 3)
    # show dataset
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    plt.figure(figsize=[10, 10])
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])

    plt.show()

    # to gray and normalaization

    train_images_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_images])
    test_images_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_images])
    train_images_gray = train_images_gray.reshape(train_images_gray.shape[0], 32, 32, 1)
    test_images_gray = test_images_gray.reshape(test_images_gray.shape[0], 32, 32, 1)
    # show  gray dataset
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    plt.figure(figsize=[10, 10])
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images_gray[i], cmap='gray')
        plt.xlabel(class_names[train_labels[i][0]])
    return train_images, test_images, train_images_gray,test_images_gray

# make model
def getModel():
    model = Sequential()

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
    # summary

    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
#fit
def get_gray_photo(image):
    train_images, test_images, train_images_gray, test_images_gray = prep()
    model = keras.models.load_model('model/model_0.h5')
    color_img = model.predict(image)




if __name__ == '__main__':
    train_images, test_images, train_images_gray, test_images_gray = prep()
    model = getModel()
    model.fit(train_images_gray, train_images, epochs=1, batch_size=32, validation_data=(test_images_gray, test_images))
    #predict
    model.save('model/model_0.h5')
    color_img = model.predict(test_images_gray)


    # image from the CFAR10 dataset
    plt.figure(figsize=(10, 10))

    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.title('gray')
        plt.imshow(test_images_gray[i], cmap='gray')
    plt.show()
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.title('predict')
        plt.imshow(cv2.cvtColor(color_img[i], cv2.COLOR_BGR2RGB))
    plt.show()
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.title('color')
        plt.imshow(test_images[i])
    plt.show()

    # image from  out  of the CFAR10 dataset
    im_ho = cv2.imread('/content/hours.png')
    plt.imshow(im_ho)

    image_input_1 = cv2.resize(im_ho, (32, 32))
    image_input = cv2.cvtColor(image_input_1, cv2.COLOR_BGR2GRAY)
    img_g = image_input.astype('float32') / 255.
    plt.imshow(img_g, cmap='gray')
    plt.show()

    img_out = image_input_1.reshape(-1, 32, 32, 1)
    preds = model.predict(img_out)
    predicted_class = np.argmax(preds)
    plt.imshow(image_input_1)
    plt.show()

