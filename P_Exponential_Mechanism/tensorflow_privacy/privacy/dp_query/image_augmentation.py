from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.analysis import cifar10_input



class CIFAR10:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        # reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 32, 32, 3)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 32, 32, 3)
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # rescale values
        self.x_train /= 255.0
        self.x_test /= 255.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        # self.y_train = to_categorical(self.y_train, 10)
        self.y_test = self.y_test.reshape(-1)

    def trian_data_augmentation(self):
        data_dir='/home/zfy/cifar-10-batches-py'
        images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=self.train_size)
        return images_train, labels_train

    def test_data_crop(self):
        data_dir='/home/zfy/cifar-10-batches-py'
        images_train, labels_train = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=self.test_size)
        return images_train, labels_train

    def data_augmentation(self, augment_size=10000):
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.05,
            width_shift_range=0.07,
            height_shift_range=0.07,
            horizontal_flip=True,
            vertical_flip=False,
            data_format="channels_last",
            zca_whitening=True)
        # fit data for zca whitening
        image_generator.fit(self.x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[randidx].copy()
        y_augmented = self.y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                           batch_size=augment_size, shuffle=False).next()[0]
        self.x_train[randidx]=x_augmented.copy()
        self.y_train[randidx]=y_augmented.copy()
        # append augmented data to trainset
        # self.x_train = np.concatenate((self.x_train, x_augmented))
        # self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        x_train=self.x_train
        y_train=self.y_train.reshape(-1)
        return x_train,y_train

    def next_train_batch(self, batch_size):
        randidx = np.random.randint(self.train_size, size=batch_size)
        epoch_x = self.x_train[randidx]
        epoch_y = self.y_train[randidx]
        return epoch_x, epoch_y

    def next_test_batch(self, batch_size):
        randidx = np.random.randint(self.test_size, size=batch_size)
        epoch_x = self.x_test[randidx]
        epoch_y = self.y_test[randidx]
        return epoch_x, epoch_y

    def shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]