import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class ModelHandler:
    def __init__(self):
        self.model = keras.models.load_model("Models/second.h5")

    def predict(self, image):
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = Image.fromarray(image)
        my_image = img_to_array(image)

        my_image = img_to_array(my_image)
        my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        # my_image = preprocess_input(my_image)

        # model.summary()
        pred = self.model.predict(my_image)
        class_name = self.get_class_by_id(np.argmax(pred))
        return class_name

    def get_class_by_id(self, id):
        classes = {0: 'Speed limit (20km/h)',
                   1: 'Speed limit (30km/h)',
                   2: 'Speed limit (50km/h)',
                   3: 'Speed limit (60km/h)',
                   4: 'Speed limit (70km/h)',
                   5: 'Speed limit (80km/h)',
                   6: 'End of speed limit (80km/h)',
                   7: 'Speed limit (100km/h)',
                   8: 'Speed limit (120km/h)',
                   9: 'No passing',
                   10: 'No passing veh over 3.5 tons',
                   11: 'Right-of-way at intersection',
                   12: 'Priority road',
                   13: 'Yield',
                   14: 'Stop',
                   15: 'No vehicles',
                   16: 'Veh > 3.5 tons prohibited',
                   17: 'No entry',
                   18: 'General caution',
                   19: 'Dangerous curve left',
                   20: 'Dangerous curve right',
                   21: 'Double curve',
                   22: 'Bumpy road',
                   23: 'Slippery road',
                   24: 'Road narrows on the right',
                   25: 'Road work',
                   26: 'Traffic signals',
                   27: 'Pedestrians',
                   28: 'Children crossing',
                   29: 'Bicycles crossing',
                   30: 'Beware of ice/snow',
                   31: 'Wild animals crossing',
                   32: 'End speed + passing limits',
                   33: 'Turn right ahead',
                   34: 'Turn left ahead',
                   35: 'Ahead only',
                   36: 'Go straight or right',
                   37: 'Go straight or left',
                   38: 'Keep right',
                   39: 'Keep left',
                   40: 'Roundabout mandatory',
                   41: 'End of no passing',
                   42: 'End no passing veh > 3.5 tons'}
        result_class = classes.get(id)
        return result_class

    def create_new_model(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        train_gen = ImageDataGenerator()
        train_data = train_gen.flow_from_directory("D:/GTSRB/Final_Training/Images",
                                                   target_size=(32, 32),
                                                   batch_size=128,
                                                   color_mode='grayscale')

        model = Sequential()
        model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(43, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        tensorboard = keras.callbacks.TensorBoard(log_dir="Models/logs/second", histogram_freq=1)
        model.fit(train_data, epochs=20, callbacks=[tensorboard])

        keras.models.save_model(filepath="Models/second.h5", model=model)


    def train_existing_model(self):
        model = keras.models.load_model("Models/first.h5")
        opt = keras.optimizers.Adam(learning_rate=0.1)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        train_gen = ImageDataGenerator()
        train_data = train_gen.flow_from_directory("D:/GTSRB/Final_Training/Images",
                                                   target_size=(32, 32),
                                                   batch_size=128,
                                                   color_mode='grayscale')

        model.fit(train_data, epochs=20)
        keras.models.save_model(filepath="Models/second.h5", model=model)

        model.fit()
    def test_model(self):

        model = keras.models.load_model("Models/second.h5")
        model.summary()
        counter = 0
        wrong_classes = []
        my_image = load_img("Test/model/STOP_sign.jpg", target_size=(32, 32), color_mode='grayscale')

        my_image = img_to_array(my_image)
        my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        #my_image = preprocess_input(my_image)


        #model.summary()
        pred = model.predict(my_image)
        print(np.argmax(pred))



"""
    def test_model(self):

        model = keras.models.load_model("Models/second.h5")
        model.summary()
        counter = 0
        wrong_classes = []
        for i in range(0, 43):
            file = "Test/model/"
            if i < 10:
                file += "0{0}.ppm".format(i)
            else:
                file += "{0}.ppm".format(i)
            my_image = load_img(file, target_size=(32, 32), color_mode='grayscale')

            my_image = img_to_array(my_image)
            my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
            # my_image = preprocess_input(my_image)

            # model.summary()
            pred = model.predict(my_image)

            print("Expected: {0}, got: {1}".format(i, np.argmax(pred)))
            if i != np.argmax(pred):
                counter += 1
                wrong_classes.append(i)
        print("Error: {0}%".format(round((counter / 43) * 100), 2))
        print("wrong classes: {0}".format(wrong_classes))
"""