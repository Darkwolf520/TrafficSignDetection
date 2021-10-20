import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

class ModelHandler:
    def __init__(self, modelName="MobileNetV2.h5", createNewModel=False):
        fullName = modelName
        self.model = ""
        self.shape = ""
        if not createNewModel:
            if modelName.find(".h5") == -1:
                fullName += ".h5"
            self.model = keras.models.load_model("Models/"+fullName)
            self.shape = self.model.input.shape[1]
            img = np.zeros((self.shape, self.shape, 3), dtype=int)
            #img = Image.fromarray(img)
            #img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            #az első predict lassú, emiatt a videó ne álljon le, mert nincsenek még betöltve az erőforrások
            self.model.predict(img)
        self.classes = {0: 'Speed limit (20km/h)',
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
                   42: 'End no passing veh > 3.5 tons',
                   43: 'Noise'}

    def predict(self, image):
        h, w, c = image.shape
        if h == 0 or w == 0:
            print("SIZE PROBLEM: height: {0}, width: {1}".format(h, w))
            return self.get_noise_class()
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        result_class = self.classes.get(id)
        return result_class

    def get_noise_class(self):
        result = self.classes.get(len(self.classes)-1)
        return result



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


    def train_existing_model(self, modelName):
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

    def train_model(self, modelName):

        """
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        """
        filePath = "Models/" + modelName
        if filePath.find(".h5") == -1:
            filePath += ".h5"



        train_gen = ImageDataGenerator(width_shift_range=[-10, 10],
                                       height_shift_range=0.2,
                                       horizontal_flip=True,
                                       rotation_range=20,
                                       brightness_range=[0.3,1.0],
                                       zoom_range=[0.5,1.0],
                                       validation_split=0.2)

        train_gen = ImageDataGenerator(validation_split=0.2)

        train_data = train_gen.flow_from_directory("D:/GTSRB/Final_Training/Images",
                                                   target_size=(224, 224),
                                                   batch_size=64,
                                                   color_mode='rgb',
                                                   subset="training")

        valid_data = train_gen.flow_from_directory("D:/GTSRB/Final_Training/Images",
                                                   target_size=(224, 224),
                                                   batch_size=16,
                                                   color_mode='rgb',
                                                   subset="validation")

        model = keras.models.load_model(filePath)

        model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

        model.summary()
        tensorboard = keras.callbacks.TensorBoard(log_dir="Models/logs/" + modelName, histogram_freq=1)
        earlyStopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0.0001, patience=15, mode="max")
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filePath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1)
        with tf.device('/gpu:0'):
            model.fit(train_data, validation_data=valid_data, epochs=200, callbacks=[tensorboard, earlyStopping, model_checkpoint_callback])

        #keras.models.save_model(filepath=filePath, model=model)

    def create_model(self, modelName):
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(400, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(44, activation='softmax'))
        model.summary()
        keras.models.save_model(filepath="Models/"+ modelName + ".h5", model=model)

    def create_model2(self, modelName):
        model = Sequential()
        model.add(Convolution2D(64, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
        model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(44, activation='softmax'))
        model.summary()
        keras.models.save_model(filepath="Models/"+ modelName + ".h5", model=model)

    def create_model3(self, modelName ):
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(44, activation='softmax'))
        model.summary()
        keras.models.save_model(filepath="Models/"+ modelName + ".h5", model=model)

    def create_model4(self, modelName):
        model = MobileNetV2()
        model.summary()
        for l in model.layers[:113]:
            l.trainable = False
        print(model.layers[112].trainable)
        print(model.layers[113].trainable)

        tmp = model.layers[-2]
        last = Dense(44, activation='softmax')(tmp.output)
        model = Model(inputs=model.input, outputs=last)
        model.summary()
        keras.models.save_model(filepath="Models/" + modelName + ".h5", model=model)

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