#imports
import numpy as np
import tensorflow as tensor
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Dropout # Ai API
from tensorflow.keras.models import Sequential
import os

# Model Loader

from tensorflow.keras.models import load_model


#image cleaning
import cv2
import imghdr

# visualizer:
from matplotlib import pyplot as mplt


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy # recall mesurements for saving memory models

# Data directory path

class Classifier():
    def __init__(self):
        self.model = Sequential()
        happy = "HappyPeople"
        sad = "SadPeople"
        self.dir_file = 'data'
        self.img_files = ['jpeg','jpg','bmp','png'] # all standerdized file types



    def Parser(self):
        files = os.listdir(self.dir_file)
        files.remove('.DS_Store')

        for image_class in files:
            for image in os.listdir(os.path.join(self.dir_file,image_class)):
                image_path = os.path.join(self.dir_file,image_class,image)
                try:
                    img = cv2.imread(image_path)
                    tip = imghdr.what(image_path)
                    if tip not in self.img_files:
                        print("image not in lists {}".format(image_path))
                        os.remove(image_path)
                except Exception as e:
                    print('issues with image {}'.format(image_path))
                    os.remove(image_path)

        data = tensor.keras.utils.image_dataset_from_directory('data') # building the data pipeline

        data_iterator = data.as_numpy_iterator() # iterating through the pipeline
        batch = data_iterator.next()# accessing the pipeline
        data = data.map(lambda x,y: (x/255,y))
        data.as_numpy_iterator().next()

        return batch,data

    def imgRep(self,batch,data):
        fig, ax = mplt.subplots(ncols=4,figsize=(20,20))
        for idx, img in enumerate(batch[0][:4]):
            ax[idx].imshow(img.astype(int))
            ax[idx].title.set_text(batch[1][idx])
            mplt.show()


        train_size = int(len(data)*0.5)
        val_size = int(len(data)*0.3)
        test_size = int(len(data)*0.2)

        print(train_size+val_size+test_size)

        train = data.take(train_size)
        val = data.skip(train_size).take(val_size)
        test = data.skip(train_size+val_size).take(test_size)
        print(len(train))
        print(len(val))
        print(len(test))

        return train,val,test

    # Building the Deep Learning Model
    def Build(self):


        self.model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(32, (3,3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(16, (3,3), 1, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile('adam', loss=tensor.losses.BinaryCrossentropy(), metrics=['accuracy'])

        self.model.summary()

    # Training the Model

    def Train(self):
        logging_Directory = 'logs'
        tensorboard_callback = tensor.keras.callbacks.TensorBoard(log_dir=logging_Directory)

    # Evaluation
    def Evaluation(self,test):
        precision = Precision()
        recall = Recall()
        accuracy = BinaryAccuracy()

        for batch in test.as_numpy_iterator():
            X, y = batch
            yhat = self.model.predict(X)
            precision.update_state(y, yhat)
            recall.update_state(y, yhat)
            accuracy.update_state(y, yhat)


    def Save(self,model):
        model.save(os.path.join('models', 'imageclassifier.h5'))

    def Model_Execution(self,img):
        pred_img = cv2.imread(img)
        model_execution = load_model('imageclassifier.h5')
        resize = tensor.image.resize(pred_img, (256,256))
        return model_execution.predict(np.expand_dims(resize / 255, 0))

