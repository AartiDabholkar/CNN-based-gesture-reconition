# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:36:33 2018

@author: mypc
"""

import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical 
from PIL import Image
from win32com.client import Dispatch
import matplotlib.pyplot as plt  
import math  
import cv2  


img_width, img_height = 224, 224  
   
top_modal_weights_paths = 'bottleneck_fc_model.h5'  
train_data_dirs = 'C:\\python\\sign_letters\\training_data'  
validation_data_dirs = 'C:\\python\\sign_letters\\testing_data'  
   
 # number of epochs to train top model  
epochs = 50  
 # batch size used by flow_from_directory and predict_generator  
batch_size = 16  

model_new = applications.VGG16(include_top=False, weights='imagenet') 

#for training data
datagen1 = ImageDataGenerator(rescale=1. / 255)  
   
generator1 = datagen1.flow_from_directory(  
     train_data_dirs,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_train_samples = len(generator1.filenames)  
num_classes = len(generator1.class_indices)  
   
predict_size_train_new = int(math.ceil(nb_train_samples / batch_size))  
   
bottleneck_features_train_new = model_new.predict_generator(  
     generator1, predict_size_train_new)  
   
np.save('bottleneck_features_train.npy', bottleneck_features_train_new)  

#for validation data
generator1 = datagen1.flow_from_directory(  
     validation_data_dirs,  
     target_size=(img_width, img_height),  
     batch_size=batch_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_validation_samples = len(generator1.filenames)  
   
predict_size_validation_new = int(math.ceil(nb_validation_samples / batch_size))  
   
bottleneck_features_validation_new = model_new.predict_generator(  
     generator1, predict_size_validation_new)  
   
np.save('bottleneck_features_validation.npy', bottleneck_features_validation_new)  
#In order to train the top model, we need the class labels for each of the training/validation samples.
# We use a data generator for that also. We also need to convert the labels to categorical vectors.
datagen_top = ImageDataGenerator(rescale=1./255)  
generator_top = datagen_top.flow_from_directory(  
         train_data_dirs,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode='categorical',  
         shuffle=False)  
   
nb_train_samples = len(generator_top.filenames)  
num_classes = len(generator_top.class_indices)  
   
 # load the bottleneck features saved earlier  
train_data = np.load('bottleneck_features_train.npy')  
   
 # get the class lebels for the training data, in the original order  
train_labels = generator_top.classes  
   
 # convert the training labels to categorical vectors  
train_labels = to_categorical(train_labels, num_classes=num_classes)  

#do the same for validation data as well
generator_top = datagen_top.flow_from_directory(  
         validation_data_dirs,  
         target_size=(img_width, img_height),  
         batch_size=batch_size,  
         class_mode=None,  
         shuffle=False)  
   
nb_validation_samples = len(generator_top.filenames)  
   
validation_data = np.load('bottleneck_features_validation.npy')  
   
validation_labels = generator_top.classes  
validation_labels = to_categorical(validation_labels, num_classes=num_classes)


#now create and train a model
model = Sequential()  
model.add(Flatten(input_shape=train_data.shape[1:]))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation='sigmoid'))  
   
model.compile(optimizer='rmsprop',  
              loss='categorical_crossentropy', metrics=['accuracy'])  
   
history = model.fit(train_data, train_labels,  
          epochs=epochs,  
          batch_size=batch_size,  
          validation_data=(validation_data, validation_labels))  
   
model.save_weights(top_modal_weights_paths)  
   
(eval_loss, eval_accuracy) = model.evaluate(  
     validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("[INFO] Loss: {}".format(eval_loss))  

#plot the training history
plt.figure(1)  
   
 # summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
# summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()  
#Now we're ready to train our model. We call the two functions in sequence,

#We first load and pre-process the image,
image_path = 'C:\\python\\train\\K\\008.jpg' 
img_test=Image.open(image_path)
plt.imshow(img_test),plt.title("Image selected for Testing ")
   
orig = cv2.imread(image_path)  

   
print("[INFO] loading and preprocessing image...")  
image = load_img(image_path, target_size=(224, 224))  
image = img_to_array(image)  
   
 # important! otherwise the predictions will be '0'  
image = image / 255  
   
image = np.expand_dims(image, axis=0)


# build the VGG16 network  
model = applications.VGG16(include_top=False, weights='imagenet')  
   
 # get the bottleneck prediction from the pre-trained VGG16 model  
bottleneck_prediction = model.predict(image)  
   
 # build top model  
model = Sequential()  
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation='sigmoid'))  
   
model.load_weights(top_modal_weights_paths)  
   
 # use the bottleneck prediction on the top model to get the final classification  
class_predicted = model.predict_classes(bottleneck_prediction)  

#Finally, we decode the prediction and show the result,
inID = class_predicted[0] 
class_dictionary = generator_top.class_indices  
inv_map = {v: k for k, v in class_dictionary.items()}  
label = inv_map[inID]  


speak = Dispatch("SAPI.SpVoice")
#speak.Voice = speak.GetVoices("gender=female")[1]; 


 # get the prediction label  
print("Image ID: {}, Label: {}".format(inID, label))  
   
 # display the predictions with the image  
cv2.putText(orig, "Predicted: {}".format(label), (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)  
speak.Speak("Predicted class is:"+label)
#cv2.imshow("Classification", orig)  
plt.imshow(orig),plt.title("Classification")
cv2.waitKey(0)  
cv2.destroyAllWindows()  
  
 
