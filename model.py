from matplotlib import pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input


# imports the mobilenet mode
#
# l and discards the last 1000 neuron layer.
base_model2 = tf.keras.applications.MobileNet(input_shape=(224, 224, 3), include_top=False,
                                              weights='imagenet')
model2 = Sequential()
model2.add(base_model2)
model2.add(GlobalAveragePooling2D())
model2.add(Dense(64, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Dense(2, activation='sigmoid'))
model2.summary()

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('dataset/train',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)
test_set = train_datagen.flow_from_directory('dataset/test',
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='categorical')


model2.compile(optimizer='Adam', loss='categorical_crossentropy',
               metrics=['accuracy'])

history = model2.fit_generator(generator=train_generator,

                               epochs=15,
                               verbose=1,
                               validation_data=test_set)

model2.save("alg/FinalModel.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'], 'r',
         label='Training accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('# Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("alg/FinalModel.png")

plt.show()
acc = history.history['accuracy'][-1]
print(acc)
