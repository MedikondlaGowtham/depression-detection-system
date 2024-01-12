from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Activation
from keras.regularizers import l2


model = Sequential()
model.add(Conv2D(filters=32, padding="same", activation="relu",
          kernel_size=3, strides=2, input_shape=(224, 224, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=32, padding="same", activation="relu", kernel_size=3))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))


model.add(Dense(2, kernel_regularizer=l2(0.01), activation="softmax"))
model.summary()

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, rescale=1./255)  # included in our dependencies

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


model.compile(optimizer="adam", loss="squared_hinge", metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,

                              epochs=15,
                              verbose=1,
                              validation_data=test_set)

model.save("alg/svm.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'], 'r',
         label='Training accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('# Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("alg/svm.png")

plt.show()
acc = history.history['accuracy'][-1]
print(acc)
