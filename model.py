import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

def generator(train_images, test_images):
    root = "train" 
    train_images1 = os.listdir(root)
    root2 = "test"
    test_images2 = os.listdir(root2)
    len(train_images1) 

    train_datagen = ImageDataGenerator(
        rescale=1./255,  
        shear_range=0.2,  
        zoom_range=0.2,  
        horizontal_flip=True,  
        rotation_range=20,  
        width_shift_range=0.2,  
        height_shift_range=0.2,  
        fill_mode='nearest',  
        channel_shift_range=0.2, 
        validation_split=0.2 
    )                         

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(train_images,
                                                     target_size=(64,64),
                                                     color_mode='rgb',
                                                     shuffle=True,
                                                     subset='training',
                                                     seed=None,
                                                     class_mode='categorical',
                                                     classes=['label_0', 'label_1', 'label_2', 'label_3', 
                                                              'label_4', 'label_5', 'label_6', 'label_7']
                                                 )

    test_set = test_datagen.flow_from_directory(test_images,
                                                target_size=(64,64),
                                                color_mode='rgb',
                                                shuffle=False,
                                                seed=None,
                                                class_mode='categorical'
                                              )

    validation_set = train_datagen.flow_from_directory(train_images,
                                                       target_size=(64,64),
                                                       color_mode='rgb',
                                                       shuffle=True,
                                                       seed=None,
                                                       subset='validation',       
                                                       class_mode='categorical',
                                                       classes=['label_0', 'label_1', 'label_2', 'label_3', 
                                                                'label_4', 'label_5', 'label_6', 'label_7']
                                                   )
    return training_set, test_set, validation_set


def model(num_classes, learning_rate):
    np.random.seed(42)  # consistency for this model
    num_classes_training_set = num_classes
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes_training_set, activation='softmax'))
    opt = Adam(learning_rate=learning_rate)
    model.summary()
    plot_model(model, show_shapes=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    train_images_path = input("Path to training images: ")
    test_images_path = input("Path to test_images: ")

    num_classes = int(input("Enter number of classes: "))
    learning_rate = float(input("Learning rate: "))

    training_set, test_set, validation_set = generator(train_images_path, test_images_path)

    your_model = model(num_classes, learning_rate)
    history = your_model.fit(training_set, epochs=100, validation_data=validation_set)

if __name__ == "__main__":
    main()


