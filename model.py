import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import imread
import cv2
import shutil



def image_processing(train):
    root = train
    frames = os.listdir(root)
    images = []
    for frame in frames:
        image = imread(os.path.join(root, frame))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        med_filt = cv2.medianBlur(gray, 5)
        guss_filt = cv2.GaussianBlur(gray,(9,9), 0)
        histo = cv2.equalizeHist(gray)
        edge = cv2.Canny(gray,100,200)
        images.append(gray)
        images.append(med_filt)
        images.append(guss_filt)
        images.append(histo)
        images.append(edge)
    
    images_array = np.array(images)

    return images_array

def create_folder(txt_file, output_folder):
    with open(txt_file) as f:
        lines = f.readlines()
        file_paths = []
        labels = []
    for line in lines:
        line = line.strip().split()
        file_paths.append(line[0])
        labels.append(int(line[1]))
    y = np.array(labels)
    
    for label in range(8):
        folder_name = f'label_{label}'
        os.makedirs(os.path.join(output_folder, folder_name), exist_ok=True)

    for idx in range(len(file_paths)):
        label = y[idx]
        source_path = file_paths[idx]
        destination_folder = os.path.join(output_folder, f'label_{label}')
        destination_path = os.path.join(destination_folder, os.path.basename(source_path))
        shutil.copy(source_path, destination_path)

    # Return the path to the main folder 'train'
    return output_folder


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
    train_images_path = input("Path to training images")
    test_images_path = input("Path to test_images: ")
    output_folder = "processed_images"

    num_classes = int(input("Enter number of classes: "))
    learning_rate = float(input("Learning rate: "))

    train_images_path = create_folder(train_images_path, output_folder)

    processed_images = image_processing(train_images_path)

    training_set, test_set, validation_set = generator(output_folder, test_images_path)

    num_classes_training_set = num_classes
    your_model = model(num_classes_training_set, learning_rate)
    history = your_model.fit(training_set, epochs=100, validation_data=validation_set)

if __name__ == "__main__":
    main()

