import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping

def training():
    # Dataset with MRI scans
    data_dir = 'dataset/'  # 4 folders inside: glioma, meningioma, notumor, pituitary

    # Image parameters
    img_height = 244
    img_width = 244
    batch_size = 16

    # Data settings
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Training settings
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )


    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Defining CNN model
    model = models.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')  # 4 diseases
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Training the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_data, epochs=10, validation_data=val_data, verbose=1, callbacks = [early_stopping])

    # Saving the model
    model.save('mri_disease_classifier.h5')

    # Saving class labels
    class_labels = list(train_data.class_indices.keys())
    print("Class Labels:", class_labels)
