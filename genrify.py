# Authors: Alexander Cox, Ernst-Richard Kausche, Ava Sato
import time
import tensorflow as tf

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

NUM_CLASSES = len(GENRES)

IMG_SIZE = (288, 432, 3)


def zfnet():
    """ Creates a ZFNet tensorflow model as a tensorflow Sequential

    :return: Tensorflow Sequential object representing the ZFNet CNN architecture
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(96, (7, 7), strides=(2, 2), activation='relu', input_shape=IMG_SIZE),
        tf.keras.layers.MaxPooling2D(3, strides=2),
        tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x)),

        tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(3, strides=2),
        tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x)),

        tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),

        tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),

        tf.keras.layers.MaxPooling2D(3, strides=2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(4096),

        tf.keras.layers.Dense(4096),

        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])


def gtzanet():
    """ Creates a GTZANet model as a tensorflow Sequential

    :return: Tensorflow sequential object representing the GTZANet CNN architecture
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), activation='relu', input_shape=IMG_SIZE),
        tf.keras.layers.MaxPooling2D(3, strides=2),

        tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(3, strides=2),

        tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(3, strides=2),

        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(3, strides=2),

        tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(3, strides=2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])


if __name__ == '__main__':
    print(f'Process Started at {time.strftime("%H:%M:%S", time.localtime())}')
    start_time = time.time()

    print("Generating Data Sets...")
    # Generate training set and validation set from images_original directory with 20% saved for validation
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        'Data/images_original', validation_split=0.2,
        subset="both",
        seed=12345,
        image_size=IMG_SIZE[:-1],
        batch_size=32
    )

    # Autotune datasets for efficient memory use
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Initialize model
    model = gtzanet()

    # Compile model with accuracy and top-5-categorical-accuracy metrics
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(5)]
    )

    # Initialize learning rate reduction callback with minimum of 0.00001
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001)

    # Train model on train_ds over 100 epochs
    model.fit(train_ds, batch_size=128, validation_data=val_ds, epochs=100, callbacks=[reduce_lr])

    print(f'Finished after {round(time.time() - start_time, 3)}s')
