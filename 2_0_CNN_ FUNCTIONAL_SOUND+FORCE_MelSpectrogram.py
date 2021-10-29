# Purpose: Convolution Neural Network + ANN
# Author: Binayak Bhandari
# Date: 17 March, 2020
# Woosong University
# Modified from the source code
# Source code : https://github.com/musikalkemist/DeepLearningForAudioWithPython/tree/master/16-%20How%20to%20implement%20a%20CNN%20for%20music%20genre%20classification/code

import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.utils import plot_model
from keras.layers import Flatten
import os
import pandas as pd
from keras.callbacks import CSVLogger # Save history of each epochs
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===================== AUDIO DATA====================
DATASET_PATH = "Mel_Spectrogram_for_10_40sec_Audio_Preprocessed_Comparison_Study.json"

# ================= FORCE DATA ======================
FORCE_DATA = 'Force_Data.npy'   # Shape is (1600, 130, 3)
FORCE_LABEL = 'Force_Label.npy' # Shape is (1600, 4)

# Change two places: line number 25 (for folder name ) and second, line 249(for optimizer)
# ============ CONSTANTS used in the program =====================
save_dir = os.path.join(os.getcwd(), 'CNN_Sound+Force_MelSpectrogram') #RAdam AdamW
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_name = 'MelSpectrogram_model'
maxlen = 130

# =============OTHER CONSTANTS =====================
random_states =  [154, 306, 743, 877, 558]
test_accuracies = []
history_loss = []
history_accuracy = []
history_val_loss = []
history_val_accuracy = []
# Define the function to load the Audio and Force data


def load_data(dataset_path):
    if dataset_path.endswith('.json'):
        with open(dataset_path, "r") as fp:
            data = json.load(fp)

        # Convert lists/dictionary into numpy array
        Sound_input = np.array(data["Mel_Spectrogram"])
        Sound_label = np.array(data["labels"])

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        Sound_label = Sound_label.reshape(len(Sound_label), 1)
        Sound_label = onehot_encoder.fit_transform(Sound_label)
        print(Sound_label)
        print("Data successfully loaded")
        print(f'The shape of sound input is {Sound_input.shape} and shape of sound level is {Sound_label}')
        return Sound_input, Sound_label
    elif dataset_path.endswith('.npy'):
        data = np.load(dataset_path)
        return data

# ================== FUNCTION for plotting===================


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.show()


def prepare_datasets_CNN_sound(Sound_data, Force_data, Sound_label, Force_label, test_size, validation_size):
    # create train, validation and test split
    Sound_train, Sound_test, Force_train, Force_test, Sound_label_train, Sound_label_test,\
        Force_label_train, Force_label_test= train_test_split(Sound_data,
                                                              Force_data,
                                                              Sound_label,
                                                              Force_label,
                                                              test_size=test_size)

    Sound_test, Sound_validation, Force_test, Force_validation, \
        Sound_label_test, Sound_label_validation, Force_label_test, Force_label_validation = train_test_split(Sound_test,
                           Force_test,
                           Sound_label_test,
                           Force_label_test,
                           test_size=validation_size)

    # add an axis to input sets
    Sound_train = Sound_train[..., np.newaxis] # Add new axis for channel as CNN expect 4D data (samples, width, breath, channel)
    Sound_validation = Sound_validation[..., np.newaxis]
    Sound_test = Sound_test[..., np.newaxis]

    return Sound_train, Sound_validation, Sound_test, Force_train, Force_validation, Force_test, \
           Sound_label_train, Sound_label_validation, Sound_label_test, Force_label_train, Force_label_validation, Force_label_test

def model_configure(Sound_data, Force_data):
    """Generates functional model with CNN sound model + MLP force model"""
    print(f"Shape of Sound_data is {Sound_data.shape}") # (800, 130, 132, 1)
    # SOUND
    Input_Layer_Sound = layers.Input(shape=((Sound_data.shape[1:]))) # First layer is shape

    # 1st conv layer
    First = keras.layers.Conv2D(32, (3, 3), activation='relu')(Input_Layer_Sound) # , input_shape=input_shape
    First = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(First)
    First = keras.layers.BatchNormalization()(First)

    # 2nd conv layer
    Second = keras.layers.Conv2D(32, (3, 3), activation='relu')(First)
    Second = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Second)
    Second = keras.layers.BatchNormalization()(Second)

    # 3rd conv layer
    Third = keras.layers.Conv2D(32, (2, 2), activation='relu')(Second)
    Third = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(Third)
    Third = keras.layers.BatchNormalization()(Third)
    Third = keras.layers.Dense(64, activation='relu')(Third)
    Third = keras.layers.Dropout(0.3)(Third)

    # flatten output and feed it into dense layer
    Sound = keras.layers.Flatten()(Third)

    #   FORCE
    # First layer is always shape
    Input_Layer_Force = layers.Input(shape=(Force_data.shape[1:]))
    print(f'Force Input shape is {Input_Layer_Force}')
    First = keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(Input_Layer_Force)  # First hidden layer
    Second = keras.layers.Dropout(0.3)(First)
    Third = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(Second)  # Second hidden layey
    Fourth = keras.layers.Dropout(0.3)(Third)
    Fifth = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(Fourth)  # Third hidden layer
    Fifth = keras.layers.Dropout(0.3)(Fifth)
    Force = Flatten()(Fifth)

    # CONCATENATE SOUND & FORCE
    merged = keras.layers.concatenate([Sound, Force])

    second_2_final = keras.layers.Dense(512, activation='relu')(merged)
    final_layer = keras.layers.Dropout(0.3)(second_2_final)

    outputs = keras.layers.Dense(4, activation="softmax")(final_layer)
    # Define the model by providing inputs and outputs
    model = keras.Model(inputs=[Input_Layer_Sound, Input_Layer_Force], outputs=outputs)

    # Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])  # sparse_
    dot_img_file = os.path.join(save_dir, 'CNN_Sound+Force_MelSpectrogram.png')
    plot_model(model, to_file=dot_img_file, dpi=200)
    return model


def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # Insert new axis at the benning to make array shape (1, 130, 132, 1)

    # perform prediction
    prediction = model.predict(X, verbose=2)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))

# =========================EXECUTION START ===========================
if __name__ == "__main__":
    for idx in range(len(random_states)):

        # Load Audio data
        Sound_inputs, Sound_label = load_data(DATASET_PATH)
        # Load Force data
        Force_input = load_data(FORCE_DATA)
        Force_label = load_data(FORCE_LABEL)

        # Spliting the data into train (50%) and test set (50%)
        Sound_train, Sound_validation, Sound_test, Force_train, Force_validation, Force_test, \
        Sound_label_train, Sound_label_validation, Sound_label_test, Force_label_train, Force_label_validation, Force_label_test = prepare_datasets_CNN_sound(Sound_inputs,
                                                                                                    Force_input,
                                                                                                    Sound_label,
                                                                                                    Force_label,
                                                                                                    test_size=0.5,
                                                                                                    validation_size = 0.5)
        # create network
        #input_shape = (X_train.shape[1], X_train.shape[2], 1)

        # Train the network
        model = model_configure(Sound_train, Force_train)
        model.summary()
        earlystop = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=0, mode='auto')
        csv_logger = CSVLogger(os.path.join(save_dir, f'{model_name}_history_log_{str(idx)}.csv'), append=True)
        history = model.fit([Sound_train, Force_train], Force_label_train,
                            validation_data=([Sound_validation, Force_validation], Force_label_validation),
                            epochs=500,
                            batch_size=32,
                            callbacks=[earlystop, csv_logger],

                            verbose = 2) #shuffle=True,

        # ================== Save model and weights ====================
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, f'{model_name}_{idx + 1}.h5')
        model.save_weights(model_path)
        print('Saved trained model at %s ' % model_path)

        # ================ Evaluate on the model (Inference) ==============================
        score, acc = model.evaluate([Sound_test, Force_test], Force_label_test, batch_size=8, verbose=2)  # Sound_label_test
        print('Test loss:', score)
        print('Test accuracy:', acc)

        # =============== Save the history of each model as csv ============================
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = os.path.join(save_dir, f'{model_name}_history_{idx + 1}.csv')
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

# plot accuracy/error for training and validation
plot_history(history)

# ================================ CONFUSION MATRIX============================
num_classes = 4
class_names = ['Fine', 'Smooth', 'Rough', 'Coarse']
#result = model.predict([Sound_test, Sound_label_test])
result = model.predict([Sound_test, Force_test])
confusion_mtx = confusion_matrix(Sound_label_test.argmax(axis=-1), result.argmax(axis=-1))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
disp.plot()
plt.savefig(os.path.join(save_dir, 'Confusion_Matrix_CNN_Mel_Spectrogram.png'), dpi=300)
plt.show()

print("Confusion matrix")
print(disp.confusion_matrix)
