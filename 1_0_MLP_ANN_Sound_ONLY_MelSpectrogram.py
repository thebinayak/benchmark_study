# Purpose: Multilayer Perceptrion implementation
# Author: Binayak Bhandari
# Date: 11 March, 2020
# Woosong University
'''from google.colab import files
uploaded = files.upload()'''

'''
In the previous code (we have preprocessed the data for Inputs and Outputs (lables or targets)), 
1. we have extracted the all the different genres and mapped them
2. We extracted the labels
3. We extracted the MSCCs and/or Mel_Spectrograms
'''

'''In this case we will
1. Load the dataset
2. Split the data into train and test sets
3. Using Tensorflow and Keras we're going to built the network architecture
4. Compile the network
5. Train the network
'''

import os
import json
import numpy as np
from  sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import Flatten
import pandas as pd
from keras.callbacks import CSVLogger # Save history of each epochs
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder

# ===================== AUDIO DATA====================
DATASET_PATH = "Mel_Spectrogram_for_10_40sec_Audio_Preprocessed_Comparison_Study.json"

# =============OTHER CONSTANTS =====================
random_states =  [154, 306, 743, 877, 558]
test_accuracies = []
history_loss = []
history_accuracy = []
history_val_loss = []
history_val_accuracy = []


# Change two places: line number 42 (for folder name ) and second, line XXX(for optimizer)
# ============ CONSTANTS used in the program =====================
save_dir = os.path.join(os.getcwd(), 'MLP_Audio_only_MelSpectrum')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_name = 'MelSpectrogram_model'
maxlen = 130  # (orig = 323)

# Load the data using function
def load_data(dataset_path): # The data is stored in the json file
    if dataset_path.endswith('.json'):
        with open(dataset_path, "r") as fp:
            data = json.load(fp)
        # Convert lists into numpy arrays because labels and MelSpectrogram are saved in json as list
        Sound_input = np.array(data["Mel_Spectrogram"])
        Sound_label = np.array(data["labels"])
        print("Data succesfully loaded!")

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        Sound_label = Sound_label.reshape(len(Sound_label), 1)
        Sound_label = onehot_encoder.fit_transform(Sound_label)
        print(Sound_label)
        print("Data successfully loaded")
        print(f'The shape of sound input is {Sound_input.shape} and shape of sound level is {Sound_label}')
        # The shape of sound input is (1600, 130, 13) and shape of sound level is [[1. 0. 0. 0.]
        return Sound_input, Sound_label
    elif dataset_path.endswith('.npy'):
        data = np.load(dataset_path)
        return data

# =================== FUNCTION for plotting ========================
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
    plt.tight_layout() # To solve the overlapping problem

    plt.show()

def prepare_datasets(Sound_data,  Sound_label, test_size, validation_size):
    # create train, validation and test split
    Sound_train, Sound_test,  Sound_label_train, Sound_label_test = train_test_split(Sound_data, Sound_label, test_size=test_size)

    Sound_test, Sound_validation,  Sound_label_test, Sound_label_validation = train_test_split(Sound_test, Sound_label_test, test_size=validation_size)

    return Sound_train, Sound_validation, Sound_test, Sound_label_train, Sound_label_validation, Sound_label_test


# ====================== EXECUTION START ===============================
if __name__=="__main__":
    for idx in range(len(random_states)):
      # Load data
      Sound_inputs, Sound_label = load_data(DATASET_PATH)

      # Spliting the data into train (50%) and test set (50%)
      Sound_train, Sound_validation, Sound_test, Sound_label_train, Sound_label_validation, Sound_label_test = prepare_datasets( Sound_inputs, Sound_label, test_size=0.5, validation_size=0.5)
      print(f'This is the shape of Force_train shape after split {Sound_train.shape}, {Sound_validation.shape}, {Sound_test.shape}, {Sound_label_train.shape}, {Sound_label_validation.shape}, {Sound_label_test.shape}')
      #inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3) # 30% of the data is reserved for test (or validation)

      # Building the architecture
      model = keras.Sequential([
                          keras.layers.Flatten(input_shape = (Sound_inputs.shape[1], Sound_inputs.shape[2])), # Input layer, which flattens multiple dimension array and flattens (interval and values of MFCCs (13)
                          keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)), # First hidden layer
                          keras.layers.Dropout(0.3),
                          keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)), # Second hidden layer
                          keras.layers.Dropout(0.3),
                          keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)), # Third hidden layer
                          keras.layers.Dropout(0.3),
                          #output Layer
                          keras.layers.Dense(4, activation="softmax")# 10 is for the 10 genre of music

      ])

      # Compile the network
      optimizer = keras.optimizers.Adam(learning_rate=0.0001)
      model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"]) # After one-hot encoded you have to use categorical_corssentropy instead of sparse_categorical_Entropy
      dot_img_file = os.path.join(save_dir, 'MLP_Sound_Only_MelSpectrogram.png')
      plot_model(model, to_file=dot_img_file, dpi=200)
      #plot_model(model, to_file='MLP_ANN_SOUND_FORCE.png')
      model.summary() # To see visually the architecture

      # Train the network
      earlystop = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=0, mode='auto')
      csv_logger = CSVLogger(os.path.join(save_dir, "model_history_log_" + str(idx) + ".csv"), append=True)
      history = model.fit(Sound_train, Sound_label_train, validation_data=(Sound_validation, Sound_label_validation),
                epochs=500,
                batch_size=32,
                callbacks=[earlystop, csv_logger],
                shuffle= True,
                verbose=2)
      # ================== Save model and weights ====================
      if not os.path.isdir(save_dir):
          os.makedirs(save_dir)
      model_path = os.path.join(save_dir, f'{model_name}_{idx + 1}.h5')
      model.save_weights(model_path)
      print('Saved trained model at %s ' % model_path)

      # ================ Evaluate on the model (Inference) ==============================
      #score, acc = model.evaluate([Sound_test, Force_test], Force_label_test, batch_size=8)
      #print('Test loss:', score)
      #print('Test accuracy:', acc)

      # =============== Save the history of each model as csv ============================
      hist_df = pd.DataFrame(history.history)
      hist_csv_file = os.path.join(save_dir, f'{model_name}_history_{idx + 1}.csv')
      with open(hist_csv_file, mode='w') as f:
          hist_df.to_csv(f)

plot_history(history)

# ================================ CONFUSION MATRIX============================
num_classes = 4
class_names = ['Fine', 'Smooth', 'Rough', 'Coarse']
#result = model.predict([Sound_test, Sound_label_test])
result = model.predict(Sound_test)
confusion_mtx = confusion_matrix(Sound_label_test.argmax(axis=-1), result.argmax(axis=-1))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
disp.plot()
plt.savefig(os.path.join(save_dir, 'Confusion_Matrix.png'), dpi=300)
plt.show()

print("Confusion matrix")
print(disp.confusion_matrix)