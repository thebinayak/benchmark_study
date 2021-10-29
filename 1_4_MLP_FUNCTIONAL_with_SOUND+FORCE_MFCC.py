# Purpose: Multilayer Perceptrion implementation
# Author: Binayak Bhandari
# Date: 11 March, 2020
# Woosong University

""" In this case we will
1. Load the dataset
2. Split the data into train and test sets
3. Using Tensorflow and Keras we're going to built the network architecture
4. Compile the network
5. Train the network
"""

import json
import numpy as np
from  sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.utils import plot_model # pip install pydot & sudo apt-get install graphviz
from keras.layers import Flatten
import os
import pandas as pd
from keras.callbacks import CSVLogger # Save history of each epochs
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder

# ================== AUDIO DATA ===========================
DATASET_PATH = "MFCC_for_10_40sec_Audio_Preprocessed_Comparison_Study.json"

# ================== FORCE DATA ===========================
FORCE_DATA = 'Force_Data.npy'   # Shape is (1600, 130, 3)
FORCE_LABEL = 'Force_Label.npy' # Shape is (1600, 4)

# =============OTHER CONSTANTS =====================
random_states =  [154, 306, 743, 877, 558]
test_accuracies = []
history_loss = []
history_accuracy = []
history_val_loss = []
history_val_accuracy = []

# Change two places: line number 25 (for folder name ) and second, line 249(for optimizer)
# ============ CONSTANTS used in the program =====================
save_dir = os.path.join(os.getcwd(), 'MLP_SOUND+FORCE_MFCC') #RAdam AdamW
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_name = 'MFCC_model'
maxlen = 130


# Load the data using function


def load_data(dataset_path): # The data is stored in the json file
    if dataset_path.endswith('.json'):
        with open(dataset_path, "r") as fp:
            data = json.load(fp)
        # Convert lists into numpy arrays because labels and MelSpectrogram are saved in json as list
        Sound_input = np.array(data["mfcc"])
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
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    plt.show()

def prepare_datasets(Sound_data, Force_data, Sound_label, Force_label, test_size, validation_size):
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

    ''''# add an axis to input sets
    Sound_train = Sound_train[..., np.newaxis]
    Sound_validation = Sound_validation[..., np.newaxis]
    Sound_test = Sound_test[..., np.newaxis]'''

    return Sound_train, Sound_validation, Sound_test, Force_train, Force_validation, Force_test, \
           Sound_label_train, Sound_label_validation, Sound_label_test, Force_label_train, Force_label_validation, Force_label_test



"""========= Building the MODEL using Functional API ============="""


def model_configure(Sound_data, Force_data):
    #  SOUND
    # First layer is always shape
    Input_Layer_Sound = layers.Input(shape=((Sound_train.shape[1:]))) # Flattens n dimension array (1600,130,13) we want last two
    print(f'Sound Input shape is {Input_Layer_Sound}')
    First     = keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(Input_Layer_Sound) #First hidden layer
    Second    = keras.layers.Dropout(0.3)(First)
    Third     = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(Second) # Second hidden layey
    Fourth    = keras.layers.Dropout(0.3)(Third)
    Fifth     = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(Fourth)# Third hidden layer
    Sound     = keras.layers.Dropout(0.3)(Fifth)
    Sound   = Flatten()(Sound)

    #   FORCE
    # First layer is always shape
    Input_Layer_Force = layers.Input(shape=(Force_train.shape[1:]))
    print(f'Force Input shape is {Input_Layer_Force}')
    FFirst = keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(Input_Layer_Force)  # First hidden layer
    FSecond = keras.layers.Dropout(0.3)(FFirst)
    FThird = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(FSecond)  # Second hidden layey
    FFourth = keras.layers.Dropout(0.3)(FThird)
    FFifth = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(FFourth)  # Third hidden layer
    Force = keras.layers.Dropout(0.3)(FFifth)
    Force = Flatten()(Force)

    # CONCATINATE SOUND & FORCE
    merged = keras.layers.concatenate([Sound, Force])

    second_2_final = keras.layers.Dense(512, activation='relu')(merged)
    final_layer = keras.layers.Dropout(0.3)(second_2_final)

    outputs = keras.layers.Dense(4, activation="softmax")(final_layer)
    # Define the model by providing inputs and outputs
    model = keras.Model(inputs=[Input_Layer_Sound, Input_Layer_Force], outputs=outputs)

    # Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"]) #sparse_
    #model.summary() # To see visually the architecture
    plot_model(model, to_file='multilayer_perceptron_graph.png')
    dot_img_file = os.path.join(save_dir, 'MLP_Sound+Force_MFCC.png')
    plot_model(model, to_file=dot_img_file, dpi=200)
    model.summary()  # To see visually the architecture
    return model


""" ====================== EXECUTION START =============================== """
if __name__=="__main__":
    for idx in range(len(random_states)):
        # Load Audio data
        Sound_inputs, Sound_label = load_data(DATASET_PATH)

        # Load Force data
        Force_input = load_data(FORCE_DATA)
        Force_label = load_data(FORCE_LABEL)

        Sound_train, Sound_validation, Sound_test, Force_train, Force_validation, Force_test, \
        Sound_label_train, Sound_label_validation, Sound_label_test, Force_label_train, \
        Force_label_validation, Force_label_test = prepare_datasets(Sound_inputs,
                                                                    Force_input,
                                                                    Sound_label,
                                                                    Force_label,
                                                                    test_size=0.5, validation_size=0.5)

        print(f'Sound_label_train shape is {Sound_label_train.shape} and Force_label_train shape is {Force_label_train.shape}')
        print(f'Sound_train shape is {Sound_train.shape} and Force_train shape is {Force_train.shape}')


        # Train the network
        model = model_configure(Sound_train, Force_train)
        model.summary()
        earlystop = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=0, mode='auto')
        csv_logger = CSVLogger(os.path.join(save_dir, "model_history_log_" + str(idx) + ".csv"), append=True)
        history = model.fit([Sound_train, Force_train], Force_label_train,
                            validation_data=([Sound_validation, Force_validation], Force_label_validation),
                            epochs=500,
                            batch_size=32,
                            callbacks=[earlystop, csv_logger],
                            shuffle=True,
                            verbose=2)

        # ================== Save model and weights ====================
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, f'{model_name}_{idx + 1}.h5')
        model.save_weights(model_path)
        print('Saved trained model at %s ' % model_path)

        # ================ Evaluate on the model (Inference) ==============================
        #score, acc = model.evaluate([Sound_test, Force_test], Force_label_test, batch_size=8)  # Sound_label_test
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
result = model.predict([Sound_test, Force_test])
confusion_mtx = confusion_matrix(Sound_label_test.argmax(axis=-1), result.argmax(axis=-1))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
disp.plot()
plt.savefig(os.path.join(save_dir, 'Confusion_Matrix_MLP_S+F_MFCC.png'), dpi=300)
plt.show()

print("Confusion matrix")
print(disp.confusion_matrix)
