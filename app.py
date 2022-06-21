import shutil

from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from flask_cors import CORS, cross_origin
from nl_cnn_1d import create_nl_cnn_model as create_1d_cnn_model
from nl_cnn_2d import create_nl_cnn_model as create_2d_cnn_model

import scipy.io as sio
import numpy as np
import keras
import tensorflow
from matplotlib import pyplot as plt
app = Flask(__name__)
CORS(app)

def get_dataset(dataset):
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    y_valid = None
    x_valid = None
    num_classes = 0
    if dataset == "Voice command":

        train_set = sio.loadmat("voice_command_dataset/vr_train_set_NRDT.mat")
        test_set = sio.loadmat("voice_command_dataset/vr_test_set_NRDT.mat")


        x_test = test_set["Samples"]
        y_test = test_set["Labels"]
        y_test = np.reshape(y_test, y_test.size)

        x_train = train_set["Samples"]
        y_train = train_set["Labels"]
        y_train = np.reshape(y_train, y_train.size)

        x_train = np.reshape(x_train, [np.shape(x_train)[0], np.shape(x_train)[1], np.shape(x_train)[2], 1])
        x_test = np.reshape(x_test, [np.shape(x_test)[0], np.shape(x_test)[1], np.shape(x_test)[2], 1])

        num_classes = np.max(y_train) + 1

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        y_test = y_test.astype('uint8')
        y_train = y_train.astype('uint8')

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)

    elif dataset == "EmoDB":
        train_set = sio.loadmat("emo_db_dataset/emo_dataset.mat")


        x_train = train_set["Samples"]
        y_train = train_set["Labels"]
        y_train = np.reshape(y_train, y_train.size)

        x_train = np.reshape(x_train, [np.shape(x_train)[0], np.shape(x_train)[1], np.shape(x_train)[2], 1])

        num_classes = np.max(y_train) + 1

        x_train = x_train.astype("float32")
        y_train = y_train.astype('uint8')

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_valid = keras.uitls.to_categorical(y_valid, num_classes)

    elif dataset == "Bonn-EEG":
        train_set = sio.loadmat("eeg_dataset/eeg_train_dataset.mat")
        test_set = sio.loadmat("eeg_dataset/eeg_test_dataset.mat")

        x_test = test_set["Samples"]
        y_test = test_set["Labels"]
        y_test = np.reshape(y_test, y_test.size)

        x_train = train_set["Samples"]
        y_train = train_set["Labels"]
        y_train = np.reshape(y_train, y_train.size)

        x_train = np.reshape(x_train, [np.shape(x_train)[0], np.shape(x_train)[2], np.shape(x_train)[1]])
        x_test = np.reshape(x_test, [np.shape(x_test)[0], np.shape(x_test)[2], 1])

        num_classes = np.max(y_train) + 1

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        y_test = y_test.astype('uint8')
        y_train = y_train.astype('uint8')

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)

    return x_train, y_train, x_test, y_test, x_valid, y_valid, num_classes


@cross_origin(origin='*')
@app.route('/create-and-train', methods=['POST'])
def create_model_and_train():  # put application's code here
    dataset = request.json.get("dataset")
    conv_dimension = request.json.get("conv_dimension")
    k = request.json.get("k")
    filtre = request.json.get("nr_filtre")
    nl0 = request.json.get("nonlinear_layer_0")
    nl1 = request.json.get("nonlinear_layer_1")
    dropout = request.json.get("dropout")
    learning_rate = request.json.get("learning_rate")
    epochs = request.json.get("epochs")
    x_train, y_train, x_test, y_test, x_valid, y_valid, num_classes = get_dataset(dataset)
    input_shape = np.shape(x_train)[1:4]

    if conv_dimension == "1D":
        model = create_1d_cnn_model(input_shape, num_classes, k=k,separ=0, flat=0, width=filtre, nl=(nl0,nl1),
                                    add_layer=0, learning_rate=learning_rate, dropout=dropout)
    else:
        model = create_2d_cnn_model(input_shape, num_classes, k=k, separ=0, flat=0, width=filtre, nl=(nl0,nl1),
                                    add_layer=0, learning_rate=learning_rate, dropout=dropout)

    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=5,
                        epochs=epochs,
                        verbose=1,  # aici 0 (nu afiseaza nimic) 1 (detaliat) 2(numai epocile)
                        validation_data=(x_valid, y_valid))
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    tensorflow.keras.utils.plot_model(model, "model.png", show_shapes=True, show_layer_names=True)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig("validation.png")

    shutil.copy("model.png", "static")
    shutil.copy("validation.png", "static")
    score = model.evaluate(x_test, y_test, verbose=1)
    return jsonify({
        "model": "/static/model.png",
        "validation": "/static/validation.png",
        "accuracy": score[1]
    })


if __name__ == '__main__':
    app.run()
