import tensorflow as tf
import tensorflow_hub as hub
from functions.bert_functions import encode_names
import os
import matplotlib.pyplot as plt
import numpy as np

def gpu_checker():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

def word_count(string):
    return(len(string.split()))


def find_max_seq_len(tokenizer, x_val):

    text = tf.ragged.constant([
        encode_names(n, tokenizer) for n in x_val])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * text.shape[0]
    input_word_ids = tf.concat([cls, text], axis=-1)

    ## Find max len for our model
    lens = [len(i) for i in input_word_ids]
    # Set a max sequence length
    return(int(max(lens)))



def tokenize_speech(
        string: str,
        tokenizer):
    """
    :return: print our tokenized input string
    """
    tokenized_string = tokenizer.tokenize(string)
    for i in tokenized_string:
        print(i, tokenizer.convert_tokens_to_ids([i]))


def plot_history(history):
    plt.style.use('ggplot')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def save_model(filename, filepath, model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model_fname = filename
    my_wd = filepath

    model.save(os.path.join(my_wd, model_fname))
    print("Model saved")

def load_model(filename, filepath):
    model_fname = filename
    my_wd = filepath

    return(tf.keras.models.load_model(os.path.join(my_wd, model_fname)))

def evaluate_model(model, x_train, y_train, x_test, y_test):
    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

def predict_class(model, input, decoder):
    prediction = model.predict(input)
    print(prediction)
    print('Speech predicted to be: ', decoder[np.argmax(prediction)])