import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from keras.utils import np_utils

import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization as tokenization
from official import nlp

# load custom functions
from load_data import load_data, encode_parties
from utils import gpu_checker, word_count, find_max_seq_len, tokenize_speech, save_model, load_model, evaluate_model, plot_history, predict_class
from bert_functions import encode_names, bert_encode

tf.get_logger().setLevel('ERROR')
gpu_checker() # check for GPU

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 3
epochs = 5
seed = 42
bert_max = 512 # max of this implementation
pls_save = True # since I've run this now, set to False
model_filename = "bert 1.0_v2"

# create train/test splits and encode our parties
x_train, x_test, Y_train, Y_test = load_data()
print(len(x_train)) # check this is correct...
y_train, y_test, decoder, encoder = encode_parties(pd.Series(Y_train), pd.Series(Y_test)) # fix this func pls

# load bert
# can we edit where bert is loaded from pls...
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2",
                            trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

# use bert_encode to encode our speeches
# max_seq_length = find_max_seq_len(tokenizer, pd.concat([x_train, x_test], axis = 0))
max_seq_length = bert_max
x_train = bert_encode(x_train, tokenizer, max_seq_length)
x_test = bert_encode(x_test, tokenizer, max_seq_length)

## Initial training...
num_class = len(decoder)  # Based on available class selection

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

output = tf.keras.layers.Dropout(rate=0.1)(pooled_output)

output = tf.keras.layers.Dense(num_class, activation='softmax', name='output')(output)

model = tf.keras.Model(
    inputs={
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
        },
        outputs=output)

# Set up epochs and steps
eval_batch_size = batch_size

train_data_size = len(y_train)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model layers and parameters...")
print(model.summary())


# save our model locally
if pls_save:
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        verbose=1)

    # evaluate final accuracy...
    evaluate_model(model, x_train, y_train, x_test, y_test)
    save_model(model_filename, os.getcwd(), model)

