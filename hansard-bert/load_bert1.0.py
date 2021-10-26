import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
from keras.utils import np_utils
import official.nlp.bert.tokenization as tokenization
from official.nlp import bert
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# load custom functions
from load_data import load_data, encode_parties
from utils import load_model, evaluate_model, plot_history, predict_class
from bert_functions import encode_names, bert_encode

model_to_load = "bert 1.0_v2"

# load in our model
# create train/test splits and encode our parties
x_train, x_test, Y_train, Y_test = load_data()
y_train, y_test, decoder, encoder = encode_parties(pd.Series(Y_train), pd.Series(Y_test)) # fix this func pls

bert_v1 = load_model(model_to_load, os.getcwd())
# manually create our decoder as I forgot to save it
tokenizerSaved = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(os.getcwd(), model_to_load, 'assets/vocab.txt'),
    do_lower_case=False)

### Accuracy checks!
# check confusion matrices for each class
test_results = bert_v1.predict(bert_encode(x_test, tokenizerSaved, 512))
predicted_categories = tf.argmax(test_results, axis=1)
true_categories = tf.concat(encoder.transform(Y_test), axis=0)
# clean up our predicted and true categories for printing in the matrix...

conf_matrix = confusion_matrix(y_pred = predicted_categories, y_true = true_categories, labels = [0,1,2]) # labels = ['Conservative', 'Labour', 'Liberal Democrat']) # produce our confusion matrix...
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                             display_labels=decoder)
disp.plot() # display our plot graphically
# print report...
print(classification_report(y_pred = predicted_categories, y_true = true_categories, target_names=decoder))

# word version of our confusion matrix
classes, counts = np.unique(np.array(true_categories), return_counts = True)
for i in range(len(decoder)):
    not_i = classes.tolist()
    not_i.pop(i)
    print("Party: ", decoder[i])

    tp = conf_matrix[i,i]
    fn = conf_matrix[i, not_i]
    print("Correctly guessed: ", tp/(sum(fn)+tp), " %")
    print("Total in class: ", str(counts[i]))

    print("True positive: ", str(tp))
    print("True negative: ", str(sum(sum(conf_matrix[not_i, :]))))


    print("False negative: ", sum(fn))
    print("False positive:", sum(conf_matrix[not_i, i]))
    print("") # add whitespace





### Playing around with our model
bojo_speech = ["the first time we have met since you defied the sceptics by winning councils and communities that Conservatives have never won in before – such as Hartlepool " \
       "in fact it’s the first time since the general election of 2019 when we finally sent the corduroyed communist cosmonaut into orbit where he belongs " \
       "and why are we back today for a traditional Tory cheek by jowler? It is because for months we have had one of the most open economies and societies " \
       "and on July 19 we decided to open every single theatre and every concert hall and night club in England and we knew that some people would still be anxious" \
       "so we sent top government representatives to our sweatiest boites de nuit to show that anyone could dance " \
       "perfectly safely and wasn’t he brilliant my friends? let’s hear it for Jon Bon Govi living proof that we, you all represent the most jiving hip happening and" \
       " generally funkapolitan party in the world and how have we managed to open up ahead of so many of our friends? You know the answer, its " \
       "because of the roll-out of that vaccine a UK phenomenon the magic potion invented in oxford university and bottled in wales " \
       "distributed at incredible speed to vaccination centres everywhere I saw the army in action in Glasgow " \
       "firing staple guns like carbines as they set up a huge vaccination centre and in Fermanagh I saw the needles go in like a collective sewing machine" \
       "and they vaccinated so rapidly that we were able to"]


bojo_input = bert_encode(string_list=list(bojo_speech),
                     tokenizer=tokenizerSaved,
                     max_seq_length=512)

predict_class(bert_v1, bojo_input, decoder) # interestingly, it's extremely confident that this is a Labour speech


eton_speech = ["Eton college is the finest in all the land"]


eton_input = bert_encode(string_list=list(eton_speech),
                     tokenizer=tokenizerSaved,
                     max_seq_length=512)

predict_class(bert_v1, eton_input, decoder) # interestingly, it's extremely confident that this is also a Labour speech

david_c_speech = ["Conservative policies, policies you campaigned on, policies we are delivering. Two hundred new academies. Ten thousand university places. Fifty thousand apprenticeships." \
                 "Corporation tax – cut. The jobs tax – axed. Police targets – smashed. Immigration – capped. The third runway – stopped. Home Information Packs – dropped. Fat cat salaries – revealed. ID Cards – abolished. The NHS – protected. Our aid promise – kept." \
                 "Quangos – closing down. Ministers' pay – coming down. A bank levy – coming up. A cancer drugs fund – up and running. £6bn of spending saved this year. An emergency budget to balance the books in five years. An EU referendum lock to protect our sovereign powers every year." \
                 "For our pensioners – the earnings link restored. For our new entrepreneurs – employees' tax reduced. And for our brave armed forces – the operational allowance doubled. " \
                 "Look what we've done in five months. Just imagine what we can do in five years. " \
                 "In five years time, our combat troops will have left Afghanistan. This party has steadfastly supported our mission there, and so will this government." \
                 "But that does not mean simply accepting what went before. In our first few weeks in office, we set a clear new direction. Focused. Hard-headed. Time-limited."]

david_c_input = bert_encode(string_list=list(david_c_speech),
                     tokenizer=tokenizerSaved,
                     max_seq_length=512)

predict_class(bert_v1, david_c_input, decoder)
