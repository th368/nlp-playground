import tensorflow as tf
import numpy as np

def encode_names(n, tokenizer):
    """
    Used exclusively in the tensorboard implementation of BERT
    for huggingface variations, use tokenize_hf
    :param n: string input to tokenize
    :param tokenizer: tokenizer to use
    :return: tokenized sentence. encode_names(string_to_test, tokenizer) for e.g.
    """

    tokens = list(tokenizer.tokenize(n))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(string_list, tokenizer, max_seq_length):
    num_examples = len(string_list)

    # tokenize our string inputs
    string_tokens = tf.ragged.constant([
        encode_names(n, tokenizer)[0:max_seq_length] for n in np.array(string_list)])
    # create N [CLS] tokens and merge onto our speech tokens
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * string_tokens.shape[0]
    input_word_ids = tf.concat([cls, string_tokens], axis=-1)

    # makes all values in our seq 1 (and then pads with 0s)
    # mask layer isn't used, so these are just dummies
    input_mask = tf.ones_like(input_word_ids).to_tensor(shape=(None, max_seq_length))


    type_cls = tf.zeros_like(cls)
    type_tokens = tf.ones_like(string_tokens)
    input_type_ids = tf.concat(
        [type_cls, type_tokens], axis=-1).to_tensor(shape=(None, max_seq_length))

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(shape=(None, max_seq_length)),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs

