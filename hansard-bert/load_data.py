import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def load_data(
        path: str = "data/hansard-speeches-v301.csv",
        # Set some additional parameters
        REQUIRED_SPEECH_LEN: int = 1000,  # use chars to keep it computationally simpler
        REQUIRED_NUM_SPEECHES: int = 1500,
        REQUIRED_COLS=['speech', 'party'],
        SPECIFIC_PARTIES: bool = True,  # lib dem, cons and labour only? - value outlined below
        MAIN_PARTIES=["Conservative", "Labour", "Liberal Democrat", "Labour (Co-op)"],
        GROUP_SIZE: int = 20000 # lib dem class has 29000 valid speeches, so don't exceed
):
    # read in our speeches
    speeches = pd.read_csv(path)

    # where a speech is less than our
    # specified len remove it
    speeches = speeches[
        speeches.speech.str.len() > REQUIRED_SPEECH_LEN
        ]

    # add some more filters and trim for our required fields
    speeches = speeches[
                       speeches["speech_class"].isin(['Speech'])
                       & speeches.party.notna()
                       ].loc[:, REQUIRED_COLS]

    # a = speeches[speeches.speech.str.len() > REQUIRED_SPEECH_LEN]
    # a.party.value_counts()
    # a.groupby(['party', 'speech_class']).count()

    # pull all parties less than our required number of speeches
    if SPECIFIC_PARTIES:
        speeches = speeches[
            speeches.party.isin(MAIN_PARTIES)
        ]
        speeches.party[speeches.party == "Labour (Co-op)"] = "Labour"
    else:
        min_speech_party = speeches["party"].value_counts()[lambda x: x > REQUIRED_NUM_SPEECHES].index.tolist()
        min_speech_party.append('Speaker')
        # rm any parties with less than our required num of speeches
        speeches = speeches[
            speeches.party.isin(min_speech_party)
        ]
    # pull out speech groups of our desired sizing
    speeches = speeches.groupby('party').sample(n=GROUP_SIZE, random_state=7)

    # return(train_test_split(speeches.speech, speeches.party, test_size=0.20, random_state=32))
    return (train_test_split(speeches.speech.values, speeches.party.values, test_size=0.20, random_state=32))


def encode_parties(y_train, y_test):
    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train, y_test], axis=0))
    encoded_y_train = encoder.transform(y_train)
    encoded_y_test = encoder.transform(y_test)
    np_utils.to_categorical(encoded_y_train)
    return np_utils.to_categorical(encoded_y_train), np_utils.to_categorical(encoded_y_test), encoder.classes_, encoder
