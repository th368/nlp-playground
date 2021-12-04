import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def load_data(
    path: str = "data/pq.csv",
    REQUIRED_COLS = ["department", "question", "answer"],
    MAX_GROUP_SIZE: int = 10000 # set the maximum class size so our training doesn't take an eternity
):
    # load in our main dataset
    pq_answer = pd.read_csv(path)
   
    # current department list available at
    # https://www.gov.uk/government/organisations
    # import data
    current_depts = 'data/current_dept_list.txt'

    depts = []
    with open(current_depts) as infile:
        for line in infile.readlines():
            depts.append(line.strip())

    # find existing depts and change their current name
    existing_dept_lookup = pd.read_csv('data/existing_dept_lookup.csv')
    # strip whitespace...
    existing_dept_lookup[["old_dept", "new_dept"]] = existing_dept_lookup[["old_dept", "new_dept"]].apply(lambda x: x.str.strip())
    dept_lookup_dict = dict(existing_dept_lookup.values)
    pq_answer = pq_answer.replace({"department": dept_lookup_dict})

    # apply some filters to get our desired output dataset...
    depts_to_keep = pq_answer["department"].value_counts()[lambda x: x > 400].index.tolist()
    pq_answer = pq_answer[
        (pq_answer.withdrawn == 0) # remove any withdrawn questions
        & (pq_answer.department.isin(set(depts+depts_to_keep))) # find only activate depts
        & (pq_answer.question.str.len() > 0) # remove any questions with no content (if they exist...)
    ].loc[:, REQUIRED_COLS] # select only our desired columns
    
    # remove non-ascii characters from quesitons...
    pq_answer['question'] = pq_answer['question'].str.encode('ascii', 'ignore').str.decode('ascii')
    # return our final dataset in test/training splits
    return (train_test_split(pq_answer.question.values, pq_answer.department.values, test_size=0.20, random_state=32))

def encode_parties(y_train, y_test):
    encoder = LabelEncoder()
    encoder.fit(pd.concat([y_train, y_test], axis=0))
    encoded_y_train = encoder.transform(y_train)
    encoded_y_test = encoder.transform(y_test)
    np_utils.to_categorical(encoded_y_train)
    return np_utils.to_categorical(encoded_y_train), np_utils.to_categorical(encoded_y_test), encoder.classes_, encoder