# !/usr/bin/env python
# coding: utf-8
#
# In[ ]:
#

from pandas import DataFrame
from preprocessing_classification import *
from joblib import load

BEST_MODEL_PATH = "resources/best_model.pkl"

model = load(BEST_MODEL_PATH)


def inference(path):
    '''
    path: a DataFrame
    result is the output of function which should be
    somethe like: [0,1,1,1,0]
    0 -> Lost
    1 -> Won
    '''

    result = []

    dataset = read_data(path)

    # ''''''
    first = []
    list_prediction = list(dataset['Stage'])
    for a in list_prediction:
        if a == 'Won':
            first.append(1)
        elif a == 'Lost':
            first.append(0)
        else:
            first.append(3)
    print(first)
    # '''''
    dataset = preprocess(dataset)
    dataset = new_feature(dataset)
    # get products
    dataset = dataset.drop(['Customer', 'Agent', 'SalesAgentEmailID', 'ContactEmailID',
                            'Created Date', 'Close Date', 'avg_sale_cyc', 'Stage'], axis=1)
    s = (dataset.dtypes == 'object')
    products = list(s[s].index)
    # getting one on encoded columns
    dataset = encoder(dataset, products)

    pred = model.predict(dataset)

    list_prediction = list(pred)
    for a in list_prediction:
        if a == 'Won':
            result.append(1)
        else:
            result.append(0)
    print(result)

    return result


inference("dataset.xls")
