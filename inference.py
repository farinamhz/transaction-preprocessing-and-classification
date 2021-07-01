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
    deal_class = dataset['Stage']

    # ''''''
    # first = []
    # list_prediction = list(dataset['Stage'])
    # for a in list_prediction:
    #     if a == 'Won':
    #         first.append(1)
    #     elif a == 'Lost':
    #         first.append(0)
    #     else:
    #         first.append(3)
    # print(first)
    # '''''
    dataset = preprocess(dataset)
    dataset = new_feature(dataset)
    # get products
    dataset = dataset.drop(['Customer', 'Agent', 'SalesAgentEmailID', 'ContactEmailID',
                            'Created Date', 'Close Date', 'avg_sale_cyc', 'Stage'], axis=1)
    # dataset = dataset.drop(['Customer', 'Agent', 'SalesAgentEmailID', 'ContactEmailID',
    #                         'Created Date', 'Close Date', 'avg_sale_cyc'], axis=1)
    # s = (dataset.dtypes == 'object')
    # products = list(s[s].index)
    # dataset = encoder(dataset, products)
    dataset = dataset.join(pd.get_dummies(dataset['Product'])).drop('Product', axis=1)

    pred = model.predict(dataset)

    list_prediction = list(pred)
    for a in list_prediction:
        if a == 'Won':
            result.append(1)
        else:
            result.append(0)
    print(result)
    print('Accuracy:\n', accuracy_score(deal_class, pred), end='\n\n')
    return result


inference("test.xls")
