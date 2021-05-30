from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

OH_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

# def read_data():
interactions = pd.read_excel('interactions.xlsx')
dataset = pd.read_excel('dataset.xls')
dataset_len = len(dataset)


def plot_product_close_value():
    f, ax = plt.subplots(figsize=(12, 12))
    sb.boxplot(x='Product', y='Close_Value', hue='Stage', data=dataset, ax=ax)
    plt.show()


# read_data()
# plot_product_close_value()

def new_feature():
    global dataset
    '''''
    # opportunity Win rate for a sales rep --> dataset['win_rate']
    for agent in dataset['Agent'].unique():
        dataset.loc[(dataset['Agent'] == agent), 'win_rate'] = (
                    (dataset['Stage'] == 'Won').where(dataset['Agent'] == agent).sum() / (
                        dataset['Agent'] == agent).sum())
    '''''
    # date difference between dates in data --> dataset['DateDiff']
    date_format = "%m/%d/%Y"
    dataset['Close Date'] = pd.to_datetime(dataset['Close Date'], date_format)
    dataset['Created Date'] = pd.to_datetime(dataset['Created Date'], date_format)
    dataset['DateDiff'] = dataset['Close Date'] - dataset['Created Date']
    dataset["DateDiff"] = (dataset["DateDiff"]).dt.days

    # Avg Sale cycle for a company --> sales_data['avg_sale_cyc']
    for prod in dataset['Product'].unique():
        dataset.loc[(dataset['Product'] == prod), 'avg_sale_cyc'] = dataset['DateDiff'].where(
            dataset['Product'] == prod).sum() / ((dataset['Product']
                                                  == prod).where(dataset['Stage'] == 'Won').sum())
    # print(dataset.head())


new_feature()


# function to one hot encode columns
def OHE(df, col):
    encprod = pd.DataFrame(OH_enc.fit_transform(df[col]))
    encprod.index = df.index
    encprod.columns = OH_enc.get_feature_names(col)
    df = pd.concat([df, encprod], axis=1)
    df = df.drop(col, axis=1)
    return df


# PreProcessing and Correlation matrix
def correlation_matrix():
    global dataset, dataset_len
    for i in range(0, dataset_len):
        if dataset.loc[i, 'DateDiff'] < dataset.loc[i, 'avg_sale_cyc']:
            dataset.loc[i, 'DateDiff'] = 1
        else:
            dataset.loc[i, 'DateDiff'] = 0
    # print(dataset)
    dataset_class = dataset
    progress_dataset = dataset.loc[dataset.Stage == 'In Progress', dataset.columns]
    progress_dataset.reset_index(drop=True, inplace=True)
    progress_dataset = progress_dataset.drop(['Customer', 'Agent', 'SalesAgentEmailID', 'ContactEmailID', 'Stage',
                                              'Created Date', 'Close Date', 'avg_sale_cyc'], axis=1)
    dataset_class = dataset_class.loc[dataset_class.Stage != 'In Progress', dataset_class.columns]
    dataset_class.reset_index(drop=True, inplace=True)
    dataset_class = dataset_class.drop(['Customer', 'Agent', 'SalesAgentEmailID', 'ContactEmailID',
                                        'Created Date', 'Close Date', 'avg_sale_cyc'], axis=1)
    # Label Binarizer
    lb = LabelBinarizer()
    lb.fit_transform(dataset_class['Stage'])
    deal_class = dataset_class['Stage']
    dataset_class = dataset_class.drop(['Stage'], axis=1)

    s = (dataset_class.dtypes == 'object')
    products = list(s[s].index)

    # getting one on encoded columns
    dataset_class = OHE(dataset_class, products)
    progress_dataset = OHE(progress_dataset, products)

    # corr and plot
    correlation = dataset_class.corr()
    # plt.figure(figsize=(14, 14))
    sb.heatmap(data=correlation, square=True, annot=True, vmin=-1, vmax=1, center=0,
               cmap='coolwarm', annot_kws={"size": 5, "weight": "bold"})
    plt.show()


correlation_matrix()


# def plotCorrelationMatrix(df, graphWidth):
#     df = df.dropna('columns') # drop columns with NaN
#     df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
#     if df.shape[1] < 2:
#         print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
#         return
#     corr = df.corr()
#     print(corr)
#     plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
#     corrMat = plt.matshow(corr, fignum = 1)
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.gca().xaxis.tick_bottom()
#     plt.colorbar(corrMat)
#     plt.title('Correlation Matrix for transactions', fontsize=15)
#     plt.show()
#
#
# plotCorrelationMatrix(dataset, dataset.get_width())
