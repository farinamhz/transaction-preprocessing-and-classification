import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OH_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

# read_data
interactions = pd.read_excel('interactions.xlsx', index_col=0)
dataset = pd.read_excel('dataset.xls', index_col=0)


def interaction_analysis():
    global interactions, dataset
    analyser = SentimentIntensityAnalyzer()
    scores = []
    text = interactions['Extracted Interaction Text']
    for sentence in text:
        score = analyser.polarity_scores(sentence)
        scores.append(score)

    # Converting List of Dictionaries into Dataframe
    score_dataframe = pd.DataFrame(scores)
    interactions = pd.concat([interactions, score_dataframe.compound],axis=1)
    # merging interaction score in sales pipeline file
    # Calculating product acceptance rate for as product of a company or won to total opportunities for a company
    #
    for em in interactions.fromEmailId.unique():
        # interactions.loc[(interactions['fromEmailId']==em),'compound']=(interactions['compound'].where(interactions['fromEmailId']==em)).mean()
        dataset.loc[dataset['SalesAgentEmailID'] == em, 'interaction_score'] =\
            (interactions['compound'].where(interactions['fromEmailId'] == em)).mean()
    for cm in dataset.ContactEmailID.unique():
        dataset.loc[dataset['ContactEmailID'] == cm, 'prod_acc_rate'] =\
            (dataset.loc[dataset['ContactEmailID'] == cm, 'Product']).where(dataset.Stage == 'Won').count() /\
            (dataset.loc[dataset['ContactEmailID'] == cm, 'Product']).count()
    print(dataset['prod_acc_rate'])
    # Finish In(9)/ Birdie shape link


# drop NaN
dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
dataset = dataset.reset_index()

# drop index column
dataset.drop(["index"], axis=1, inplace=True)

dataset_len = len(dataset)

'''''
def plot_product_close_value():
    f, ax = plt.subplots(figsize=(12, 12))
    sb.boxplot(x='Product', y='Close_Value', hue='Stage', data=dataset, ax=ax)
    plt.show()
'''''


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
    # dataset_class includes won or fail AND progress_class includes In progress
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

    # get products
    s = (dataset_class.dtypes == 'object')
    products = list(s[s].index)

    # getting one on encoded columns
    dataset_class = OHE(dataset_class, products)
    progress_dataset = OHE(progress_dataset, products)

    # corr and plot
    correlation = dataset_class.corr()
    plt.figure(figsize=(14, 14))
    sb.heatmap(data=correlation, square=True, annot=True, vmin=-1, vmax=1, center=0,
               cmap='coolwarm', annot_kws={"size": 15}, linewidths=2, linecolor='black', )
    plt.show()

    return dataset_class, deal_class, progress_dataset


dataset_class, deal_class, progress_dataset = correlation_matrix()


lb = MultiLabelBinarizer()
lb.fit_transform(dataset_class.values.tolist())
# train
X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)


def print_report(name, y_test_internal, pred_internal):
    # print report
    print('*** {} Classification Report ***'.format(name), end='\n\n')
    print('Confusion Matrix:\n', confusion_matrix(y_test_internal, pred_internal), end='\n\n')
    print('Classification Report:\n', classification_report(y_test_internal, pred_internal), end='\n\n')
    print('Accuracy:\n', accuracy_score(y_test_internal, pred_internal), end='\n\n')


'''''
def logistic_regression():
    # train
    X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    # print report
    print_report("Logistic Regression", y_test, pred)


logistic_regression()
'''''


def random_forest():
    # # train
    # X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    pred = rf_classifier.predict(X_test)

    # print report
    print_report("Random Forest", y_test, pred)


random_forest()


def decision_tree():
    # # train
    # X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    pred = dt_classifier.predict(X_test)

    # print report
    print_report("Decision Tree", y_test, pred)


decision_tree()


def gaussian_naive_bayes():
    # # train
    # X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)

    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    pred = nb_classifier.predict(X_test)

    # print report
    print_report("Gaussian Naive Bayes", y_test, pred)


gaussian_naive_bayes()


def knn():
    # # train
    # X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)

    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    pred = knn_classifier.predict(X_test)

    # print report
    print_report("KNN", y_test, pred)


knn()
