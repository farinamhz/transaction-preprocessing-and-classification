import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import *
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# read_data
interactions = pd.read_excel('interactions.xlsx')


def read_data(path):
    return pd.read_excel(path)


dataset = read_data('dataset.xls')

'''''
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
'''''
'''''
# drop NaN
dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
dataset = dataset.reset_index()
'''''


def preprocess(dataset):
    dataset.drop('Unnamed: 0', axis=1, inplace=True)
    # fill NaN with mean
    # mean = dataset.Close_Value.mean()
    # dataset.Close_Value.fillna(mean, inplace=True)

    # fill NaN with mean of each product
    dataset["Close_Value"] = dataset.groupby("Product").transform(lambda x: x.fillna(x.mean()))
    return dataset


dataset = preprocess(dataset)


def new_feature(dataset):
    # opportunity Win rate for a sales rep --> dataset['win_rate']
    for agent in dataset['Agent'].unique():
        dataset.loc[(dataset['Agent'] == agent), 'win_rate'] = (
                (dataset['Stage'] == 'Won').where(dataset['Agent'] == agent).sum() / (dataset['Agent'] == agent).sum())

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
    for i in range(0, len(dataset)):
        if dataset.loc[i, 'DateDiff'] < dataset.loc[i, 'avg_sale_cyc']:
            dataset.loc[i, 'DateDiff'] = 1
        else:
            dataset.loc[i, 'DateDiff'] = 0
    return dataset


dataset = new_feature(dataset)


# function to one hot encode columns
def encoder(df, col):
    OH_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encprod = pd.DataFrame(OH_enc.fit_transform(df[col]))
    encprod.index = df.index
    encprod.columns = OH_enc.get_feature_names(col)
    df = pd.concat([df, encprod], axis=1)
    df = df.drop(col, axis=1)
    return df


# PreProcessing and Correlation matrix
def preprocess_2():
    global dataset

    # dataset_class includes won or lost AND progress_class includes In progress
    dataset_class = dataset
    progress_dataset = dataset.loc[dataset.Stage == 'In Progress', dataset.columns]
    progress_dataset.reset_index(drop=True, inplace=True)
    progress_dataset = progress_dataset.drop(['Customer', 'Agent', 'SalesAgentEmailID', 'ContactEmailID', 'Stage',
                                              'Created Date', 'Close Date', 'avg_sale_cyc'], axis=1)
    dataset_class = dataset_class.loc[dataset_class.Stage != 'In Progress', dataset_class.columns]
    dataset_class.reset_index(drop=True, inplace=True)
    dataset_class = dataset_class.drop(['Customer', 'Agent', 'SalesAgentEmailID', 'ContactEmailID',
                                        'Created Date', 'Close Date', 'avg_sale_cyc'], axis=1)
    # print(dataset_class.head())

    # Label Binarizer
    lb = LabelBinarizer()
    lb.fit_transform(dataset_class['Stage'])
    deal_class = dataset_class['Stage']
    dataset_class = dataset_class.drop(['Stage'], axis=1)

    # get products
    s = (dataset_class.dtypes == 'object')
    products = list(s[s].index)

    # getting one on encoded columns
    dataset_class = encoder(dataset_class, products)
    progress_dataset = encoder(progress_dataset, products)

    '''''
    # corr and plot
    correlation = dataset_class.corr()
    plt.figure(figsize=(14, 14))
    sb.heatmap(data=correlation, square=True, annot=True, vmin=-1, vmax=1, center=0,
               cmap='coolwarm', annot_kws={"size": 15}, linewidths=2, linecolor='black', )
    plt.show()
    '''''
    return dataset_class, deal_class, progress_dataset


dataset_class, deal_class, progress_dataset = preprocess_2()

lb = MultiLabelBinarizer()
lb.fit_transform(dataset_class.values.tolist())

# train
# default test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, test_size=0.2, random_state=0)


def print_report(name, y_test_internal, pred_internal):
    # print report
    print('*** {} Classification Report ***'.format(name), end='\n\n')
    # print('Confusion Matrix:\n', confusion_matrix(y_test_internal, pred_internal), end='\n\n')
    # print('Classification Report:\n', classification_report(y_test_internal, pred_internal), end='\n\n')
    print('Accuracy:\n', accuracy_score(y_test_internal, pred_internal), end='\n\n')


def random_forest():
    # # train
    # X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)
    '''''
    train_scoreNum, test_scoreNum = validation_curve(
        RandomForestClassifier(),
        X=X_train, y=y_train,
        param_name='n_estimators',
        param_range=np.arange(1, 1000),
        cv=3, scoring="accuracy")
    plt.plot(np.arange(1, 1000), train_scoreNum,
             marker='o', markersize=5,
             color='blue', label='Training Accuracy')
    plt.plot(np.arange(1, 1000), test_scoreNum,
             marker='o', markersize=5,
             color='green', label='Validation Accuracy')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    '''''
    rf_classifier = RandomForestClassifier()

    # print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))

    rf_classifier.fit(X_train, y_train)

    # Save the model as a pickle in a file
    joblib.dump(rf_classifier, 'resources/best_model.pkl')

    # Load the model from the file
    # rf_classifier_from_joblib = joblib.load('resources/best_model.pkl')
    # pred = rf_classifier_from_joblib.predict(X_test)

    # predicting for in progress deals
    # pre_inprogress = rf_classifier.predict(progress_dataset)
    # pre_dataset = dataset.loc[dataset.Deal_Stage == 'In Progress', dataset.columns]
    # pre_dataset.reset_index(drop=True, inplace=True)
    # pre = pd.Series(pre_inprogress)
    # pre.value_counts()
    # pre_dataset = pd.concat([pre_dataset, pre], axis=1)
    # print(pre_dataset.head(100))

    # printing the confusion matrix
    '''''
    LABELS = ['Won', 'Lost']
    conf_matrix = confusion_matrix(y_test, pred)
    plt.figure(figsize=(12, 12))
    sb.heatmap(conf_matrix, xticklabels=LABELS,
               yticklabels=LABELS, annot=True, fmt="d", cmap='coolwarm')
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    '''''

    # print report
    # print_report("Random Forest", y_test, pred)


random_forest()


def decision_tree():
    # # train
    # X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    pred = dt_classifier.predict(X_test)

    # print report
    print_report("Decision Tree", y_test, pred)


# decision_tree()


def gaussian_naive_bayes():
    # # train
    # X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)

    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    pred = nb_classifier.predict(X_test)

    # print report
    print_report("Gaussian Naive Bayes", y_test, pred)


# gaussian_naive_bayes()


def knn():
    # # train
    # X_train, X_test, y_train, y_test = train_test_split(dataset_class, deal_class, random_state=0)

    '''''
    # dimensionality reduction
    pca = make_pipeline(StandardScaler(), PCA(n_components=2))
    pca.fit(X_train, y_train)
    '''''

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    '''''
    # dimensionality reduction and show plot
    X_embedded = pca.transform(X_train)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=['red' if item == 'Won' else 'blue' for item in y_train.tolist()],
                s=30, cmap='Set1')
    plt.show()
    '''''
    pred = knn_classifier.predict(X_test)

    print()
    list_prediction = list(pred)
    result = []
    for a in list_prediction:
        if a == 'Won':
            result.append(1)
        else:
            result.append(0)
    print(result)
    print_report("KNN", y_test.values, pred)

# knn()
