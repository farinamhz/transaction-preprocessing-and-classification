from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer
# analyser = SentimentIntensityAnalyzer()
# interactions = pd.read_excel('interactions.xlsx')
# scores = []
# text = interactions['Extracted Interaction Text']
# # print(text)
# for sentence in text:
#     score = analyser.polarity_scores(sentence)
#     scores.append(score)
#
# # Converting List of Dictionaries into Dataframe
# dataFrame = pd.DataFrame(scores)
# interactions= pd.concat([interactions,dataFrame.compound],axis=1)
# dataset=pd.read_excel('dataset.xls')
# #merging interaction score in sales pipeline file
# #Calculating product acceptance rate for as product of a company or won to total opportunities for a company
# #
# for em in interactions.fromEmailId.unique():
#      #interactions.loc[(interactions['fromEmailId']==em),'compound']=(interactions['compound'].where(interactions['fromEmailId']==em)).mean()
#     dataset.loc[dataset['SalesAgentEmailID'] == em, 'interaction_score']=(interactions['compound'].where(interactions['fromEmailId'] == em)).mean()
# for cm in dataset.ContactEmailID.unique():
#     dataset.loc[dataset['ContactEmailID'] == cm, 'prod_acc_rate']= (dataset.loc[dataset['ContactEmailID'] == cm, 'Product']).where(dataset.Stage == 'Won').count() / (dataset.loc[dataset['ContactEmailID'] == cm, 'Product']).count()
# print(dataset['prod_acc_rate'])
# Finish In(9)/ Birdie shape link



# plots parts from In(11) to In(15) /Birdie shape link
# f, ax = plt.subplots(figsize=(12, 12))
# sb.boxplot(x='Product',y='Close_Value',hue='Stage',data=sales_data,ax=ax)
# plt.show()
# sb.barplot(y='Product',x='Close_Value',hue='Stage',data=sales_data)
# plt.show()
# sb.countplot(y='Product',hue='Stage',data=sales_data)
# plt.show()
# sb.barplot(y='Customer',x='prod_acc_rate',data=sales_data,ax=ax)
# plt.show()



# In(17) preprocessing
#######################Farinam's Work##########################



