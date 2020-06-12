# -*- coding: utf-8 -*-
"""# NLP Assignment"""
#section : this is the NLP assignement on Naive Bayes Text Classifier
import pandas as pd
import numpy as np
#importing the data from csv file.
data = pd.read_csv('/Tweets.csv')
data.columns
data.head()
#data has following columns
#Index(['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence',
       'negativereason', 'negativereason_confidence', 'airline',
       'airline_sentiment_gold', 'name', 'negativereason_gold',
       'retweet_count', 'text', 'tweet_coord', 'tweet_created',
       'tweet_location', 'user_timezone'],
      dtype='object')
#out of which text, airline and airline_sentiment is important for us
# I am diving the data into 90% and 10%.
#90% is used for training the model and 10% is used for testing the model
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(data['text'],data['airline_sentiment'],test_size = 0.1)
#converting them to list data structures
X_train.to_list
Y_train.to_list

#creating the word vector as done in IR and web search.
#I am creating dictionary out of text field and airline_sentiment field 
from sklearn.feature_extraction.text import CountVectorizer
vectorXTrain = CountVectorizer()
vectorXTrain.fit(X_train)
countXTrain = vectorXTrain.transform(X_train)
countXTrain.shape

#calculating the tf-idf scores for the above calculation

from sklearn.feature_extraction.text import TfidfTransformer
vectorizerXTrain = TfidfTransformer()
finalXTrain = vectorizerXTrain.fit_transform(countXTrain)
finalXTrain.shape

#generating the naive bayes model for our prepared data
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(finalXTrain,Y_train)

#taking the rest 10% data to test it

Xtest = vectorXTrain.transform(X_test)
Xtestidf = vectorizerXTrain.transform(Xtest)
predicted = clf.predict(Xtestidf)
for x in predicted:
  print(x)

#precision accuracy and recall values
from sklearn import metrics
from sklearn.metrics import accuracy_score
print("Accuracy" , accuracy_score(Y_test,predicted))
print(metrics.classification_report(Y_test,predicted))
