#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:20:41 2018

@author: group6
"""
import pandas as pd
import numpy as np
import re
import nltk
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pos_words = pd.read_csv("positive-words.txt", header=None)
neg_words = pd.read_csv("negative-words.txt", header=None)


def preprocess_tweets(tweet):
  tweet=tweet.lower()
  tweet=re.sub('((www.[^\s]+)|(https?://[^\s]+))','URL', tweet)
  tweet=re.sub('@[^\s]+','AT_USER',tweet)
  tweet=re.sub('[\s]+', ' ', tweet)
  tweet=re.sub(r'#([^\s]+)', r'\1', tweet)
  return tweet
preprocess_tweets("@V_DEL_ROSSI: Me #dragging myself to the gym https://t.co/cOjM0mBVeY")

stopWords = pd.read_csv('stopwords.txt').values


nltk.download('punkt')
def removePunctuations(tweet):
  punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  no_punct_tweet = ""
  for char in tweet:
    if char not in punctuations:
       no_punct_tweet = no_punct_tweet + char
  return(no_punct_tweet)

def word_tokenizer(tweet):
  tweet=removePunctuations(tweet)
  tokenized_tweets = nltk.word_tokenize(tweet)
  filtered_sentence = [w for w in tokenized_tweets if not w in stopWords]
  return filtered_sentence


def getfeaturevector(tokenized_tweet):
    pos_words_count = 0
    neg_words_count = 0
    neutral_words_count = 0
    for token in tokenized_tweet:
        if (token in pos_words[0].values):
            pos_words_count += 1
        elif (token in neg_words[0].values):
            neg_words_count += 1
        else:
            neutral_words_count += 1
            
    return np.array([pos_words_count, neg_words_count, neutral_words_count])


def trainModel(X_train,Y_train):

    svc = svm.SVC(kernel='linear')  
    svc.fit(X_train, Y_train)
    
    predLog = svc.predict(X_train)
    print(predLog)
    return svc

def testModel(model, X_test,Y_test):
    Y_predict = model.predict(X_test)
    return accuracy_score(Y_test,Y_predict)
    
tweet = "Neither, Man nor machine can about angry bad stupid great priyanka is awesome, replace its all creator really"
print("------------------------------")
processed_tweet = preprocess_tweets(tweet)
print(processed_tweet)
tokenized_tweet = word_tokenizer(processed_tweet)
print(tokenized_tweet)
tweetFeatures = getfeaturevector(tokenized_tweet)
print(tweetFeatures)

print("------------------------------")

print("Loading data ------------------------------")
data= pd.read_csv("twitter_train.csv",encoding='cp1252')
data.drop(data.columns[[0]], axis=1, inplace=True)
X_train = []
Y_train = []

for index, row in data.iterrows():
       processed_tweet = preprocess_tweets(row['SentimentText'])
       tokenized_tweet = word_tokenizer(processed_tweet)
       tweetFeatures = getfeaturevector(tokenized_tweet)
       X_train.append(tweetFeatures)
       Y_train.append(int(row['Sentiment']))

X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train, test_size=0.2, random_state=46)
print("training model ------------------------------")      
model = trainModel(X_train,Y_train)
print("testing model ------------------------------")      
accuracy = testModel(model,X_test,Y_test)
print(accuracy)

consumer_key = '3znWdatdhro9sgkdEPYv7LG3k'
consumer_secret = 'KhIfIuN12q21H2WZXJdnrU6L4TeSZY80G0Bs8CuuKeZBDQlktU'
access_token = '72797128-tm5e8tDTcGOnwu2aR1qDTStsjV3K8skmCpRwK2SXb'
access_secret = 'Gbs9KKnWwHweJcFnQy3t480PLgkV5OIbiwfk9nDFt4rHu'

import tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)  

X_tweet=[]
for i in tweepy.Cursor(api.search, q='#election2018', lang = 'en', full_text=True).items(10):
    #print(i._json['text'])
    
    processed_tweet = preprocess_tweets(i._json['text'])
    tokenized_tweet = word_tokenizer(processed_tweet)
    tweetFeatures = getfeaturevector(tokenized_tweet)
    X_tweet.append(tweetFeatures)
    #print(myStreamListener.processTweet(i._json['text']), end='\n\n\n')


Y_predict = model.predict(X_tweet)
j=0
for i in X_tweet:
  print("{} :{}".format(i,Y_predict[j]))
  j +=1
total = len(Y_predict)
positive = 0
negative = 0

for i in Y_predict:
  if (i==0):
    negative+=1
  else:
    positive+=1
print(" total :{}, positive:{}, negative:{}".format(total, positive, negative))

#print(" positive " + str((positive/total)*100) + "%")
print(" negative " + str((negative/total)*100) + "%")