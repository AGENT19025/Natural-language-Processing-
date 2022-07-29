# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:30:57 2022

@author: Vishwas
"""
#dataset importing 
import pandas as pd

messages = pd.read_csv('SpamClassifier-master/smsspamcollection/SMSSpamCollection',sep='\t',names=["label","message"])

#data cleaning and preprocessing 
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
corpus=[]

for i in range (len(messages)):
    review=re.sub('[^a-zA-Z]',' ', messages['message'][i])
    review =review.lower()
    review= review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('English')]
    review= ' '.join(review)
    corpus.append(review)
    
#creating a bag of word 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x= cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y= y.iloc[:,1].values

#train set split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.20,random_state=0)

#training test model by using naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(x_train,y_train)
y_pred=spam_detect_model.predict(x_test)


    
