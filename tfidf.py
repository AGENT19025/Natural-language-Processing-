# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:30:07 2022

@author: Vishwas
"""

import nltk

paragraph =""" The company was co-founded by Chris Berend and Andrew Morse. Berend started his career in media by working for six years as a director of content at ESPN. Through this experience, he gained the knowledge that led him to become head of video for Bloomberg Media Group. After leaving Bloomberg, Berend became the senior vice president of CNN’s digital video.[4] Morse started his media career by working as a desk assistant for ABC News and eventually became a producer. Morse spent a total of 15 years at ABC. After leaving ABC, Morse became the head of Bloomberg Television, then left Bloomberg, and joined CNN in 2013. Morse is currently the Executive Vice President and General Manager of CNN Digital Worldwide.

In 2015, while both working at CNN, Berend and Morse came up with the idea for Great Big Story. Berend, who oversees strategy/operations and audience development for Great Big Story, described wanting the company to be "fundamentally optimistic, but not naïve or sunshine-y".[5] Morse described wanting Great Big Story to have "remarkable feats of storytelling".[6]

Video content
Great Big Story created videos that go into categories and they have subcategories within them. The five main categories are Human Condition, Frontiers, Planet Earth, Flavors, and Origins.

Human Condition
Human Condition videos are primarily about people. Within the Human Condition category, the subcategories are "More than a Day Job", "Defiant", "No Way!?", and "Music to my Ears".[7] More than a Day Job covers crafters, artist, innovators and regular people doing their job. These videos include a story about the man that runs the last manual scoreboard or the first autistic actor to be cast as the lead in a play. Defiant videos are about people who break expectations and societal norms that are put in for them. These videos include stories about a bodybuilder with 1 arm and no legs or a Division 1 college football player that is completely blind. No Way videos are stories about people that many people don’t know. One of these videos is about how a high school project inspired the 50-star American flag. The last section in Human Condition is Music to our Ears. Music to our Ears covers stories about songs and musicians that may be surprising. One of these stories includes one about the missing people choir. This choir is full of families and individuals of missing children who come together as a choir, turning their grief into hope.

Frontiers
The Frontiers category contains subcategories called "Portraits of the Artist", "Pushing Boundaries", "Wild World of Sports", and "Life in Space" with Leland Melvin.[8] Portraits of the Artist videos are about artists who share their art and their lives with the world. These include videos about an artist who built a human sized-bird's nest or a rancher who builds sculptures from scrap metal. Life in Space with Leland Melvin is a section that is much different from the others. This subcategory is where Leland Melvin describes how to eat, sleep, and live in outer space through animated videos. The next subsection is the Wild World of Sports. The Wild World of Sports introduces the view to new adventurous spots that they have never heard of like unicycle football and varsity lumberjack. The last subcategory is Pushing Boundaries. This is a broad category that can be about anything from an artist who creates instruments from wine glasses to a slackliner who walks between mountains. """

#cleaning the texts
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
corpus=[]
wordnet=WordNetLemmatizer()
sentences=nltk.sent_tokenize(paragraph)

for i in range (len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review= review.split()
    review =[wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('English')) ]
    review =' '.join(review)
    corpus.append(review)
    
#creating the  tf-idf model

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
x=cv.fit_transform(corpus).toarray()

    




