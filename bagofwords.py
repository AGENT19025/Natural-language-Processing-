# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:44:34 2022

@author: Vishwas
"""

import nltk

paragraph=""" It is important to note that a paragraph does not have a minimum or maximum number of sentences that it must have to fit the definition of a paragraph. Some writers will opt to use very short paragraphs, while others will include dozens of sentences in their paragraphs. It is also important to know that most writers separate lines of dialogue into paragraphs, so if a character only speaks a single line, it will be its own paragraph.

Keeping that in mind, there is a general agreement on the format of a standard paragraph, which especially applies to informational and argumentative or persuasive writing. A paragraph should be divided into three distinct sections that each serve a purpose to the paragraph as a whole.

Topic Sentence - The topic sentence is the sentence that lays out a preview of what the paragraph will be about. Think of it as a preview of the paragraph. It puts the reader's mindset into the right place to digest the information.
Supporting Sentence - The supporting sentence is the most important part of the paragraph and provides details to support the topic sentence. If the topic sentence is about frogs, then the details should be about frogs as well. This section can be longer or shorter depending on the content and the writer's preference.
Concluding Sentence- The concluding sentence wraps up the paragraph and ties back to the topic sentence. A conclusion could be a reiteration of the topic, an opinion based on the topic, or a way to wrap up the paragraph with a more general statement.
The following paragraph is an example of a simple paragraph that follows the basic paragraph form.

1 There are many different kinds of animals that live in China. 2 Tigers and leopards are animals that live in China's forests in the north. 2 In the jungles, monkeys swing in the trees, and elephants walk through the brush. The cities in China are filled with millions of people. 2 There are camels in the deserts in China that people use for transportation. 3 Lots of different kinds of animals make their home in China.

In the example paragraph, the topic sentence, labeled with a 1, states the paragraph's main idea. The reader can rightfully assume that the content will have to do with animals in China.
The supporting sentences, indicated by the 2, are separated into sentences that give more specific information about animals that live in China. Each detail directly supports the main topic.
The concluding sentence, indicated by 3, restates the topic sentence in a slightly different way to help wrap up the paragraph.
The highlighted sentence in the paragraph is important to note because it is a detail that does not relate closely enough to the topic sentence to be included in the essay. While the sentence does mention China, it would be better placed under a paragraph about people in China or the cities of China.
Along with the general format of a paragraph, two kinds of paragraphs bear mentioning. They are a brief paragraph and a piece paragraph. A brief paragraph is fairly straightforward since it contains the definition in its name. A brief paragraph is a short paragraph generally consisting of less than ten sentences in total."""

#cleaning the text
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer


ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
