# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:42:55 2022

@author: Vishwas
"""
import nltk


paragraph =""" First published in The Atlantic Monthly in 1981, “Cathedral” is today known as one of Raymond Carver’s finest works. When it opens, we meet a narrator whose wife is expecting a visit from an old friend, a blind man. Dissatisfied and distrusting of people not like him, our narrator struggles to connect until the blind man asks him to describe a cathedral to him. 

 “Cathedral” is one of Carver’s own personal favorites, and deservedly so. His characteristic minimalist style is devastating as the story builds up to a shattering moment of emotional truth — an ultimate reminder that no-one else can capture the quiet sadness of working-class people like him. """
 
 #tokenizing paragraph into sentences 
sentence=nltk.sent_tokenize(paragraph)

#tokenizing papagraph into words

words = nltk.word_tokenize(paragraph)


