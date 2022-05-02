# -*- coding: utf-8 -*-

import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re


stopwords = nltk.corpus.stopwords.words('english')
addi_sw = ['im','shouldve','youd','youll','youre','youve']
stopwords.extend(addi_sw)
stopwords.remove("not")


df = pd.read_csv('tokibbi_chat_log_2022_04_07-01.34.30_PM.csv', index_col=0)

df.dropna(axis=0, inplace=True)

# To train our model, we only need the text and sentiment columns 
text = df[['chat','sentiment']]

vectorizer = TfidfVectorizer(norm=None, binary=False, ngram_range=(1, 2))

# We remove any usernames, hashtags, links, stopwords, and punctuation from our chat messages
text['no_users_links'] = text['chat'].apply(lambda x: re.sub('(@|#)([A-Za-z0-9_-]+\s?)|(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)','',x.lower()))
text['no_punct'] = text['no_users_links'].str.replace(r'([^\w\s]|[_])+', '')
text['cleaned'] = text['no_punct'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stopwords]))

pos = text[text['sentiment'] == 1]['cleaned']
neu = text[text['sentiment'] == 0]['cleaned']
neg = text[text['sentiment'] == -1]['cleaned']

X_train, X_test, y_train, y_test = train_test_split(text['cleaned'], text['sentiment'], test_size = .25, random_state=42, stratify = text['sentiment'])

X_train_vec = vectorizer.fit_transform(X_train)




