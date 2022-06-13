# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
import nltk


stopwords = nltk.corpus.stopwords.words('english')
addi_sw = ['hi','hii','hai','haii','im','shouldve','youd','youll','youre','youve']
stopwords.extend(addi_sw)
stopwords.remove("not")

df1 = pd.read_csv('tokibbi_chat_log_2022_04_07-01.34.30_PM.csv', index_col=0)
df2 = pd.read_csv('itsryanhiga_chat_log_2022_04_07-01.02.08_PM.csv', index_col=0)
df = pd.concat([df1, df2], axis=0, ignore_index=True)

df['sentiment'].value_counts()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].apply(lambda x: x - timedelta(hours=5))

df.dropna(axis=0, inplace=True)

text = df[['chat','sentiment']]

# We remove any usernames, hashtags, links, stopwords, and punctuation from our chat messages
# We also try to adjust the spelling on words such as 'soooooo' to 'soo'
# While not perfect, it reduces different variations such as 'sooo', 'soooooooo', etc.
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

wnl = WordNetLemmatizer()

# To train our model, we only need the text and sentiment columns 
text = df[['chat','sentiment']]

text['no_users_links_punct'] = text['chat'].apply(lambda x: re.sub('(#)|(@)([A-Za-z0-9_-]+\s?)|[0-9]|(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*?)|([^\w\s]|[_])+','',x.lower()))
text['spell_corrected'] = text['no_users_links_punct'].apply(lambda x: re.sub(r'(.)\1+', r'\1\1', x))
text['cleaned'] = text['spell_corrected'].apply(lambda x: ' '.join([wnl.lemmatize(word, pos='v') for word in word_tokenize(x) if word not in stopwords]))

# We stratify based on the Y column as the 'negative' class is under-represented 
# in comparison to the other two classes, so our train/test splits will have equal 
# values of each class
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text['cleaned'], text['sentiment'], test_size = .20, random_state=42, stratify = text['sentiment'])


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

tv = TfidfVectorizer(binary=False, ngram_range=(1, 3))
tv_lr = LogisticRegression(max_iter=200)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score


def lr_cv(X, Y, pipeline, splits=5, average_method='macro'):
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X,Y): 
        fit_lr = pipeline.fit(X.iloc[train],Y.iloc[train])
        predictions = fit_lr.predict(X.iloc[test])
        scores = fit_lr.score(X.iloc[test],Y.iloc[test])
        
        accuracy.append(scores * 100)
        precision.append(precision_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('              negative    neutral     positive')
        print('precision:',precision_score(Y.iloc[test], predictions, average = None))
        recall.append(recall_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('recall:   ',recall_score(Y.iloc[test], predictions, average = None))
        f1.append(f1_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('f1 score: ', f1_score(Y.iloc[test], predictions, average = None))
        print('-'*50)
    
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
    

# Conducted model comparisons between Logistic Regression and Naive Bayes
# Also within each model, tested between the cleaned dataset and SMOTE oversampling 
# for the minority class, but there was not a noticeable increase in average recall 
# in relation to the drop in average precision 

from sklearn.pipeline import Pipeline 

original_pipeline = Pipeline([
    ('vectorizer', tv),
    ('classifier', tv_lr)
    ])

lr_cv(text['cleaned'], text['sentiment'], original_pipeline)


from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

smt = SMOTE(random_state = 42, k_neighbors = 3)
SMOTE_pipeline = make_pipeline(tv, smt, tv_lr)

lr_cv(text['cleaned'], text['sentiment'], SMOTE_pipeline)


from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB()

def nb_cv(X, Y, pipeline, splits=5, average_method='macro'):
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X,Y): 
        fit_nb = pipeline.fit(X.iloc[train],Y.iloc[train])
        predictions = fit_nb.predict(X.iloc[test])
        scores = fit_nb.score(X.iloc[test],Y.iloc[test])
        
        accuracy.append(scores * 100)
        precision.append(precision_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('              negative    neutral     positive')
        print('precision:',precision_score(Y.iloc[test], predictions, average = None))
        recall.append(recall_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('recall:   ',recall_score(Y.iloc[test], predictions, average = None))
        f1.append(f1_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('f1 score: ', f1_score(Y.iloc[test], predictions, average = None))
        print('-'*50)
    
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))

nb_pipeline = Pipeline([
    ('vectorizer', tv),
    ('classifier', MNB)
    ])

nb_cv(text['cleaned'], text['sentiment'], nb_pipeline)

nb_cv(text['cleaned'], text['sentiment'], SMOTE_pipeline)

# Attempted to reduce the number of repeat messages from the overall dataset 
# to get more even distributions, but it resulted in a decrease of 8-10% 
# across all metrics, likely because of how many messages were taken out of the dataset (~2000). 


