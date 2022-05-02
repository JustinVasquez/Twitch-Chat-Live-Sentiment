# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:09:28 2022

@author: jvasq
"""

import pandas as pd
import datetime
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib import style

style.use('fivethirtyeight')

def clean_df(log_name):
    df = pd.read_csv(log_name+'.csv', index_col=0)
    
    # The timestamps produced by the TwitchIO api are 5 hours ahead of my timezone
    # So I convert the column to datetime in order to adjust the hour accordingly
    # And get the correct timestamps in the graphs
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].apply(lambda x: x - timedelta(hours=5))

    df.dropna(axis=0, inplace=True)
    
    return df


def df_graphs(df):

    # This graph plots the top 10 chatters by message count
    active_chatters = df.groupby('chatter').chat.count().sort_values(ascending=False)
    
    ax = sns.barplot(x=active_chatters.index[:10], y=active_chatters[:10])
    
    # This graph plots the percentage of messages based on their sentiment
    text_df = df[['chat','sentiment']]
    text_df['sentiment'].value_counts()
    pct_sentiment = (text_df['sentiment'].value_counts())/(text_df['sentiment'].count())
    
    colors = sns.color_palette('pastel')
    ax2 = plt.pie(pct_sentiment, labels=pct_sentiment.index, colors = colors, autopct='%.2f%%')
    
    # This graph plots how many messages are sent per minute 
    time_idx = df.set_index('timestamp')
    msg_per_min = time_idx['chat'].resample('1T').count()
    
    ax3 = sns.lineplot(x=msg_per_min.index, y=msg_per_min)
    
    plt.show()