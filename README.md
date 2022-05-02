# Twitch-Chat-Live-Sentiment
Analyzing a Twitch channel's chat sentiment in realtime 
======
## Context
Over the course of the pandemic, I began watching Twitch live streams as I found them to be a refreshing change of pace akin to a live podcast rather than a typical TV show. Apart from the rare prerecorded clip usually in collaboration with a sponsor, the content streamers broadcast is live along with genuine reactions to the games they playing, the videos they watch, and the other creators they interact with. As such, I wanted to track how viewers, colloquially known as "chat", react in realtime to the content being presented. 

## Dataset
I utilized Twitch's API, TwitchIO, to connect to channels to save messages being sent in chat to train my model, as well as to connect for live analysis. 

## Analysis Process
1. Collecting chat messsages and manually classifying (positive, neutral, negative)
2. Data cleaning
3. Exploratory Data Analysis
4. Training sentiment analysis model 
5. Defining functions to produce live graphs 
