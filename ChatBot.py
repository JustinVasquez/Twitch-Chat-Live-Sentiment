# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os
from TwitchBot import TwitchBot
from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')

# Accesses the .env file that holds your Twitch access token
load_dotenv()
token = os.getenv('TWITCH_ACCESS_TOKEN')

# Specifies which channel to connect to
channel = 'tokibbi'

bot = TwitchBot(token, channel, classifier)
bot.run()
