# -*- coding: utf-8 -*-

from twitchio.ext import commands
from colorama import Fore, Style
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib import style
from ChatProcessor import clean_df, df_graphs



date = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
timestamp = []
chatter = []
chat_log = []
p_n = []

class TwitchBot(commands.Bot):

    def __init__(self, token, channel, classifier):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot
        super().__init__(token=token, prefix='?', initial_channels=[channel])
        self.classifier = classifier
        self.channel = channel

    async def event_ready(self):
        # We are logged in and ready to chat and use commands...
        print(f'Logged in as | {self.nick}')
        #print(f'User id is | {self.user_id}')
        

    async def event_message(self, message):
        if message.echo:
            return 
        #print(message.content)
        timestamp.append(message.timestamp)
        chatter.append(message.author.name)
        chat_log.append(message.content)
        sentiment = self.classifier(message.content)
        mylist = sentiment[0]
        score = mylist['score']
        sentclass = mylist['label']
        p_n.append(sentclass)
        #print(chat_log)
        if (sentclass == 'NEGATIVE'):
            print(Fore.LIGHTRED_EX + f'author: {message.author.name} sent: {message.content} score: {score} sentiment: {sentclass}')
        else:
            print(Fore.GREEN + f'author: {message.author.name} sent: {message.content} score: {score} sentiment: {sentclass}')
        print('Messages gathered: ', len(chat_log), '\n')
        
        await self.handle_commands(message)
        
        if len(chat_log) >= 1:
             c_dict = {'timestamp':timestamp,'chatter':chatter,'chat':chat_log,'sentiment':p_n}
             c_df = pd.DataFrame(c_dict)
             log_name = str(f'{self.channel}_chat_log_{date}')
             c_df.to_csv(log_name + ".csv")
             
        df = clean_df(log_name = str(f'{self.channel}_chat_log_{date}'))
        
        df_graphs(df) 

    @commands.command()
    async def hello(self, ctx: commands.Context):
        # Send a hello back!
        await ctx.send(f'Hello {ctx.author.name}!')


