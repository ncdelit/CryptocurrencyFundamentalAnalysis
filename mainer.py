import praw
import pprint
from nltk.corpus import stopwords
from collections import Counter
import re
from numpy import transpose
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import numpy as np
from defs import *
import pandas as pd
import numpy as np
import nltk
import operator
import collections
orderedDict = collections.OrderedDict()
from collections import OrderedDict

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def unicodetoascii(text):

    uni2ascii = {
            ord('\xe2\x80\x99'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\x9c'.decode('utf-8')): ord('"'),
            ord('\xe2\x80\x9d'.decode('utf-8')): ord('"'),
            ord('\xe2\x80\x9e'.decode('utf-8')): ord('"'),
            ord('\xe2\x80\x9f'.decode('utf-8')): ord('"'),
            ord('\xc3\xa9'.decode('utf-8')): ord('e'),
            ord('\xe2\x80\x9c'.decode('utf-8')): ord('"'),
            ord('\xe2\x80\x93'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x92'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x94'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x94'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x98'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\x9b'.decode('utf-8')): ord("'"),

            ord('\xe2\x80\x90'.decode('utf-8')): ord('-'),
            ord('\xe2\x80\x91'.decode('utf-8')): ord('-'),

            ord('\xe2\x80\xb2'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb3'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb4'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb5'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb6'.decode('utf-8')): ord("'"),
            ord('\xe2\x80\xb7'.decode('utf-8')): ord("'"),

            ord('\xe2\x81\xba'.decode('utf-8')): ord("+"),
            ord('\xe2\x81\xbb'.decode('utf-8')): ord("-"),
            ord('\xe2\x81\xbc'.decode('utf-8')): ord("="),
            ord('\xe2\x81\xbd'.decode('utf-8')): ord("("),
            ord('\xe2\x81\xbe'.decode('utf-8')): ord(")"),

                            }
    return text.decode('utf-8').translate(uni2ascii).encode('ascii')

"""""""""""""""""""""""""""""""""""
PRAW REDDIT API CONNECTION
"""""""""""""""""""""""""""""""""""
reddit = praw.Reddit(client_id='dVYcYUslaGprfg',
                     client_secret='',
                     user_agent='',
                     username='',
                     refreshtoken = '',
                     password='')



subreddit = reddit.subreddit('CryptoCurrency')
print(subreddit.display_name)  # Output: redditdev
print(subreddit.title)         # Output: reddit Development

"""""""""""""""""""""""""""""""""""
INPUTS
"""""""""""""""""""""""""""""""""""
Terms = ["whitepaper","whitepapers","gitHub","usecase","use-case", "fundamental","telegram","roadmap","commits","adoption","intrinsic","valuation","repository","CryptoTechnology","bitcointalk","discord","gitter"]

otherWords = ['I','You','It','The','If','So','like','I\'m','also','It\'ll'] # other words removed

NumberOfComments = 500
TopPercentage = 1
TopN = 2000

"""""""""""""""""""""""""""""""""""
USER SELECTION
"""""""""""""""""""""""""""""""""""
#users =['arsonbunny']

import pickle

with open('database2.db', 'rb') as fh: db2 = pickle.load(fh)

users = db2['Users'] #Load Selected sophisticated users from database

"""""""""""""""""""""""""""""""""""
FUNCTION CALLS
"""""""""""""""""""""""""""""""""""
userswithcomments = GetComments(users,NumberOfComments,reddit)

usersS = ScoreUsers(users,userswithcomments,1,Terms)[0]
usersSind = ScoreUsers(users,userswithcomments,1,Terms)[1]

Merged = CleanComments(NumberOfComments,userswithcomments,users,usersSind)

SubsetCount = FrequencyTable(Merged,otherWords)

projects = ProjectDetermination(SubsetCount,TopN)

sentences = CommentsToSentences(userswithcomments,usersSind)

WordCoin = ComputeSentimentTable(sentences,projects)

ComputeAnalysisTable(WordCoin,projects)

AssociatedWords(WordCoin,otherWords)
