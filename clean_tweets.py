#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 09 9:03 PM 2021
Created in PyCharm
Created as disaster_tweets/clean_tweets.py

@author: Lynnie Saade
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, words

stop = set(stopwords.words('english'))
words_en = set(words.words())


def testlam(x):
    return x


# Remove 'internet garbage' (e.g. urls, emojis)
# these functions use regular expressions. re.compile is a command to search for
# the expression in question

# the logic for remove_URL is
# r->makes it so you can use backslashes literally rather than as the start
# of some special expression
# ? -> matches to preceding https or anything less (like http)
# S -> matches to any character that is not whitespace
# + ->matches to a repetition of anything preceding, so there can be multiple http's
# and www's

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


# for remove_html the logic is
# . ->matches anything
# * -> matches to repititions of anything preceding
# ? -> matches to anything preceding or anything less
# in short this finds and gets rid of anything like <...>

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emotion faces
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags 
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# the u means unicode.

def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    # this creates a map such that punctuation marks are mapped to blank spaces
    return text.translate(table)  # this applies it to the string table.
    # punctuation gets replaced by blank space
    # this also removes hashtag signs and @ signs


def only_letters(dataset):
    dclean = dataset.copy()
    dclean['text'] = dclean['text'].apply(lambda x: remove_html(x))
    dclean['text'] = dclean['text'].apply(lambda x: remove_URL(x))
    dclean['text'] = dclean['text'].apply(lambda x: remove_emoji(x))
    dclean['text'] = dclean['text'].apply(lambda x: remove_punct(x))
    # tokenize texts (split into words) - this gives structure to previously unstructured text
    dclean.text = dclean.text.apply(lambda x: word_tokenize(x))
    # convert words to lower case to normalize - this will shrink vocabulary without much loss in information
    dclean.text = dclean.text.apply(lambda x: [word.lower() for word in x])
    # remove remaining tokens that are not alphabetic
    dclean.text = dclean.text.apply(lambda x: [word for word in x if word.isalpha()])
    # remove stopwords (of, the, and etc.) (pretty useless in terms of classification)
    dclean.text = dclean.text.apply(lambda x: [word for word in x if not word in stop])
    # remove non-english words
    dclean.text = dclean.text.apply(lambda x: [word for word in x if word in words_en])
    # stem words
    porter = PorterStemmer()
    dclean.text = dclean.text.apply(lambda x: [porter.stem(word) for word in x])
    return dclean


# based on above, lets make a preprocessing function that vectorizer will use:
def preProcess(text):
    result = remove_html(text)
    result = remove_URL(result)
    result = remove_emoji(result)
    result = remove_punct(result)
    result = word_tokenize(result)
    result = [w.lower() for w in result]
    result = [w for w in result if w.isalpha()]
    result = [w for w in result if not w in stop]
    result = [w for w in result if w in words_en]
    porter = PorterStemmer()
    result = [porter.stem(w) for w in result]
    return result