#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 27 7:20 PM 2021
Created in PyCharm
Created as disaster_tweets/classify_tweets

@author: Dylan Neff, Dylan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score


def main():
    pars = init_pars()
    data = get_data(pars['path'] + pars['train_file'])
    # visualize(train)
    train, test, validation = split_data(data, pars['test_size'], pars['validation_size'], pars['split_seed'])
    train, test = clean_data(train, test)
    train_tokens, test_tokens = transform_data(train, test)
    model = model_data(train_tokens, train['target'])
    model_eval = eval_model(model, test_tokens, test['target'])

    print(model_eval)

    print('donzo')


def init_pars():
    pars = {'path': 'C:/Users/Dylan/Desktop/disaster_tweets/',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'test_size': 0.2,
            'validation_size': 0.2,
            'split_seed': 34}

    return pars


def visualize(data):
    hist_char_len(data, False)
    hist_word_len(data, False)
    plt.show()


def split_data(data, test_size, validation_size, seed):
    """
    Split original data into train, test, and validation sets.
    :param data: Original data to split
    :param test_size: Size of test set as percentage of original data set
    :param validation_size: Size of validation set as percentage of original data set
    :param seed: Seed to use when splitting data
    :return:
    """
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    train, validation = train_test_split(data, test_size=validation_size / (1 - test_size), random_state=seed)

    return train, test, validation


def clean_data(train, test):
    return train, test


def transform_data(train, test):
    vec_binary = CountVectorizer(binary=True)
    train_tokens = vec_binary.fit_transform(train['text'])
    test_tokens = vec_binary.transform(test['text'])
    # print(train_tokens)
    # train['tokens'] = train_tokens
    # test['tokens'] = test_tokens
    # input()

    return train_tokens, test_tokens


def model_data(train_tokens, train_target):
    model_bnb = BernoulliNB()
    # print([(row['target'], row['tokens']) for i, row in train.iterrows()])
    # print(train['tokens'])
    # print(train['tokens'].shape, train['target'].shape)
    model_bnb.fit(train_tokens, train_target)

    return model_bnb


def eval_model(model, test_tokens, test_target):
    pred = model.predict(test_tokens)
    acc_score = accuracy_score(test_target, pred)

    return acc_score


def get_data(path):
    data = pd.read_csv(path)

    return data


def hist_char_len(data, show=False):
    """
    Histogram the number of characters in each tweet.
    :param data: Twitter data dataframe or dictionary with 'text' key containing list of tweets
    :param show: True to show plot when function is called, false to postpone plt.show() call
    :return:
    """
    tweet_chars = [len(x) for x in data['text']]
    fig, ax = plt.subplots()
    ax = sns.histplot(tweet_chars, ax=ax, discrete=True)
    ax.set_xlabel('Tweet Character Length')
    if show:
        plt.show()


def hist_word_len(data, show=False):
    """
    Histogram the number of words in each tweet, defined as character groups separated by a space.
    :param data: Twitter data dataframe or dictionary with 'text' key containing list of tweets
    :param show: True to show plot when function is called, false to postpone plt.show() call
    :return:
    """
    tweet_words = [len(x.split(' ')) for x in data['text']]
    fig, ax = plt.subplots()
    ax = sns.histplot(tweet_words, ax=ax, discrete=True)
    ax.set_xlabel('Tweet Word Length')
    if show:
        plt.show()


if __name__ == '__main__':
    main()
