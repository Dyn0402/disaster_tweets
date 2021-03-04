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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# def vectorize(train, test, vec_pars={'binary': True}):
#     vec_binary = CountVectorizer(**vec_pars)
#     train_tokens = vec_binary.fit_transform(train)
#     test_tokens = vec_binary.transform(test)
#
#     return train_tokens, test_tokens
#
# train_vec, test_vec = train['target'], test['target']  # Create new train/test dataframes with targets. Keeps train/target from being modified.
# train_vec['features'], test_vec['features'] = vectorize(train['text'], test['text'])  # Vectorize text and add as 'features' column to dataframes
#
# def eval_vs_hyperpar(train, test, model_func, var_par={}, hyper_pars={}):
#     """
#     Train and evaluate model for each parameter value given in var_par. Plot each evaluation metric as a function of this parameter.
#     :param train: Training data
#     :param test: Test data
#     :param model_func: Function to fit model and generate evaluation metrics
#     :param var_par: Dictionary with hyperparameter name as key and list of parameter values as value. The model will be evaluated at each value of this list. Currently only impelemted for single dictionary entry.
#     :param hyper_pars: Additional hyper_parameters to pass to model. These remain static between evaluations of var_par values.
#     :return:
#     """
#     for var_val in list(var_par.values())[0]:
#         set_pars = hyper_pars.copy()
#         model_func(train, test, )
#
# def random_forest(train, test, hyper_pars={}):
#     vec_binary = CountVectorizer(binary=True)
#     train_tokens = vec_binary.fit_transform(train['text'])
#     test_tokens = vec_binary.transform(test['text'])
#     model_rf = RandomForestClassifier(**hyper_pars)
#     model_rf.fit(train_tokens, train['target'])
#     eval_metrics = eval_model(model_rf, test_tokens, test['target'], 'Random Forest')
#     return eval_metrics

#Kaggle uses F1_Score for evaluation. We may want to generalize to fbeta score in the scenario where this model is actually implemented.

beta = 2  # Weight
fbeta_scorer = make_scorer(fbeta_score, beta=beta)

def main():
    pars = init_pars()
    data = get_data(pars['path'] + pars['train_file'])
    # visualize(train)
    train, test, validation = split_data(data, pars['test_size'], pars['validation_size'], pars['split_seed'])
    train, test = clean_data(train, test)
    train_tokens, test_tokens = transform_data(train, test)
    models = model_data(train_tokens, train['target'])
    model_evals = {}
    for model_name, model in models.items():
        model_evals.update({model_name: eval_model(model, test_tokens, test['target'], model_name)})

    print(model_evals)

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
    model_rf = RandomForestClassifier()
    model_rf.fit(train_tokens, train_target)

    return {'Bernoulli Naive Bayes': model_bnb, 'Random Forest': model_rf}


def eval_model(model, test_data, test_target, name='Model'):
    """
    Evaluate model based on multiple metrics and print/plot results.
    :param model: Model to be used in predicting test data. Must have predict method that takes test_data and returns
    predictions to compare to test_target
    :param test_data: Test data to feed model in order to make predictions to compare to test_target
    :param test_target: Test target values to compare to model prediction on test_data
    :param name: Name of model being evaluated for plotting/printing purposes, defaults to 'model'
    :return: Dictionary of evaluation metrics and their values.
    """
    pred = model.predict(test_data)
    acc = accuracy_score(test_target, pred)
    f1 = f1_score(test_target, pred)

    print(f'{name} Performance:')
    print(f'Accuracy Score: {acc}')
    print(f'F1 Score: {f1}')

    cf = confusion_matrix(test_target, pred)
    ax = sns.heatmap(cf, annot=True)
    ax.set_title(f'{name} Confusion Matrix')
    ax.set_xlabel('Predicted Classification')
    ax.set_ylabel('True Classification')
    plt.show()

    return {'accuracy_score': acc, 'f1_score': f1}


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
