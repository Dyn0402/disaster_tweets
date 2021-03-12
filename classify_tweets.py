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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, make_scorer
from sklearn.pipeline import Pipeline

from clean_tweets import preProcess, testlam

from IPython.display import display


def main():
    pars = init_pars()
    data = get_data(pars['path'] + pars['train_file'])
    # visualize(train)
    train, test = train, test = train_test_split(data, test_size=pars['test_size'], random_state=pars['split_seed'])
    # train, test = clean_data(train, test)

    beta = 2  # beta = 1 equivalent to f1 score
    scorers = {'Accuracy Score': accuracy_score, 'F1 Score': f1_score,
               f'F{beta} Score': lambda x, y: fbeta_score(x, y, beta=beta)}

    fbeta_scorer = make_scorer(fbeta_score,
                               beta=beta)  # fbeta is genaeralized f1 where beta is the weight of precision relative to recall
    gs_scorers = {'Accuracy Score': make_scorer(accuracy_score), 'F1 Score': make_scorer(f1_score),
                  f'F{beta} Score': fbeta_scorer}  # Use these for grid searches

    count_vec_1 = CountVectorizer(tokenizer=testlam,
                                  preprocessor=preProcess,
                                  analyzer='word',
                                  ngram_range=(1, 2))
    # count_vec_1 = CountVectorizer(tokenizer=(lambda x: x),
    #                               analyzer='word',
    #                               ngram_range=(1, 2))
    # for txt in train.text:
    #     print(type(txt), txt)
    X_train_countvec = count_vec_1.fit_transform(train.text)

    MultiNB2 = MultinomialNB()
    param = {'alpha': np.arange(0.01, 2.5, .01)}
    gs2 = GridSearchCV(MultiNB2, param, scoring=gs_scorers['F1 Score'], cv=5, n_jobs=-1, verbose=1)
    gs2_fit = gs2.fit(X_train_countvec, train.target)
    gs2_res = gs2_fit.cv_results_
    display(pd.DataFrame(gs2_res).sort_values('mean_test_score', ascending=False).head())

    pipe = Pipeline([('cvec', count_vec_1), ('mnb', MultinomialNB())], memory=pars['path'][:-1])
    param = dict(mnb__alpha=np.arange(0.01, 2.5, 1.3))
    gspipe = GridSearchCV(pipe, param, scoring=gs_scorers['F1 Score'], cv=5, n_jobs=-1, verbose=1)
    gspipe_fit = gspipe.fit(train.text, train.target)
    gspipe_res = gspipe_fit.cv_results_
    display(pd.DataFrame(gspipe_res).sort_values('mean_test_score', ascending=False).head())

    # train_tokens, test_tokens = transform_data(train, test)
    # models = model_data(train_tokens, train['target'])
    # model_evals = {}
    # for model_name, model in models.items():
    #     model_evals.update({model_name: eval_model(model, test_tokens, test['target'], model_name)})

    # print(model_evals)

    print('donzo')


def init_pars():
    pars = {'path': 'C:/Users/Dylan/Desktop/disaster_tweets/',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'test_size': 0.2,
            'split_seed': 42}

    return pars


def visualize(data):
    hist_char_len(data, False)
    hist_word_len(data, False)
    plt.show()


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
