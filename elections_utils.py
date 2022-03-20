import nltk
import pandas as pd
import numpy as np
import preprocessor as pre
import matplotlib.pyplot as plt
import random
random.seed(10) #it will be useful to have the same dictionary every time, in order to write it only once
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'Trump', '', tweet)
    tweet = re.sub(r'trump', '', tweet)
    tweet = re.sub(r'TRUMP', '', tweet)
    tweet = re.sub(r'Donald', '', tweet)
    tweet = re.sub(r'donald', '', tweet)
    tweet = re.sub(r'DONALD', '', tweet)
    tweet = re.sub(r'Biden', '', tweet)
    tweet = re.sub(r'biden', '', tweet)
    tweet = re.sub(r'BIDEN', '', tweet)
    tweet = re.sub(r'Joe', '', tweet)
    tweet = re.sub(r'joe', '', tweet)
    tweet = re.sub(r'JOE', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)
    return tweets_clean


def count_tweets(result, tweets, ys):
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word,y)
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1
            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
                #only for debugging:
                #print(word)
    return result


def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    vocab = []
    for a in freqs:
        vocab.append(a[0])
    vocab=list(dict.fromkeys(vocab))
    V = len(vocab)
    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:
            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]
        # else, the label is negative
        else:
            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]
    # Calculate D, the number of documents
    D = len(train_y)
    # Calculate D_pos, the number of positive documents
    D_pos = (train_y==1).sum()
    # Calculate D_neg, the number of negative documents
    D_neg = (train_y==0).sum()
    # Calculate logprior
    logprior = np.log(float(D_pos / D_neg))
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        if (word,1.0) in freqs:
            freq_pos = freqs[(word,1.0)]
        else:
            freq_pos=0
        if (word,0.0) in freqs:
            freq_neg = freqs[(word,0.0)]
        else:
            freq_neg=0
        # calculate the probability that each word is positive, and negative
        p_w_pos = float(freq_pos+1)/(N_pos+V)
        p_w_neg = float(freq_neg+1)/(N_neg+V)
        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    return logprior, loglikelihood


def naive_bayes_predict(tweet, logprior, loglikelihood):
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)
    # initialize probability to zero
    p = 0
    # add the logprior
    p += logprior
    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]
    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    accuracy = 0
    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0
        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)
    # error is the average of the absolute values of the differences between y_hats and test_y
    error = 0
    for i in range(len(y_hats)):
        if y_hats[i] != test_y[i]:
            error += 1
    error = error / len(y_hats)
    # Accuracy is 1 minus the error
    accuracy = 1 - error
    return accuracy


def average_log_posterior(tweets, logprior, loglikelihood):
    average=0
    for tweet in tweets:
        average+=naive_bayes_predict(tweet,logprior,loglikelihood)
    average=average/len(tweets)
    return average


def average_posterior(tweets, logprior, loglikelihood):
    average=0
    for tweet in tweets:
        a=np.exp(naive_bayes_predict(tweet,logprior,loglikelihood))
        average+=a
        print(a)
    average=average/len(tweets)
    return average