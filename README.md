# 2020_elections

## work in progress

Data is taken from https://www.kaggle.com/manchunhui/us-election-2020-tweets under CC0 license

The algorithm has been implemented from scratch following tutorials in https://www.coursera.org/learn/classification-vector-spaces-in-nlp

Naive Bayes applied to tweets in order to find the most impactful words

Code is divided in 3 parts. In the first one, a sentiment analysis-like Naive Bayes algorithm has been trained on part of the data, in order to separate tweets referring to Biden to ones referring to Trump. Notice that names have been removed from the tweets and the corpus contains both positive and negative tweets.
Even if the accuracy is not great, it is possible to evaluate the posteriors of the words and find the highest (and lowest) ones.

In the second part, a sentiment analysis NB has been trained on the standard twitter dataset and then used to perform sentiment analysis on the political corpus. Tweets referring to Biden seem slightly more positive.

Using the sentiment analysis of part 2, tweets referring to both candidates have been split into positive and negative and then I tried to infer who each tweet in the test set is referring to. Not much improvement unfortunately. The idea behind this last attemp was that maybe some words are present in positive tweets about Trump and negative about Biden, for instance "maga".

In the future, we may try a more complex model, but the point of the analysis was exactly to find relations between single words.

Written with Francesco Marengo
