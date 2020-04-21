import matplotlib
matplotlib.use('Agg')
from main import TwitterScraper
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import Phrases, Phraser
import re
import numpy as np
from scipy import signal
import pdb
from sklearn.cluster import KMeans
import spacy
import multiprocessing
from collections import defaultdict
import pandas as pd
import random
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

from gensim.models import Word2Vec
from fse.models.base_s2v import BaseSentence2VecModel

from sklearn import manifold

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import os

import matplotlib.cm as cm
from langdetect import detect

REAL = np.float32


def sif_embeddings(sentences, model, vocab_freq, alpha=1e-3):
    """Compute the SIF embeddings for a list of sentences
    Parameters
    ----------
    sentences : list
        The sentences to compute the embeddings for
    model : `~gensim.models.base_any2vec.BaseAny2VecModel`
        A gensim model that contains the word vectors and the vocabulary
    alpha : float, optional
        Parameter which is used to weigh each individual word based on its probability p(w).
    Returns
    -------
    numpy.ndarray
        SIF sentence embedding matrix of dim len(sentences) * dimension
    """

    vlookup = vocab_freq  # Gives us access to word index and count
    vectors = model  # Gives us access to word vectors
    size = model.vector_size  # Embedding size

    Z = sum(vlookup.values())

    output = []

    # Iterate all sentences
    for s in sentences:
        v = np.zeros(size, dtype=REAL)  # Summary vector
        # Iterate all words
        count = 0
        for w in s.split():
            # A word must be present in the vocabulary
            if w in vectors and w in vlookup:
                v += (alpha/(alpha + (vlookup[w] / Z))) * vectors[w]
                count += 1
        if count > 0:
            v = v/count
            output.append(v)
    return np.column_stack(tuple(output)).astype(REAL)

class ModelCreator:

    def __init__(self, encoding):

        '''
        :param encoding: can take one of two values, either 'screen_name' or 'follows'
                         'screen_name' sorts by screen_names (does not work yet)
                         'follows' sorts by who they follow
        '''

        self.scraper = TwitterScraper()
        self.all_tweets = self.scraper.fetch_all_tweets(group_by = encoding)
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    def get_tweets(self, bias):

        try:

            return self.all_tweets[bias]

        except KeyError:

            return None

    def cleaning(self, doc):
        # Lemmatizes and removes stopwords
        # doc needs to be a spacy Doc object

        txt = [token.lemma_ for token in self.nlp(doc) if not token.is_stop and len(token.lemma_) > 1]
        # Word2Vec uses context words to learn the vector representation of a target word,
        # if a sentence is only one or two words long,
        # the benefit for the training is very small
        if len(txt) > 2:
            return ' '.join(txt)

    def clean_tweets(self, biases):

        tweets = []

        N = 2000

        for bias in biases:

            user_tweets = self.get_tweets(bias)

            if user_tweets is None:

                continue


            choices = np.random.choice(len(user_tweets), N, replace=False)

            for i in range(len(user_tweets)):

                if i not in choices:

                    continue

                ex = user_tweets[i]['full_text'].lower()

                clean_tweet = self.cleaning(re.sub("[^A-Za-z']+", ' ', re.sub(r'http\S+', '', ex)))

                if clean_tweet is not None:

                    if detect(clean_tweet) == 'en':

                        tweets.append(clean_tweet.split())

        if len(tweets) < 200:

            return None

        phrases = Phrases(tweets, min_count=30, progress_per=10000)
        bigram = Phraser(phrases)
        tweets = bigram[tweets]
        word_freq = defaultdict(int)
        for sent in tweets:
            for i in sent:
                word_freq[i] += 1
        print(len(word_freq))
        print(sorted(word_freq, key=word_freq.get, reverse=True)[:100])
        cores = multiprocessing.cpu_count()
        w2v_model = Word2Vec(min_count=20,
                             window=5,
                             size=300,
                             sample=6e-5,
                             alpha=0.03,
                             min_alpha=0.0007,
                             negative=20,
                             workers=cores - 1)
        w2v_model.build_vocab(tweets, progress_per=10000)
        w2v_model.train(tweets, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        w2v_model.init_sims(replace=True)

        if len(biases) == 1:
            w2v_model.save(biases[0] + ".model")
            bigram.save(biases[0] + "_bigram_model.pkl")
        else:
            w2v_model.save("all_relev.model")
            bigram.save("all_bigram_relev_model.pkl")

        return w2v_model, bigram

class TweetEncoder:

    def __init__(self, encoding):

        '''
        :param encoding: can take one of two values, either 'screen_name' or 'follows'
                         'screen_name' sorts by screen_names (does not work yet)
                         'follows' sorts by who they follow
        '''

        self.scraper = TwitterScraper()
        self.all_tweets = self.scraper.fetch_all_tweets(group_by = encoding)
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

    def cleaning(self, doc):
        # Lemmatizes and removes stopwords
        # doc needs to be a spacy Doc object

        txt = [token.lemma_ for token in self.nlp(doc) if not token.is_stop and len(token.lemma_) > 1]
        # Word2Vec uses context words to learn the vector representation of a target word,
        # if a sentence is only one or two words long,
        # the benefit for the training is very small
        if len(txt) > 2:
            return ' '.join(txt)

    def get_tweets(self, bias):

        """
        :param screen_name: screen_name of the twitter account
        :return: returns tweets either from a screen_name or following a screen_name
        """

        return self.all_tweets[bias]

    def encode_tweets(self, bias, model, bigram = None, relev_terms = None):

        """
        :param model: word2vec model being used to encode each tweet
        :param screen_name: screen_name of relevant twitter account
        :return: saves word2vec for each word in the tweets as numpy array
        """

        user_tweets = self.get_tweets(bias)
        tweets = []

        for i in range(len(user_tweets)):

            ex = user_tweets[i]['full_text'].lower()

            if relev_terms is not None:

                bacon = False

                for relev_term in relev_terms:

                    if relev_term in ex:

                        bacon = True
                        break
            if not bacon:

                continue

            clean_tweet = self.cleaning(re.sub("[^A-Za-z']+", ' ', re.sub(r'http\S+', '', ex)))

            if clean_tweet is not None:

                tweets.append(clean_tweet)

        if bigram:
            tweets = bigram[tweets]

        try:
            cv = CountVectorizer(tweets, min_df = 30)
            cv.fit_transform(tweets)
            freq = cv.vocabulary_
            np.save("./Data/" + str(bias) + "/tweet_vecs_relev.npy", sif_embeddings(tweets, model, freq))

        except ValueError:
            return None

if __name__ == '__main__':

    BIASES = {"patribotics": -38, "Bipartisanism": -26, "fwdprogressives": -25, "HuffPost": -22, "MSNBC": -19,
              "washingtonpost": -10, "CNNPolitics": -8, "CNN": -7, "propublica": -5, "NPR": -5, "PBS": -5, "nytimes": -4, "ABC": 0,
              "business": 1, "CBSNews": 4, "Forbes": 5, "thehill": 10, "weeklystandard": 18, "TheTimesNUSA": 20,
              "amconmag": 27, "FoxNews": 27, "foxandfriends": 28, "OANN": 28, "realDailyWire": 28, "BreitbartNews": 34, "newswarz": 38}
    screen_names = list(BIASES.keys()) #+ ["foxandfriends", "CNNPolitics"]

    # model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    #

    relev_terms = ["coronavirus", "covid",
                   "trump", "wh", "press",
                   "media"]

    # model_c = ModelCreator("follows")
    # model_c.clean_tweets(screen_names)

    # for screen_name in screen_names:
    #
    #     model_c.clean_tweets([screen_name])

    # model = Word2Vec.load('all_relev.model')
    # bigram = Phraser.load("all_bigram_relev_model.pkl")
    # enc = TweetEncoder('follows')
    N = 0
    data = []
    regression_vals = []
    names = []
    #
    for bias in screen_names:

        print(str(bias) + " is being Vectorized.")

        names.append(str(bias))
        try:
            os.mkdir("./Data/" + str(bias) + "/")
        except FileExistsError:
            pass
        # enc.encode_tweets(bias, model, bigram, relev_terms)
        try:
            dat = np.load("./Data/" + str(bias) + "/tweet_vecs_relev.npy")
        except FileNotFoundError:
            continue

        if dat is not None:
            if dat.shape[1] >= N:
                data.append(dat)
                # data.append(dat[:, np.random.choice(dat.shape[1], N, replace=False)])
                regression_vals += [BIASES[bias]] * data[-1].shape[1]
                print(data[-1].shape)

    print(names)

    data_collected = np.hstack(tuple(data)).T
    trues = ~np.any(np.isnan(data_collected), axis = 1)
    data_collected = data_collected[trues, :]
    regression_vals = np.array(regression_vals)[trues] #+ np.random.normal(0, 5, data_collected.shape[0])

    scaler = StandardScaler()
    scaler.fit(data_collected)
    X = scaler.transform(data_collected)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, regression_vals, test_size=0.2)

    ridge = LinearRegression()
    ridge.fit(X_train, y_train)
    print(ridge.score(X_test, y_test))
    visualizer = ResidualsPlot(ridge)
    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    plt.xlabel("Predicted Bias Value", fontsize = 15)
    plt.ylabel("Residual Bias Value", fontsize = 15)
    plt.savefig('Linear_residuals.png')
    plt.close()

    plt.scatter(regression_vals, ridge.predict(X))
    plt.savefig('Linear_plot.png')
