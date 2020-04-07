from main import TwitterScraper
from gensim.models import KeyedVectors
import re
import numpy as np
from scipy import signal
import pdb
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import os

import matplotlib.cm as cm


class TweetEncoder:

    def __init__(self, encoding):

        '''
        :param encoding: can take one of two values, either 'screen_name' or 'follows'
                         'screen_name' sorts by screen_names (does not work yet)
                         'follows' sorts by who they follow
        '''

        self.scraper = TwitterScraper()
        self.all_tweets = self.scraper.fetch_all_tweets(group_by = encoding)

    def get_tweets(self, screen_name):

        """
        :param screen_name: screen_name of the twitter account
        :return: returns tweets either from a screen_name or following a screen_name
        """

        return self.all_tweets[screen_name]

    def encode_tweets(self, model, screen_name):

        """
        :param model: word2vec model being used to encode each tweet
        :param screen_name: screen_name of relevant twitter account
        :return: saves word2vec for each word in the tweets as numpy array
        """

        user_tweets = self.get_tweets(screen_name)
        tweets = []
        tweet_sizes = []

        for i in range(len(user_tweets)):

            try:

                ex = user_tweets[i]['full_text'].lower()

            except KeyError:

                print('This tweet is truncated' + str(i))

            ex = re.sub('[,\.!?:/…]', '', ex).split()
            ex = [word for word in ex if len(word) > 0 and word in model]

            if len(ex) > 0:

                tweet_vec = np.column_stack(tuple([model[word] for word in ex]))
                tweets.append(tweet_vec)
                tweet_sizes.append(tweet_vec.shape[1])

            else:

                print(ex)

        np.save("./Data/" + screen_name + "/tweets.npy", np.hstack(tuple(tweets)))
        np.save("./Data/" + screen_name + "/tweet_sizes.npy", np.array(tweet_sizes))
        return tweets, tweet_sizes

    def average_tweet_vectors(self, screen_name):

        """
        :param screen_name: screen_name of relevant twitter account
        :return: saves average of each of words in word2vec format for each tweet
        """

        tweets = np.load("./Data/" + screen_name + "/tweets.npy")
        tweet_sizes = np.load("./Data/" + screen_name + "/tweet_sizes.npy")

        tweet_vecs = np.zeros((tweets.shape[0], len(tweet_sizes)))
        start = 0

        for i in range(len(tweet_sizes)):

            tweet_vecs[:, i] = np.mean(tweets[:, start: start + tweet_sizes[i]], axis = 1)
            start += tweet_sizes[i]

        np.save("./Data/" + screen_name + "/tweet_vector_mean.npy", tweet_vecs)
        return tweet_vecs

    def conv_tweet_vectors(self, screen_name):

        """
        :param screen_name: screen_name of relevant twitter account
        :return: saves average of convolution of each of words in word2vec format for each tweet
        """

        tweets = np.load("./Data/" + screen_name + "/tweets.npy")
        tweet_sizes = np.load("./Data/" + screen_name + "/tweet_sizes.npy")

        tweet_vecs = np.zeros((tweets.shape[0], len(tweet_sizes)))
        start = 0

        for i in range(len(tweet_sizes)):

            tweet = tweets[:, start: start + tweet_sizes[i]]

            try:
                tweet = signal.convolve2d(tweet, np.ones((1, 3))/3, mode = 'valid')

            except ValueError:
                pass

            tweet = np.mean(tweet, axis=1)

            tweet_vecs[:, i] = tweet.reshape((tweet.shape[0]))
            start += tweet_sizes[i]

        np.save("./Data/" + screen_name + "/tweet_vector_conv.npy", tweet_vecs)
        return tweet_vecs

    def prune_tweets(self, screen_name, data, terms):

        """
        :param screen_name: screen_name of relevant twitter account
        :param data: word2vec data of the tweets to be compared
        :param terms: relevant terms in word2vec format to compare to other tweets
        :return: tweets that are considered relevant to the other terms
        """

        sub = 0
        pruned = []

        for t in range(data.shape[1]):

            if len(re.sub('[,\.!?:/…]', '', self.get_tweets(screen_name)[t - sub]["full_text"]).split()) < 2:
                sub += 1
                break

            vals = [(np.dot(data[:, t - sub], term)/(np.linalg.norm(term) * np.linalg.norm(data[:, t - sub])))
                    for term in terms]

            if np.mean(vals) > 0.34 and np.all(np.array(vals) > 0.3):

                pruned.append((self.get_tweets(screen_name)[t - sub]["full_text"], vals))

        return pruned


class Clustering:

    def __init__(self, model_type, data, group_sizes, labels):

        """
        :param model_type: clustering model being used
        :param data: word2vec data of all te tweets
        :param group_sizes: size of each of the groups
        :param labels: name for each group
        """

        self.model_type = model_type
        self.data = data
        self.group_sizes = group_sizes
        self.group_labels = labels

    def evaluate_models(self, max_num):

        """
        :param max_num: max number of clusters to explore
        :return: saves figure of the metric for each cluster number
        """

        tot = []
        tot1 = []
        tot2 = []

        for num_clusters in range(2, max_num):

            self.model = self.model_type(num_clusters, max_iter = 500)
            self.model.fit(self.data)
            tot.append(self.model.inertia_)
            # tot1.append(self.model.aic(self.data))
            # tot2.append(self.model.bic(self.data))

        plt.scatter(range(2, max_num), tot)
        # plt.scatter(range(2, max_num), tot1)
        # plt.scatter(range(2, max_num), tot2)
        # plt.legend(["AIC", "BIC"])

        plt.savefig("./ElbowMethod")
        plt.close()
        return None

    def silhouette_eval(self, max_num):

        """
        :param max_num: max number of clusters to explore
        :return: saves figures of silhouette score for each cluster number
        """

        for n_clusters in range(2, max_num):

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            ax1.set_xlim([-0.5, 1])

            ax1.set_ylim([0, len(self.data) + (n_clusters + 1) * 10])

            clusterer = self.model_type(n_clusters)
            cluster_labels = clusterer.fit_predict(self.data)
            silhouette_avg = silhouette_score(self.data, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(self.data, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.45, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            pca = PCA(n_components = 2)
            ax2.scatter(pca.fit_transform(self.data)[:, 0], pca.fit_transform(self.data)[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

            plt.savefig("./Silhouettes/silhouette" + str(n_clusters) + ".png")
            plt.close()

    def set_cluster_num(self, n):

        """
        :param n: number of clusters for a model
        :return: fits model to the data
        """

        self.model = self.model_type(n, max_iter = 500)
        self.model.fit(self.data)
        return None

    def evaluate_cluster(self):

        """
        :return: saves the figure placing each group in a cluster and plots as a histogram
        """

        prediction = self.model.predict(self.data)

        plt.hist([prediction[sum(list([0] + self.group_sizes)[:i]):sum(list([0] + self.group_sizes)[:i + 1])] for i in range(1, len(self.group_sizes) + 1)])

        plt.legend(self.group_labels)
        plt.savefig("./individuals.png")
        plt.close()

        return None


if __name__ == '__main__':

    screen_names = ['NPR', 'MSNBC', 'CNNPolitics', 'BostonGlobe', 'nytimes', 'FoxNews', 'foxandfriends']
    data = []
    data_sizes = []
    # model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

    # np.save("./Data/health.npy", model['health'])
    # np.save("./Data/care.npy", model['care'])
    # #
    # word1 = np.load("./Data/health.npy")
    # word2 = np.load("./Data/care.npy")

    # enc = TweetEncoder("follows")
    N = 180#3150

    for screen_name in screen_names:

        print(screen_name + " is being Vectorized.")
        # try:
        #     os.mkdir("./Data/" + screen_name + "/")
        # except FileExistsError:
        #     pass
        # enc.encode_tweets(model, screen_name)
        # enc.average_tweet_vectors(screen_name)
        # enc.conv_tweet_vectors(screen_name)
        dat = np.load("./Data/" + screen_name + "/tweet_vector_conv.npy")
        data.append(dat[:, np.random.choice(dat.shape[1], N, replace=False)])
        data_sizes.append(data[-1].shape[1])
        # print(enc.prune_tweets(screen_name, data[-1], [word1, word2]))

    data_collected = np.hstack(tuple(data))

    clus = Clustering(KMeans, normalize(data_collected.T), data_sizes, screen_names)
    clus.evaluate_models(30)
    # clus.silhouette_eval(30)

    clus.set_cluster_num(7)
    clus.evaluate_cluster()
