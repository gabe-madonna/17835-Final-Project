import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pdb
from main import TwitterScraper
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler

BIASES = {"patribotics": -38, "Bipartisanism": -26, "fwdprogressives": -25, "HuffPost": -22, "MSNBC": -19,
          "washingtonpost": -10, "CNNPolitics": -7, "propublica": -5, "NPR": -5, "PBS": -5, "nytimes": -4, "ABC": 0,
          "business": 1, "CBSNews": 4, "Forbes": 5, "thehill": 10, "weeklystandard": 18, "TheTimesNUSA": 20,
          "amconmag": 27, "FoxNews": 27, "OANN": 28, "realDailyWire": 28, "BreitbartNews": 34, "newswarz": 38}

# plot 1: Similarity matrix

# networks = [(BIASES[name], name) for name in BIASES]
# networks.sort()
#
# models = [(Word2Vec.load(network + '.model'), Phraser.load(network + "_bigram_model.pkl"), network)
#           for _, network in networks]
#
# for i, network_1 in enumerate(models):
#
#     fig, ax = plt.subplots(figsize=(30, 30))
#     model, bigram, network = network_1
#     word_vecs = []
#     terms = model.wv.index2entity[:100]
#
#     for term in terms:
#
#         if term in model:
#
#             word_vecs.append(model[term])
#
#     try:
#         data = np.array(word_vecs)
#         pca = PCA(n_components=2)
#         new_data = pca.fit_transform(data)
#         plt.xlim(min(new_data[:, 0]) - 0.1, max(new_data[:, 0]) + 0.1)
#         plt.ylim(min(new_data[:, 1]) - 0.1, max(new_data[:, 1]) + 0.1)
#
#         for j, term in enumerate(terms):
#
#             if term in model:
#
#                plt.text(new_data[j, 0], new_data[j, 1], term, fontsize = 30)
#
#         plt.title(network + " Most Common Words", fontsize = 40)
#         plt.xlabel("Collapsed Dimension 1", fontsize = 35)
#         plt.ylabel("Collapsed Dimension 2", fontsize = 35)
#         plt.savefig(network + '.png')
#         plt.close()
#
#     except ValueError:
#         continue


# plot 2: most similar words to trump bynetwork

terms = ["pandemic", "realdonaldtrump"]

networks = [(BIASES[name], name) for name in BIASES]
networks.sort()

models = [(Word2Vec.load(network + '.model'), Phraser.load(network + "_bigram_model.pkl"), network)
          for _, network in networks]

bacon = [np.all(np.array([term in model for term in terms])) for model, _, _ in models]

models_relev = [model for i, model in enumerate(models) if bacon[i]]

fig, ax = plt.subplots(figsize=(40, 20))
plt.xlabel("Networks", fontsize = 30)
plt.xlim(-1, len(models_relev))
plt.ylabel("Terms", fontsize = 30)
plt.ylim(0, 0.5)
bias_used = set()
N = 10

for i, network_1 in enumerate(models_relev):

    model_1, bigram_1, name = network_1
    print(name)
    similar_words = model_1.most_similar(positive=terms, topn=N)

    for j, (word, _) in enumerate(similar_words):

        plt.text(i, 0.5*(j + 0.5)/N, word, fontsize = 15)
    plt.xticks([i for i in range(len(models_relev))],
                [str(BIASES[network]) for model, bigram, network in models_relev],
               fontsize = 25)

plt.savefig("topwords.png")
plt.close()

# plot 3: most common words by network


# plot 4: similarity to patribotics and Fox

# liberal = 'patribotics'
# conserv = 'OANN'
#
# model_google = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
# model_patribotics = Word2Vec.load(liberal + '.model')
# model_oann = Word2Vec.load(conserv + '.model')
#
#
# hf, ha = plt.subplots(1, 1)
# plt.xlabel(conserv)
# plt.xlim(-1, 1)
# plt.ylabel(liberal)
# plt.ylim(-1, 1)
#
# terms = ['democrats', 'republicans', 'pelosi',
#          'coronavirus', 'liberal', 'guns', 'climate',
#          'president']
#
# for term in terms:
#
#     try:
#
#         x = cosine_similarity(model_google[term].reshape((1, 300)), model_oann[term].reshape((1, 300)))
#         y = cosine_similarity(model_google[term].reshape((1, 300)), model_patribotics[term].reshape((1, 300)))
#
#         print((x, y, term))
#
#         plt.text(x, y, term)
#
#     except KeyError:
#
#         continue
#
# plt.savefig('terms.png')

# plot 5: lienar regression

