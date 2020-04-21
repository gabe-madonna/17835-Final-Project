from main import TwitterScraper, BIASES, merge_lists, gen_freq_dict
import spacy
import re
import matplotlib.pyplot as plt

NLP = spacy.load('en_core_web_sm', disable=['ner', 'parser'])


def parse_words(text):
    return remove_articles(clean_text(text))


def clean_text(text):
    return re.sub("[^A-Za-z']+", ' ', re.sub(r'http\S+', '', text.lower()))


def remove_articles(text):
    if len(text) > 100000:
        return merge_lists([remove_articles(text[i*100000:min(len(text), (i+1)*100000)]) for i in range(len(text)//100000)])
    else:
        return [token.lemma_ for token in NLP(text) if not token.is_stop and len(token.lemma_) > 1]


def filter_blacklist_words(d):
    black_list = {"rt", "de", "en", "amp", "que", "deo", "um", "la", "el", "una", }
    for w in black_list:
        if w in d:
            d.pop(w)

# plot 1: Similarity matrix


# plot 2: most similar words to trump bynetwork


# plot 3: most common words by network
def plot3():
    top_word_dict = {}
    tweet_dict = TwitterScraper.fetch_all_tweets(group_by="follows")
    networks = ["patribotics", "MSNBC", "NPR", "business", "OANN", "newswarz"]
    network_labels = ["Patribotics", "MSNBC", "NPR", "Bloomerbg", "Fox News", "OANN", "News Wars"]

    for network in networks:
        tweets = tweet_dict[network]
        full_text = " ".join([tweet["full_text"] for tweet in tweets])
        words = parse_words(full_text)
        word_freq = gen_freq_dict(words)
        filter_blacklist_words(word_freq)
        top_words = sorted(list(word_freq.keys()), key=lambda w: word_freq[w], reverse=True)[:10]
        if network in BIASES:
            top_word_dict[network] = top_words
        print("{} {}: {}".format(BIASES.get(network, "X"), network, top_words))

    # networks = sorted(list(top_word_dict.keys()), key=lambda network: BIASES[network])
    biases = [BIASES[n] for n in networks]
    labels = [str(biases[i]) + "\n" + "\n".join(top_word_dict[network][:5]) for i, network in enumerate(networks)]

    fig, ax = plt.subplots()
    # ax.bar(biases, [0]*len(networks), 0.9, label='Men')
    ax.set_xticks(biases)
    ax.set_xticklabels(labels)
    ax.yaxis.set_visible(False)
    ax.set_title("Most Popular Words By Network Bias")
    fig.tight_layout()
    plt.show()
    plt.save_fig("most_common_words.png")
    print()

# plot 4: similarity to patribotics and Fox


# plot 5: lienar regression


if __name__ == '__main__':
    plot3()
