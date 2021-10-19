import codecs
import random
import string

import gensim
from nltk.stem.porter import PorterStemmer

random.seed(123)


def remove_backslash(paragraph):
    for item in paragraph:
        if "\r\n" in item:
            a, b, c = item.partition("\r\n")
            yield a
            yield c
        else:
            yield item


# task 1
def get_clean_text():
    f = codecs.open("pg3300.txt", "r", "utf-8")
    text = f.read()
    f.close()

    # 1.2 splitting on double newline aka paragraphs
    paragraphs = text.split("\r\n\r\n")

    # remove whitespace
    p1 = [line.strip() for line in paragraphs]
    # remove empty items
    p1 = list(filter(lambda x: (x != ""), p1))
    # 1.3 remove all paragraphs containing gutenberg
    p1 = list(filter(lambda x: ("gutenberg" not in x.lower()), p1))
    # lowercase
    p1_lowercase = [x.lower() for x in p1]
    # 1.4 tokenize
    p1_tokenized = [x.split(" ") for x in p1_lowercase]
    # 1.5 remove punctuation
    table = str.maketrans(dict.fromkeys(string.punctuation))
    p1_tokenized = [[word.translate(table) for word in paragraph] for paragraph in p1_tokenized]
    # remove backslash characters
    p1_tokenized = [list(remove_backslash(paragraph)) for paragraph in p1_tokenized]
    # 1.6
    stemmer = PorterStemmer()
    p1_stemmed = [[stemmer.stem(word) for word in paragraph] for paragraph in p1_tokenized]

    return p1_stemmed


# task 2
def dict_building(paragraphs):
    # create dictionary
    dictionary = gensim.corpora.Dictionary(paragraphs)
    # generate stopword list
    with open("common-english-words.txt", "r") as stopword_file:
        stopwords = stopword_file.read().split(",")
    # add check if the stopword exists in the dictionary to ensure that it doesn't crash
    stopword_ids = [dictionary.token2id[x] for x in stopwords if x in dictionary.token2id]
    # filter dictionary to remove stopwords
    dictionary.filter_tokens(stopword_ids)
    # create bag of words of each document / paragraph
    bag_of_words = [dictionary.doc2bow(paragraph) for paragraph in paragraphs]
    return bag_of_words, dictionary


# task 3
def retrieval_models(bag_of_words, dictionary):
    # 3.1 create model
    tfidf_model = gensim.models.TfidfModel(corpus=bag_of_words)
    # 3.2 create weights
    tfidf_corpus = tfidf_model[bag_of_words]
    # 3.3 create matrix similarity
    tfidf_matrix_similiarity = gensim.similarities.MatrixSimilarity(tfidf_corpus)
    # 3.4 create LSI-model
    lsi_model = gensim.models.LsiModel(corpus=bag_of_words, id2word=dictionary, num_topics=100)
    lsi_corpus = lsi_model[bag_of_words]
    lsi_matrix_similarity = gensim.similarities.MatrixSimilarity(lsi_corpus)
    print(lsi_model.show_topics())
    return


if __name__ == '__main__':
    pgs = get_clean_text()
    bow, filtered_dict = dict_building(pgs)
    retrieval_models(bow, filtered_dict)
