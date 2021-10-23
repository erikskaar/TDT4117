import codecs
from itertools import groupby
import random
import string
import re

import gensim
from gensim.corpora import dictionary
from nltk.stem.porter import PorterStemmer

random.seed(123)

stemmer = PorterStemmer()

# This import may have to be changed to just 'pg3300.txt' depending on your setup
f = codecs.open("assignment_3/pg3300.txt", "r", "utf-8")
text = f.read()
f.close()

# # 1.2 splitting on double newline aka paragraphs
def splitter(lines):
    for separator, iteration in groupby(lines.splitlines(True), key=str.isspace):
        if not separator:
            yield ''.join(iteration)


# 1.3 remove all paragraphs containing gutenberg 
def make_paragraphs(file):
    paragraps = []
    for p in splitter(file):
        if 'Gutenberg'.casefold() not in p.casefold():
            paragraps.append(p)
    return paragraps

def get_clean_text(document):
    tokenized = []
    for d in document:
        tokenized.append(re.sub("[^\w]", " ", d).split())
    stemmed = []
    for d in tokenized:
        words_stemmed = []
        for word in d:
            words_stemmed.append(stemmer.stem(word).lower())
        stemmed.append(words_stemmed)
    return stemmed

'''
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
    f = codecs.open("assignment_3/pg3300.txt", "r", "utf-8")
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

    return p1_stemmed, paragraphs
'''

paragraphs = make_paragraphs(text)
paragraphs_fixed = paragraphs
paragraphs_fixed = get_clean_text(paragraphs_fixed)
#paragraphs_fixed, paragraphs = get_clean_text()

# task 2
def dict_building(paragraphs):
    # create dictionary
    dictionary = gensim.corpora.Dictionary(paragraphs)
    # generate stopword list
    with open("assignment_3/common-english-words.txt", "r") as stopword_file:
        stopwords = stopword_file.read().split(",")
    # add check if the stopword exists in the dictionary to ensure that it doesn't crash
    stopword_ids = [dictionary.token2id[x] for x in stopwords if x in dictionary.token2id]
    # filter dictionary to remove stopwords
    dictionary.filter_tokens(stopword_ids)
    # create bag of words of each document / paragraph
    bag_of_words = [dictionary.doc2bow(paragraph) for paragraph in paragraphs]
    return bag_of_words, dictionary

bow, dictionary = dict_building(paragraphs_fixed)


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
    print('---3.5 first 3 LSI topics---')
    print(lsi_model.show_topics(num_topics=3))
    return tfidf_model, tfidf_matrix_similiarity, lsi_model, lsi_corpus, lsi_matrix_similarity

tfidf_model, tfidf_matrix_similiarity, lsi_model, lsi_corpus, lsi_matrix_similarity = retrieval_models(bow, dictionary)


# task 4
# 4.1 apply tranformations

    
    
    
def transform(query: list):
    # 'query' shouldn't need to be a list but didn't have time to fix..
    
    # tokenize
    query_tokenized = [x.split(" ") for x in query][0]

    # remove punctuation
    table = str.maketrans(dict.fromkeys(string.punctuation))
    query_tokenized = [word.translate(table) for word in query_tokenized]
    
    # stem
    stemmer = PorterStemmer()
    query_stemmed = [stemmer.stem(word) for word in query_tokenized]
    return query_stemmed

query = ['How taxes influence Economics?']

query = transform(query)

# convert to bow
query = dictionary.doc2bow(query)
print(query)

# 4.2 Convert BOW to TF-IDF representation
tfidf_index = tfidf_model[query]

print('---4.2---')
for word in tfidf_index:
    word_index = word[0]
    word_weight = word[1]
    print("index", word_index, "| word:", dictionary.get(word_index, word_weight), "| weight:", word_weight)


# 4.3 
print('---4.3---')

doc2sim = enumerate(tfidf_matrix_similiarity[tfidf_index])
top_results = sorted(doc2sim, key=lambda x: x[1], reverse=True)[:3]
# printing the top 3 on the assignemnt example format
for result in top_results:
    doc = paragraphs[result[0]]
    doc = doc.split('\n')
    print("\n[Document %d]" % result[0])
    # printing 5 lines (or all lines available if the result is less than 5 lines)
    print_range = len(doc)
    for line in range(min(print_range, 5)):
        print(doc[line])
        

# 4.4 - part 1
print('\n---4.4---')
print('---4.4.1---')
lsi_query = lsi_model[query]
# sort according to assignment hint
topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
for topic in enumerate(topics):
    t = topic[1][0]
    print("\n[Topic %d]" % t)
    print(lsi_model.show_topics()[t])
    


print('---4.4.2---')
# find similar documents
lsi_doc2sim = enumerate(lsi_matrix_similarity[lsi_query])

# sort according to assignment hint
lsi_documents = sorted(lsi_doc2sim, key=lambda kv: -abs(kv[1]))[:3]
# printing the top 3 on the assignemnt example format
for result in lsi_documents:
    doc = paragraphs[result[0]]
    doc = doc.split('\n')
    print("\n[Document %d]" %result[0])
    for line in range(5):
        print(doc[line])