import math

from data import split_text, corpus, total_word_cat, distinct_words_cat, informative_words, count_values_from_list, \
    category_length, total_docs, total_word_corpus, distinct_words_corpus

sigma = 0.5


# implementation of unigram model
def unigramModel(document, category_i):

    # p(Ci) probability of category
    probability = math.log(category_length[category_i]/ total_docs, 10)

    words = split_text(document)

    for word in words:

        probability += getProb(word, category_i)

    return probability


# implementation of unigram model by using informative_words
def unigramModel_with_informative_words(document, category_i):

    # p(Ci) probability of category
    probability = math.log(category_length[category_i]/ total_docs, 10)

    words = split_text(document)

    for word in words:

        if (word in informative_words):
            probability += getProb(word, category_i)


    return probability


# this func returns probability of given word in corpus which is used in smoothing
def probInCorpus(word_i):

    count = 1

    for cat in corpus:
        if word_i in corpus[cat]:
            count += len(corpus[cat][word_i])

    return (count/total_word_corpus)


# returns unigram probability of given word in given category
def getProb(word_i, category_i):

    word_count = 0

    if(word_i in corpus[category_i]):
        word_count = len(corpus[category_i][word_i])

    total_words = total_word_cat[category_i]

    prob =  ((max(word_count-sigma, 0)) / total_words) +\
            ((sigma * (distinct_words_corpus) / total_word_corpus) / total_word_corpus)

    prob = math.log(prob, 10)

    return prob


# implementation of bigram model
def bigramModel(document, category_i):

    # p(Ci) probability of category
    probability = math.log(category_length[category_i]/ total_docs, 10)

    words = split_text(document)

    size = len(words)

    probability += getProb(words[0], category_i)

    for i in range (1, size):

        prob = getBigramProb(words[i], words[i-1], category_i)

        probability += prob


    return probability


# returns bigram probability of given words in given category
def getBigramProb(word_i2, word_i1, category_i):

    total_count = 0

    bigram = 0

    prob = 0

    Beta = 0

    p_BG = probInCorpus(word_i2)

    if(word_i1 in corpus[category_i]):

        total_count = len(corpus[category_i][word_i1])

        bigram = count_values_from_list(corpus[category_i][word_i1], word_i2)

    if(total_count > 0):

        prob = (max((bigram - sigma), 0) / total_count)

    for cat in corpus:

        if(word_i1 in corpus[cat]):

            Beta += len(set(corpus[cat][word_i1]))

    prob += ((sigma * Beta) / total_word_corpus) * p_BG

    if(prob>0):
        prob = math.log(prob, 10)


    return prob


#  classifies the given document with bigram model
def classify_bigram(doc):

    prob = 0
    category_ = 'not'
    index = 0


    for cat in corpus:

        tmp = bigramModel(doc, cat)

        if(tmp >= prob or index == 0):
            prob = tmp
            category_ = cat

        index += 1
        print(cat)
        print(tmp)

    return category_

#  classifies the given document with unigram model
def classify_unigram(doc):

    prob = 0
    category_ = 'not'
    index = 0


    for cat in corpus:

        tmp = unigramModel(doc, cat)

        if(tmp > prob or index == 0):
            prob = tmp
            category_ = cat

        index += 1
        print(cat)
        print(tmp)

    return category_

#  classifies the given document with unigram model using features
def classify_unigram_with_informative_words(doc):

    prob = 0
    category_ = 'not'
    index = 0


    for cat in corpus:

        tmp = unigramModel_with_informative_words(doc, cat)

        if(tmp >= prob or index == 0):
            prob = tmp
            category_ = cat

        index += 1
        print(cat)
        print(tmp)

    return category_
