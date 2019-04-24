import math
import pickle


# this func parses the train data in given path, and extract the needed data for calculating
# IG, the data is stored in pickle for further
def parseFile(path):

    with open(path) as f:
        lines  =  f.readlines()

    doc_id = 1

    doc_name = 'doc_'

    dataset = {}

    corpus_words = {}

    for line in lines:

        doc = (doc_name+str(doc_id))

        token = line.split('@@@@@@@@@@')

        cat = token[0]

        text = token[1]

        if(cat in dataset):
            dataset[cat][doc] ={}


        else:

            dataset[cat] = {}
            dataset[cat][doc] = {}


        text = text.replace('\n','')

        doc_id += 1

        words = text.split(' ')
        length = len(words)
        for i in range (1, length-1):

            if(words[i] in dataset[cat][doc]):

                 val = dataset[cat][doc][words[i]]

                 val = val + 1

                 dataset[cat][doc][words[i]] = val

            else:

                dataset[cat][doc][words[i]] = 1

            if not (words[i] in corpus_words):
                corpus_words[words[i]] = 1
            else:
                corpus_words[words[i]] = corpus_words[words[i]]+1

    with open('dataset_feature.pickle', 'wb') as file:

        pickle.dump(dataset, file)

    with open('corpus_words.pickle', 'wb') as file:

        pickle.dump(corpus_words, file)

    return dataset


# parseFile('/Users/zahra_abasiyan/PycharmProjects/NLP/classification/train_test_data/HAM-Train.txt')

with open('dataset_feature.pickle', 'rb') as file:
    dataset = pickle.load(file)


with open('corpus_words.pickle', 'rb') as file:
    corpus_words = pickle.load(file)


#  this func calculates IG of given word
def getInformationGain(word):

    N = 0

    # docs containing w
    N_W = 0

    # docs not containing w
    N_W_not = 0

    for cat in dataset:

        N += len(dataset[cat])

        for doc in dataset[cat]:

            if word in dataset[cat][doc]:

                N_W += 1

            else:

                N_W_not += 1

    tmp_1 = 0

    tmp_2 = 0

    cat_prob = 0

    for cat in dataset:

        # docs in category cat
        N_I = len(dataset[cat])

        cat_prob += -1 * (N_I/N) * math.log((N_I/N), 2)

        # docs in category cat containing w
        N_IW = 0

        for doc in dataset[cat]:

            if word in dataset[cat][doc]:

                N_IW += 1

        # docs in category ci not containing w
        N_IW_not = N_I - N_IW


        if (N_IW >0):

            tmp_1 += (N_IW / N_W) * math.log((N_IW / N_W) , 2)

        if(N_IW_not >0):

            tmp_2 += (N_IW_not / N_W_not) * math.log((N_IW_not / N_W_not), 2)

    # Information Gain
    IG = (N_W / N) * tmp_1 + (N_W_not / N) * tmp_2 + cat_prob

    return IG


#  return IG of all of the words in train corpus
def cal_IG_of_all_words():

    features = []

    for word in corpus_words:
        features.append((word, getInformationGain(word)))

    features = sorted(features, key=lambda t: t[1], reverse=True)

    with open('features2.pickle', 'wb') as file:
        pickle.dump(features, file)

    return


# cal_IG_of_all_words()
