import pickle

directory = '/Users/zahra_abasiyan/PycharmProjects/NLP/classification/'


# this function reads the file in given path and creates a dictionary from train data
# in this dictionary each category is a key to other dictionary which
# every distinct word in that category is a key to array of its following words
# structure is like this --> cat:word:[following words]
# after creating this dictionary it is saved with pickle

def parseFile(path):

    with open(path) as f:
        lines  =  f.readlines()

    corpus = {}
    category_length = {}

    for line in lines:

        token = line.split('@@@@@@@@@@')

        cat = token[0]

        text = token[1]

        tokens = text.split(' ')

        length = len(tokens) - 2

        if not (cat in corpus):
            corpus[cat] = {}
            category_length[cat] = 1

        else:
            tmp = category_length[cat]
            tmp +=1
            category_length[cat] = tmp

        for i in range (1, length+1):

            if(tokens[i] in corpus[cat]):

                arr = corpus[cat][tokens[i]]

                if(i == length):
                    arr.append('.')

                else:
                    arr.append(tokens[i + 1])

                corpus[cat][tokens[i]] = arr

            else:

                if (i == length):
                    corpus[cat][tokens[i]] = ['.']

                else:
                    corpus[cat][tokens[i]] = [tokens[i+1]]

    with open('category.pickle', 'wb') as file:

        pickle.dump(corpus, file)

    with open('category_length.pickle', 'wb') as file:

        pickle.dump(category_length, file)

    return corpus


#  this func parses the test data file in given path and returns an array of tuples
# including each document and its tag like this --> [(document, tag)]
def loadTest(path):

    with open(path) as f:
        lines = f.readlines()

    testSet=[]
    length = 0

    for line in lines:

        token = line.split('@@@@@@@@@@')
        cat = token[0]
        text = token[1]

        if(len(text.split(' '))> length):
            length = len(text.split(' '))

        testSet.append((text, cat))

    return testSet


# this func removes  given values from given list
def remove_values_from_list(the_list, val):

   return [value for value in the_list if value != val]


# this func returns count of given values in given list
def count_values_from_list(the_list, val):

   return len([value for value in the_list if value == val])


# this func returns tokens of given text
def split_text(text):

    words = text.split(' ')

    arr = []

    for i in range(1, len(words)-1):

        arr.append(words[i])

    return arr


#  load needed pickles, variables, and data

parseFile('train_test_data/HAM-Train.txt')

with open((directory + 'category.pickle'), 'rb') as file:
    corpus = pickle.load(file)


with open((directory + 'category_length.pickle'), 'rb') as file:
    category_length = pickle.load(file)


with open('features2.pickle', 'rb') as file:
    features = pickle.load(file)


top_features = features [0:200]

informative_words = []

for elem in top_features:

    informative_words.append(elem[0])

total_word_cat = {}
distinct_words_cat ={}
total_word_corpus = 0
total_words = []

for cat in corpus:

    index = 0

    for word in corpus[cat]:
        index+= len(corpus[cat][word])
        total_words.append(word)

    total_word_cat[cat] = index
    total_word_corpus += index
    distinct_words_cat[cat] = len(corpus[cat])

total_docs = 0

for cat in category_length:
    total_docs += category_length[cat]


distinct_words_corpus = len(total_words)

def write_informative_words_to_file():

    file = open('features2.txt','w')

    for elem in top_features:
        tmp = str(elem[1]) + " : " + (elem[0])+"\n"
        file.write(tmp)

    file.close()


# write_informative_words_to_file()