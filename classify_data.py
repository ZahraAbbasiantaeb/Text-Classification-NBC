from sklearn.metrics import classification_report, confusion_matrix
from classifiers import classify_unigram, classify_unigram_with_informative_words, classify_bigram
from data import loadTest, directory

test_data = loadTest(directory + 'train_test_data/HAM-Test.txt')

predict = []
label = []

# test the model with unigram model

for elem in test_data:

    tmp = classify_bigram(elem[0])
    predict.append(tmp)
    label.append(elem[1])
    print('****')


print(classification_report(label, predict))

print(confusion_matrix(label, predict))

print(label)

print(predict)