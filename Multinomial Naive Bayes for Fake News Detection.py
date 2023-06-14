

"""**Libraries**"""

import time, glob
import shutil
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

"""**Code**

**Multinomial Naive Bayes With Stopwords**
"""

def text_preprocess(text):
    
    # remove punctuation symbols from the text
    text = re.sub('''[٪!%`‘’")'(.،؟:۔]''', ' ', text)
    # place space between a digit and a text
    text = re.sub('(\d+(\.\d+)?)', r' \1 ', text)
    # remove any garbage letters from the text
    text = text.replace(u'\ufeff', '')
    # split the text into words
    w1 = text.split()

    return w1

def concat_data(doc_name, url):

    doc_name = 'all_' + str((int(time.time()))) + ".txt"
    # combine url paths of all the files
    filenames = glob.glob(url)
    
    with open(doc_name, 'w') as outfile:
        for fname in filenames:
          # read all the files given the url path
            with open(fname, 'r') as readfile:
                infile = readfile.read()
                # write the read lines into the output file
                for line in infile:
                    outfile.write(line)
                outfile.write("\n")

def concat_train():

    # read text from both the fake file and real file and concatenate it into one file
    filenames = ['train_real.txt', 'train_fake.txt']
    with open('train_corpus.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def doc_concat():

    train_fake_url = '/content/gdrive/My Drive/Train/Fake/*.txt'
    concat_data('train_fake', train_fake_url)
    train_real_url = '/content/gdrive/My Drive/Train/Real/*.txt'
    concat_data('train_real', train_real_url)
    concat_train()
    test_doc1_url = '/content/gdrive/My Drive/Test/Real/*.txt'
    concat_data('test_doc1', test_doc1_url)
    test_doc0_url = '/content/gdrive/My Drive/Test/Fake/*.txt'
    concat_data('test_doc0', test_doc0_url)


def create_vocab():

    # open the train corpus file containing fake and real file text
    f1 = open('train_corpus.txt', 'r')
    # read the lines from the corpus
    text = f1.read()
    # split lines into words
    w1 = text_preprocess(text)

    # check if the word is added in vocab if not then add it
    vocab = []
    for w2 in w1:
        if w2 not in vocab:
            vocab.append(w2)

    # removing any duplicates
    vocab = list(set(vocab))
    # calculate the length of vocab
    vocab_len = len(vocab)

    return vocab, vocab_len

def train_Nw(file_path):

    # calculate word count for different classes i.e fake and real
    file_text = open(file_path, 'r')
    # read lines from the file
    file_lines = file_text.read()
    # preprocess the lines and split it in words
    file_words = text_preprocess(file_lines)
    # calculate length of the words
    file_word_count = len(file_words)

    return file_word_count

def cal_word_count():

    train_fake_file = 'train_fake.txt'
    train_fake_word_count = train_Nw(train_fake_file)
    train_real_file = 'train_real.txt'
    train_real_word_count = train_Nw(train_real_file)

    return train_fake_word_count, train_real_word_count, train_fake_file, train_real_file

def count_train_doc(file_path):

    # count how many files are in fake class and real class
    list = os.listdir(file_path) 
    fake_num_files = len(list)

    return fake_num_files


def count_doc_Nc():

    train_fake_path = '/content/gdrive/My Drive/Train/Fake'
    fake_num_files = count_train_doc(train_fake_path)
    train_real_path = '/content/gdrive/My Drive/Train/Real'
    Real_num_files = count_train_doc(train_real_path)
    N_total_num_files = Real_num_files + fake_num_files

    return fake_num_files, Real_num_files, N_total_num_files

def prior_prob(fake_num_files, Real_num_files, N_total_num_files):

    # calculate prior prob for a given class
    fake_prior = fake_num_files / N_total_num_files
    Real_prior = Real_num_files / N_total_num_files

    return fake_prior, Real_prior


def Ni(vocab, file_path, counter_dict):

    # count tokens of words in given class doc
    f3 = open(file_path, 'r')
    # read lines
    text = f3.read()
    # preprocess and split in words
    w1 = text_preprocess(text)
    for word in w1:
      # check if the word is present in vocab
        if word in vocab:
            counter_dict[word] += 1

    return counter_dict

def cal_Ni(vocab, train_fake_file, train_real_file):

    tr_Ni = Counter()
    tf_Ni = Counter()
    train_real_Ni = Ni(vocab, train_real_file, tr_Ni)
    train_fake_Ni = Ni(vocab, train_fake_file, tf_Ni)

    return train_fake_Ni, train_real_Ni

def cond_prob(counter_dict, word_count, vocab_len, vocab, cp_dict):
    
    # calculate conditional probability for each word of text in given class
    for each_word in vocab:
        for elem in counter_dict.elements():
            if each_word in elem:
                cp_dict[elem] = float((counter_dict[elem] + 1))/ float((word_count + vocab_len))
    return cp_dict

def cal_condprob(train_fake_Ni, train_fake_word_count, vocab_len, train_real_Ni, train_real_word_count, vocab):

    cp_real = dict()
    cp_fake = dict()
    condprob_fake = cond_prob(train_fake_Ni, train_fake_word_count, vocab_len, vocab, cp_fake)
    condprob_real = cond_prob(train_real_Ni, train_real_word_count, vocab_len, vocab, cp_real)

    return condprob_fake, condprob_real

def words(file_path):

    # read lines from the test files
    file = open(file_path, 'r')
    lines = file.read()
    # preprocess the text and split in words
    words = text_preprocess(lines)
    return words

def read_words():
    file1 = 'test_doc1.txt'
    test_doc1_words = words(file1)
    file2 = 'test_doc0.txt'
    test_doc0_words = words(file2)

    return test_doc1_words, test_doc0_words

def news(doc_words, condprob_fake, condprob_real, vocab, doc_name, fake_prior, Real_prior):
    
    # calculate score of prior + conditional probility and check which is max, then classify the doc in class
    score = dict()
    # taking log of prior 
    score['Fake'] = math.log10(fake_prior)
    score['Real'] = math.log10(Real_prior)
    
    for w1 in doc_words:
        for w2 in vocab:
            if w1 in w2:
                if w1 in condprob_real:
                # taking log of conditional probability
                    score['Real'] += math.log10(condprob_real[w1])
                if w1 in condprob_fake:
                    score['Fake'] += math.log10(condprob_fake[w1])
    
    # check max value in terms of probability of each class and classify doc with max probability in respective class
    if score['Real'] > score['Fake']:
        print(f"{doc_name} is Real News")
        return 'Real'
    else:
        print(f"{doc_name} is Fake News")
        return 'Fake'

def detect_news(test_doc1_words, test_doc0_words, condprob_fake, condprob_real, vocab, fake_prior, Real_prior):

    # checking if the classification is right
    # note: test_doc0 true class is 'Real' where as test_doc1 true class is 'Fake'
    TP = TN = FP = FN = 0
    doc1_class = news(test_doc1_words, condprob_fake, condprob_real, vocab, 'Test doc1', fake_prior, Real_prior)
    doc0_class = news(test_doc0_words, condprob_fake, condprob_real, vocab, 'Test doc0', fake_prior, Real_prior)

    # when test_doc1 is classified correctly
    if doc1_class == 'Real':
        TP = 1
    # when test_doc1 is classified incorrectly
    if doc1_class == 'Fake':
        FN = 1
    # when test_doc0 is classified incorrectly
    if doc0_class == 'Real':
        FP = 1
    # when test_doc1 is classified correctly
    if doc0_class == 'Fake':
        TN = 1

    print(f''''TP: '{TP}, 'TN: '{TN}, 'FP: '{FP}, 'FN: '{FN}''')

    return TP, TN, FP, FN

def evaluate_model(TP, TN, FP, FN):

    # evaluating the model using evaluation metrics of accuracy, precision, recall and Fi-score
    accuracy = 100.0*(float(TP + TN)/float(TP + TN + FP + FN))
    print(f"accuracy: {accuracy}")

    precision = TP / (TP + FP)
    print(f"precision: {precision:4.2f}")

    recall = TP / (TP + FN)
    print(f"recall: {recall:4.2f}")

    f1_score = 2 * precision * recall / (precision + recall)
    print(f"F1-score: {f1_score:4.2f}")

def main():
    # Train Multinomial NB
    doc_concat()
    vocab, vocab_len = create_vocab()
    train_fake_word_count, train_real_word_count, train_fake_file, train_real_file = cal_word_count()
    fake_num_files, Real_num_files, N_total_num_files = count_doc_Nc()
    fake_prior, Real_prior = prior_prob(fake_num_files, Real_num_files, N_total_num_files)
    train_fake_Ni, train_real_Ni = cal_Ni(vocab, train_fake_file, train_real_file)
    condprob_fake, condprob_real = cal_condprob(train_fake_Ni, train_fake_word_count, vocab_len, train_real_Ni, train_real_word_count, vocab)
    
    # Test Multinomial NB
    test_doc1_words, test_doc0_words = read_words()
    TP, TN, FP, FN = detect_news(test_doc1_words, test_doc0_words, condprob_fake, condprob_real, vocab, fake_prior, Real_prior)
    evaluate_model(TP, TN, FP, FN)

if __name__ == '__main__':
    main()

"""**Multinomial Naive Bayes Without Stopwords**"""

def read_stop_words():

    stop_text = open('stopwords-ur.txt', 'r')
    # read words from stop words file
    stop_lines = stop_text.read()
    # split the words and store them
    stop_words = stop_lines.split()

    return stop_words

def text_preprocess(text, stop_words):
    
    # remove punctuation symbols from the text
    text = re.sub('''[٪!%`‘’")'(.،؟:۔]''', ' ', text)
    # place space between a digit and a text
    text = re.sub('(\d+(\.\d+)?)', r' \1 ', text)
    # remove any garbage letters from the text
    text = text.replace(u'\ufeff', '')
    # split the text into words
    w1 = text.split()
    # remove stop words from text
    w1 = [x for x in w1 if x not in stop_words]

    return w1

def concat_data(doc_name, url):

    doc_name = 'all_' + str((int(time.time()))) + ".txt"
    # combine url paths of all the files
    filenames = glob.glob(url)
    
    with open(doc_name, 'w') as outfile:
        for fname in filenames:
          # read all the files given the url path
            with open(fname, 'r') as readfile:
                infile = readfile.read()
                # write the read lines into the output file
                for line in infile:
                    outfile.write(line)
                outfile.write("\n")

def concat_train():

    # read text from both the fake file and real file and concatenate it into one file
    filenames = ['train_real.txt', 'train_fake.txt']
    with open('train_corpus.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def doc_concat():

    train_fake_url = '/content/gdrive/My Drive/Train/Fake/*.txt'
    concat_data('train_fake', train_fake_url)
    train_real_url = '/content/gdrive/My Drive/Train/Real/*.txt'
    concat_data('train_real', train_real_url)
    concat_train()
    test_doc1_url = '/content/gdrive/My Drive/Test/Real/*.txt'
    concat_data('test_doc1', test_doc1_url)
    test_doc0_url = '/content/gdrive/My Drive/Test/Fake/*.txt'
    concat_data('test_doc0', test_doc0_url)


def create_vocab(stop_words):

    # open the train corpus file containing fake and real file text
    f1 = open('train_corpus.txt', 'r')
    # read the lines from the corpus
    text = f1.read()
    # split lines into words
    w1 = text_preprocess(text, stop_words)
    # check if the word is added in vocab if not then add it
    vocab = []
    for w2 in w1:
        if w2 not in vocab:
            vocab.append(w2)
    # removing any duplicates
    vocab = list(set(vocab))
    # calculate the length of vocab
    vocab_len = len(vocab)

    return vocab, vocab_len

def train_Nw(file_path, stop_words):

    # calculate word count for different classes i.e fake and real
    file_text = open(file_path, 'r')
    # read lines from the file
    file_lines = file_text.read()
    # preprocess the lines and split it in words
    file_words = text_preprocess(file_lines, stop_words)
    # calculate length of the words
    file_word_count = len(file_words)

    return file_word_count

def cal_word_count(stop_words):

    train_fake_file = 'train_fake.txt'
    train_fake_word_count = train_Nw(train_fake_file, stop_words)
    train_real_file = 'train_real.txt'
    train_real_word_count = train_Nw(train_real_file, stop_words)

    return train_fake_word_count, train_real_word_count, train_fake_file, train_real_file

def count_train_doc(file_path):

    # count how many files are in fake class and real class
    list = os.listdir(file_path)
    fake_num_files = len(list)

    return fake_num_files


def count_doc_Nc():

    train_fake_path = '/content/gdrive/My Drive/Train/Fake'
    fake_num_files = count_train_doc(train_fake_path)
    train_real_path = '/content/gdrive/My Drive/Train/Real'
    Real_num_files = count_train_doc(train_real_path)
    N_total_num_files = Real_num_files + fake_num_files

    return fake_num_files, Real_num_files, N_total_num_files

def prior_prob(fake_num_files, Real_num_files, N_total_num_files):


    # calculate prior prob for a given class
    fake_prior = fake_num_files / N_total_num_files
    Real_prior = Real_num_files / N_total_num_files

    return fake_prior, Real_prior


def Ni(vocab, file_path, counter_dict, stop_words):

    # count tokens of words in given class doc
    f3 = open(file_path, 'r')
    # read lines
    text = f3.read()
    # preprocess and split in words
    w1 = text_preprocess(text, stop_words)
    for word in w1:
      # check if the word is present in vocab
        if word in vocab:
            counter_dict[word] += 1

    return counter_dict

def cal_Ni(vocab, train_fake_file, train_real_file, stop_words):

    tr_Ni = Counter()
    tf_Ni = Counter()
    train_real_Ni = Ni(vocab, train_real_file, tr_Ni, stop_words)
    train_fake_Ni = Ni(vocab, train_fake_file, tf_Ni, stop_words)

    return train_fake_Ni, train_real_Ni

def cond_prob(counter_dict, word_count, vocab_len, vocab, cp_dict):
    
    # calculate conditional probability for each word of text in given class
    for each_word in vocab:
        for elem in counter_dict.elements():
            if each_word in elem:
                cp_dict[elem] = float((counter_dict[elem] + 1))/ float((word_count + vocab_len))
    return cp_dict

def cal_condprob(train_fake_Ni, train_fake_word_count, vocab_len, train_real_Ni, train_real_word_count, vocab):

    cp_real = dict()
    cp_fake = dict()
    condprob_fake = cond_prob(train_fake_Ni, train_fake_word_count, vocab_len, vocab, cp_fake)
    condprob_real = cond_prob(train_real_Ni, train_real_word_count, vocab_len, vocab, cp_real)

    return condprob_fake, condprob_real

def words(file_path, stop_words):

    # read lines from the test files
    file = open(file_path, 'r')
    lines = file.read()
    # preprocess the text and split in words
    words = text_preprocess(lines, stop_words)
    return words

def read_words(stop_words):
    file1 = 'test_doc1.txt'
    test_doc1_words = words(file1, stop_words)
    file2 = 'test_doc0.txt'
    test_doc0_words = words(file2, stop_words)

    return test_doc1_words, test_doc0_words

def news(doc_words, condprob_fake, condprob_real, vocab, doc_name, fake_prior, Real_prior):
    
    # calculate score of prior + conditional probility and check which is max, then classify the doc in class
    score = dict()
    # taking log of prior
    score['Fake'] = math.log10(fake_prior)
    score['Real'] = math.log10(Real_prior)
    
    for w1 in doc_words:
        for w2 in vocab:
            if w1 in w2:
                if w1 in condprob_real:
                # taking log of conditional probability
                    score['Real'] += math.log10(condprob_real[w1])
                if w1 in condprob_fake:
                    score['Fake'] += math.log10(condprob_fake[w1])
    # check max value in terms of probability of each class and classify doc with max probability in respective class
    if score['Real'] > score['Fake']:
        print(f"{doc_name} is Real News")
        return 'Real'
    else:
        print(f"{doc_name} is Fake News")
        return 'Fake'

def detect_news(test_doc1_words, test_doc0_words, condprob_fake, condprob_real, vocab, fake_prior, Real_prior):

    # checking if the classification is right
    # note: test_doc0 true class is 'Real' where as test_doc1 true class is 'Fake'
    TP = TN = FP = FN = 0
    doc1_class = news(test_doc1_words, condprob_fake, condprob_real, vocab, 'Test doc1', fake_prior, Real_prior)
    doc0_class = news(test_doc0_words, condprob_fake, condprob_real, vocab, 'Test doc0', fake_prior, Real_prior)
    
    # when test_doc1 is classified correctly
    if doc1_class == 'Real':
        TP = 1
    # when test_doc1 is classified incorrectly
    if doc1_class == 'Fake':
        FN = 1
    # when test_doc0 is classified incorrectly
    if doc0_class == 'Real':
        FP = 1
    # when test_doc1 is classified correctly
    if doc0_class == 'Fake':
        TN = 1

    print(f''''TP: '{TP}, 'TN: '{TN}, 'FP: '{FP}, 'FN: '{FN}''')

    return TP, TN, FP, FN

def evaluate_model(TP, TN, FP, FN):

    # evaluating the model using evaluation metrics of accuracy, precision, recall and Fi-score
    accuracy = 100.0*(float(TP + TN)/float(TP + TN + FP + FN))
    print(f"accuracy: {accuracy}")

    precision = TP / (TP + FP)
    print(f"precision: {precision:4.2f}")

    recall = TP / (TP + FN)
    print(f"recall: {recall:4.2f}")

    f1_score = 2 * precision * recall / (precision + recall)
    print(f"F1-score: {f1_score:4.2f}")

def main():
    # Train Multinomial NB
    doc_concat()
    stop_words = read_stop_words()
    vocab, vocab_len = create_vocab(stop_words)
    train_fake_word_count, train_real_word_count, train_fake_file, train_real_file = cal_word_count(stop_words)
    fake_num_files, Real_num_files, N_total_num_files = count_doc_Nc()
    fake_prior, Real_prior = prior_prob(fake_num_files, Real_num_files, N_total_num_files)
    train_fake_Ni, train_real_Ni = cal_Ni(vocab, train_fake_file, train_real_file, stop_words)
    condprob_fake, condprob_real = cal_condprob(train_fake_Ni, train_fake_word_count, vocab_len, train_real_Ni, train_real_word_count, vocab)
    
    # Test Multinomial NB
    test_doc1_words, test_doc0_words = read_words(stop_words)
    TP, TN, FP, FN = detect_news(test_doc1_words, test_doc0_words, condprob_fake, condprob_real, vocab, fake_prior, Real_prior)
    evaluate_model(TP, TN, FP, FN)

if __name__ == '__main__':
    main()

