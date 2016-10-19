#!/usr/bin/python

import collections
import math
import os
import random

import numpy as np
import tensorflow as tf
from random import shuffle

import sys, getopt
import os

from collections import namedtuple

Dataset = namedtuple('Dataset','sentences labels')

num_classes = 3
learning_rate = 0.01
num_epochs = 10
embedding_dim = 10
label_to_id = {'World': 0, 'Entertainment': 1, 'Sports': 2}
unknown_word_id = 0


def create_label_vec(label):
   # Generate a label vector for a given classification label.
   label_vec = [0] * num_classes
   label_vec[label_to_id[label]] = 1
   return label_vec

def tokenize(sens):
    # Tokenize a given sentence into a sequence of tokens.
    import nltk
    return nltk.word_tokenize(sens)

def map_token_seq_to_word_id_seq(token_seq, word_to_id):
    return [map_word_to_id(word_to_id,word) for word in token_seq]

def map_word_to_id(word_to_id, word):
    # map each word to its id.
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['$UNK$']

def build_vocab(sens_file_name):
    data = []
    with open(sens_file_name) as f:
        for line in f.readlines():
            tokens = tokenize(line)
            data.extend(tokens)
    count = [['$UNK$', 0]]
    sorted_counts = collections.Counter(data).most_common()
    count.extend(sorted_counts)
    word_to_id = dict()
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    print('size of vocabulary is %s. ' % len(word_to_id))
    return word_to_id


def read_labeled_dataset(sens_file_name, label_file_name, word_to_id):
    sens_file = open(sens_file_name)
    label_file = open(label_file_name)
    data = []
    for label in label_file:
        label = label.strip()
        sens = sens_file.readline()
        word_id_seq = map_token_seq_to_word_id_seq(tokenize(sens), word_to_id)
        data.append((word_id_seq, create_label_vec(label)))
    print("read %d sentences from %s ." % (len(data), sens_file_name))
    sens_file.close()
    label_file.close()
    return data

def read_dataset(sens_file_name, word_to_id):
    sens_file = open(sens_file_name)
    data = []
    for sens in sens_file:
        word_id_seq = map_token_seq_to_word_id_seq(tokenize(sens), word_to_id)
        data.append(word_id_seq)
    print("read %d sentences from %s ." % (len(data), sens_file_name))
    sens_file.close()
    return data


def train_eval(word_to_id, train_dataset, dev_dataset, test_dataset):
    print 'train and eval start'
    num_words = len(word_to_id)
    # Initialize the placeholders and Variables. E.g.
    # label tensor
    correct_label = tf.placeholder(tf.float32, shape=[num_classes])
    # sentences tensor
    input_sens = tf.placeholder(tf.int32, shape=[None])

    #build sentences embedding as embedding1 and labels embedding as embedding2 with embdedding_dim dimensions
    embeddings1 = tf.Variable(tf.random_uniform([num_words, embedding_dim], -1.0/embedding_dim, 1.0/embedding_dim)) # num_words * 10
    embeddings2 = tf.Variable(tf.random_uniform([num_classes, embedding_dim], -1.0/embedding_dim, 1.0/embedding_dim)) # 3 * 10
    with tf.Session() as sess:
        # retrieves rows of the params tensor
        embed1 = tf.nn.embedding_lookup(embeddings1, input_sens) # n * 10

        # Computes the sum of elements across dimensions of a tensor, and keep the dimension numbers.
        tmp_m1 = tf.reduce_sum(embed1, 0)  # 1 * 10

        # reshape the tensor
        sum_rep1 = tf.reshape(tmp_m1, [1, embedding_dim]) # 1*10

        # embed2 = tf.nn.embedding_lookup(embeddings2, correct_label)
        # tmp_m2 = tf.reduce_sum(embed2, 0)
        # sum_rep2 = tf.reshape(tmp_m2, [num_classes, embedding_dim])

        # Get a 1*3 dimension tensor
        # y : maximum likelihood estimation
        y = tf.nn.softmax(tf.matmul(sum_rep1, embeddings2, transpose_b=True)) # 3*1


        cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(y), reduction_indices=[1])) # 1*1

        #evaluation code, assume y is the estimated probability vector of each class
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(correct_label, 0))
        accuracy = tf.cast(correct_prediction, tf.float32)
        prediction = tf.cast(tf.argmax(y, 1), tf.int32)
        sess.run(tf.initialize_all_variables())

        # Build SGD optimizer
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        for epoch in range(num_epochs):
            print 'epoch start'
            shuffle(train_dataset)
            for data in train_dataset:
                train_step.run(feed_dict={input_sens: data[0], correct_label: data[1]})

            # The following line computes the accuracy on the development dataset in each epoch.
            print('Epoch %d : %s .' % (epoch,compute_accuracy(accuracy,input_sens, correct_label, dev_dataset)))

        print('Accuracy on the test set : %s.' % compute_accuracy(accuracy,input_sens, correct_label, test_dataset))
        # input_sens is the placeholder of an input sentence.
        test_results = predict(prediction, input_sens, test_dataset)
    return test_results


def compute_accuracy(accuracy,input_sens, correct_label, eval_dataset):
    num_correct = 0
    for (sens, label) in eval_dataset:
        num_correct += accuracy.eval(feed_dict={input_sens: sens, correct_label: label})
    print('#correct sentences is %s ' % num_correct)
    return num_correct / len(eval_dataset)


def predict(prediction, input_sens, test_dataset):
    test_results = []
    for (sens, label) in test_dataset:
        test_results.append(prediction.eval(feed_dict={input_sens: sens}))
    return test_results


def write_result_file(test_results, result_file):
    with open(result_file, mode='w') as f:
         for r in test_results:
             f.write("%d\n" % r)


def main(argv):
    trainSensFile = ''
    trainLabelFile = ''
    devSensFile = ''
    devLabelFile = ''
    testSensFile = ''
    testLabelFile = ''
    testResultFile = ''
    try:
        opts, args = getopt.getopt(argv, "hd:", ["dataFolder="])
    except getopt.GetoptError:
        print('fastText.py -d <dataFolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('fastText.py -d <dataFolder>')
            sys.exit()
        elif opt in ("-d", "--dataFolder"):
            trainSensFile = os.path.join(arg, 'sentences_train.txt')
            devSensFile = os.path.join(arg, 'sentences_dev.txt')
            testSensFile = os.path.join(arg, 'sentences_test.txt')
            trainLabelFile = os.path.join(arg, 'labels_train.txt')
            devLabelFile = os.path.join(arg, 'labels_dev.txt')
            #testLabelFile = os.path.join(arg, 'labels_test.txt')
            testResultFile = os.path.join(arg, 'test_results.txt')
        else:
            print("unknown option %s ." % opt)
    word_to_id_train = build_vocab('sentences_train.txt')
    train_dataSet = read_labeled_dataset('sentences_train.txt', 'labels_train.txt', word_to_id_train)

    # word_to_id_dev = build_vocab('sentences_dev.txt')
    #dev_dataSet = read_labeled_dataset('sentences_dev.txt', 'labels_dev.txt', word_to_id_train)

    test_dataSet = read_labeled_dataset('sentences_test.txt', 'labels_test.txt', word_to_id_train)

    test_results = train_eval(word_to_id_train, train_dataSet, test_dataSet, test_dataSet)
    write_result_file(test_results, 'test_results.txt')


if __name__ == "__main__":
   main(sys.argv[1:])

