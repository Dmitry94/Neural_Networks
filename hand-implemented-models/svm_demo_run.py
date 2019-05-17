#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-

"""SVM classifier demo run."""

import sys
sys.path.append('..')

import argparse
import time
import numpy as np
import cifar_read_utils

from SVMClassifier import SVMClassifier


def parse_args():
    """Create, parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        'Demo applications for demonstration SVMClassifier work.')

    parser.add_argument('cifar_path', help='Path to CIFAR data.')

    parser.add_argument('--bsize', default=64, type=int,
                        help='Size of one feeding data batch.')

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Training learning rate.')

    parser.add_argument('--reg-lambda', default=1e-3, type=float,
                        help='L2-regularization lambda.')

    args = parser.parse_args()
    return args


def main():
    """Application entry point."""
    args = parse_args()

    train_batches, test_batch = cifar_read_utils.read_ciraf_10(args.cifar_path)
    classifier = SVMClassifier(args.lr, args.reg_lambda, args.bsize)

    # Merge batches
    train_data = train_batches[0]['data']
    train_labels = np.array(train_batches[0]['labels'])
    for i in range(1, len(train_batches)):
        cur_data = train_batches[i]['data']
        cur_labels = np.array(train_batches[i]['labels'])

        train_data = np.concatenate((train_data, cur_data))
        train_labels = np.concatenate((train_labels, cur_labels))

    #
    print('Testing implemented svm classifier...')
    # Train
    start = time.clock()
    classifier.train(train_data, train_labels)
    end = time.clock()
    print('SVM training time = {}'.format(end - start))
    # Predict
    start = time.clock()
    predictions = classifier.predict(test_batch['data'])
    end = time.clock()
    print('SVM predicting time = {}'.format(end - start))
    accuracy = np.mean(predictions == test_batch['labels'])
    print('SVM accuracy = {}'.format(accuracy))


if __name__ == '__main__':
    main()
