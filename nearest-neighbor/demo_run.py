#!/usr/bin/env python3.6
# -*- coding: UTF-8 -*-

"""Demo application for NN classifier."""

import sys
sys.path.append('..')
import cifar_read_utils

import argparse
import time
import numpy as np

from NNClassifier import NNClassifier


def parse_args():
    """Create, parse and return command line arguments."""
    parser = argparse.ArgumentParser('This demo shows, how NN classifier '
                                     'works.')

    parser.add_argument('cifar_path', help='Path to the CIFAR data.')

    parser.add_argument('--k', help='Count of neighbors to analyze.',
                        default=1)

    parser.add_argument('--metric', help='Metric to use',
                        choices=['L1', 'L2'], default='L1')

    args = parser.parse_args()
    return args


def main():
    """Application entry point."""
    args = parse_args()

    train_batch, test_batch = cifar_read_utils.read_ciraf_10(args.cifar_path)

    classifier = NNClassifier()

    start = time.clock()
    classifier.train(train_batch[0]['data'], train_batch[0]['labels'])
    end = time.clock()
    print('Training time = {}'.format(end - start))

    start = time.clock()
    predictions = classifier.predict(test_batch['data'], args.k, args.metric)
    end = time.clock()
    print('Prediction time = {}'.format(end - start))

    test_accuracy = np.mean(predictions == test_batch['labels'])
    print('Got accuracy = {} at test data.'.format(test_accuracy))


if __name__ == '__main__':
    main()
