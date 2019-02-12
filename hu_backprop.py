#!/usr/bin/env python3
import numpy as np
from io import StringIO
from random import random
from math import exp

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = ""



#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32).reshape(len(xs),NUM_FEATURES,1) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
        model = []
        layer1 = []
        layer2 = []
        for item in w1:
            layer1.append({'weights': list(item)})
        for item in w2:
            layer2.append({'weights': list(item)})
        model.append(layer1)
        model.append(layer2)

    else:
        model=[]
        w1 = [{'weights': [random() for i in range(NUM_FEATURES)]} for i in range(args.hidden_dim)]
        model.append(w1)
        w2 = [{'weights': [random() for i in range(args.hidden_dim + 1)]}]
        model.append(w2)
    return model


    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):

    def forward_prop(model, data):
        inputs = data
        for layer in model:
            new_input = []
            for neuron in layer:
                weights = neuron['weights']
                activation = weights[-1]
                for i in range(len(weights) - 1):
                    if type(inputs[i]) == float:
                        activation += weights[i] * inputs[i]
                    else:
                        activation += weights[i] * inputs[i][0]
                a = 1.0 / (1.0 + exp(-activation))
                neuron['output'] = a
                new_input.append(neuron['output'])
            inputs = new_input
        return inputs

    def back_prop(model, ex_output):
        for i in reversed(range(len(model))):
            layer = model[i]
            errors = []
            if i != len(model) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in model[i + 1]:
                        error += (neuron['weights'][j] * neuron['change'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    if type(ex_output[j]) == float:
                        errors.append(ex_output[j] - neuron['output'])
                    else:
                        errors.append(ex_output[j][0] - neuron['output'])

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['change'] = errors[j] * (neuron['output'] * (1.0 - neuron['output']))

    def update_weights(model, data, lr):
        for i in range(len(model)):
            inputs=data[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in model[i - 1]]
            for neuron in model[i]:
                for j in range(len(inputs)):
                    if type(inputs[j])==float:
                        neuron['weights'][j] += lr * neuron['change'] * inputs[j]
                    else:
                        neuron['weights'][j] += lr * neuron['change'] * inputs[j][0]
                neuron['weights'][-1] += lr * neuron['change']


    train_input=train_xs.tolist()
    train_output=train_ys.tolist()
    acc1 = 0.0
    for iter in range(args.iterations):
        for i in range(len(train_input)):
            forward_prop(model,train_input[i])
            back_prop(model,train_output[i])
            update_weights(model, train_input[i], args.lr)
        if not args.nodev:
            acc2 = test_accuracy(model,dev_ys,dev_xs)
            if acc2<acc1:
                print('Accuracy decreases, iterations stop')
                break
            acc1=acc2
    return model


def test_accuracy(model, test_ys, test_xs):
    correct=0

    def forward_prop(model, inputs):
        for layer in model:
            new_input=[]
            for neuron in layer:
                weights=neuron['weights']
                activation=weights[-1]
                for i in range(len(weights)-1):
                    activation += weights[i] * inputs[i]
                if activation < 0:
                    a=1.0 - 1.0/(1.0 + exp(activation))
                else:
                    a=1.0 / (1.0 + exp(-activation))
                neuron['output'] = a
                new_input.append(neuron['output'])
            inputs = new_input

    for i in range(len(test_xs)):
        forward_prop(model,test_xs[i])
        p=0
        if model[1][0]['output']>=0.5:
            p=1
        elif model[1][0]['output']<0.5:
            p=0
        if p==test_ys[i]:
            correct+=1
    accuracy=correct/len(test_ys)

    return accuracy

def extract_weights(model):
    w1 = []
    w2 = []
    for neuron in model[0]:
        w1.append(neuron['weights'])
    for neuron2 in model[1]:
        w2.append(neuron2['weights'])
    return np.array(w1),np.array(w2)




def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()
