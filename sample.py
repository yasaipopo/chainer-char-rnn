# coding:utf-8
import time
import math
import sys
import argparse
import cPickle as pickle
import codecs
import os.path

import numpy as np
from chainer import cuda, Variable, FunctionSet
import chainer.functions as F
from CharRNN import CharRNN, make_initial_state
import MeCab
import re
mecab = MeCab.Tagger ("-Ochasen")

def add_text(state):
    node = mecab.parseToNode(args.primetext)
    while node:
        if node.surface=="":
            node=node.next
            continue
        i=node.surface+"::"+node.feature
        sys.stdout.write(i.decode("utf-8").split("::")[0])
        prev_char = np.ones((1,), dtype=np.int32) * vocab[i]
        if args.gpu >= 0:
            prev_char = cuda.to_gpu(prev_char)

        state, prob = model.forward_one_step(prev_char, prev_char, state, train=False)
        node = node.next
    return state,prob,prev_char

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

#%% arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model',      type=str,   default='cv/latest.chainermodel')
parser.add_argument('--vocabulary', type=str,   default='')

parser.add_argument('--seed',       type=int,   default=123)
parser.add_argument('--sample',     type=int,   default=1)
parser.add_argument('--primetext',  type=str,   default='')
parser.add_argument('--length',     type=int,   default=2000)
parser.add_argument('--gpu',        type=int,   default=0)
parser.add_argument('--delimiter',  type=str,   default="-=-=-=-=-=-=-=-")

args = parser.parse_args()
#delimiter
delimiter_a = []
if args.delimiter!="":
    node = mecab.parseToNode(args.delimiter)
    while node:
        delimiter_a.append(node.surface+"::"+"_".join(node.feature.split(",")[0:2]))
        node=node.next


np.random.seed(args.seed)

if os.path.isdir(args.model):
    args.model += "/latest.chainermodel"

# load vocabulary

if args.vocabulary!='':
    vocab = pickle.load(open(args.vocabulary, 'rb'))
else:
    vocab = pickle.load(open(os.path.dirname(args.model)+"/vocab.bin", 'rb'))

ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c

# load model
model = pickle.load(open(args.model, 'rb'))
n_units = model.embed.W.data.shape[1]

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# initialize generator
state = make_initial_state(n_units, batchsize=1, train=False)
if args.gpu >= 0:
    for key, value in state.items():
        value.data = cuda.to_gpu(value.data)

prev_char = np.array([0], dtype=np.int32)
if args.gpu >= 0:
    prev_char = cuda.to_gpu(prev_char)

if len(args.primetext) > 0:
    state,prob,prev_char = add_text(state)



delimiter_hit_count=0
for i in xrange(args.length):
    state, prob = model.forward_one_step(prev_char, prev_char, state, train=False)

    if args.sample > 0:
        probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
        probability /= np.sum(probability)
        index = np.random.choice(range(len(probability)), p=probability)
    else:
        index = np.argmax(cuda.to_cpu(prob.data))
    sys.stdout.write(ivocab[index].decode("utf-8").split("::")[0])
    prev_char = np.array([index], dtype=np.int32)
    if args.gpu >= 0:
        prev_char = cuda.to_gpu(prev_char)


    if len(delimiter_a)>0 and len(args.primetext) > 0:
        if ivocab[index]==delimiter_a[delimiter_hit_count]:
            delimiter_hit_count+=1
            if delimiter_hit_count==len(delimiter_a):
                state,prob,prev_char=add_text(state)
                delimiter_hit_count=0
        else:
            delimiter_hit_count=0

    #print ivocab[index].decode("utf-8")



print


