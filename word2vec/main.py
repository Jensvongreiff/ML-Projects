import random

import numpy as np

import preprocessing as pp
import os
import model as mod
import pickle

random.seed(None)

preprocessing_done = True

# PREPROCESSING
review = pp.load_dataset("task03_data (1).npy")
processed_data, vocabulary, counts = pp.vocab(review, multidata=True)
numbered_vocab, numbered_counts, inv_numbered_vocab = pp.indicize_vocab(vocabulary, counts)
test_data = ['I', 'also', 'got', 'a', '_unk', '_unk', 'not', 'with', '_unk', 'which', 'was', 'amazing', '_unk', '_unk',
             'the', 'Vegas', '_unk', '_unk', 'was', 'so', 'good', '_unk', 'what']
if not preprocessing_done:
    context = pp.window_slide(processed_data, 2, inv_numbered_vocab)

    # To save (serialize) the list to a file:
    with open("context.pkl", "wb") as f:
        pickle.dump(context, f)

# Later, to load the list back into memory:
load = False
if load:
    with open("context.pkl", "rb") as f:
        print('Loading...')
        context = pickle.load(f)
    print('Data Loaded')

# MODEL
Embedding = mod.Model(70, 201)
Embedding.init_weights()
#print(Embedding.one_hot([1,4,7]))
# y_pred = Embedding.forward_prop([1,4,7], y_true)
# cross = Embedding.cross_entropy(y_true, y_pred)
# dU, dV = Embedding.back_prop()
epochs = 15
best_loss = 100
i = 0
train_on = False
if train_on:
    while i < epochs:
        for context_batch in pp.generate_minibatches(context, 1000):
            x, y_true = Embedding.training_pairs(context_batch)
            y_pred, loss = Embedding.forward_prop(x, y_true, vector_wise_loss=False)
            if loss < best_loss:
                best_loss = loss
                np.save("U_weights.npy", Embedding.U)
                np.save("V_weights.npy", Embedding.V)
            #print('y ' + str(y_pred))
            dU, dV = Embedding.back_prop()
            #print(dU)
            #print("\n")
            #print(dV)
            #print('error')
            #print(y_pred - y_true)
            #print('y_true')
            #print(y_true)
            Embedding.training_step(dU, dV, 0.01)
            i += 1

test = True
if test:
    U_loaded = np.load("U_weights.npy")
    V_loaded = np.load("V_weights.npy")
    next = 'to'
    while True:
        if next is '_unk':
            random_int = np.random.randint(0, 201)
            next = numbered_vocab[random_int]
        inputski = Embedding.one_hot([inv_numbered_vocab[next]])
        outputski = mod.softmax(inputski@(U_loaded@V_loaded), axis=1)
        buf = np.zeros_like(outputski)
        buf = (outputski == outputski.max()).astype(int)
        outputski = buf
        next = numbered_vocab[np.argmax(outputski)]
        print(next)


# print(y_pred)
# print(cross)
# print(dU)
# print(dV)
# print(np.shape(dU.T) == np.shape(dV))
