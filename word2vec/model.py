import numpy as np


def softmax(x, axis):  # computes vectorwise softmax
    x = x - np.max(x, axis=axis, keepdims=True)
    summation = np.sum(np.exp(x), axis=axis, keepdims=True)
    ans = np.exp(x) / summation

    return ans


class Model:

    def __init__(self, embedding_size, vocab_size):
        self.vocab_size = vocab_size  # Dim N, vector of all vocab
        self.embedding_size = embedding_size  # Dim D, size of embedding

        self.back_buf = None
        self.U = None  # NxD
        self.V = None  # DxN
        self.loss = None

        self.init_weights()

    def init_weights(self):  # initializes weights for U,V matrices, resets backprop carryover vals
        self.back_buf = None
        self.loss = None
        self.V = np.random.normal(0, np.sqrt(6. / (self.embedding_size + self.vocab_size)),
                                  (self.embedding_size, self.vocab_size))
        self.U = np.random.normal(0, np.sqrt(6. / (self.embedding_size + self.vocab_size)),
                                  (self.vocab_size, self.embedding_size))

    def one_hot(self, phrase_as_indices):
        phrase_len = len(phrase_as_indices)
        one_hot = np.zeros((phrase_len, self.vocab_size))

        for index, token in enumerate(phrase_as_indices):
            # print(str(token) + ' and ' + str(index))
            # print(np.shape(one_hot))
            one_hot[index, token] = 1
        return one_hot

    def cross_entropy(self, y_true, y_pred, vector_wise=False):  # calculates cross entropy loss, assumes y's are SxN
        cross_ent = np.zeros(shape=(1, np.shape(y_true)[0]))
        iterator = 0
        epsilon = 1e-6
        for vec_true, vec_pred in zip(y_true, y_pred):
            pre_sum = [(true*(np.log(max(pred, epsilon)))) for true, pred in zip(vec_true, vec_pred)]
            summation = -1*np.sum(pre_sum)
            cross_ent[0, iterator] = summation
            iterator += 1
        if vector_wise is False:
            cross_ent = np.sum(cross_ent)*(1/np.shape(cross_ent)[1])
        return cross_ent

    def forward_prop(self, x, y_true, vector_wise_loss=False):
        # print('x: ' + str(np.shape(x)))
        embedding = x@self.U
        # print('embedding: ' + str(np.shape(embedding)))
        pre_soft = embedding@self.V
        # print('presoft: ' + str(np.shape(pre_soft)))
        y_pred = softmax(pre_soft, axis=1)

        self.back_buf = x, embedding, y_pred, y_true
        self.loss = self.cross_entropy(y_true, y_pred, vector_wise_loss)
        print("Loss is: " + str(self.loss))

        return y_pred, self.loss

    def back_prop(self):
        x, embedding, y_pred, y_true = self.back_buf
        # print('y_pred: ' + str(np.shape(y_pred)))
        # print('y_true: ' + str(np.shape(y_true)))
        dV = embedding.T@(y_pred-y_true)
        dU = x.T@(self.V@(y_pred-y_true).T).T

        max_norm = 5.0
        norm_dU = np.linalg.norm(dU)
        if norm_dU > max_norm:
            dU = dU * (max_norm / norm_dU)
        norm_dV = np.linalg.norm(dV)
        if norm_dV > max_norm:
            dV = dV * (max_norm / norm_dV)

        return dU, dV

    def training_step(self, dU, dV, learning_rate):
        self.U = self.U - learning_rate*dU
        self.V = self.V - learning_rate*dV

    def training_pairs(self, context):
        input_set = []
        true_val_set = []
        for x, y in context:
            input_set.append(x)
            true_val_set.append(y)
        input_set = self.one_hot(input_set)
        true_val_set = self.one_hot(true_val_set)
        return input_set, true_val_set
