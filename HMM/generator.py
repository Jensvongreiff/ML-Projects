import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import preprocessing as pp
matplotlib.use("TkAgg")

class HMM_Parameters:

    def __init__(self, n_states, n_symbols):
        """ Makes three randomly initialized stochastic matrices `self.A`, `self.B`, `self.pi`.

        Parameters
        ----------
        n_states: int
                  number of possible values for Z_t.
        n_symbols: int
                  number of possible values for X_t.

        Returns
        -------
        None

        """
        self.A = self.random_mat(n_states, n_states)
        self.B = self.random_mat(n_states, n_symbols)
        self.pi = self.random_mat(1, n_states).transpose()

    def random_mat(self, I, J):
        """
        Returns a randomly initialized stochastic matrix with shape (I, J),
        where each row is a valid probability distribution (non-negative, sums to 1).
        """
        x = np.full((I, J), (1 / J))
        x += np.random.randn(I, J) * (1.0 / (J * J))
        x /= np.sum(x, axis=1, keepdims=True)
        return x


class HMM_Gen:

    def __init__(self, corpus, n_states):
        self.corpus = corpus.copy()
        flattened_data = [token for review in corpus for token in review]  # flattens data into a single list
        wordList = []
        for word in flattened_data:  # creates unique word list
            if word not in wordList: wordList.append(word)

        self.wordList = wordList
        self.wordDict = dict(enumerate(self.wordList))  # Creates dict from unique word list
        self.n_symbols = len(self.wordList)  # Dimension N, number of vocab
        self.n_states = n_states  # Dimension K, number of latent states
        self.parameters = HMM_Parameters(self.n_states, self.n_symbols)

    def forwards(self, sequence):  # Computes forward for sequence(length) where sequence is the numbers that correspond
        # to the words,returns alphas organized by row: [[a1],[a2],...] so (KxT)
        seqLen = len(sequence)
        pi = self.parameters.pi.flatten()
        B = self.parameters.B
        A = self.parameters.A
        initAlpha = pi * B[:, sequence[0]]
        alphas = np.ones(shape=(seqLen, len(pi)))
        alphas[0, :] = initAlpha
        i = 1
        while i < seqLen:
            next = A.T @ alphas[i - 1, :]
            next = B[:, sequence[i]] * next
            alphas[i, :] = next
            i += 1
        return alphas

    def backwards(self, sequence):  # Same structure for forwards, but for betas (KxT)
        seqLen = len(sequence)
        pi = self.parameters.pi.flatten()
        B = self.parameters.B
        A = self.parameters.A
        betas = np.ones(shape=(seqLen, len(pi)))
        for i in reversed(range(seqLen - 1)):
            next = B[:, sequence[i + 1]] * betas[i + 1, :]
            next = A @ next
            betas[i, :] = next
        return betas

    def log_likelyhood(self, alphas):  # Computes log likelyhood for a set of observations
        end = np.shape(alphas)[0]
        prob = np.sum(alphas[end-1, :])
        log_like = np.log(prob + 1e-300)
        return log_like

    def log_like_corpus(self):  # Computes log likelyhood over the entire corpus, functions as a loss measure
        log_like = 0
        for phrase in self.corpus:
            _, _, buf = self.forwardsBackwards(phrase)
            log_like += buf
        return log_like

    def phrase_to_number(self, phrase):  # takes in list[str] outputs list[int] according to dictionary
        inv_dict = {word: index for index, word in self.wordDict.items()}
        numWord = [inv_dict[word] for word in phrase]
        return numWord

    def num_to_phrase(self, numList):  # opposite of phrase_to_number
        phrase = [self.wordList[num] for num in numList]
        return phrase

    def forwardsBackwards(self, phrase):
        indices = self.phrase_to_number(phrase)
        alphas = self.forwards(indices)
        betas = self.backwards(indices)
        log_like = self.log_likelyhood(alphas)

        return alphas, betas, log_like

    def E_step(self, phrase):
        indices = self.phrase_to_number(phrase)
        alphas, betas, log_like = self.forwardsBackwards(phrase)
        phraseLen = len(phrase)

        A = self.parameters.A
        B = self.parameters.B

        # calculate gamma (TxK)
        gamma = np.ones(shape=np.shape(alphas))
        for i in range(phraseLen):
            numerator = alphas[i, :]*betas[i, :]
            denominator = np.sum(numerator)
            gamma[i, :] = numerator / denominator

        # calculate chi (K,K,T)
        chi = np.ones(shape=(self.n_states, self.n_states, phraseLen-1))
        for i in range(phraseLen-1):
            buf = betas[i+1, :]*B[:, indices[i+1]]
            numerator = (alphas[i, :, None]*A)*buf[None, :]
            denominator = np.sum(numerator)
            chi[:, :, i] = numerator / denominator

        return gamma, chi, indices

    def M_step(self, gamma, chi, indices):
        # updates pi
        pi = np.reshape(gamma[0, :], np.shape(self.parameters.pi))

        # Prep sum terms
        phraseLen = np.shape(gamma)[0]
        gammaSum = np.sum(gamma, axis=0)
        gammaRed = gamma[0:phraseLen-1, :]
        gammaSumRed = np.sum(gammaRed, axis=0)
        chiSum = np.sum(chi, axis=2)

        # calculate and update A
        A = np.divide(
            chiSum,
            gammaSumRed[:, None],
            out=np.zeros_like(chiSum),
            where=gammaSumRed[:, None] != 0
        )

        # calculate and update B
        B_temp = np.zeros((self.n_states, self.n_symbols))
        for k in range(self.n_states):
            for t in range(len(indices)):
                B_temp[k, indices[t]] += gamma[t, k]
            B_temp[k, :] /= np.sum(gamma[:, k])
        B = B_temp

        return pi, A, B

    def train(self, iterations, plot=True):
        likelihood_prog = []
        loss = self.log_like_corpus()
        likelihood_prog.append(loss)
        print('Initial Conditions: ' + 'have a log likelyhood of: ' + str(loss))
        for i in range(iterations):
            A = np.zeros_like(self.parameters.A)
            B = np.zeros_like(self.parameters.B)
            pi = np.zeros_like(self.parameters.pi)

            for phrase in self.corpus:  # Batching for the whole corpus
                gamma, chi, indices = self.E_step(phrase)
                temppi, tempA, tempB = self.M_step(gamma, chi, indices)
                A += tempA
                B += tempB
                pi += temppi
                #print('done')
            # Normalize to make them proper stochastic matrices
            self.parameters.pi = pi / np.sum(pi)
            self.parameters.A = A / A.sum(axis=1, keepdims=True)
            self.parameters.B = B / B.sum(axis=1, keepdims=True)
            #calc loss
            loss = self.log_like_corpus()
            print('Iteration: ' + str(i) + ' has a log likelyhood of: ' + str(loss))
            likelihood_prog.append(loss)
        # plot likelihood_prog
        if plot:
            plt.figure()
            plt.plot(range(iterations+1), likelihood_prog, label='Likelihood')
            plt.title('Iterations v Likelihood')
            plt.xlabel('Iterations')
            plt.ylabel('Likelihood')
            plt.show()

    def generate_sentence(self, sentence_length):
        """ Given the model parameter,generates an observed
            sequence of length `sentence_length`.
            Hint: after generating a list of word-indices like `x`, you can convert it to
                  an actual sentence as `self.X_to_sentence(x)`

        Parameters
        ----------
        sentence_length : int,
                        length of the generated sentence.

        Returns
        -------
        sent : a list of words, like ['the' , 'food' , 'was' , 'good']
               a sentence generated from the model.
        """

        A = self.parameters.A  # [KxK]
        B = self.parameters.B  # [KxV]
        pi = self.parameters.pi.flatten()  # [Kx1] ------

        x = []
        state_probs = pi
        for _ in range(sentence_length):
            obs_probs = state_probs @ B
            x.append(np.argmax(obs_probs))
            state_probs = state_probs @ A

        return self.num_to_phrase(x)
