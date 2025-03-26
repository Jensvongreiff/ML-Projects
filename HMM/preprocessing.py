import numpy as np
from collections import Counter

# from typing import List, Sequence, Tuple, Dict
# from numpy.typing import NDArray

# data is a dictionary with two keys, reviews_5star and reviews_1star
# these keys correspond to a list of lists; the inner list contains pieces of or entire tokenized reviews
# Each word corresponds to a single token
# data['key'][review][word]

# what we need for the model: input, output: unique word vector

UNKNOWN = '_unk'


def thing():  # ignore
    data = np.load("task03_data (1).npy", allow_pickle=True)
    data_checker = data.item()
    reviews_5star = data_checker['reviews_5star']
    print(data_checker.keys())
    print(data_checker['reviews_5star'])
    rev_len = []
    for rev in reviews_5star:
        rev_len.append(len(rev))
    print(rev_len)


def load_dataset(data):  # loads data set into a list[list[list]]
    data = np.load(data, allow_pickle=True).item()
    return [data[key] for key in data]


def vocab(full_data, multidata=False, vocab_size=200):  # takes in a list[list[list]] if multidata = True, otherwise list[list] of
    # your data, returns the filtered data(only 200 most common
    # words stay in the corpus) in chronological order as a list, the vocabulary(the 200 most common words and
    # UNKNOWN) as a tuple, and an incidence rate of the vocabulary, where vocab[index] corresponds to counts[index]
    # as a tuple
    if multidata:
        for sections in full_data:  # incorporates all data into a single list of lists
            data = [sections[i] for i in range(len(sections))]
    else:
        data = full_data
    flattened_data = [token for review in data for token in review]  # flattens data into a single list
    vocabulary, counts = zip(
        *Counter(flattened_data).most_common(vocab_size))  # sorts vocab and counts by 200 most commonly occurring
    processed_data = [word if word in vocabulary else UNKNOWN for word in flattened_data]  # filters the data
    vocabulary = vocabulary + (UNKNOWN,)
    counts = counts + (processed_data.count(UNKNOWN),)
    return processed_data, vocabulary, counts


def indicize_vocab(vocabulary, counts):  # takes in vocabulary and count tuples, returns three dictionary objects:
    # numbered_vocab = {index : word}, numbered_counts = {index : count}, inv_numbered_vocab = {word : index}
    numbered_vocab = dict(enumerate(vocabulary))
    numbered_counts = dict(enumerate(counts))
    inv_numbered_vocab = {word: index for index, word in numbered_vocab.items()}
    return numbered_vocab, numbered_counts, inv_numbered_vocab


def window_slide(phrase_ls, window_size, vocab_to_number):  # return list[tuple] of co-occurance in terms of the indices
    # that correspond to the vocabulary, takes in phrase = str, window_size = int, vocab_to_number = aforementioned
    # inv_numbered_vocab from indicize_vocab()
    context = []
    # Rework phrase into a usable list
    if str(phrase_ls) is phrase_ls: phrase_ls = phrase_ls.split(' ')
    # jesus christ
    for spec_word in phrase_ls:
        for index, word in enumerate(phrase_ls):
            if word is spec_word:
                slicer_1 = index - window_size
                if slicer_1 < 0: slicer_1 = 0
                slicer_2 = index + window_size
                if slicer_2 > len(phrase_ls): slicer_2 = len(phrase_ls) - 1
                subphrase = phrase_ls[slicer_1:slicer_2 + 1]  # extracts context window
                # print(subphrase)
                for target in subphrase:
                    if target is not spec_word:
                        context.append((vocab_to_number[spec_word], vocab_to_number[target]))
        # print(context)
        # print(str(spec_word) + ' done')
    return context


def generate_minibatches(data, batch_size):
    """Yield successive minibatches from the data."""
    print('Generating minibatch')
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

#
# def get_token_pairs_from_window(sequence: Sequence[str], window_size: int, token_to_index: Dict[str, int]) -> Sequence[Tuple[int, int]]:
#     """ Collect all ordered token pairs from a sentence (sequence) that are at most `window_size` apart.
#     Note that duplicates should appear more than once, e.g. for "to be to", (to, be) should be returned more than once, as "be"
#     is in the context of both the first and the second "to".
#
#     Args:
#         sequence (Sequence[str]): The sentence to get tokens from
#         window_size (int): The maximal window size
#         token_to_index (Dict[str, int]): Mapping from tokens to numerical indices
#
#     Returns:
#         Sequence[Tuple[int, int]]: A list of pairs (token_index, token_in_context_index) with pairs of tokens that co-occur, represented by their numerical index.
#     """
#     sequence = sequence.split(' ')
#     newlist2 = []
#     c=0
#     d=0
#     e=0
#     for word in sequence:
#
#         if sequence.index(word, d) + window_size >= len(sequence):
#             for x in range(len(sequence) - sequence.index(word, d)):
#                 if x == 0:
#                     c = 1
#                 else:
#                     newlist2.append(( token_to_index[word], token_to_index[sequence[sequence.index(word, d) + x]] ))
#         else:
#             for x in range(window_size+1):
#                 if x == 0:
#                     c = 1
#                 else:
#                     newlist2.append(( token_to_index[word], token_to_index[sequence[sequence.index(word, d) + x]] ))
#
#         if sequence.index(word, d) < window_size:
#             for x in range(sequence.index(word, d)+1):
#                 if x == 0:
#                     c = 1
#                 else:
#                     newlist2.append(( token_to_index[word], token_to_index[sequence[sequence.index(word, d) - x]] ))
#         else:
#             for x in range(window_size+1):
#                 if x == 0:
#                     c = 1
#                 else:
#                     newlist2.append(( token_to_index[word], token_to_index[sequence[sequence.index(word, d) - x]] ))
#         d += 1
#
#     return newlist2
