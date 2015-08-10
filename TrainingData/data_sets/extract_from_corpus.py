import os
import random
import pickle
import numpy as np
from TrainingData.load_corpus.load_corpus_data import get_cds_words

# seed rng for sampling of contexts in function 'get_data_set'
random.seed(6675)


# helper functions

def get_file_path():
    return os.path.dirname(os.path.realpath(__file__))


def scale_to_unit_interval(np_array):
    return np_array * 1.0 / np.max(np_array)


def next_words_to_int(next_words):
    unique_next_words = list(set(next_words))
    to_int = dict(zip(unique_next_words, range(len(next_words))))
    ints = [to_int[word] for word in next_words]
    return ints


def sample_right_context_words(target_word, window_size, sent, index, vocabulary, right_context_words, target_words):
    # sampled sentence-internal window to the right
    # only consider context words within this window
    t = random.randint(1, window_size)
    to_right = min(len(sent), index + t + 1)

    for next_word_idx in range(index + 1, to_right):
        context_word = sent[next_word_idx]
        if context_word in vocabulary:
            # (context_word, target_word) pair
            right_context_words.append(context_word)
            target_words.append(target_word)


def increment_left_context_vector(target_word, index, window_size, sent, left_context_words, left_context_vectors,
                                  tokens_to_int):
    # sentence internal window to the left
    # only consider context words within this window
    from_left = max(0, index - window_size)
    for cxword_index in range(from_left, index):
        context_word = sent[cxword_index]

        if context_word in left_context_vectors:
            # increment count in left_context_vector
            column_idx = tokens_to_int[context_word]
            left_context_vectors[target_word][column_idx] += 1
            # keep track of the number of left-context words
            left_context_words.add(context_word)

########################################################################################################################


# get / save data set functions

def create_and_save_data_set(file_name, vocabulary, window_size):
    """ Call 'get_data_set' and save result to file. """
    target_word_ints, right_context_word_ints, embeddings_dict = get_data_set(vocabulary, window_size)
    assert len(right_context_word_ints) == len(target_word_ints)
    data_set = {'target_words': target_word_ints,
                'right_context_words': right_context_word_ints,
                'embeddings_dict': embeddings_dict}

    data_dir = get_file_path() + '/textual/'
    pickle.dump(data_set, open(data_dir + file_name, 'wb'))


def get_data_set(vocabulary, window_size):
    """
    Extract and return a data set consisting of normalized frequency vectors for the left context and of
    sampled target-context word pairs for the right context.

    In the training corpus, consider in turn each sentence $S$ of length $N$. Consider each word $w_n$ at position
    $n <= N$ as a target word iff $w_n$ is included in the vocabulary $V$. Extract the left context of $w_n$ and sample
    words from the right context of $w_n$. This is done as follows:

    - extracting the left context:

            For each target word in the vocabulary $V$, create a left-context vector $v$ of zeros before iterating
            through the corpus. Each value $v_i$ in $v$ corresponds to a word $w_i$ in the vocabulary, and each word
            $w_i$ in the vocabulary corresponds to a value $v_i$ in $v$.
            When iterating through the corpus, for a given target word $w_n$, if $w_i$ occurs within
            a sentence-internal window of $window_size$ words to the left of $w_n$, increment $v_i$ by one.
            After having gone through the entire corpus, normalize $v$ to unit interval.
            $v$ is thus a normalized vector of left-context word frequencies.

    - sampling words from the right context:

            Given target word $w_n$, sample an integer $t$ from the uniform distribution ${1, ... window_size}$.
            Then, consider each word $w_j$ within a sentence-internal window of $t$ words to the right of $w_n$
            as a right-context word. If $w_j$ is in the vocabulary, add the target-context word pair $(w_n, w_j)$ to
            the training set.

    :type vocabulary:       array
    :param vocabulary:      an array of word strings, in the format 'word-pos_tag', where 'pos_tag' is one
                            of 'v' (verb), 'n' (noun), 'adj' (adjective), 'fn' (function word / closed class word).

    :type window_size:      int
    :param window_size:     sentence-internal window of words to the left and right of target words within which
                            potential context words will be considered

    :return: target_words           list of target word strings

    :return: right_context_words    list of right-context word strings. each right-context word at position
                                    $i$ in 'right_context_words' occurred in the right context of the
                                    target word string at position $i$ in 'target_words'

    :return: embeddings_dict        dictionary mapping  each target word strings to a left-context vector
    """
    print 'Extracting data set...'
    # map each token from the vocabulary to a unique integer
    tokens_to_int = dict(zip(vocabulary, range(len(vocabulary))))

    # store left-context-vectors by target word index
    left_context_vectors = {w: np.zeros(len(vocabulary)) for w in vocabulary}

    left_context_words = set()      # keeps track of all left-context words (for which we increment counts in
                                    # left-context vectors)

    target_words = []               # tokens for which we collect words from the right context, stored as integers
    right_context_words = []        # words sampled from the right context of target words, stored as integers
                                    # each context word at index $i$ in 'right_context_word_ints' is sampled
                                    # from the right context of the target word at position $i$ in 'target_word_ints'

    # list of lists of tokens; each list of tokens is a sentence containing tagged tokens
    words_by_sents = get_cds_words(collapse_function_words=True)

    for sent in words_by_sents:

        for index, target_word in enumerate(sent):

            if target_word in vocabulary:

                sample_right_context_words(target_word, window_size, sent, index, vocabulary, right_context_words,
                                           target_words)

                increment_left_context_vector(target_word, index, window_size, sent, left_context_words,
                                              left_context_vectors, tokens_to_int)

    print '...done. Got a data set with %s right-context words and frequencies for %s / %s left-context words.' % \
          (len(set(right_context_words)), len(left_context_words), len(vocabulary))
    print 'Number of sampled right-context words: %s' % len(right_context_words)

    # normalize left-context vectors to unit interval
    embeddings = [scale_to_unit_interval(v) for v in left_context_vectors.values()]
    embeddings_dict = dict(zip(left_context_vectors.keys(), embeddings))

    return target_words, right_context_words, embeddings_dict