from TrainingData.data_sets.extract_from_corpus import create_and_save_data_set
from TrainingData.load_corpus.load_corpus_data import get_target_words


# exclude the following words, all longer than three syllables, which is not supported by the system ued to
# construct phonological feature vectors
more_than_three_syllables = ['actually', 'anybody', 'investigator', 'helicopter', 'interesting', 'cookie_monster',
                             'definitely', 'refrigerator',  'oh_my_goodness',  'humpty_dumpty', 'interested',
                             'everybody', 'father_christmas', 'helicopter', 'alligator', 'caterpillar', "everybody's",
                             'hippopotamus']

# also exclude the empty string, which occasionally occurs in the corpus
empty_string = ['']


# get vocabulary -- this is a list of the 2000 most frequent word strings,
#  with each word is given in the format 'word-pos_tag'.
# there are four POS tags: 'v' (verb), 'n' (noun), 'adj' (adjective) 'fn' (function word / closed class word)
vocabulary = get_target_words(2000, tags={'n', 'v', 'adj', 'fn'},
                               exclude_words=more_than_three_syllables + empty_string)

# the vocabulary composition by POS tag is as follows:
# nouns:        1010
# adjectiveS:   166
# closed class: 522
# verbs:        302


# go through the corpus and extract the text-based part of the data set, consisting of (1) vectors with frequency counts
# for each target word's left context and sampled (2) words from the right context; save extracted data set to disk
create_and_save_data_set('CDS', vocabulary=vocabulary, window_size=3)