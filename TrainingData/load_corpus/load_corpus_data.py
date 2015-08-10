import operator
from collections import Counter, defaultdict
from CHILDES.pickled.load_pickled import unpickle_cds

# the most important CHILDES tags (according to the CHAT manual -- the actual tags used in some corpora may differ

# 1  Adjective             --      ADJ
# 2  Adverb                --      ADV
# 3  Communicator          --      CO
# 4  Conjunction           --      CONJ
# 5  Determiner            --      DET
# 6  Filler                --      FIL
# 7  Infinitive marker to  --      INF
# 8  Noun                  --      N
# 9  Proper Noun           --      N:PROP
# 10 Number                --      DET:NUM
# 11 Particle              --      PTL
# 12 Preposition           --      PREP
# 13 Pronoun               --      PRO
# 14 Quantifier            --      QN
# 15 Verb                  --      V
# 16 Aux. (inc. modals)    --      V:AUX
# 17 WH words              --      WH

# list of POS tags that are considered closed class word tags
function_tags = {'adv', 'cop', 'ptl', 'co', 'rel', 'post', 'neg', 'pro', 'det', 'coord', 'prep', 'num', 'meta', 'mod',
                 'part', 'aux', 'conj', 'wh', 'inf', 'qn'}


def collapse_function_tags(pos_tag):
    # replace 'pos_tag' with a single function word
    # tag 'fn', if 'pos_tag' is a function word tag
    if pos_tag in function_tags:
        return 'fn'
    else:
        return pos_tag


def get_cds_words(collapse_function_words=True):
    """
    Load CDS from disk and return lower-cased tokens, as a list of sentences, where a sentence is a list of
    POS-ttagged tokens. Tokens are in the format 'word-pos_tag'. Optionally replace all closed class / function word
    POS tags with the single tag 'fn'.
    """
    sentences = []
    CDS = unpickle_cds()
    for file_name in CDS:
        for s in CDS[file_name]:
            sentences.append([])
            for w, pos_tag in s:
                w = w.lower()
                if collapse_function_words:
                    sentences[-1].append(w + '-' + collapse_function_tags(pos_tag))
                else:
                    sentences[-1].append(w + '-' + pos_tag)
    return sentences


def get_word_frequencies():
    """ Return a list of (word, frequency) tuples, sorted by frequency, form most to least frequent. """
    tagged_corpus = unpickle_cds()
    words = []
    for file_ in tagged_corpus:
        for sentence_ in tagged_corpus[file_]:
            sentence_ = [(token, collapse_function_tags(pos_tag)) for token, pos_tag in sentence_]
            words += sentence_
    counted_tokens = Counter(words)
    word_frequencies = sorted(counted_tokens.items(), key=operator.itemgetter(1))
    word_frequencies.reverse()
    return word_frequencies


def get_target_words(extract_this_many_words, tags=('all_tags',), exclude_words=()):
    """
    Get the 'extract_this_many_words' most frequent words. Return only words whose POS tag is included in 'tags'.
    Skip words that are included in 'exclude_words'.
    """
    context_words = []
    word_frequencies = get_word_frequencies()
    pos_dict = defaultdict(int)
    n_words = 0
    for word_pos, frequency in word_frequencies:

        word, pos_tag = word_pos
        if word in exclude_words:
            continue

        if pos_tag in tags or tags == ('all_tags',):
            pos_dict[word_pos[1]] += 1
            context_words.append(word + '-' + pos_tag)
            n_words += 1

        if n_words == extract_this_many_words:
            print 'Extracted the following number of words, by POS tag:'
            for pos_tag in pos_dict:
                print pos_tag, pos_dict[pos_tag]
            print

            return context_words
