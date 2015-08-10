import os
import pickle
from nltk.corpus.reader import CHILDESCorpusReader


def get_file_path():
    return os.path.dirname(os.path.realpath(__file__))


def pickle_tagged_sents(speakers, file_name, max_age=2000, ENG_NA=True, ENG_UK=True, stemmed=False):
    """
    Create and pickle a dictionary with CHILDES file names as keys and sentences of tagged words as values.
    A tagged sentence is list of (word, POS tag) tuples.

    :param speakers:    list of speaker IDs whose utterances will be considered
    :param file_name:   name of pickled output file
    :param max_age:     the maximum child age in months for which you want to consider corpus files. If this is a large
                        number (e.g. the default 2000), no files will be excluded on the basis of age.
    :param ENG_NA:      if True, will consider US corpora
    :param ENG_UK:      if True, will consider UK corpora
    :param stemmed:     if True, sentences will be lists of (stem, POS tag) tuples; else, they will be lists of
                        (token, POS tag) tuples
    """
    sent_dict = dict()

    if ENG_NA:
        corpus_reader = CHILDESCorpusReader(get_file_path() + '/Corpora/ENG-NA/', u'.*.xml')
        to_sent_dict(sent_dict, corpus_reader, max_age, speakers, stemmed)
    if ENG_UK:
        corpus_reader = CHILDESCorpusReader(get_file_path() + '/Corpora/ENG-UK/', u'.*.xml')
        to_sent_dict(sent_dict, corpus_reader, max_age, speakers, stemmed)

    path_ = os.getcwd() + '/pickled/' + file_name
    pickle.dump(sent_dict, open(path_, "wb"))


def to_sent_dict(sent_dict, corpus_reader, max_age, speakers, stemmed):
    """ Go through files and store cleaned sentences in 'sent_dict' """
    for filename in corpus_reader.fileids():

        age = corpus_reader.age(filename, month=True)[0]

        # skip the file if age is undefined (safest possible option)
        if age is None:
            continue
        elif age > max_age:
            continue

        sents = corpus_reader.tagged_sents(filename, speaker=speakers, stem=stemmed)
        sent_dict[filename] = clean_sents(sents)


def clean_sents(sents):
    new_sents = []
    for s in sents:
        # sometimes we get empty sentences
        if len(s) == 0:
            continue
        converted = [convert_word_pos(w_pos) for w_pos in s]
        new_sents.append(converted)
    return new_sents


def convert_word_pos(w_pos):

    # sometimes words come as single strings, instead of as a (word, pos tag) tuple
    if type(w_pos) is str:
        w = w_pos.lower()
        # if we have any a single string, instead of the expected tuple, annotate with 'None' (a string)
        w_pos = w, str(None)

    # if 'w_pos' is not a string, it is a (word, POS tag) tuple
    w, pos = w_pos

    # there is a small number of words with non-ASCII characters
    # quick solution: replace the offending word with the string 'not_in_ascii'
    # the number of affected words is so small that this doesn't make a detectable difference (107 in NA and UK corpora)
    if not is_ascii(w):
        print 'Replacing %s with "not_in_ascii"' % w
        w = 'not_in_ascii'

    # some words are annotated with affix tags
    # in such cases, the tag follows a hyphen ('-') after the word
    if '-' in w:
        w = w.split('-')[0]

    # in case the word was unintelligible to the transcriber, plus possibly other conditions, the POS tag is None
    # we convert it to str so we can apply string functions to all POS tags without checking for its type in the future
    if pos is None:
        return w, str(pos)

    # sometimes a tag is further specified by information following a colon
    # we are only interested in the general tag to the left of the colon
    if ':' in pos:
        pos = pos.split(':')[0]

    # for some reason, a small number of tags has a '0' to the left
    if '0' in pos:
        pos = pos.lstrip('0')

    return w.lower(), pos.lower()


def is_ascii(w):
    # check if 'w' contains only ASCII chars
    try:
        w.decode('ascii')
        return True
    except UnicodeEncodeError:
        return False


# lists of speaker IDs

all_adults = ["AD1", "AD2", "AD3", "ADU", "ANG", "ART", "AU2", "AU3", "AUD", "AUN", "BOB", "CAR", "CAT", "CLA", "CLI",
              "COL", "COU", "DAD", "DAN", "DEB", "DOU", "EDN", "ELL", "ELS", "ENV", "ERI", "ERN", "EXP", "FAD", "FAI",
              "FAT", "FRI", "GAI", "GIN", "GLO", "GR1", "GR2", "GRA", "GRD", "GRF", "GRM", "GUE", "HAR", "HEL", "HOU",
              "HUS", "IN2", "INV", "JAC", "JAM", "JAN", "JEA", "JEN", "JES", "JIM", "JJJ", "JUD", "JUI", "KAR", "KEI",
              "KEL", "KEN", "KKK", "KRI", "KUR", "LAU", "LEA", "LES", "LIN", "LLL", "LOI", "MAD", "MAI", "MAN", "MEL",
              "MIC", "MOT", "NAN", "NEI", "NNN", "NOE", "NON", "OBS", "OP1", "OP2", "OPE", "OTH", "PER", "RAC", "RAN",
              "REB", "REP", "RIC", "ROG", "RRR", "RUS", "RYA", "SIL", "SSS", "SUS", "TCA", "TEA", "TEL", "TER", "THA",
              "TO1", "TO2", "TO3", "TOM", "TOY", "TTT", "UNC", "URS", "VI1", "VI2", "VIS", "WAL", "WAY", "WEN", "WOR",
              "WWW"]

father_mother = ["FAT", "MOT"]

child = ["CHI"]



if __name__ == '__main__':
    # pickle CDS by mother and father only
    pickle_tagged_sents(speakers=father_mother, file_name='CDS.p', stemmed=False)

    # pickle utterances by all adults in the corpora
    # pickle_tagged_sents(speakers=all_adults, file_name='CDS_all_adults.p', stemmed=False)