from PatPho import pat_pho
from Celex.EPWRepo import epw_repo


def get_phoneme_vectors(words, left):
    """
    Return a dictionary mapping word strings to phoneme vectors.

    :type words:     array
    :param words:    word strings for which you want to obtain phonological vectors.

    :type left:     bool
    :param left:    if True, phoneme vectors will emphasize beginning of words; else, emphasize endings of words
                    (see parameter 'left' in 'PatPho' class in /PatPho/PatPho.py
    """
    word_vector_mapping = dict()
    for word in words:
        t_without_pos = word.split('-')[0]
        phonemes = epw_repo.get_phonologicalform(t_without_pos)
        v = pat_pho.get_phon_vector(phonemes, left)
        word_vector_mapping[word] = v
    return word_vector_mapping


def get_primary_stress_vectors(words):
    """
    Return a dictionary mapping word strings to lexical stress vectors.

    :type words:     array
    :param words:    word strings for which you want to obtain phonological vectors.
    """
    word_vector_mapping = dict()
    for word in words:
        t_without_pos = word.split('-')[0]
        v = epw_repo.get_primary_stress_pattern(t_without_pos)
        word_vector_mapping[word] = v
    return word_vector_mapping