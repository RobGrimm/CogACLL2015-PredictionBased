class PatPho(object):

    def __init__(self):
        """
        Python re-implementation of of PatPho -- a system for converting sequences of phonemes to vector representations
        that capture phonological similarity of words.

        The system is described in:

            Li, P., & MacWhinney, B. (2002). PatPho: A phonological pattern generator for neural networks.
                Behavior Research Methods, Instruments, & Computers, 34(3), 408-415.

        The original C implementation can be found here (June 2015): http://www.personal.psu.edu/pul8/patpho_e.shtml
        """
        self.syllabic_grid = None
        self.idx = None
        self.init_syllabic_grid_and_index()

        self.current_phonemes = None

        self.vowels = {"i", "I", "e", "E", "&", "@", "3", "V", "a", "U", "u", "O", "A", "Q"}

        # map phonemes to their vector representations
        self.phonemes = {"i": [0, 1, 0, 1, 1], "I": [0, 1, 0, 0, 1], "e": [0, 1, 1, 0, 1], "E": [0, 1, 1, 1, 0],
                         "&": [0, 1, 1, 0, 0], "@": [1, 1, 1, 0, 0], "3": [1, 1, 0, 0, 1], "V": [1, 1, 1, 1, 0],
                         "a": [1, 1, 1, 0, 0], "U": [1, 0, 0, 1, 1], "u": [1, 0, 0, 0, 1], "O": [1, 0, 1, 0, 1],
                         "A": [1, 0, 1, 0, 0], "Q": [1, 0, 1, 1, 0], 'VO': [0, 0, 0, 0, 0],
                         "p": [0, 0, 0, 0, 0, 1, 0], "t": [0, 0, 1, 1, 0, 1, 0], "k": [0, 1, 1, 0, 0, 1, 0],
                         "b": [1, 0, 0, 0, 0, 1, 0], "d": [1, 0, 1, 1, 0, 1, 0], "g": [1, 1, 1, 0, 0, 1, 0],
                         "m": [1, 0, 0, 0, 0, 0, 1], "n": [1, 0, 1, 1, 0, 0, 1], "N": [1, 1, 1, 0, 0, 0, 1],
                         "l": [1, 0, 1, 1, 0, 1, 1], "r": [1, 0, 1, 1, 1, 1, 0], "f": [0, 0, 0, 1, 1, 0, 0],
                         "v": [1, 0, 0, 1, 1, 0, 0], "s": [0, 0, 1, 1, 1, 0, 0], "z": [1, 0, 1, 1, 1, 0, 0],
                         "S": [0, 1, 0, 0, 1, 0, 0], "Z": [1, 1, 0, 0, 1, 0, 0], "j": [1, 1, 0, 1, 0, 1, 1],
                         "h": [0, 1, 1, 1, 0, 1, 1], "w": [1, 1, 1, 0, 0, 1, 1], "T": [0, 0, 1, 0, 1, 0, 0],
                         "D": [1, 0, 1, 0, 1, 0, 0], "C": [0, 1, 0, 1, 1, 0, 0], "J": [1, 1, 0, 1, 1, 0, 0],
                         'CO': [0, 0, 0, 0, 0, 0, 0]}


    def init_syllabic_grid_and_index(self):
        """ initialize trisyllabic consonant-vowel (CO-VO) grid """
        self.idx = 0
        self.syllabic_grid = ['CO', 'CO', 'CO', 'VO', 'VO', 'CO', 'CO', 'CO', 'VO', 'VO', 'CO', 'CO', 'CO', 'VO', 'VO',
                              'CO', 'CO', 'CO']


    def index_to_next_vowel(self):
        """ increment self.idx to next empty vowel postion in syllabic grid """
        self.idx += self.syllabic_grid[self.idx:].index("VO")


    def index_to_next_consonant(self):
        """ increment self.idx to next empty consonant postion in syllabic grid """
        self.idx += self.syllabic_grid[self.idx:].index("CO")


    def insert_phoneme_into_grid(self, phoneme):
        """" place 'phoneme' on the syllabic grid """
        if phoneme in self.vowels:
            try:
                self.index_to_next_vowel()
                self.syllabic_grid[self.idx] = phoneme
            except ValueError:
                print 'Word is too long: %s' % self.current_phonemes


        elif phoneme in self.phonemes:
            try:
                self.index_to_next_consonant()
                self.syllabic_grid[self.idx] = phoneme
            except ValueError:
                print 'Word is too long: %s' % self.current_phonemes

        else:
            raise Exception('Unknown phoneme in %s (%s chars long): %s' % (self.current_phonemes,
                                                                           len(self.current_phonemes), phoneme))

    def get_phon_vector(self, phonemes, left=True):
        """
        Convert the phoneme sequence to a vector representation.

        :type left:     bool
        :param left:    if True, place phonemes on consonant-vowel grid starting from the left (left-justified format),
                        which emphasizes similarities of word-vectors at the beginning; else, place phonemes on the grid
                        going from right to left, which emphasizes similarities of word endings
        """
        # for debugging
        self.current_phonemes = phonemes

        if not left:
            phonemes = phonemes[::-1]

        # go through the phonemes and insert them into the metrical grid
        for p in phonemes:
            self.insert_phoneme_into_grid(p)

        # convert syllabic grid to vector
        if not left:
            self.syllabic_grid = self.syllabic_grid[::-1]

        phon_vector = []
        for i in self.syllabic_grid:
            phon_vector.extend(self.phonemes[i])

        # reset syllabic grid
        self.init_syllabic_grid_and_index()

        return phon_vector


pat_pho = PatPho()


if __name__ == "__main__":
    # some test cases
    print pat_pho.get_phon_vector('@Uld')           # adjective 'old'
    print pat_pho.get_phon_vector('@uld', False)    # adjective 'old', right-justified
    print pat_pho.get_phon_vector('weIt')           # verb 'wait'
    print pat_pho.get_phon_vector('hI@')            # verb 'hear'