import numpy as np
from itertools import combinations

class decoder:
    def __init__(self, k=10, parities=None):
        # generate the k valid outputs of each model
        self.k = k
        self.checks = [[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6],
                        [7, 7], [8, 8], [9, 9]]]
        if parities is not None:
            self.checks.append(parities)
        self.modelOutputs = []
        for check in self.checks:
            self.modelOutputs.append(self.generateCodeword(self.k,
                                                           check))

    def generateCodeword(self, k, checks):
        # generate the k valid outputs of a model defined by its parity checks
        codewords = []
        for i in range(k):
            sysBits = np.zeros(k, dtype='uint8')
            sysBits[i] = 1
            code = np.zeros(len(checks), dtype='uint8')
            for j in range(len(checks)):
                for idx in checks[j]:
                    code[j] |= sysBits[idx]
            codewords.append(code)
        return codewords

    def decode(self, v, selectedMods=[0, 1]):
        # input v: a binary numpy array of datatype 'uint8'
        #       selectedMods: ascending ordered model indices
        codewords = []  # generate the k valid outputs of the selected models
        for i in range(self.k):
            code = []
            for m in selectedMods:
                code += list(self.modelOutputs[m][i])
            codewords.append(np.array(code, dtype='uint8'))
        distance = self.k
        decision = np.random.randint(self.k)  # first make a random decision
        # enable random selection when distances tie up
        idxPool = np.random.permutation(self.k)
        for idx in idxPool:
            dis = sum(v != codewords[idx])
            if dis <= distance:
                decision = idx
                distance = dis
        return decision


def test():
    dc = decoder()
    for i in range(10):
        expected = np.zeros(10, dtype='uint8')
        expected[i] = 1
        assert (dc.modelOutputs[0][i] == expected).all()

        expected = np.zeros(5, dtype='uint8')
        expected[int(i / 2)] = 1
        assert (dc.modelOutputs[1][i] == expected).all()

    assert (dc.modelOutputs[2][4] == np.array([0, 0, 1, 0, 0])).all()
    assert (dc.modelOutputs[3][5] == np.array([1, 0, 0, 0, 0])).all()
    assert (dc.modelOutputs[4][6] == np.array([0, 0, 0, 1, 0])).all()
    assert (dc.modelOutputs[5][7] == np.array([0, 0, 0, 0, 1])).all()

    v = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert dc.decode(v, [0]) == 0
    v = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    assert dc.decode(v, [0, 1]) == 1
    # errorneous codeword equally close to 1, 2, and 3
    v = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    for i in range(10):
        assert dc.decode(v, [0, 1]) in [1, 2, 3]


#test()
