from itertools import combinations
import numpy as np

def bit(idx):
    return 1 << idx

def createParityMap(numClasses):
    bits = int(np.log2(numClasses)) + 1
    mapping = {}
    for b0, b1 in combinations(range(bits), 2):
        mapping[(b0,b1)] = []
        for c in range(numClasses):
            c0 = (c & bit(b0)) >> b0
            c1 = (c & bit(b1)) >> b1
            if c0 ^ c1:
                mapping[(b0,b1)].append(c)
    return mapping
        
if __name__ == '__main__':
    mapping = createParityMap(100)
    print(len(mapping))
    for key in sorted(mapping.keys()):
        print(key)
        print(len(mapping[key]))
