from itertools import combinations
import numpy as np

def bit(idx):
    return 1 << idx

def createParityMap(numClasses):
    bits = int(np.log2(numClasses)) + 1
    mapping = {}
#    for b0, b1 in combinations(range(bits), 2):
#        mapping[(b0,b1)] = []
#        for c in range(numClasses):
#            c0 = (c & bit(b0)) >> b0
#            c1 = (c & bit(b1)) >> b1
#            if c0 ^ c1:
#                mapping[(b0,b1)].append(c)
    for p0, p1 in combinations(range(numClasses), 2):
        mapping[(p0,p1)] = [p0, p1]
    return mapping
        
if __name__ == '__main__':
    mapping = createParityMap(10)
    print(len(mapping))
    i = 1
    for key in sorted(mapping.keys()):
        print(key, i)
        i += 1
        print(len(mapping[key]))
