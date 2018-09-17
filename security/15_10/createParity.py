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
#    for p0, p1 in combinations(range(numClasses), 2):
#        mapping[(p0,p1)] = [p0, p1]
#    for p0, p1, p2 in combinations(range(numClasses), 3):
#        mapping[(p0,p1,p2)] = [p0, p1, p2]
#    gMatrix = np.array([[1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#                        [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
#                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
#                        [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
#                        [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
#                        [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#                        [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
#                        [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0]])

#    gMatrix = np.array([[0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
#                        [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
#                        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
#                        [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
#                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
#                        [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
#                        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
#                        [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
#                        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
#                        [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]])

    gMatrix = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                        [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
                        [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                        [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
                        [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
                        [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
                        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]])
    
    for p in range(gMatrix.shape[1]):
        k = np.where(gMatrix[:, p] == 1)[0].tolist()
        k = tuple(k)
        mapping[k] = k
    return mapping
        
if __name__ == '__main__':
    mapping = createParityMap(10)
    print(len(mapping))
    for key in sorted(mapping.keys(), key=lambda a: (len(a), a), reverse=True):
        print(key)
        print(len(mapping[key]))
