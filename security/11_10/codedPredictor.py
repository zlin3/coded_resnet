import numpy as np
import decoder
from itertools import combinations

numClasses = 10

def convertOnehot(prediction):
    onehot = np.zeros((prediction.shape[0], prediction.max()+1))
    onehot[np.arange(prediction.shape[0]), prediction] = 1
    return onehot.astype(int)

def createCodewords():
    codewords = None
    for i in range(0, 2):
        if not i:
            pred = np.loadtxt('prediction_%d.txt' % i, dtype=int)
            pred = convertOnehot(pred)
        else:
            pred = np.loadtxt('parities.txt', dtype=int)
        if codewords is None:
            codewords = pred
        else:
            codewords = np.hstack((codewords, pred))
    return codewords

def decode(codewords, parities):
    dec = decoder.decoder(numClasses, parities)
    ret = [dec.decode(np.array(codewords[i], dtype='uint8')) for i in range(len(codewords))]
    return ret


if __name__ == '__main__':
    truth = np.loadtxt('truth.txt')
    parities = []
    for p1, p2 in combinations(range(numClasses), 2): 
        parities.append([p1,p2])
    
    codewords = createCodewords()
    numParities = int(codewords.shape[1]) - numClasses
    neededParities = 45
    #for trial in range(10):
    for comb in combinations(range(numParities), neededParities):
        idx = np.sort(np.array(comb))
        base = np.array([i for i in range(10)])
        c_idx = np.append(base, 10+idx)
        parities_test = []
        for i in idx:
            parities_test.append(parities[i])
        pred = decode(codewords[:, c_idx], parities_test)
        print ('accuracy is: %f' % (sum(pred == truth) / truth.shape[0])) 
       
