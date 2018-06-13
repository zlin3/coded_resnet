import numpy as np
from itertools import combinations

numClasses = 10

def decode(pred, finalPred):
    fullSet = set(range(numClasses))
    parities = []
    for p1, p2 in combinations(range(numClasses), 2): 
        parities.append(set([p1,p2]))

    idx0 = np.where(pred == 0)[0]
    idx1 = np.where(pred == 1)[0]

    #inter0 = fullSet - parities[idx0[0]]
    #inter1 = parities[idx1[0]]
    
    #np.random.shuffle(idx0)
    #np.random.shuffle(idx1)
    #for idx in idx0:
    #    ret = inter0.intersection(fullSet - parities[idx])
    #    if ret:
    #        inter0 = ret
    #for idx in idx1:
    #    ret = inter1.intersection(parities[idx])
    #    if ret:
    #        inter1 = ret
    count = [0 for i in range(10)]
    for idx in idx0:
        for p in (fullSet - parities[idx]):
            count[p] += 1 
    inter0 = np.argmax(count)
    #print (count)
    count = [0 for i in range(10)]
    for idx in idx1:
        for p in parities[idx]:
            count[p] += 1  
    inter1 = np.argmax(count)
    #print (count)
    finalPred.append(inter0)
    if inter0 != inter1:
        print (inter0, inter1)

def main():
    parities_pred = np.loadtxt('parities.txt')
    finalPred = []
    for i in range(parities_pred.shape[0]):
    #for i in range(37,38):
        decode(parities_pred[i], finalPred)
    finalPred = np.array(finalPred).reshape(-1,1)
    np.savetxt('finalPred.txt', finalPred, fmt='%d')

if __name__ == '__main__':
    main()
