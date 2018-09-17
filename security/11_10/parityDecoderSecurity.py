import numpy as np
from itertools import combinations
from createParity import createParityMap
from random import shuffle
numClasses = 10

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def decode(pred, finalPred):
#    fullSet = set(range(numClasses))
#    parities = []
#    for p1, p2 in combinations(range(numClasses), 2): 
#        parities.append(set([p1,p2]))
#    for p1, p2, p3 in combinations(range(numClasses), 3): 
#        parities.append(set([p1,p2,p3]))
    fullSet = set(range(numClasses))
    parities = []
    parityMap = createParityMap(numClasses)
    mappingKeys = sorted(parityMap.keys(), key=lambda a: (len(a), a))
    for key in mappingKeys:
        parities.append(set(parityMap[key]))

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
    #if inter0 != inter1:
    #    print (inter0, inter1)

def decode_soft(pred, finalPred, pid, idx, suspect):
    fullSet = set(range(numClasses))
    parities = []
    parityMap = createParityMap(numClasses)
    mappingKeys = sorted(parityMap.keys(), key=lambda a: (len(a), a), reverse=False)
    #for key in mappingKeys:
    for key in idx:
        parities.append(set(parityMap[mappingKeys[key]]))
    idx0 = np.where(pred < 0.5)[0]
    idx1 = np.where(pred >= 0.5)[0]

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
    count0 = [0 for i in range(10)]
    for idx in idx0:
        val = 1 - pred[idx]
        for p in (fullSet - parities[idx]):
            count0[p] += val 
    inter0 = np.argmax(count0)
    count1 = [0 for i in range(10)]
    for idx in idx1:
        val = pred[idx]
        for p in parities[idx]:
            count1[p] += val  
    inter1 = np.argmax(count1)
    #finalPred.append(inter0)
    #if inter0 != inter1:
    #    #print (pid, count0, count1)
    #    #print (inter0, inter1)
    #    count0 = softmax(count0)
    #    count1 = softmax(count1)
    #    if count1[inter1] > count0[inter0]:
    #        finalPred[-1] = inter1

    countTotal = np.array(count0) + np.array(count1)
    inter = np.argmax(countTotal)
    finalPred.append(inter)
    for i, parity in enumerate(parities):
        if inter in parity:
            if pred[i] < 0.5:
                suspect[i] += 1
        else:
            if pred[i] >= 0.5:
                suspect[i] += 1


def evaluate(parities_pred, softmaxes, truth, idx):
    finalPred = []
    suspect = [0 for i in range(softmaxes.shape[1])]
    for i in range(parities_pred.shape[0]):
    #for i in range(26,28):
        #decode(parities_pred[i], finalPred)
        decode_soft(softmaxes[i].copy(), finalPred, i, idx, suspect)
    finalPred = np.array(finalPred)
    print ('accuracy is: %f' % (sum(finalPred == truth) / truth.shape[0])) 
    finalPred = finalPred.reshape(-1,1)
    np.savetxt('finalPred_soft.txt', finalPred, fmt='%d')
    print (suspect)

def main():
    parities_pred = np.loadtxt('parities_new.txt')
    softmaxes_orig = np.loadtxt('softmaxes_new.txt')
    truth = np.loadtxt('truth.txt')
    numParities = int(softmaxes_orig.shape[1])
    neededParities = 11
    for trial in range(11):
        softmaxes = softmaxes_orig.copy()
        #softmaxes[:,trial] = np.random.rand(softmaxes.shape[0])
        softmaxes[:,trial] = 0.9
        #softmaxes[:,7] = np.random.rand(softmaxes.shape[0])
        check = []
        for comb in combinations(range(numParities), neededParities):
            #idx = np.sort(np.random.choice(range(numParities), neededParities, replace=False))
            idx = np.sort(np.array(comb))
            check.append(idx)
        #shuffle(check)
        for idx in check:
            print (idx)
            evaluate(parities_pred[:, idx].copy(), softmaxes[:, idx].copy(), truth, idx)

if __name__ == '__main__':
    main()
