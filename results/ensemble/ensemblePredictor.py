import numpy as np
from scipy import stats

def createPredictions(num_models):
    preds = None
    for i in range(0, num_models):
        pred = np.loadtxt('prediction_%d.txt' % i, dtype=int)
        pred = pred.reshape(pred.shape[0], 1)
        if preds is None:
            preds = pred
        else:
            preds = np.hstack((preds, pred))
    return preds

def vote(preds):
    ret = stats.mode(preds, axis=1)[0]
    return ret


if __name__ == '__main__':
    for numEnsemble in range(30, 46):
        pred = np.loadtxt('prediction_0.txt', dtype=int)
        truth = np.loadtxt('truth.txt', dtype=int)
        precision = np.sum(pred == truth) / truth.shape[0]
        preds = createPredictions(numEnsemble)
        ensemblePred = vote(preds)
        ensemblePred = ensemblePred.reshape((-1,))
        ensemblePrecision = np.sum(ensemblePred == truth) / truth.shape[0]
        print ('uncoded precision is: %f' % precision)
        print ('ensemble precision with %d models is: %f' % (numEnsemble, ensemblePrecision))

