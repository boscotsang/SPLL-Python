__author__ = 'BoscoTsang'

import copy

import numpy
import scipy
from sklearn.cluster import KMeans


def SPLL(X1, X2, PARAM=None):
    if PARAM is None:
        k = 3
    else:
        k = PARAM
    Ch = numpy.zeros(2)
    ps = numpy.zeros(2)
    s = numpy.zeros(2)
    Ch[0], ps[0], s[0] = Log_LL(X1, X2, k)
    Ch[1], ps[1], s[1] = Log_LL(X2, X1, k)

    ind = numpy.argmax(s)
    Change = Ch[ind]
    st = s[ind]
    pst = ps[ind]
    return Change, pst, st


def Log_LL(X1, X2, k):
    assert X1.shape[1] == X2.shape[1]
    n = X1.shape[1]
    kmean = KMeans(k, init=X1[:k, :], max_iter=100)
    kmean.fit(X1)
    labels = kmean.predict(X1)
    means = kmean.cluster_centers_
    SC = []
    label_prior = numpy.zeros((1, k))
    for i in xrange(k):
        label_idx = labels == i
        label_prior[0, i] = numpy.sum(label_idx)
        if label_prior[0, i] < 1:
            SC.append(numpy.zeros(n))
        else:
            if X1[label_idx, :].shape[0] > 1:
                SC.append(numpy.diag(numpy.var(X1[label_idx, :], axis=0, ddof=1)).flatten())
            else:
                SC.append(numpy.diag(numpy.var(X1[label_idx, :], axis=0, ddof=0)).flatten())
    SC = numpy.asarray(SC)
    label_count = copy.deepcopy(label_prior)
    label_prior /= X1.shape[0]
    scov = numpy.sum(SC * numpy.tile(label_prior.T, (1, n ** 2)), axis=0)
    scov = numpy.reshape(scov, (n, n))
    z = numpy.array(numpy.diag(scov))
    indexvarzero = z < numpy.spacing(1)
    if numpy.sum(indexvarzero) == 0:
        invscov = numpy.linalg.inv(scov)
    else:
        z[indexvarzero] = numpy.min(z[~indexvarzero])
        invscov = numpy.diag(1. / z)

    LogLikelihoodTerm = numpy.zeros(X2.shape[0])
    for j in xrange(X2.shape[0]):
        xx = X2[j, :]

        DistanceToMeans = numpy.zeros(k)
        for jj in xrange(k):
            if label_count[0, jj] > 0:
                DistanceToMeans[jj] = numpy.dot(numpy.dot((means[jj, :] - xx), invscov), (means[jj, :] - xx).T)
            else:
                DistanceToMeans[jj] = numpy.inf
        LogLikelihoodTerm[j] = numpy.min(DistanceToMeans)
    st = numpy.mean(LogLikelihoodTerm)
    pst = min(scipy.stats.chi2.cdf(st, n), 1 - scipy.stats.chi2.cdf(st, n))
    Change = pst < 0.05
    return Change, pst, st


if "__main__" == __name__:
    #a = numpy.array([[0.3, 1.4, 0.9], [0.2, 1.2, 0.7], [0.1, 1.0, 0.77], [0.4, 1.8, 0.9], [0.33, 1.3, 0.7]])
    #b = numpy.array([[0.3, 1.2, 0.9], [0.2, 1.4, 0.7], [0.1, 1.3, 0.77], [0.4, 1.0, 0.9], [0.33, 1.8, 0.7]])
    a = numpy.random.random((4, 4))
    b = numpy.random.random((4, 4)) + 0.1
    print a
    print b
    change, pst, st = SPLL(a, b, 3)
    print change, pst, st
