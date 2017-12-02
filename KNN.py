import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import euclidean
from scipy import stats
from matplotlib import pyplot as plt


def calc_all_distancies(data_x, unknown):
    '''
        Function calculates distancies between each pairs of known and unknown points
    '''
    dists = np.sqrt(np.sum(np.power(data_x - unknown[:,np.newaxis], 2),axis=2))

    return  dists


def predict(dists, data_y, k):
    '''
        Function predicts the class of the unknown point by the k nearest neighbours
    '''
    num_pred = dists.shape[0]
    y_pred = np.zeros(num_pred)

    for j in range(num_pred):
        dst = dists[j]
        x_closest = np.argsort(dst)
        x_k = x_closest[:k]
        y_closest = data_y[x_k]
        y_pred[j] = stats.mode(y_closest, axis=None).mode
        print(y_pred.shape)

    return y_pred

def accuracy(predicted,real):
    '''
        Calculates accuracy percentage
    '''
    correct = sum(predicted == real)
    total = len(predicted)
    return 100*correct/total


def compare_k(data_x, data_y, test_x, test_y, kmin=1, kmax=50, kstep=4):
    '''
        Main comparing function
    '''
    k = list(range(kmin, kmax, kstep))
    steps = len(k)
    features = np.zeros((steps, 3))

    print('Evaluating distancies started')

    t0 = time.time()
    distancies = calc_all_distancies(data_x, test_x)
    miss = []
    t = time.time()
    s1 = data_x.shape[0]
    s2 = test_x.shape[0]

    print('Distancies completed in %d seconds for %dx%d' % (t - t0, s1, s2))

    for j in range(steps):
        t0 = time.time()
        yk = predict(distancies, data_y, k[j])

        t = time.time() - t0
        features[j][0] = k[j]
        features[j][1] = accuracy(yk, test_y)
        features[j][2] = t
        cond = yk != test_y
        miss.append({
            'k': k[j],
            'acc': features[j][1],
            'x': test_x[cond]}
        )

        print('k={0}, accuracy = {1}%, time = {2} sec'.format(k[j], features[j][1], features[j][2]))

    return features, miss

num_observations = 300
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([-2, 3], [[2, .75], [.75, 2]], num_observations)


X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((np.zeros(num_observations),
               np.ones(num_observations)))

ratio = 0.67
l = len(X)
n = int(l * ratio)
ind = np.random.permutation(l)

X = X[ind]
Y = Y[ind]

x_trn = X[:n]
y_trn = Y[:n]
x_tst = X[n:]
y_tst = Y[n:]

res, ms = compare_k(x_trn, y_trn, x_tst, y_tst,1,201,20)

# initial data
fig = plt.figure()
plt.scatter(x1[:, 0], x1[:, 1], color='c',label='class1')
plt.scatter(x2[:, 0], x2[:, 1], color='y',label='class2')
# randomly selected data
plt.scatter(x_tst[:,0],x_tst[:,1],color='b',label='test')
plt.legend(loc='best')


# missidentifies for k = value
plt.figure()
plt.scatter(x1[:, 0], x1[:, 1], color='c', label='class1')
plt.scatter(x2[:, 0], x2[:, 1], color='y', label='class2')
plt.scatter(ms[-1]['x'][:,0],ms[-1]['x'][:,1],color='r',label='missidenity,k=%d'%ms[-1]['k'])
plt.legend(loc='best')
plt.xlabel('x1')
plt.ylabel('x2')
plt.figure()

# accuracy plot
k = plt.scatter(res[:, 0], res[:, 1])
plt.ylim(min(res[:, 1]) - 2, max(res[:, 1])+1, 4)
plt.xlabel('k')
plt.ylabel('accuracy, %')
plt.show()

