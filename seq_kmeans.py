import numpy as np
from chainer import Variable, optimizers, Chain, cuda
import chainer.functions as F
import chainer.links as L
import chainer, cupy
import chainer.initializers
import random
from chainer_edit_distance import soft_edit_distance, edit_distance


def one_hot_encoding(X, dict_alphabet, max_seq_length, smooth=1.):
    out = np.zeros((len(X), len(dict_alphabet), max_seq_length), dtype=np.float32)
    if smooth < 1:
        out[:] = (1 - smooth) / (len(dict_alphabet) - 1)
    for i, seq in enumerate(X):
        l = len(seq)
        for j, c in enumerate(seq):
            out[i, dict_alphabet[c], j] = smooth
        out[i, :, l:] = 0
    return out


class KMeansLayer(chainer.Link):
    def __init__(self, n_centroid, length, tau=2, init_W=None):
        super(KMeansLayer, self).__init__()
        with self.init_scope():
            if init_W is not None:
                initializer = chainer.initializers._get_initializer(init_W)
            else:
                initializer = chainer.initializers.Normal()
            self.centroid = chainer.Parameter(initializer)
            self.mask = None
            self.n_centroid = n_centroid
            self.length = length
            self.tau = tau

    def __call__(self, x, inference=False, indexes=None):
        if self.mask is None:
            self.mask = self.xp.zeros((self.n_centroid, x.shape[1], x.shape[2]), dtype=np.float32)
            self.mask[:, :, :self.length] = 1
            self.centroid.initialize((self.n_centroid, x.shape[1], x.shape[2]))
        I = np.broadcast_to(np.arange(x.shape[0]), (self.n_centroid, x.shape[0])).T
        J = np.broadcast_to(np.arange(self.n_centroid), I.shape)
        I = np.ravel(I)
        J = np.ravel(J)

        centers = F.softmax(self.centroid) * self.mask
        if indexes is None:
            if inference:
                d = edit_distance(x[I], centers.data[J]).astype(np.float32)
                d = cupy.reshape(d, (x.shape[0], self.n_centroid))
                centroid_indexes = np.argmin(cupy.asnumpy(d), axis=1)
                d = d[np.arange(len(centroid_indexes)), centroid_indexes]
                return F.mean(d), centroid_indexes
            else:
                with chainer.no_backprop_mode():
                    d = edit_distance(x[I], centers[J].data)
                    d = cupy.reshape(d, (x.shape[0], self.n_centroid))
                    centroid_indexes = np.argmin(cupy.asnumpy(d), axis=1)
                d = soft_edit_distance(x, centers[centroid_indexes], self.tau)
                w = self.xp.zeros(len(d), dtype=np.float32)
                unique_centroid = np.unique(centroid_indexes)
                for u in unique_centroid:
                    ind = np.where(centroid_indexes == u)[0]
                    w[ind] = 1. / (len(ind) * len(unique_centroid))

                return F.sum(d * w), centroid_indexes
        else:
            d = soft_edit_distance(x, centers[indexes], self.tau)
            return d

    def get_centroid(self):
        centers = F.softmax(self.centroid).data[:, :, :self.length]
        return centers


class SoftKMeansLayer(chainer.Link):
    def __init__(self, n_centroid, length, tau1=2., tau2=2., init_W=None):
        super(SoftKMeansLayer, self).__init__()
        with self.init_scope():
            if init_W is not None:
                initializer = chainer.initializers._get_initializer(init_W)
            else:
                initializer = chainer.initializers.Normal()
            self.centroid = chainer.Parameter(initializer)
            self.mask = None
            self.n_centroid = n_centroid
            self.length = length
            self.tau1 = tau1
            self.tau2 = tau2

    def __call__(self, x, inference=False):
        if self.mask is None:
            self.mask = self.xp.zeros((self.n_centroid, x.shape[1], x.shape[2]), dtype=np.float32)
            self.mask[:, :, :self.length] = 1
            self.centroid.initialize((self.n_centroid, x.shape[1], x.shape[2]))
        I = np.broadcast_to(np.arange(x.shape[0]), (self.n_centroid, x.shape[0])).T
        J = np.broadcast_to(np.arange(self.n_centroid), I.shape)
        I = np.ravel(I)
        J = np.ravel(J)

        centers = F.softmax(self.centroid) * self.mask
        if inference:
            d = edit_distance(x[I], centers.data[J]).astype(np.float32)
            d = cupy.reshape(d, (x.shape[0], self.n_centroid))
            centroid_indexes = np.argmin(cupy.asnumpy(d), axis=1)
            d = d[np.arange(len(centroid_indexes)), centroid_indexes]
            return F.mean(d), centroid_indexes
        else:
            d = soft_edit_distance(x[I], centers[J], self.tau1)
            d = F.reshape(d, (x.shape[0], self.n_centroid))
            coef = F.softmax(-d * self.tau2)
            S = F.broadcast_to(F.sum(coef, axis=0), coef.shape)

            d = F.sum(d * coef / S)
            return d / self.n_centroid
            #d = F.min(d, axis=1)
            return F.mean(d)

    def get_centroid(self):
        centers = F.softmax(self.centroid).data[:, :, :self.length]
        return centers


class SeqKmeans():
    def __init__(self, n_centroid, centroid_length, alphabet, use_gpu=True, tau=2):
        self.model = None
        self.optimizer = None
        self.centroid_length = centroid_length
        self.n_centroid = n_centroid
        self.tau = tau
        self.use_gpu = use_gpu
        self.alphabet = alphabet
        self.dict_alphabet = {alphabet[i]: i for i in range(len(alphabet))}
        self.max_length = None

    def get_initialize_points(self, X, smooth, n_centroid):
        X = cupy.array(one_hot_encoding(X, self.dict_alphabet, self.max_length, smooth), dtype=np.float32)
        I = np.ravel(np.broadcast_to(np.arange(len(X)), (len(X), len(X))).T)
        J = np.ravel(np.broadcast_to(np.arange(len(X)), (len(X), len(X))))
        d = edit_distance(X[I], X[J]).reshape((len(X), len(X)))
        d = cupy.asnumpy(d)
        out = [random.randint(0, len(X)-1)]
        for i in range(n_centroid - 1):
            min_d = np.min(d[:, out], axis=1)
            new_point = np.random.choice(len(min_d), 1, p=min_d / np.sum(min_d))
            out.append(new_point)
        return cupy.asnumpy(X)[out, :, :]

    def fit(self, X, mini_batch=1000, subsample_batch=100, n_iter=100, step_per_iter=10, init_smooth=0.8,
            init_scale=0.1, lr=0.1, optimizer='SGD'):
        L = np.array([len(seq) for seq in X])
        self.max_length = np.max(L)

        init = X[np.where(L == self.centroid_length)[0]]
        init = np.unique(init)
        if len(init) > self.n_centroid * 100:
            init = init[np.random.choice(len(init), self.n_centroid * 100, replace=False)]

        init_seq = self.get_initialize_points(init, init_smooth, self.n_centroid)
        """init_seq = one_hot_encoding(init, self.dict_alphabet, self.max_length, init_smooth)
        init_seq[np.where(init_seq != 0)] = np.log(init_seq[np.where(init_seq != 0)])
        noise = np.random.gumbel(0, 1, init_seq.shape)
        init_seq[np.where(init_seq != 0)] += noise[np.where(init_seq != 0)]
        init_seq *= init_scale"""
        init_seq= np.transpose(np.transpose(init_seq, (1, 0, 2)) - np.mean(init_seq, axis=1), (1, 0, 2))

        self.model = Chain(
            kmeans=KMeansLayer(self.n_centroid, self.centroid_length, init_W=init_seq, tau=self.tau)
        )
        self.optimizer = {'Adam': optimizers.Adam(lr),
                          'Momentum': optimizers.MomentumSGD(lr),
                          'SGD': optimizers.SGD(lr)}[optimizer]
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-6))
        if self.use_gpu:
            self.model.to_gpu()

        with chainer.using_config('train', True):
            lcurve = []
            for i in range(n_iter):
                self.model.cleargrads()
                indexes = np.random.choice(len(X), mini_batch)
                x = X[indexes]
                x = one_hot_encoding(x, self.dict_alphabet, self.max_length)
                if self.use_gpu:
                    x = cupy.array(x)
                with chainer.no_backprop_mode():
                    _, labels = self.model.kmeans(x, inference=True)
                labels_indexes = [np.where(labels == u)[0] for u in np.unique(labels)]
                for j in range(step_per_iter):
                    indexes = []
                    for row in labels_indexes:
                        indexes += np.random.choice(row, subsample_batch // len(labels_indexes)).tolist()

                    loss = self.model.kmeans(x[indexes], indexes=labels[indexes])
                    loss = F.mean(loss)
                    loss.backward()
                    lcurve.append(float(loss.data))
                    self.optimizer.update()
                    print(i, j, np.mean(lcurve[-10:]))

        return np.array(lcurve)

    def transform(self, X, batchsize=1000):
        labels = []
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                for i in range(0, len(X), batchsize):
                    print(i)
                    x = X[i: i + batchsize]
                    x = one_hot_encoding(x, self.dict_alphabet, self.max_length)
                    if self.use_gpu:
                        x = cupy.array(x)
                    loss, indexes = self.model.kmeans(x, inference=True)
                    labels.append(indexes)
        return np.concatenate(labels)

    def get_centroid(self):
        return cupy.asnumpy(self.model.kmeans.get_centroid())


class SoftSeqKmeans():
    def __init__(self, n_centroid, centroid_length, alphabet, use_gpu=True, tau=2):
        self.model = None
        self.optimizer = None
        self.centroid_length = centroid_length
        self.n_centroid = n_centroid
        self.tau = tau
        self.use_gpu = use_gpu
        self.alphabet = alphabet
        self.dict_alphabet = {alphabet[i]: i for i in range(len(alphabet))}
        self.max_length = None

    def fit(self, X, batchsize=100, n_iter=100, init_smooth=0.8, init_scale=0.1, lr=0.01, optimizer='Momentum'):
        L = np.array([len(seq) for seq in X])
        self.max_length = np.max(L)

        init = X[np.where(L == self.centroid_length)[0]]
        init = np.unique(init)
        init = init[np.random.choice(len(init), self.n_centroid, replace=False)]
        print(init)
        init_seq = one_hot_encoding(init, self.dict_alphabet, self.max_length, init_smooth)
        init_seq[np.where(init_seq != 0)] = np.log(init_seq[np.where(init_seq != 0)])
        noise = np.random.gumbel(0, 1, init_seq.shape)
        init_seq[np.where(init_seq != 0)] += noise[np.where(init_seq != 0)]
        init_seq *= init_scale
        init_seq= np.transpose(np.transpose(init_seq, (1, 0, 2)) - np.mean(init_seq, axis=1), (1, 0, 2))

        self.model = Chain(
            kmeans=SoftKMeansLayer(self.n_centroid, self.centroid_length, init_W=init_seq, tau1=self.tau)
        )

        self.optimizer = {'Adam': optimizers.Adam(lr),
                          'Momentum': optimizers.MomentumSGD(lr),
                          'SGD': optimizers.SGD(lr)}[optimizer]

        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-6))
        if self.use_gpu:
            self.model.to_gpu()

        with chainer.using_config('train', True):
            lcurve = []
            for i in range(n_iter):
                self.model.cleargrads()
                indexes = np.random.choice(len(X), batchsize)
                x = X[indexes]
                x = one_hot_encoding(x, self.dict_alphabet, self.max_length)
                if self.use_gpu:
                    x = cupy.array(x)
                loss = self.model.kmeans(x[indexes])
                loss.backward()
                lcurve.append(float(loss.data))
                self.optimizer.update()
                print(i, np.mean(lcurve[-10:]))

        return np.array(lcurve)

    def transform(self, X, batchsize=1000):
        labels = []
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                for i in range(0, len(X), batchsize):
                    print(i)
                    x = X[i: i + batchsize]
                    x = one_hot_encoding(x, self.dict_alphabet, self.max_length)
                    if self.use_gpu:
                        x = cupy.array(x)
                    loss, indexes = self.model.kmeans(x, inference=True)
                    labels.append(indexes)
        return np.concatenate(labels)

    def get_centroid(self):
        return cupy.asnumpy(self.model.kmeans.get_centroid())