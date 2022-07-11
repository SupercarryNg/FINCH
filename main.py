from finch import FINCH

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

import umap

import os
from time import time
from collections import Counter


class Clustering:
    def __init__(self, dataset='wine', method='FINCH'):
        self.dataset = dataset
        self.method = method
        self.data = None
        self.lbs = None
        self.res = None

    def load_data(self):

        print('Loading the {} Dataset...'.format(self.dataset))
        minmax_scale = preprocessing.MinMaxScaler()  # Normalization method

        if self.dataset == 'wine':
            df = pd.read_csv('data/wine.txt', sep=",", index_col=None)
            lbs = df.cls  # lbs means the Ground Truth Labels -> shape of (N, 1)
            feature = df.iloc[:, 1:]  # feature that we use for clustering later -> shape of (N, num of feature)
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            feature = minmax_scale.fit_transform(feature)
            self.data = feature
            self.lbs = lbs

        elif self.dataset == 'mnist':
            df = pd.read_csv('data/mnist.csv', index_col=None)
            lbs = df.label
            feature = df.iloc[:, 1:]
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            feature = minmax_scale.fit_transform(feature)
            self.data = feature
            self.lbs = lbs

    def fit(self):
        if self.method == 'FINCH':
            print('Fitting {} dataset with {} method...'.format(self.dataset, self.method))
            start = time()
            c, num_clust, req_c = FINCH(self.data)  # c: (N, P) pred labels for each sample
            end = time()
            runTime = end - start
            print("RunTime：", runTime)
            self.res = c

    @staticmethod
    def reduce(data):
        reducer = umap.UMAP(min_dist=0.9, random_state=42, n_components=2, n_neighbors=100)
        embedding = reducer.fit_transform(data)  # reduce the data dimension for visualization
        return embedding

    def visualization(self):
        path = './Results/' + self.method + '/' + self.dataset + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        # we wanna plot each result of clustering hierarchy
        embedding = self.reduce(self.data)
        n = len(embedding)
        for i in range(self.res.shape[1]):
            pred_lbs = self.res[:, i]
            counter = Counter(pred_lbs)  # {0: 137, 1: 15, 2: 16, ...}

            # Use a for loop to plot scatter of different clusters
            for j in range(len(np.unique(pred_lbs))):
                plt.plot(embedding[:, 0][pred_lbs == j], embedding[:, 1][pred_lbs == j], '.', ms=3, alpha=0.5,
                         label='cluster {}: {:.2f}%'.format(j, counter[j] / n * 100))
            plt.title('{} method ({} cluster)'.format(self.method, len(np.unique(pred_lbs))))
            plt.savefig(path + 'Partition_{}_'.format(i) + 'cluster_plot')
            plt.show()

        counter_gt = Counter(self.lbs)
        for k in range(len(np.unique(self.lbs))):
            plt.plot(embedding[:, 0][self.lbs == k], embedding[:, 1][self.lbs == k], '.', ms=3, alpha=0.5,
                     label='cluster {}: {:.2f}%'.format(k, counter_gt[k] / n * 100))

        plt.savefig(path + 'GroundTruth_' + 'cluster_plot')
        plt.show()

    def evaluation(self):
        # we need to maximize the matched accuracy
        pass


if __name__ == '__main__':
    model = Clustering(dataset='mnist')
    model.load_data()
    model.fit()
    model.visualization()





