from finch import FINCH

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

import umap

import os
from time import time
from collections import Counter

from dim_reduce import reduction
from  sklearn.cluster import DBSCAN

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

            feature = feature / 255.0
            feature = reduction(feature.values)
            self.data = feature.data.cpu().numpy()
            self.lbs = lbs

        elif self.dataset == 'seed':
            df = pd.read_csv('data/Seed_Data.csv', index_col=None)
            lbs = df.target
            feature = df.iloc[:, :-1]
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            feature = minmax_scale.fit_transform(feature)
            self.data = feature
            self.lbs = lbs

        elif self.dataset == 'box':
            df = pd.read_csv('data/boxes3.csv', index_col=None)
            lbs = df.color
            feature = df.iloc[:, :-1]
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            feature = minmax_scale.fit_transform(feature)
            self.data = feature
            self.lbs = lbs


        elif self.dataset == 'credit':
            df = pd.read_csv('data/creditcard.csv', index_col=None)
            lbs = df.Class
            feature = df.iloc[:, -1:]
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            feature = minmax_scale.fit_transform(feature)
            self.data = feature
            self.lbs = lbs


    def fit(self):
        print('Fitting {} dataset with {} method...'.format(self.dataset, self.method))
        if self.method == 'FINCH':
            start = time()
            c, num_clust, req_c = FINCH(self.data)  # c: (N, P) pred labels for each sample
            end = time()
            runTime = end - start
            print("RunTime：", runTime)
            self.res = c

        elif self.method == 'KMeans':
            start = time()
            k_means = KMeans(n_clusters=10)
            k_means.fit(self.data)
            end = time()
            runTime = end - start
            print("RunTime：", runTime)
            self.res = k_means.labels_.reshape(-1, 1)

        #eps=0.02,min_samples=10 when is box data
        #eps=0.07,min_samples=8 when is wine data,
        #eps=0.02,min_samples=10 when is mnist data
        #eps=0.04,min_samples=10 when is mnist data
        elif self.method == 'dbscan':
            start = time()
            x=self.data[:,0:2]
            dbscan= DBSCAN(eps=0.04, min_samples=10).fit(x)
            labels = dbscan.labels_
            #y_pred =dbscan.fit_predict(self.data)
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            print("total cluster is",n_clusters_)
            end = time()
            runTime = end - start
            print("RunTime：", runTime)
            self.res =labels.reshape(-1, 1)


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
    model = Clustering(dataset='seed', method='dbscan')
    model.load_data()
    model.fit()
    model.visualization()





