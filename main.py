from finch import FINCH

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

import umap

import os
from time import time
from collections import Counter

from dim_reduce import reduction

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#FA8072', '#800080',
                                                '#008080'])  # color map settings


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
            print('The length of the Dataset is {}'.format(df.shape[0]))
            feature = df.iloc[:, 1:]  # feature that we use for clustering later -> shape of (N, num of feature)
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            # feature = minmax_scale.fit_transform(feature)
            feature = feature.values
            self.data = feature
            self.lbs = lbs

        elif self.dataset == 'mnist':
            df = pd.read_csv('data/mnist.csv', index_col=None)
            lbs = df.label
            print('The length of the Dataset is {}'.format(df.shape[0]))
            feature = df.iloc[:, 1:]
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            feature = feature / 255.0
            feature = feature.values
            print('Reducing high dimension data for clustering')
            feature = reduction(feature)
            self.data = feature.data.cpu().numpy()
            self.lbs = lbs

        elif self.dataset == 'seed':
            df = pd.read_csv('data/Seed_Data.csv', index_col=None)
            lbs = df.target
            print('The length of the Dataset is {}'.format(df.shape[0]))
            feature = df.iloc[:, :-1]
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            feature = feature.values
            self.data = feature
            self.lbs = lbs

        elif self.dataset == 'box':
            df = pd.read_csv('data/boxes3.csv', index_col=None)
            lbs = df.color
            print('The length of the Dataset is {}'.format(df.shape[0]))
            feature = df.iloc[:, :-1]
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))

            # feature = minmax_scale.fit_transform(feature)
            feature = feature.values
            self.data = feature
            self.lbs = lbs

        elif self.dataset == 'credit':
            df = pd.read_csv('data/creditcard.csv', index_col=None).iloc[:10000, :]  # 16G Memory limitation
            lbs = df.Class
            print('The length of the Dataset is {}'.format(df.shape[0]))
            feature = df.iloc[:, :-1]
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))
            feature = feature.values
            # feature = minmax_scale.fit_transform(feature)
            self.data = feature
            self.lbs = lbs

    def fit(self):
        print('Fitting {} dataset with {} method...'.format(self.dataset, self.method))
        if self.method == 'FINCH':
            start = time()
            c, num_clust, req_c = FINCH(self.data, distance='euclidean')
            # c: (N, P) pred labels for each sample
            end = time()
            runTime = end - start
            print("RunTime：", runTime)
            self.res = c

        elif self.method == 'KMeans':
            start = time()
            best_cluster = 10  # initialize best cluster number
            best_s_score = -1

            for i in range(2, 15):
                k_means = KMeans(n_clusters=i)
                k_means.fit(self.data)
                tmp_res = k_means.labels_
                s_score = self.evaluation(res=tmp_res)
                if s_score > best_s_score:
                    best_cluster = i
                    best_s_score = s_score

            k_means = KMeans(n_clusters=best_cluster)
            k_means.fit(self.data)
            end = time()
            runTime = end - start
            print("RunTime：", runTime)
            self.res = k_means.labels_.reshape(-1, 1)

        # eps=0.02, min_samples=10 when is box data
        # eps=0.07, min_samples=8 when is wine data,
        # eps=0.02, min_samples=10 when is mnist data
        # eps=0.04, min_samples=10 when is mnist data
        elif self.method == 'dbscan':
            start = time()
            x = self.data[:, :2]
            minmax_scale = preprocessing.MinMaxScaler()  # Normalization method
            x = minmax_scale.fit_transform(x)
            dbscan = DBSCAN(eps=0.07, min_samples=8)
            dbscan.fit(x)
            labels = dbscan.labels_
            # y_pred =dbscan.fit_predict(self.data)
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            print("total cluster is", n_clusters_)
            end = time()
            runTime = end - start
            print("RunTime：", runTime)
            self.res = labels.reshape(-1, 1)

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

        plt.title('Ground Truth ({} cluster)'.format(self.dataset))
        plt.savefig(path + 'GroundTruth_' + 'cluster_plot')
        plt.show()

    def evaluation(self, res=None):

        if res is None:
            if self.method != 'FINCH':  # FINCH have couples of predict results
                res = self.res.ravel()
                s_score = silhouette_score(self.data, res)
                print('The silhouette_score of {} in {} dataset is: {}'.format(self.method, self.dataset, s_score))

            else:
                res = self.res
                for i in range(self.res.shape[1]):
                    s_score = silhouette_score(self.data, res[:, i])
                    print('The silhouette_score of {} of Partition{} in {} dataset is: {}'
                          .format(self.method, i, self.dataset, s_score))

        else:
            s_score = silhouette_score(self.data, res)
            return s_score


if __name__ == '__main__':
    model = Clustering(dataset='box', method='FINCH')
    model.load_data()
    model.fit()
    model.visualization()
    model.evaluation()

