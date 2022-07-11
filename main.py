from finch import FINCH

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import umap

import os
from collections import Counter


class Clustering:
    def __init__(self, dataset='wine', method='FINCH'):
        self.dataset = dataset
        self.method = method
        self.data = None
        self.lbs = None
        self.res = None

    def load_data(self):

        print('Loading the {} Dataset'.format(self.dataset))

        if self.dataset == 'wine':
            df = pd.read_csv('data/wine.txt', sep=",", index_col=None)
            lbs = df.cls  # lbs means the Ground Truth Labels -> shape of (N, 1)
            feature = df.iloc[:, 1:]  # feature that we use for clustering later -> shape of (N, num of feature)
            print('There are {} clusters in the true dataset'.format(len(lbs.unique())))
            self.data = feature
            self.lbs = lbs

        elif self.dataset == 'mnist':
            pass

    def fit(self):
        if self.method == 'FINCH':
            c, num_clust, req_c = FINCH(self.data)  # c: (N, P) pred labels for each sample
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

        for i in range(self.res.shape[1]):
            pred_lbs = self.res[:, i]
            embedding = self.reduce(self.data)
            counter = Counter(pred_lbs)  # {0: 137, 1: 15, 2: 16, ...}
            n = len(embedding)

            # Use a for loop to plot scatter of different clusters
            for j in range(len(np.unique(pred_lbs))):
                plt.plot(embedding[:, 0][pred_lbs == j], embedding[:, 1][pred_lbs == j], '.', ms=3, alpha=0.5,
                         label='cluster {}: {:.2f}%'.format(j, counter[j] / n * 100))

            plt.savefig(path + 'Partition_{}_'.format(i) + 'cluster_plot')
            plt.show()

    def evaluation(self):
        # we need to maximize the matched accuracy
        pass


if __name__ == '__main__':
    model = Clustering()
    model.load_data()
    model.fit()
    model.visualization()
    print(model.res)





