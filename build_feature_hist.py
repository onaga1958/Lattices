#!/usr/local/bin/python3.6
import matplotlib.pyplot as plt
from datasets import BigBreastCanser


FEATURE_NAMES = [
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
]


def main():
    X, y = BigBreastCanser().get_raw_X_y()
    positives = X[y == 1]
    negatives = X[y == 0]
    for ind, name in enumerate(FEATURE_NAMES):
        fig, (positve_ax, negative_ax) = plt.subplots(1, 2, sharey=True)
        fig.suptitle(name + ' distribution')
        positive_cnt, _, _ = positve_ax.hist(positives[:, ind], bins=10)
        print(positive_cnt)
        negative_cnt, _, _ = negative_ax.hist(negatives[:, ind], bins=10)
        print(negative_cnt)
        positve_ax.set_title('positives')
        negative_ax.set_title('negatives')
        plt.show()


if __name__ == '__main__':
    main()
