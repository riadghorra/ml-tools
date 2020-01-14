"VIsualize PCs after fit transform operation on a dataframe"

import pandas as pd
import matplotlib.pyplot as plt


def visualize_pc_for_classification(pca_transformed_train, y_train):
    principalDf = pd.DataFrame(data=pca_transformed_train,
                               columns=["pc_{}".format(i) for i in range(pca_transformed_train.shape[1])])
    labels = [1 if val == True else 0 for val in y_train]
    principalDf["target"] = labels

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = principalDf['target'] == target
        ax.scatter(principalDf.loc[indicesToKeep, 'pc_0']
                   , principalDf.loc[indicesToKeep, 'pc_1']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()

    return principalDf