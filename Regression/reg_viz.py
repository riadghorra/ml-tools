import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def boxplot_categorical(data, feature, target):
    data = pd.concat([data[target], data[feature]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=feature, y=target, data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)

