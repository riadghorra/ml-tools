import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_continuous_hist_w_target(dataset, column_name, range_=None):
    x1 = dataset[dataset.target == 0][column_name].tolist()
    x2 = dataset[dataset.target == 1][column_name].tolist()

    plt.hist(x1, color="r", label='False', range=range_)
    plt.hist(x2, color='g', label='True', range=range_)
    plt.gca().set(title='Frequency Histogram of {}'.format(column_name), ylabel='Frequency')
    plt.legend()
    plt.show()


def plot_categorical_hist_w_target(dataset, column_name, limit=None):
    by_col_name = dataset[column_name].value_counts(dropna=False).to_frame().reset_index().rename(
        columns={"index": column_name,
                 column_name: "count"})

    labels = by_col_name[column_name].tolist()
    target_true_counts = []
    target_false_counts = []

    if limit:
        labels = labels[limit[0]:limit[1]]

    for label in labels:
        target_true_counts.append(dataset[(dataset.target == 1) & (dataset[column_name] == label)][column_name].count())
        target_false_counts.append(
            dataset[(dataset.target == 0) & (dataset[column_name] == label)][column_name].count())

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, target_true_counts, width, label='True', color="g")
    rects2 = ax.bar(x + width / 2, target_false_counts, width, label='False', color="r")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Count by possible values and target')
    ax.set_xticks(x)
    plt.xticks(rotation=90)
    ax.set_xticklabels(labels)
    ax.legend(loc="best")

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()


def audit_categorical_feature(dataset, column_name):
    stat = dataset[column_name].value_counts(dropna=False).to_frame().reset_index().rename(
        columns={"index": column_name, column_name: "count"})
    stat["percentage"] = [x / dataset.shape[0] * 100 for x in stat["count"].tolist()]
    tr = dataset[dataset.target == 1][column_name].value_counts(dropna=False).to_frame().reset_index().rename(columns={
        "index": column_name, column_name: "count_true"})
    stat = stat.merge(tr, on=column_name)
    stat["percentage of {} which are True".format(column_name)] = stat["count_true"] * 100 / stat["count"]
    return stat


def audit_continuous_feature(dataset, column_name):
    x = dataset[column_name].astype(int).tolist()
    sns.distplot(x)
    plt.title("Distribution plot of {}".format(column_name))
    plt.show()
    sns.boxplot(x=dataset["target"], y=dataset[column_name])
    plt.title("Box plots of {} with regards to target value".format(column_name))
    plt.show()



