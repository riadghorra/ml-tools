from scipy.stats import pearsonr, spearmanr

"""Pearson correlation to evaluate linear correlation"""


def pearson(x, y):
    return pearsonr(x, y)[0]


"""Spearman correlation to evaluate non-linear correlation"""


def spearman(x, y):
    return spearmanr(x, y)[0]
