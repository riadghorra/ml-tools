import pandas as pd
import scipy.stats as ss
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


# Function to compute correlation between two categorical columns
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def mesure_performance(y_true, y_hat):
    print(classification_report(y_true, y_hat))

    print("AUC :", roc_auc_score(y_true, y_hat))
