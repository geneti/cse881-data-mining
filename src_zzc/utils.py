import time
import main
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def save_preds(file_path, preds):
    with open(file_path, 'w', encoding='utf-8') as f:
        for pred in preds:
            f.write(f'{pred}\n')


def plot_confusion_matrix(labels, gts, preds, file_name):
    array = confusion_matrix(gts, preds)
    df_cm = pd.DataFrame(array, index=labels, columns=labels)

    plt.figure()
    font_size = 9
    sn.set(font_scale=1)
    ax = sn.heatmap(df_cm,
                    cmap='Blues',
                    annot=True,
                    fmt='d',
                    annot_kws={"size": font_size})
    # ax.set(xlabel='GT', ylabel='Prediction')
    ax.set_xlabel('Prediction', fontsize=font_size)
    ax.set_ylabel('GT', fontsize=font_size)
    plt.savefig(file_name)
    # plt.show()