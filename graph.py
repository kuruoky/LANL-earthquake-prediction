import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
def save_or_show_plot(file_nm: str, save: bool):
    if save:
        plt.savefig(os.path.join('./graph/', file_nm))
    else:
        plt.show()

folder = os.path.join("data", "test")
df1 = pd.read_csv('submission.csv')
df2 = pd.read_csv('359.csv')

for i in range(len(df1['seg_id'])):
    df3 = pd.read_csv('./data/test/' + df1['seg_id'][i] + '.csv')
    fig = plt.figure(figsize=(200, 100))
    plt.plot(df3['acoustic_data'])
    plt.ylim(-100, 100)
    x1 = df1['time_to_failure'][i]
    x2 = df2['time_to_failure'][i]
    plt.twinx()
    plt.ylim(0, 15)
    plt.axhline(x1, color='r', alpha=1)
    plt.axhline(x2, color='b', alpha=1)
    save_or_show_plot(str(i) + '.png', True)
