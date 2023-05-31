import os

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

model_name_mapper = {
    'unet': 'CNN',
    'fno': 'FNO',
    'sno': 'SNO',
}

color_mapper = {
    'unet': 'palevioletred',
    'fno': 'mediumvioletred',
    'sno': 'darkslateblue',
}

color_res_equiv = 'grey'
color_no_equiv = 'grey'

def plot(results):
    res_train = 61
    plt.figure(figsize=(12, 7)) # increase figure size
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(xticks)
    plt.grid(linestyle='--' , linewidth=2, alpha=0.5)
    plt.xlabel('Resolution', fontsize=22)
    plt.ylabel('Relative Error (%)', fontsize=22)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    for result in results:
        errs = result.values
        res_grid = result.index.values
        c = color_mapper[result.name]
        p = plt.plot(res_grid, errs, lw=3.0, color=c)
        color = p[0].get_color()
        plt.scatter([res_grid[0]], [errs[0]], c=c, s=150, zorder=22, marker='+')
        ax.annotate(model_name_mapper[result.name], fontsize=26, color=color,
                xy=(175, errs[-1] + 10), xycoords='data',
                )
    ax.axvspan(res_grid[0] - 40, res_train - 1, facecolor=color_no_equiv, alpha=0.1, zorder=0)
    ax.axvspan(res_train - 1, res_grid[-1] + 40, facecolor=color_res_equiv, alpha=0.0, zorder=1)
    plt.axvline(x=res_train - 1, linewidth=3, color=color_no_equiv, linestyle='--', zorder=-3)
    plt.xlim(-5, 200)
    plt.ylim(-10, 155)
    plt.text(-2, 143, 'No Equivalence', fontsize=24, c=color_no_equiv) # increase text font size
    plt.text(90, 143, 'Representation Equivalence', fontsize=24, c=color_res_equiv) # increase text font size
    plt.scatter([res_train], [0], c='gold', s=150, label='Training Resolution', zorder=22, edgecolors='black')
    plt.tick_params(axis='both', which='major', labelsize=14) # increase tick label font size


if __name__ == '__main__':
    # you need to change these paths here
    paths = [
        'unet_epoch_499_loss_0.07949766889214516.csv',
        'fno_epoch_499_loss_0.04822678957134485.csv',
        'sno_epoch_499_loss_0.1291605643928051.csv',
    ]
    results = []
    for path in paths:
        result = pd.read_csv(path, index_col='res').squeeze("columns")
        results.append(result)
    plot(results)
    plt.show()
    print(results)