import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(font_scale=2)
sns.set_style("white")
pd.options.display.float_format = '{:,.2f}'.format

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_data_and_decision_boundary(the_data, f1, f2, target, model=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.scatterplot(x=f1, y=f2, data=the_data, style=target, hue=target, ax=ax, s=150)

    if model:
        xx, yy = make_meshgrid(the_data[f1], the_data[f2])
        plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.2)

pass