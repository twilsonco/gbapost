#!/usr/local/bin/python3

import csv, os
from datetime import datetime
import numpy as np
from sys import argv, exc_info
from mendeleev import element
from statistics import stdev, mode, mean
import glob
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pylatexenc.latexencode import unicode_to_latex
import gc


## this prevents a huge memory leak when making a large number of plots https://stackoverflow.com/a/69253468/2620767
import matplotlib
# matplotlib.use('Agg')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

"""A permissive filename sanitizer."""
import unicodedata

import re




USE_R2_ADJ = False
INCLUDE_DIFF = False
P_VALUE_IN_REGRESSION_PLOTS = False
REGRESSION_PLOTS_TITLE_INSIDE_PLOT = True



regression_plot_title_y = 0.87 if REGRESSION_PLOTS_TITLE_INSIDE_PLOT else (1.11 if P_VALUE_IN_REGRESSION_PLOTS else 1.03)
os.environ['PATH'] += ":/Library/TeX/texbin/"
plt.rcParams['text.usetex'] = True

OUTPUT_FORMAT = '.pdf'

# red = (185/255,71/255,53/255)
# blue = (58/255,117/255,135/255)
red = (224/255,117/255,71/255)
blue = (101/255,151/255,226/255)



BASIN_COMPARISON_VARLIST = ["electron density","volume","kinetic energy","density"]
BASIN_COMPARISON_VARBLACKLIST = ["boundary","deformation",]

CLEAR = ''.join(['\n']*100)

DIFF_SUFFIX = " (∆)"
def diff_func(s):
    return s + DIFF_SUFFIX


def sanitize(filename):
    """Return a fairly safe version of the filename.

    We don't limit ourselves to ascii, because we want to keep municipality
    names, etc, but we do want to get rid of anything potentially harmful,
    and make sure we do not exceed Windows filename length limits.
    Hence a less safe blacklist, rather than a whitelist.
    """
    blacklist = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", "\0"]
    reserved = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
        "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9",
    ]  # Reserved words on Windows
    filename = "".join(c for c in filename if c not in blacklist)
    # Remove all charcters below code point 32
    filename = "".join(c for c in filename if 31 < ord(c))
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.rstrip(". ")  # Windows does not allow these at end
    filename = filename.strip()
    if all([x == "." for x in filename]):
        filename = "__" + filename
    if filename in reserved:
        filename = "__" + filename
    if len(filename) == 0:
        filename = "__"
    if len(filename) > 255:
        parts = re.split(r"/|\\", filename)[-1].split(".")
        if len(parts) > 1:
            ext = "." + parts.pop()
            filename = filename[:-len(ext)]
        else:
            ext = ""
        if filename == "":
            filename = "__"
        if len(ext) > 254:
            ext = ext[254:]
        maxl = 255 - len(ext)
        filename = filename[:maxl]
        filename = filename + ext
        # Re-check last character (if there was no extension)
        filename = filename.rstrip(". ")
        if len(filename) == 0:
            filename = "__"
    return filename

class SGB_CONDENSED_BASIN_TYPE:
    MAX = 'max'
    MIN = 'min'
    MIXED = 'mixed'
    
def is_float(s):
    try:
        f=float(s)
        return np.isfinite(f)
    except:
        return False

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def manhattan(a, b): # https://www.statology.org/manhattan-distance-python/
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def compare_func(a,b):
    return manhattan(a, b)
    # return np.corrcoef(np.array([a, b]))[0,1]



# Boruta is a tool for rejecting input degrees of freedom from the dataset.
def BorutaReduceVars(X, y):
    # https://github.com/scikit-learn-contrib/boruta_py
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    from sklearn import preprocessing
    # load X and y
    # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    # X = pd.read_csv('examples/test_X.csv', index_col=0).values
    # y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    # y = y.ravel()
    
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    
    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
    
    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, max_iter=100)
    
    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)
    
    # check selected features - first 5 features are selected
    feat_selector.support_
    
    # check ranking of features
    feat_selector.ranking_
    
    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    
    return X_filtered, feat_selector
    
    # pd.DataFrame(X_filtered).to_csv("examples/out_x.csv")
    
    # print("done")


# Code source: Jaques Grobler
# License: BSD 3 clause

def r2_adj_score(X,Y):
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X, Y)
    
    rsqr_adj = max(0, 1 - (1-regr.score(X, Y))*(len(Y)-1)/(len(Y)-X.shape[1]-1))
    return rsqr_adj

def r2_score_fit(X,Y):
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X, Y)
    return regr.score(X, Y)

def r2_func(X,Y):
    x1 = X if len(X.shape) == 2 else X.reshape(-1, 1)
    if USE_R2_ADJ:
        return r2_adj_score(x1, Y)
    else:
        return r2_score_fit(x1, Y)

def p_value_from_regression(X, y):
    from scipy import stats
    lm = linear_model.LinearRegression()
    lm.fit(X,y)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    new_X = np.append(np.ones((len(X),1)), X, axis=1)
    M_S_E = (sum((y-predictions)**2))/(len(new_X)-len(new_X[0]))
    v_b = M_S_E*(np.linalg.inv(np.dot(new_X.T,new_X)).diagonal())
    s_b = np.sqrt(v_b)
    t_b = params/ s_b
    p_val =[2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in t_b]
    p_val = np.round(p_val,3)
    return p_val

def summarize_fit(X,y):
    import pandas as pd , numpy as np
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sma
    
    X2  = sma.add_constant(X)
    _1  = sma.OLS(y, X2)
    _2  = _1.fit()
    return _2.summary()
    

def LinearRegression(X, Y, titlestr="", labels=[], xlabel=None, ylabel=None, yunits=None, colors=[], shapes=[], sizes=[], legendcolors=[], legendshapes=[], filedir=None, cutoff_r_squared=0.0):
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from sklearn import linear_model
    # from sklearn.metrics import mean_squared_error, r2_score
    # from matplotlib.patches import Patch
    # from matplotlib.lines import Line2D
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X, Y)
    
    rsqr_adj = max(0, 1 - (1-regr.score(X, Y))*(len(Y)-1)/(len(Y)-X.shape[1]-1))
    
    # Make predictions using the testing set
    y_pred = regr.predict(X)
    
    rsqr = r2_score(Y,y_pred)
    
    try:
        p_val = p_value_from_regression(X, Y)
        fit_summary = summarize_fit(X, Y)
    except Exception as e:
        p_val = None
        fit_summary = ""
        print(e)
    
    if p_val is None:
        p_val = [1.,1.]
    
    if rsqr < cutoff_r_squared:
        return
    
    different_colors = False
    if len(colors) > 1:
        different_colors = any(i != j for i,j in zip(colors[:-1],colors[1:]))
    
    # Plot outputs
    fig, ax = plt.subplots()
    if len(colors) == len(Y) and len(shapes) == len(Y) and len(sizes) == len(Y):
        for i in range(len(X)):
            mi = shapes[i]
            xi = X[i]
            yi = Y[i]
            ci = colors[i] if different_colors else 'dimgrey'
            si = sizes[i] if different_colors else 100
            if len(labels) == Y.size:
                ax.scatter(xi,yi, marker=mi, color=ci, edgecolor='k', s=si, label=unicode_to_latex(labels[i]))
            else:
                ax.scatter(xi,yi, marker=mi, color=ci, edgecolor='k', s=si)
    else:
        ax.scatter(X, Y, color="black")
        if len(labels) == Y.size:
            for i,l in enumerate(labels):
                ax.annotate(unicode_to_latex(l),(X[i], Y[i]))
        
    ax.plot(X, y_pred, color=blue, linewidth=3)
    if P_VALUE_IN_REGRESSION_PLOTS:
        if " from " in titlestr:
            ax.set_title(r'\noindent {}\\{}\\(R$^2$ = {:2.4f}; R$_{{\rm{{adj}}}}^2$ = {:2.4f})\\(p$_{{b}}$ = {:2.4f}; p$_{{m}}$ = {:2.4f})'.format(unicode_to_latex(titlestr[:titlestr.find(' from ')]), unicode_to_latex("from " + titlestr[titlestr.find(' from ')+6:]), rsqr, rsqr_adj, p_val[0], p_val[1]), y=regression_plot_title_y + 0.16, pad=18, fontdict={'fontsize': 18})
        else:
            ax.set_title(r'\noindent {}\\(R$^2$ = {:2.4f}; R$_{{\rm{{adj}}}}^2$ = {:2.4f})\\(p$_{{b}}$ = {:2.4f}; p$_{{m}}$ = {:2.4f})'.format(unicode_to_latex(titlestr), rsqr, rsqr_adj, p_val[0], p_val[1]), y=regression_plot_title_y, pad=18, fontdict={'fontsize': 18})
    else:
        if " from " in titlestr:
            ax.set_title(r'\noindent {}\\{}\\(R$^2$ = {:2.4f}; R$_{{\rm{{adj}}}}^2$ = {:2.4f})'.format(unicode_to_latex(titlestr[:titlestr.find(' from ')]), unicode_to_latex("from " + titlestr[titlestr.find(' from ')+6:]), rsqr, rsqr_adj), y=regression_plot_title_y + 0.16, pad=18, fontdict={'fontsize': 18})
        else:
            ax.set_title(r'\noindent {}\\(R$^2$ = {:2.4f}; R$_{{\rm{{adj}}}}^2$ = {:2.4f})'.format(unicode_to_latex(titlestr), rsqr, rsqr_adj), y=regression_plot_title_y, pad=18, fontdict={'fontsize': 18})

    # ax.set_xticks(())
    if xlabel:
        ax.set_xlabel(unicode_to_latex(xlabel), fontdict={'fontsize': 16})
    # ax.set_yticks(())
    # if ylabel:
    #     ax.set_ylabel(unicode_to_latex(ylabel + yunits), fontdict={'fontsize': 16})
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    """
    legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
                   Line2D([0], [0], marker='o', color='w', label='Scatter',
                          markerfacecolor='g', markersize=15),
                   Patch(facecolor='orange', edgecolor='r',
                         label='Color Patch')]
                        """
    if len(legendcolors) > 0 or len(legendshapes) > 0:
        lc=[]
        ls=[]
        [(ls.append(i[0]),lc.append(i[1])) for i in sorted(zip(legendshapes,legendcolors if len(legendcolors) > 0 else ([None] * len(legendshapes))), key=lambda x:x[0][1])]
        legend_elements = [Line2D([0], [0], marker=c[0], color=(1,1,1,0.3), markerfacecolor='dimgrey', markeredgecolor='k', markersize=10, label=c[1]) for c in ls]
        if len(legendcolors):
            legend_elements += [Line2D([],[],linestyle='')] + [Line2D([0], [0], marker='o', color='w', markerfacecolor=c[0], markeredgecolor='k', markersize=10, label=c[1]) for c in lc]
        ax.legend(handles=legend_elements, loc='right', fontsize=14)
    else:
        ax.legend(fontsize=13)
    
    if filedir:
        # filepath = os.path.join(filedir,sanitize(f"{rsqr_adj if USE_R2_ADJ else rsqr:2.4f}_{titlestr}-{ylabel}_vs_{xlabel}{OUTPUT_FORMAT}"))
        # plt.savefig(filepath, pad_inches=0.2, bbox_inches='tight', dpi=300)
        filepath = os.path.join(filedir,sanitize(f"{titlestr}_{ylabel}_vs_{xlabel}_{rsqr:2.4f}_{OUTPUT_FORMAT}"))
        plt.savefig(filepath, pad_inches=0.2, bbox_inches='tight', dpi=300)
        if fit_summary != '':
            with open(filepath.replace(OUTPUT_FORMAT, ".txt"), 'w') as out:
                out.write(str(fit_summary)) 
        # filepath = os.path.join(filedir,sanitize(f"{xlabel}_vs_{ylabel}-{titlestr}_{rsqr:2.4f}{OUTPUT_FORMAT}"))
        # plt.savefig(filepath, pad_inches=0.2, bbox_inches='tight', dpi=300)
    # plt.show()
    # Clear the current axes.
    
    plt.close(fig)



def CorrelationMatrix(x, titlestr="", labels=None, filedir=None):
    # https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    # import numpy as np
    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    
    # make dataframe
    d = pd.DataFrame(data=x, columns=labels)
    
    # compute correlation matrix
    corr = d.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    ax.set_title(f"{titlestr}")
    
    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap = sns.diverging_palette(250, 30, l=65, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    if filedir:
        filepath = os.path.join(filedir,sanitize(f"{titlestr}{OUTPUT_FORMAT}"))
        plt.savefig(filepath, pad_inches=0.2, bbox_inches='tight')
    
      
    plt.close(f)
    
    return



# https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
# https://www.kaggle.com/drazen/heatmap-with-sized-markers
def heatmap(x, y, **kwargs):
    if 'title_str' in kwargs:
        title_str = kwargs['title_str']
    else:
        title_str = ""
    
    if 'file_dir' in kwargs:
        file_dir = kwargs['file_dir']
    else:
        file_dir = None
    
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 
        
    if 'palette_fine' in kwargs:
        palette_fine = kwargs['palette_fine']
        n_colors_fine = len(palette_fine)
    else:
        n_colors_fine = len(palette)
        palette_fine = palette 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette_fine[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors_fine - 1)) # target index in the color palette
            return palette_fine[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.#01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}
    
    fig = plt.figure(dpi=300)
    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot
    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order', 'title_str', 'file_dir', 'palette_fine'
    ]}
    
    # ax.scatter(
    
    ax.scatter(
        x=[x_to_num[v] for v,c in zip(x,color) if c >= 0],
        y=[y_to_num[v] for v,c in zip(y,color) if c >= 0],
        marker=marker,
        s=[value_to_size(v) for v,c in zip(size,color) if c >= 0], 
        c=[value_to_color(v) for v in color if v >= 0],
        # hatch='/' * (2 + int(np.sqrt(len(color)))//5),
        edgecolors='none',
        # linewidths=0.8,
        **kwargs_pass_on
    )
    
    #     x=[x_to_num[v] for v in x],
    #     y=[y_to_num[v] for v in y],
    #     marker=marker,
    #     s=[value_to_size(v) for v in size], 
    #     c=[value_to_color(v) for v in color],
    #     **kwargs_pass_on
    # )
    
    ax.scatter(
        x=[x_to_num[v] for v,c in zip(x,color) if c < 0],
        y=[y_to_num[v] for v,c in zip(y,color) if c < 0],
        marker=marker,
        s=[value_to_size(v) for v,c in zip(size,color) if c < 0], 
        c=[value_to_color(v) for v in color if v < 0],
        # hatch='/' * (2 + int(np.sqrt(len(color)))//5),
        edgecolors='none',
        # linewidths=0.8,
        **kwargs_pass_on
    )
    
    
    
    
    
    ax.set_xticks([v for k,v in x_to_num.items()], labels=['' for i in range(len(x_to_num))])
    if (len(x_to_num) > 10 and mean([len(l) for l in y_names]) > 8):
        ax.tick_params(axis='x', width=0, which='minor')
        ax.set_xticks([v+0.5 for k,v in x_to_num.items()], labels=['' for i in range(len(x_to_num))], minor=True)
        ax.set_xticklabels([unicode_to_latex(k) for k in x_to_num], minor=True, rotation=40, horizontalalignment='right')
    else:
        ax.set_xticks([v for k,v in x_to_num.items()], labels=['' for i in range(len(x_to_num))])
        ax.set_xticklabels([unicode_to_latex(k) for k in x_to_num], rotation=40, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([unicode_to_latex(k) for k in y_to_num])
    
    if title_str != "":
        ax.set_title(unicode_to_latex(title_str))

    # ax.grid(False, 'major')
    # ax.grid(True, 'minor')
    ax.grid(False)
    for s in ["right","top"]:#,"bottom","left"]:
        ax.spines[s].set_visible(False)
    # ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    # ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('white')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars
        
        bar_height = bar_y[1] - bar_y[0]
        # ax.barh(
        #     y=bar_y,
        #     width=[5]*len(palette), # Make bars 5 units wide
        #     left=col_x, # Make bars start at 0
        #     height=bar_height,
        #     color=palette,
        #     linewidth=0
        # )
        
        halfi = n_colors//2
        # ytmp = bar_y[halfi:]
        # widthtmp = [5]*halfi
        # lefttmp = col_x
        # heighttmp = bar_height/2
        # colortmp = 
        ax.barh(
            y=bar_y[halfi:],
            width=[5]*halfi, # Make bars 5 units wide
            left=col_x[halfi:], # Make bars start at 0
            height=bar_height,
            color=palette[halfi:],
            linewidth=0
        )
        ax.barh(
            y=bar_y[:halfi],
            width=[5]*halfi, # Make bars 5 units wide
            left=col_x[:halfi], # Make bars start at 0
            height=bar_height,
            color=palette[:halfi],
            # hatch='//',
            linewidth=0.0,
            # edgecolor='none'
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        for s in ["right","top","bottom","left"]:
            ax.spines[s].set_visible(False)
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 
    
    if file_dir:
        fname = title_str.replace('\n','_')
        filepath = os.path.join(file_dir,sanitize(f"{sanitize(fname)}_correlation{OUTPUT_FORMAT}"))
        plt.savefig(filepath, pad_inches=0.2, bbox_inches='tight')
    
    # Clear the current axes.
     
    plt.close(fig)


def corrplot(data, size_scale=500, marker='s', titlestr="", filedir=None):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        # palette=list(sns.diverging_palette(145, 300, s=60, n=10)),
        # palette_fine=list(sns.diverging_palette(145, 300, s=60, n=2^8)),
        palette=list(sns.diverging_palette(250, 30, l=65, n=10)),
        palette_fine=list(sns.diverging_palette(250, 30, l=65, n=2^8)),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale,
        title_str=titlestr,
        file_dir=filedir
    )



def CorrelationMatrix1(x, titlestr="", filedir=None, labels=None):
    d = pd.DataFrame(data=x, columns=labels)
    corr = d.corr(method=r2_func)
    corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool_)) # mask top half of matrix
    np.nan_to_num(corr, copy=False)
    size = max(50, 1000 - 56 * corr.shape[0])
    try:
        corrplot(corr, titlestr=titlestr, filedir=filedir, size_scale=size)
    except Exception as e:
        print(f"Exception during correlationmatrix1 for {titlestr}: {e}")
    return

def CorrelationBarChart(x_in, x_over=None, titlestr="", filedir=None, labels=None, xlabel="", ylabel="", x_over_suffix="", x_over_sort=False):
    if x_over:
        if not labels:
            labels = [i+1 for i in range(len(x_in))]
            xx = sorted([(i,j) for i,j in zip(x_in,x_over)], key=lambda x:x[1] if x_over_sort else x[0])
            xy = np.array([[j,i,k] for i,(j,k) in enumerate(xx)])
        else:
            x = x_in
            xy = np.array(sorted([(i,j,k) for i,j,k in zip(x,labels,x_over)], key=lambda x:x[2] if x_over_sort else x[0]))
    else:
        if not labels:
            labels = [i+1 for i in range(len(x_in))]
            x = sorted(x_in)
        else:
            x = x_in
        
        xy = np.array(sorted([(float(i),j) for i,j in zip(x,labels)]))
        
        
    y = list(range(xy.shape[0]))
    
    xy = np.transpose(xy)
    
    
    
    fig = plt.figure(dpi=300)
    if x_over:
        plt.hlines(y=y, xmin=0, xmax=xy[2].astype(float), color=[blue if float(i) < 0. else red for i in xy[2].astype(float)], alpha=0.2, linewidth=8)
        xmins = [min([float(xy[0,i]), float(xy[2,i]),0.]) for i in range(len(y))]
    else:
        xmins = [min(0.,float(xy[0,i])) for i in range(len(y))]
    plt.hlines(y=y, xmin=0, xmax=xy[0].astype(float), color=[blue if float(i) < 0. else red for i in xy[0].astype(float)], alpha=0.8, linewidth=3)
    
    # dotted grey lines to connect the Y axis labels to the bars
    minx = min(xmins)
    x2 = np.array([i for i in xmins if i > minx])
    y2 = [i for i in y if xmins[i] > minx]
    plt.hlines(y=y2, xmin=minx, xmax=x2.astype(float), color='grey', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.gca().set(xlabel="Correlation coefficient")
    ax_list = fig.axes
    for s in ["right","top"]:
        ax_list[0].spines[s].set_visible(False)
        
    ax_list[0].axvline(0,linestyle='-', color='k', linewidth=0.8)
    plt.yticks(y, xy[1])
    
    if x_over:
        custom_lines = [Line2D([0], [0], color=blue, alpha=0.8, lw=2),
                Line2D([0], [0], color=blue, alpha=0.2, lw=5)]
        max_val = max(x_in)
        min_val = min(x_over)
        plt.legend(custom_lines, [unicode_to_latex(f"{ylabel} vs {xlabel}"), unicode_to_latex(f"{ylabel} vs {xlabel}{x_over_suffix}")], loc='lower right', fontsize=8, framealpha=0.2)
    
    if titlestr != "":
        plt.title(unicode_to_latex(titlestr))
    
    if filedir:
        filestr = titlestr.replace('\n','_')
        filepath = os.path.join(filedir,sanitize(f"{filestr}{OUTPUT_FORMAT}"))
        plt.savefig(filepath, pad_inches=0.2, bbox_inches='tight')
    
    # Clear the current axes.
      
    plt.close(fig)
    
    return


def CorrelationMultiVarBarChart(x_i, x_over=None, titlestr="", filedir=None, labels_in=None, xlabels=None, ylabel="", x_over_suffix="", x_over_sort=False):
    # x_in is [numvars X numregions]
    
    
    
        
    # determine sorting
    if x_over and x_over_sort:
        sort_vals = {j:i for i,(j,k) in enumerate(sorted(enumerate(x_over[0]), key=lambda x:x[1]))}
    else:
        sort_vals =  {j:i for i,(j,k) in enumerate(sorted(enumerate(x_i[0]), key=lambda x:x[1]))}
    
    if not labels_in:
        labels = [i+1 for i in range(len(x_i[0]))]
    else:
        labels = [i[1] for i in sorted(enumerate(labels_in),key=lambda x:sort_vals[x[0]])]
    
    num_sys = len(labels_in)
    num_vars = len(xlabels)
    num_sys_rows = (num_vars + 1)
    num_rows = num_sys * num_sys_rows
    
    fig = plt.figure(figsize=(6.4,2.8 + 0.09 * num_rows), dpi=300)
    
    xmins = np.ones(num_rows) * 2
    
    for xi,x_in in enumerate(x_i):
        y = [(i * num_sys_rows) + (num_vars - xi - 1) for i in range(num_sys)]
        
        if x_over:
            xy = np.array(sorted([(i,j,k,xj) for xj,(i,j,k) in enumerate(zip(x_in,labels,x_over[xi]))], key=lambda x:sort_vals[x[3]]))
        else:
            xy = np.array(sorted([(float(i),j,xj) for xj,(i,j) in enumerate(zip(x_in,labels))], key = lambda x:sort_vals[x[2]]))
        
        xy = np.transpose(xy)
        
        if x_over:
            for i,yi in enumerate(y):
                xmins[yi] = min([float(xy[0,i]), float(xy[2,i]),0.])
        else:
            for i,yi in enumerate(y):
                xmins[yi] = min(0.,float(xy[0,i]))
        
        if x_over:
            plt.hlines(y=y, xmin=0, xmax=xy[2].astype(float), color=[blue if float(i) < 0. else red for i in xy[2].astype(float)], alpha=0.2, linewidth=7 if xi == 0 else 3)
        plt.hlines(y=y, xmin=0, xmax=xy[0].astype(float), color=[blue if float(i) < 0. else red for i in xy[0].astype(float)], alpha=0.8, linewidth=3 if xi == 0 else 1)
    
    plt.gca().set(xlabel=unicode_to_latex(f"Correlation coefficient to {ylabel}"))
    
    tick_pos = list(range(num_rows))
    tick_lab = []
    for l in labels:
        for v in xlabels:
            tick_lab.append(r"\textit{" + unicode_to_latex(v) + r"}")
        tick_lab.append(r"\textbf{" + unicode_to_latex(l) + r"}")
    plt.yticks(tick_pos, tick_lab, fontsize=10 if num_rows <= 30 else 8 if num_rows <= 60 else 6)
    
    ax_list = fig.axes
    for s in ["right","top"]:
        ax_list[0].spines[s].set_visible(False)
    
    # dotted grey lines to connect the Y axis labels to the bars
    minx = min(xmins)
    x2 = np.array([i for i in xmins if i > minx and i <= 1.])
    y2 = [i for i,j in enumerate(xmins) if j > minx and j <= 1.]
    plt.hlines(y=y2, xmin=minx, xmax=x2.astype(float), color='lightgrey', linestyle=':', linewidth=.8)
    
    # lines delineating regions
    [ax_list[0].axhline(i,linestyle='-', color='grey', linewidth=.8) for i in range(num_vars,num_rows,num_sys_rows)]
    
    # vertical line at x=0
    ax_list[0].axvline(0,linestyle='-', color='k', linewidth=0.8)
    
    if x_over:
        custom_lines = [Line2D([0], [0], color=blue, alpha=0.8, lw=2),
                        Line2D([0], [0], color=blue, alpha=0.2, lw=5)]
        all_vals = [j for i in x_i for j in i] + [j for i in x_over for j in i]
        max_val = max(all_vals)
        min_val = min(all_vals)
        plt.legend(custom_lines, [unicode_to_latex(f"{ylabel} vs Property"), unicode_to_latex(f"{ylabel} vs Property{x_over_suffix}")], loc='lower right', fontsize=8, framealpha=0.2)
    
    if titlestr != "":
        plt.title(unicode_to_latex(titlestr))
    
    if filedir:
        filestr = titlestr.replace('\n','_')
        filepath = os.path.join(filedir,sanitize(f"{filestr}{OUTPUT_FORMAT}"))
        plt.savefig(filepath, pad_inches=0.2, bbox_inches='tight')
    
    # Clear the current axes.
    
    plt.close(fig)
    
    return

def main():
    date_time_stamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    
    print(f"Starting GBA system comparison\n(using input: {' '.join(argv)})\n\n")
    
    ##################################
    # IMPORT FILES AND INITIAL PROCESSING
    ################################## 
    
    
    
    file_paths = [f for f in os.listdir(os.getcwd()) if ".csv" in f]
    
    print(f"Loading data from {len(file_paths)} file_paths in \"{os.getcwd()}\"...\n\n")

    # Import data
    all_input_files = [list(csv.DictReader(open(logfile, 'r'))) for logfile in file_paths]
    
    
    if len(all_input_files) == 0:
        print("No data found. Quitting...")
        return False
    
    # get list of systems
    systems = {}
    for s in all_input_files:
        s_name = s[0]['Dataset name']
        systems[s_name] = {}
    
    # get regions and their atomic components in each system
    for sys in systems.keys():
        atoms = {}
        regions = {}
        minmax_basins = {}
        for s in all_input_files:
            s_name = s[0]['Dataset name']
            if sys == s_name:
                for line in s:
                    if '' in [line[k] for k in ['Dataset name', 'Atom name', 'Region name']]:
                      continue
                    a_name = line['Atom name'].split(':')[0].replace(' ','')
                    r_name = line['Region name']
                    if r_name.replace(' ','') != a_name:
                        if 'full' not in a_name:
                            if "Max " not in r_name and "Min " not in r_name:
                                if r_name not in regions:
                                    regions[r_name] = {}
                                    regions[r_name]['minmax_basin_map'] = {}
                                if "Max " in line['Atom name'] or "Min " in line['Atom name']:
                                    cb_name = a_name + line['Atom name'][line['Atom name'].find(": "):]
                                    regions[r_name]['minmax_basin_map'][a_name] = cb_name
                                regions[r_name][a_name] = {k:(float(v) if " (with boundary error)" not in k else v) for k,v in line.items() if is_float(v) and k not in ['Dataset name', 'Atom name', 'Region name']}
                            else:
                                r_name = r_name[:r_name.find(":")].strip().replace(' ','') + r_name[r_name.find(":"):]
                                minmax_basins[r_name] = {k:(float(v) if " (with boundary error)" not in k else v) for k,v in line.items() if is_float(v) and k not in ['Dataset name', 'Atom name', 'Region name']}
                    
                    else:
                        atoms[a_name] = {k:float(v) for k,v in line.items() if is_float(v) and "boundary" not in k and k not in ['Dataset name', 'Atom name', 'Region name']}
        systems[sys] = {'atoms':atoms, 'regions':regions, "minmax_basins":minmax_basins}
    
    # initial processing of data 
    for sk, sv in systems.items():
        # map SGB regions to particular max/min basins
        for rk, rv in sv['regions'].items():
            if "Cage " in rk:
                continue
            def_var1 = rk.split(' from ')[1]
            minmax_basin_map = {}
            for cb1k, cb1v in rv.items():
                if cb1k in rv['minmax_basin_map']:
                    minmax_basin_map[cb1k] = rv['minmax_basin_map'][cb1k]
                    continue
                for cb2k, cb2v in sv['minmax_basins'].items():
                    found = False
                    def_var2 = cb2k.split(' from ')[1]
                    if def_var1 == def_var2 and cb2k[:cb2k.find(':')] == cb1k:
                        found = True
                        for vk, vv in cb1v.items():
                            found &= vk in cb2v and vv == cb2v[vk]
                    if found:
                        minmax_basin_map[cb1k] = cb2k
                        break
            if len(minmax_basin_map) != len([k for k in rv.keys() if "_map" not in k]):
                print(f"Failed to associate {cb1k} in {rk} of {sk} with its corresponding condensed max or min basin!")
            systems[sk]['regions'][rk]['minmax_basin_map'] = minmax_basin_map
    
        # get min/max basins that don't take part in any special gradient bundles
        for ak in sv['atoms'].keys():
            lone_minmax_basins = set()
            for cbk in sv['minmax_basins'].keys():
                if cbk[:cbk.find(":")] == ak:
                    found = False
                    for rk, rv in sv['regions'].items():
                        found = (cbk in list(rv['minmax_basin_map'].values()))
                        if found:
                            break
                    if not found:
                        lone_minmax_basins.add(cbk)
            systems[sk]['atoms'][ak]['lone_minmax_basins'] = lone_minmax_basins
        
        # determine the sgbs composure of min/max or mixed
        sgb_types = {}
        for rk,rv in sv['regions'].items():
            if "Cage " in rk:
                continue
            ismax = all(["Max" in rv['minmax_basin_map'][ak] for ak in rv.keys() if "_map" not in ak])
            ismin = all(["Min" in rv['minmax_basin_map'][ak] for ak in rv.keys() if "_map" not in ak])
            if ismax:
                sgb_types[rk] = SGB_CONDENSED_BASIN_TYPE.MAX
            elif ismin:
                sgb_types[rk] = SGB_CONDENSED_BASIN_TYPE.MIN
            else:
                sgb_types[rk] = SGB_CONDENSED_BASIN_TYPE.MIXED
        systems[sk]['region_types'] = sgb_types
        
        # fix derived regional (deformation and te/electron) values, which come out wrong for regions when computed per-dGB as done
        # by the GBA export tool
        # fix deformation variables
        for vd in ['charge','energy','volume']:
            vn = f"Deformation {vd}"
            vp = f"Deformation positive {vd}"
            vm = f"Deformation negative {vd}"
            for rn in ['atoms','regions','minmax_basins']:
                for rk,rv in sv[rn].items():
                    if rn == 'regions':
                        for ak,av in rv.items():
                            if "_map" in ak:
                                continue
                            if all([any([v.lower() in vk.lower() for vk in av.keys()]) for v in [vn,vp,vm]]):
                                vn1 = [vk for vk in av.keys() if vn.lower() in vk.lower()][0]
                                vp1 = [vk for vk in av.keys() if vp.lower() in vk.lower()][0]
                                vm1 = [vk for vk in av.keys() if vm.lower() in vk.lower()][0]
                                systems[sk][rn][rk][ak][vn1] = av[vp1] + av[vm1]
                    else:
                        if all([any([v in vk for vk in rv.keys()]) for v in [vn,vp,vm]]):
                            vn1 = [vk for vk in rv.keys() if vn in vk][0]
                            vp1 = [vk for vk in rv.keys() if vp in vk][0]
                            vm1 = [vk for vk in rv.keys() if vm in vk][0]
                            systems[sk][rn][rk][vn1] = rv[vp1] + rv[vm1]
        
        # te per electron (and valence electron) (naming based on that resulting from use of ADF tape41 files in GBA
        for vd in ['','valence ']:
            v_te = "Kinetic energy density"
            v_rho = f"{'electron ' if vd == '' else vd}density"
            v_te_per_rho = f"Te per {vd}electron"
            for rn in ['atoms','regions','minmax_basins']:
                for rk,rv in sv[rn].items():
                    if rn == 'regions':
                        for ak,av in rv.items():
                            if "_map" in ak:
                                continue
                            if all([any([v.lower() in vk.lower() for vk in av.keys()]) for v in [v_te,v_rho,v_te_per_rho]]):
                                v_te1 = [vk for vk in av.keys() if v_te.lower() in vk.lower()][0]
                                v_rho1 = [vk for vk in av.keys() if v_rho.lower() in vk.lower()][0]
                                v_te_per_rho1 = [vk for vk in av.keys() if v_te_per_rho.lower() in vk.lower()][0]
                                systems[sk][rn][rk][ak][v_te_per_rho1] = av[v_te1] / max(av[v_rho1],1e-8)
                    else:
                        if all([any([v.lower() in vk.lower() for vk in rv.keys()]) for v in [v_te,v_rho,v_te_per_rho]]):
                            v_te1 = [vk for vk in rv.keys() if v_te.lower() in vk.lower()][0]
                            v_rho1 = [vk for vk in rv.keys() if v_rho.lower() in vk.lower()][0]
                            v_te_per_rho1 = [vk for vk in rv.keys() if v_te_per_rho.lower() in vk.lower()][0]
                            systems[sk][rn][rk][v_te_per_rho1] = rv[v_te1] / max(rv[v_rho1],1e-8)
                
                    
    ##################################
    # GET INITIAL SYSTEM, ATOM ASSOCIATIONS, AND SYSTEM NICKNAMES
    ################################## 
    
    # have the user confirm the association of atoms in the neutral (initial) system to those of the perturbed (final) systems
    # first they specify the neutral system
    sorted_sys_names = sorted(list(systems.keys()))
    str1 = '\n'.join([f"{i+1}. {s}" for i,s in enumerate(sorted_sys_names)])
    initial_sys = False
    while not initial_sys:
        initial_sys = input("Select the initial, neutral system to which the other systems are being compared (enter a number)"+f"\n\n{str1}\n\nEnter a number: ")
        try:
            s_name = sorted_sys_names[int(initial_sys) - 1]
            initial_sys = s_name
        except:
            print(f"\n\nYou entered: {initial_sys}, which was not valid.\nLet's try that again...\n\n")
            initial_sys = False
    print(f"Selected initial system: {initial_sys}")
    
        # get latest file as a starting point, if present
    # Get list of all files only in the given directory
    list_of_files = [f for f in glob.glob(os.path.join(os.getcwd(),'*')) if os.path.isfile(f) and "_AtomAssociations.txt" in f]
    str1 = ''
    use_old_data = False
    if len(list_of_files) > 0:
        use_old_data = (input("Use input data from last run?  (enter for yes, \'n\' for no)") == '')
        if use_old_data:
            # Sort list of files based on last modification time in ascending order
            list_of_files = sorted( list_of_files,
                                    key = os.path.getmtime,
                                    reverse = True)
            with open(list_of_files[0]) as f:
                str1 = f.read()
                
    if not use_old_data:
        # create map of atoms in the perturbed systems to atoms of the initial system, starting with a one-to-one assumption
        # store map in text file for the user to edit
        for k,v in systems.items():
            if k == initial_sys:
                str1 += f"[{k}] (0.0; System comparison property eg energy; property short name eg ∆E; units eg kcal/mol) {k}\n"
                for a in v['atoms'].keys():
                    str1 += f"[{a}] {a}\n"
            else:
                str1 += f"[{k}] (0.0) {k} --> {initial_sys}\n"
                for a in v['atoms'].keys():
                    str1 += f"{a} --> {a} --> {a}\n"
            str1 += "\n"
    
    file_name1 = f"{date_time_stamp}_AtomAssociations.txt"
        
    with open(file_name1, 'w') as f:
        f.write(str1)
    
    input(f"""

Step 1 of 3) TIME TO SPECIFY SYSTEM ENERGY AND ASSOCIATE ATOMS IN THE FINAL SYSTEM(S) WITH THEIR COUNTERPARTS IN THE INITIAL SYSTEM.

Press enter to open a file in which you'll define system nicknames [in brackets], energies (in parentheses), and atom associations. It will open for you, but can be found at \"{os.path.join(os.getcwd(),file_name1)}\"

It's recommended to run gradient bundle analysis on all the atoms that you want to compare between initial and final systems, but you can take advantage of symmetry in order to analyze fewer atoms in gba and then using the method below to "copy" the symmetry degenerate atoms into their would-be positions in the lower-symmetry final systems.

Even when you analyze all atoms in all systems, you still need to ensure the final system atoms point to their counterparts in the initial system.

To allow for symmetry-degenerate atoms, you specify two associations, one internal to the system (i.e. an atom in the system pointing to an different atom in the same system), and one that points to an atom in the initial, unperturbed system.

In this file you can define which atoms in the initial/final systems map to which atoms. The triple association allows you to 'create' atoms that simply point to a different atom, thereby having a full list of atoms in a system where you perhaps only analyzed the symmetry-unique atoms.

Eg        C3 --> C1 --> C2         means that there will be an atom in the output called 'C3' that points to 'C1' in the specified system, that then is compared to 'C2' in the initial system.

example file before being edited:

    [react57_cube_matthew_TZP_NFC_MO62X_COSMO4_SP] (0.0) react57_cube_matthew_TZP_NFC_MO62X_COSMO4_SP --> react1b_cube_matthew_TZP_NFC_MO62X_COSMO4_QTAIM
    C36 --> C36 --> C36
    C47 --> C47 --> C47
    C52 --> C52 --> C52
    H39 --> H39 --> H39
    H63 --> H63 --> H63
    H8 --> H8 --> H8
    O4 --> O4 --> O4
    O9 --> O9 --> O9
    
    [react1b_cube_matthew_TZP_NFC_MO62X_COSMO4_QTAIM] (0.0; System comparison property eg energy full name; short name eg mathematical; units eg kcal/mol) react1b_cube_matthew_TZP_NFC_MO62X_COSMO4_QTAIM
    [C44] C44
    [C51] C51
    [C53] C53
    [H40] H40
    [H64] H64
    [H8] H8
    [O4] O4
    [O9] O9
    
    [minus10_cube_matthew_TZP_NFC_MO62X_COSMO4_SP] (0.0) minus10_cube_matthew_TZP_NFC_MO62X_COSMO4_SP --> react1b_cube_matthew_TZP_NFC_MO62X_COSMO4_QTAIM
    C44 --> C44 --> C44
    C51 --> C51 --> C51
    C53 --> C53 --> C53
    H40 --> H40 --> H40
    H64 --> H64 --> H64
    H8 --> H8 --> H8
    O4 --> O4 --> O4
    O9 --> O9 --> O9


same file after being edited:

    [KSI-Y57] (10.0) react57_cube_matthew_TZP_NFC_MO62X_COSMO4_SP --> react1b_cube_matthew_TZP_NFC_MO62X_COSMO4_QTAIM
    C36 --> C36 --> C53
    C47 --> C47 --> C44
    C52 --> C52 --> C51
    H39 --> H39 --> H40
    H63 --> H63 --> H64
    H8 --> H8 --> H8
    O4 --> O4 --> O4
    O9 --> O9 --> O9
    
    [KSI NEF] (6.8; Reaction barrier energy ∆E; kcal/mol) react1b_cube_matthew_TZP_NFC_MO62X_COSMO4_QTAIM
    [C3] C44
    [C2] C51
    [C1] C53
    [H2] H40
    [H1] H64
    [H3] H8
    [O2] O4
    [O1] O9
    
    [KSI EEF r-] (8.6) minus10_cube_matthew_TZP_NFC_MO62X_COSMO4_SP --> react1b_cube_matthew_TZP_NFC_MO62X_COSMO4_QTAIM
    C44 --> C44 --> C44
    C51 --> C51 --> C51
    C53 --> C53 --> C53
    H40 --> H40 --> H40
    H64 --> H64 --> H64
    H8 --> H8 --> H8
    O4 --> O4 --> O4
    O9 --> O9 --> O9



The atom association file for your systems will now open. Please edit it so that the atoms in each indicated system point to their correct counterparts in the initial system: {initial_sys}

Press enter to continue...\n\n""")
    
    os.system(f"open {file_name1}")
    
    input("Once you've verified the associations and made any necessary changes, save the file, return to this window and then press enter. (note that you can also copy-paste the contents of a previous atom associations file if you've already done this before for these systems)\n\nPress enter to continue, once you've made any necessary edits and saved the file...")
        

    
    num_errors = 1
    systems_copy = {k:v for k,v in systems.items()}
    while num_errors > 0:        
        # Read the file back in to get the associations
        with open(file_name1, 'r') as f:
            str2 = f.read().strip()
        
#         try:
        num_errors = 0
        for si,sys in enumerate(str2.split('\n\n')):
            lines = sys.split('\n')
            s_nickname = lines[0][lines[0].find("[")+1:lines[0].find("]")].strip()
            
            # TODO want to be able to provide a file with system energies to prevent having to enter them here
            
            
                
            s_name = lines[0][lines[0].find(")")+2:].strip()
            if len(s_name.split(' --> ')) > 1:
                s_name = s_name.split(' --> ')[0]
                try:
                    s_energy = float(lines[0][lines[0].find("(")+1:lines[0].find(")")])
                except Exception as e:
                    s_energy = lines[0][lines[0].find("(")+1:lines[0].find(")")]
                    print(f"Energy of {s_nickname}, {s_energy}, is not valid; enter a floating point number (error: {e})")
                    num_errors += 1
                    continue
            else:
                systems[initial_sys]['nickname'] = s_nickname
                s_info = lines[0][lines[0].find("(")+1:lines[0].find(")")].split(';')
                energy_nickname_full = s_info[1].strip()
                energy_nickname = s_info[2].strip()
                energy_units = s_info[3].strip()
                try:
                    s_energy = float(s_info[0])
                except Exception as e:
                    s_energy = lines[0][lines[0].find("(")+1:lines[0].find(")")]
                    print(f"Energy of {s_nickname}, {s_energy}, is not valid; enter a floating point number (error: {e})")
                    num_errors += 1
                    continue
                
                systems[s_name]['energy'] = s_energy
                atom_nicknames = {}
                for a in lines[1:]:
                    atom_nickname = a[a.find("[")+1:a.find("]")].strip()
                    atom_name = a[a.find(']')+1:].strip()
                    atom_nicknames[atom_name] = atom_nickname
                systems[s_name]['atom_nicknames'] = atom_nicknames
                systems[s_name]['atom_map'] = {k:k for k in systems[s_name]['atoms'].keys()}
                continue
            atom_map = {}
            internal_atom_map = {}
            for a in lines[1:]:
                a123 = [ai.strip().replace(' ','') for ai in a.split(' --> ')]
                if a123[2] not in systems[initial_sys]['atoms']:
                    print(f"Atom {a123[0]} (points to {a123[1]} in {s_name}) then points to {a123[2]}, which is not in {initial_sys}!")
                    num_errors += 1
                    continue
                elif a123[1] not in systems[s_name]['atoms']:
                    print(f"Atom {a123[0]} points to {a123[1]} which is not in {s_name}!")
                    num_errors += 1
                    continue
                
                # process pointer atoms, updating minmax basins and creating new regions as necessary.
                # only new logical bonds will be made, as it's feasible that a ring/cage could have e.g. three pairs of the same pair of atoms.
                # with bonds, can assume 2 atoms per interaction, so if a bond already has two atoms and both are the targets of new pointer atoms,
                # assume those two pointers constitute a separate bond that essentially points to that of the target atoms. 
                if a123[0] not in systems[s_name]['atoms']:
                    systems[s_name]['atoms'][a123[0]] = systems[s_name]['atoms'][a123[1]]
                    minmax_basins = {}
                    # copy minmax basins for pointer atom
                    for cbk,cbv in systems[s_name]['minmax_basins'].items():
                        if a123[1] in cbk:
                            minmax_basins[cbk.replace(a123[1],a123[0])] = cbv
                    systems[s_name]['minmax_basins'].update(minmax_basins)
                    
                if a123[1] not in atom_map:
                    atom_map[a123[1]] = a123[2] # maps existing atom to existing atom in initial system
                atom_map[a123[0]] = a123[2] # maps new pointer to existing atom in initial system
                internal_atom_map[a123[0]] = a123[1] # maps new pointer atom to existing atom
                
            systems[s_name]['atom_map'] = atom_map
            
            # first match up two-member bonds, then single atom bonds
            single_atom_regions = []
            region_map = {}
            minmax_basin_correlations = {}
            for rk,rv in systems[s_name]['regions'].items():
                if len(rv) == 2: # map and one atom, are the 2 regions
                    single_atom_regions.append(rk)
                    continue
                
                cor_list = []
                lvals = [sum(av[k] for ak,av in rv.items() if "_map" not in ak) for k in rv[next(rvk for rvk in rv.keys() if "_map" not in rvk)].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                for rki,rvi in systems[initial_sys]['regions'].items():
                    if len(rk) > 5 and len(rki) > 5 and rk[:5] == rki[:5] and all([int(systems[s_name]['atom_map'][a] in rvi) for a in rv.keys() if "_map" not in a]):
                        rvals = [sum(av[k] for ak,av in rvi.items() if "_map" not in ak) for k in rvi[next(rvk for rvk in rvi.keys() if "_map" not in rvk)].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                        cor = compare_func(lvals, rvals)
                        cor_list.append([cor,rki])
                        # region_map[rk] = rki
                        # break
                if len(cor_list) > 1:
                    minmax_basin_correlations[rk] = sorted(cor_list, key = lambda x: x[0])
                elif len(cor_list) > 0:
                    cor_list[0][0] = 0
                    minmax_basin_correlations[rk] = [cor_list[0]]
            
            if any(len(c) > 0 for c in minmax_basin_correlations.values()):
                for r_type in ["Bond ","Ring ","Cage "]:
                    col_strs = sorted([r for r,v in minmax_basin_correlations.items() if r_type in r and len(v)>1 and v[0][0] > 0.])
                    row_strs = sorted(list({v[1] for col in col_strs for v in minmax_basin_correlations[col] if r_type in v[1]}))
                    
                    if len(col_strs) == 0 or len(row_strs) == 0:
                        continue
                    
                    cost_matrix = np.ndarray(shape=(len(row_strs),len(col_strs)))
                    cost_matrix.fill(-1)
                    for cj, col in enumerate(col_strs):
                        for row in minmax_basin_correlations[col]:
                            ci = row_strs.index(row[1])
                            cost_matrix[ci,cj] = row[0]
                    cost_max = np.max(cost_matrix)
                    for cj in range(len(col_strs)):
                        for ci in range(len(row_strs)):
                            if cost_matrix[ci,cj] < 0.:
                                cost_matrix[ci,cj] = cost_max + 1e100
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    for cj in range(len(col_strs)):
                        for ci in range(len(row_strs)):
                            if cost_matrix[ci,cj] > cost_max:
                                cost_matrix[ci,cj] = -1.
                    minmax_basin_correlations.update({col_strs[cj]:[[cost_matrix[ci,cj],row_strs[ci]]] for ci,cj in zip(row_ind,col_ind)})
                    
                
            region_map = {rk:rv[0][1] for rk,rv in minmax_basin_correlations.items() if len(rv) > 0}
            systems[s_name]['region_map_differences'] = {rk:rv[0][0] for rk,rv in minmax_basin_correlations.items() if len(rv) > 0}
            
            minmax_basin_correlations = {}
            for rk in single_atom_regions:
                rv = systems[s_name]['regions'][rk]
                cor_list = []
                
                if "Cage " in rk:
                    rkeys = list([rkk for rkk in rv.keys() if "_map" not in rkk])
                    lvals = [sum(av[k] for ak,av in rv.items() if "_map" not in ak) for k in rv[next(rkk for rkk in rv.keys() if "_map" not in rkk)].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                    for rki,rvi in systems[initial_sys]['regions'].items():
                        if rki not in region_map.values() and len(rk) > 5 and len(rki) > 5 and rk[:5] == rki[:5] and all([int(systems[s_name]['atom_map'][a] in rvi) for a in rv.keys() if "_map" not in a]):
                            rvals = [sum(av[k] for ak,av in rvi.items() if "_map" not in ak) for k in rvi[next(rkk for rkk in rvi.keys() if "_map" not in rkk)].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                            cor = compare_func(lvals, rvals)
                            cor_list.append([cor,rki])
                            # region_map[rk] = rki
                            # break
                    minmax_basin_correlations[rk] = sorted(cor_list, key = lambda x: x[0])
                else:
                    lvals = [sum(av[k] for ak,av in rv.items() if "_map" not in ak) for k in rv[next(rkk for rkk in rv.keys() if "_map" not in rkk)].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                    for rki,rvi in systems[initial_sys]['regions'].items():
                        if rki not in region_map.values() and len(rk) > 5 and len(rki) > 5 and rk[:5] == rki[:5] and all([int(systems[s_name]['atom_map'][a] in rvi) for a in rv.keys() if "_map" not in a]):
                            rvals = [sum(av[k] for ak,av in rvi.items() if "_map" not in ak) for k in rvi[next(rkk for rkk in rvi.keys() if "_map" not in rkk)].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                            cor = compare_func(lvals, rvals)
                            cor_list.append([cor,rki])
                            # region_map[rk] = rki
                            # break
                    minmax_basin_correlations[rk] = sorted(cor_list, key = lambda x: x[0])
            
            for r_type in ["Bond ","Ring ","Cage "]:
                col_strs = sorted([r for r,v in minmax_basin_correlations.items() if r_type in r and len(v)>1 and v[0][0] > 0.])
                row_strs = sorted(list({v[1] for col in col_strs for v in minmax_basin_correlations[col] if r_type in v[1]}))
                
                if len(col_strs) == 0 or len(row_strs) == 0:
                    continue
                
                cost_matrix = np.ndarray(shape=(len(row_strs),len(col_strs)))
                cost_matrix.fill(-1)
                for cj, col in enumerate(col_strs):
                    for row in minmax_basin_correlations[col]:
                        ci = row_strs.index(row[1])
                        cost_matrix[ci,cj] = row[0]
                cost_max = np.max(cost_matrix)
                for cj in range(len(col_strs)):
                    for ci in range(len(row_strs)):
                        if cost_matrix[ci,cj] < 0.:
                            cost_matrix[ci,cj] = cost_max + 1e100
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for cj in range(len(col_strs)):
                    for ci in range(len(row_strs)):
                        if cost_matrix[ci,cj] > cost_max:
                            cost_matrix[ci,cj] = -1.
                minmax_basin_correlations.update({col_strs[cj]:[[cost_matrix[ci,cj],row_strs[ci]]] for ci,cj in zip(row_ind,col_ind)})
                
            region_map.update({rk:rv[0][1] for rk,rv in minmax_basin_correlations.items() if len(rv) > 0})
            systems[s_name]['region_map_differences'].update({rk:rv[0][0] for rk,rv in minmax_basin_correlations.items() if len(rv) > 0})
            
        #     minmax_basin_correlations = {}
        #     for rk in single_atom_regions:
        #         if "Cage " in rk:
        #             rv = systems[s_name]['regions'][rk]
        #             cor_list = []
        #             rkeys = list([rkk for rkk in rv.keys() if "_map" not in rkk])
        #             lvals = [sum(av[k] for ak,av in rv.items() if "_map" not in ak) for k in rv[list([rkk for rkk in rv.keys() if "_map" not in rkk])[0]].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
        #             for rki,rvi in systems[initial_sys]['regions'].items():
        #                 if len(rk) > 5 and len(rki) > 5 and rk[:5] == rki[:5] and all([int(systems[s_name]['atom_map'][a] in rvi) for a in rv.keys() if "_map" not in a]):
        #                     rvals = [sum(av[k] for ak,av in rvi.items() if "_map" not in ak) for k in rvi[list([rkk for rkk in rvi.keys() if "_map" not in rkk])[0]].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
        #                     cor = compare_func(lvals, rvals)
        #                     cor_list.append([cor,rki])
        #                     # region_map[rk] = rki
        #                     # break
        #             minmax_basin_correlations[rk] = sorted(cor_list, key = lambda x: x[0])
        #         else:
        #             rv = systems[s_name]['regions'][rk]
        #             cor_list = []
        #             lvals = [sum(av[k] for ak,av in rv.items() if "_map" not in ak) for k in rv[list(rv.keys())[0]].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
        #             for rki,rvi in systems[initial_sys]['regions'].items():
        #                 if len(rk) > 5 and len(rki) > 5 and rk[:5] == rki[:5] and all([int(systems[s_name]['atom_map'][a] in rvi) for a in rv.keys() if "_map" not in a]):
        #                     rvals = [sum(av[k] for ak,av in rvi.items() if "_map" not in ak) for k in rvi[list(rvi.keys())[0]].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
        #                     cor = compare_func(lvals, rvals)
        #                     cor_list.append([cor,rki])
        #                     # region_map[rk] = rki
        #                     # break
        #             minmax_basin_correlations[rk] = sorted(cor_list, key = lambda x: x[0])
        #
        #     col_strs = sorted(list(minmax_basin_correlations.keys()))
        #     row_strs = sorted(list({v[1] for rv in minmax_basin_correlations.values() for v in rv}))
        #
        #     if False: #len(col_strs) > 1 and len(col_strs) != len(row_strs):
        #         print(f"Different number of regions on initial vs final system!")
        #                     # loop over minmax_basin_correlations pairing basins with their closest matches
        #         is_added = {}
        #         for cb1k in minmax_basin_correlations.keys():
        #             # while len(minmax_basin_correlations[cb1k]) > 0:
        #             #     do_break = True
        #             #     for c in minmax_basin_correlations[cb1k]:
        #             #         if c[1] in is_added:
        #             #             del(minmax_basin_correlations[cb1k][0])
        #             #             do_break = False
        #             #             break
        #             #     if do_break:
        #             #         break
        #             if len(minmax_basin_correlations[cb1k]) < 1:
        #                 continue
        #             a_name = cb1k[:cb1k.find(":")]
        #             def_var1 = cb1k.split(' from ')[1] if "Cage " not in cb1k else ""
        #             while len(minmax_basin_correlations[cb1k]) > 1:
        #                 do_break = True
        #                 # check that no other minmax_basin has a better correlation with cb1k's best match
        #                 for cb2k in minmax_basin_correlations.keys():
        #                     def_var2 = cb2k.split(' from ')[1] if "Cage " not in cb2k else ""
        # #                     print(f"{cb1k = }\n{cb2k = }\n")
        #                     if def_var2 == def_var1 and cb1k != cb2k and len(minmax_basin_correlations[cb2k]) > 0 and minmax_basin_correlations[cb2k][0][1] == minmax_basin_correlations[cb1k][0][1] and minmax_basin_correlations[cb2k][0][0] < minmax_basin_correlations[cb1k][0][0]:
        #                         del(minmax_basin_correlations[cb1k][0])
        #                         do_break = False
        #                         break
        #                 if do_break:
        #                     break
        #             if len(minmax_basin_correlations[cb1k]) > 0:
        #                 region_map[cb1k] = minmax_basin_correlations[cb1k][0][1]
        #                 is_added[region_map[cb1k]] = True
        #     else:
        #         cost_matrix = np.ndarray(shape=(len(col_strs),len(row_strs)))
        #         cost_matrix.fill(-1)
        #         for ci, col in enumerate(col_strs):
        #             for row in minmax_basin_correlations[col]:
        #                 cj = row_strs.index(row[1])
        #                 cost_matrix[ci,cj] = row[0]
        #         cost_max = np.max(cost_matrix)
        #         for ci in range(len(col_strs)):
        #             for cj in range(len(row_strs)):
        #                 if cost_matrix[ci,cj] < 0.:
        #                     cost_matrix[ci,cj] = cost_max + 1e100
        #         row_ind, col_ind = linear_sum_assignment(cost_matrix)
        #         minmax_basin_correlations = {col_strs[ci]:[cost_matrix[ci,cj],row_strs[cj]] for ci,cj in enumerate(col_ind)}
        #         region_map = {col_strs[ci]:row_strs[cj] for ci,cj in enumerate(col_ind)}
        #         systems[s_name]['region_map_differences'] = {col_strs[ci]:cost_matrix[ci,cj] for ci,cj in enumerate(col_ind)}
                
            
            # cost_matrix = [[ for vj in ]]
            
            
            
            systems[s_name]['region_map'] = region_map
            systems[s_name]['nickname'] = s_nickname
            systems[s_name]['energy'] = s_energy
            
            
            # update regions for pointer atom
            new_regions = {k:v for k,v in systems[s_name]['regions'].items()}
            for rk,rv in systems[s_name]['regions'].items():
                for ak,av in internal_atom_map.items():
                    if ak != av and av in rv:
                        if "Bond " not in rk or len(rv) <= 2:
                            # singly bond-wedge occupied bond, or not a bond, so add pointer atom to this region
                            new_regions[rk]['minmax_basin_map'][ak] = rv['minmax_basin_map'][av].replace(av,ak)
                            new_regions[rk][ak] = rv[av]
                        elif "Bond " in rk and len(rv) == 3:
                            # full bond, so need to make a new one for this pointer atom.
                            other_atom = [a for a in rv.keys() if a != av and "_map" not in a][0]
                            make_new_bond = True
                            cor_list = []
                            
                            # check if there is an existing bond that this pointer atom belongs to
                            # (e.g. this is a degenerate H that forms one of the C-H bonds on a C atom)
                            if make_new_bond and internal_atom_map[other_atom] == other_atom:
                                # match the correct basin on the other_atom based on correlation of bond integrals
                                lvals = [rv[av][k] + rv[other_atom][k] for k in rv[av].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                                for r2k,r2v in new_regions.items():
                                    if "Bond " in r2k and ' '.join(rk.split(" ")[2:]) == ' '.join(r2k.split(" ")[2:]) and other_atom in r2v and len(r2v) == 2:
                                        rvals = [rv[av][k] + r2v[other_atom][k] for k in rv[av].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                                        cor = compare_func(lvals, rvals)
                                        cor_list.append([cor,r2k])
                                
                            # check to see if a bond has already been made for this atom
                            # (i.e. a bond with the corresponding counterpart atom that was made in a previous iteration)
                            if make_new_bond:
                                lvals = [rv[av][k] + rv[other_atom][k] for k in rv[av].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                                for r2k,r2v in new_regions.items():
                                    if "Bond " in r2k and ' '.join(rk.split(" ")[2:]) == ' '.join(r2k.split(" ")[2:]) and len(r2v) == 2:
                                        for a2k in r2v.keys():
                                            if "_map" not in a2k and internal_atom_map[a2k] != a2k and (internal_atom_map[a2k] == other_atom or internal_atom_map[a2k] == internal_atom_map[other_atom]):
                                                rvals = [rv[av][k] + r2v[a2k][k] for k in rv[av].keys() if any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                                                cor = compare_func(lvals, rvals)
                                                cor_list.append([cor,r2k])
                            
                            if len(cor_list) > 0:
                                cor = sorted(cor_list, key = lambda x: x[0])[0]
                                if len(cor_list) == 1 or cor[0] < 100:
                                    new_regions[cor[1]][ak] = rv[av]
                                    new_regions[cor[1]]['minmax_basin_map'][ak] = rv['minmax_basin_map'][av].replace(av,ak)
                                    make_new_bond = False
                                    
                            if make_new_bond:
                                def_var = rk.split(' from ')[1]
                                bond_nums = [int(b.split(' ')[1]) for b in new_regions.keys() if "Bond " in b and def_var == b.split(' from ')[1]]
                                bond_num = 1
                                if len(bond_nums) > 0:
                                    while bond_num <= max(bond_nums):
                                        if bond_num not in bond_nums:
                                            break
                                        bond_num += 1

                                bond_name = f"Bond {bond_num} " + ' '.join(rk.split(" ")[2:])
                                    
                                new_regions[bond_name] = {ak:rv[av], 'minmax_basin_map':{ak:rv['minmax_basin_map'][av].replace(av,ak)}}
            systems[s_name]['regions'] = new_regions
            systems[s_name]['internal_atom_map'] = internal_atom_map
            
            
            # update region map for new regions
            for r1k,r1v in systems[s_name]['regions'].items():
                if "_map" not in r1k and all([internal_atom_map[a] != a for a in r1v.keys() if "_map" not in a]):
                    # found a new pointer region, so look for it's target region
                    for r2k,r2v in systems[s_name]['regions'].items():
                        if "_map" not in r2k and r2k != r1k and all([internal_atom_map[a] in r2v for a in r1v.keys() if "_map" not in a]):
                            # found target region
                            systems[s_name]['region_map'][r1k] = systems[s_name]['region_map'][r2k]
                        
                                

                    
        
        if num_errors:
            input(f"\n\nProblem(s) found with {num_errors} atom association(s). Please fix, resave the file, and press enter to check again.")
            systems = {k:v for k,v in systems_copy.items()}
            with open(file_name1, 'w') as f:
                f.write(str1)
#         except Exception as e:
#             input(f"Problem reading the edited file: {str(e)}.\n\nLet's try that again... (ctrl-c to quit)")
#             exc_type, value, exc_traceback = exc_info()
#             print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
#             systems = {k:v for k,v in systems_copy.items()}
#             with open(file_name1, 'w') as f:
#                 f.write(str1)
    
    systems = {sk:sv for sk,sv in systems.items() if 'atom_map' in sv or sk == initial_sys}
    
    ##################################
    # MAP SYSTEMS TO INITIAL SYSTEMS AND TO EACHOTHER
    ################################## 
    
    # map min/max basins to initial system based on correlation of integration values
    for sk,sv in systems.items():
        if sk == initial_sys:
            continue
        minmax_basin_correlations = {}
        minmax_basin_map = {}
        systems[sk]['minmax_basin_map_corrcoefs'] = {}
        is_added = {}
        for cb1k,cb1v in sv['minmax_basins'].items():
            cor_list = []
            a_name = cb1k[:cb1k.find(":")]
            def_var1 = cb1k.split(' from ')[1]
            
            # first see if mapped regions can be used to precicely map condensed basin
            is_found = False
            for rk1,rv1 in sv['regions'].items():
                if rk1 in sv['region_map'] and "Bond " in rk1 and len(rv1) == 3:
                    for rcb1,rcb2 in rv1['minmax_basin_map'].items():
                        if cb1k == rcb2 and len(systems[initial_sys]['regions'][sv['region_map'][rk1]]) == 3:
                            # have the condensed basin in a region; now find the corresponding condensed basin in the initial system for the corresponding region
                            for rcbi1,rcbi2 in systems[initial_sys]['regions'][sv['region_map'][rk1]]['minmax_basin_map'].items():
                                if rcbi1 == sv['atom_map'][rcb1]:
                                    minmax_basin_map[cb1k] = rcbi2
                                    is_added[rcbi2] = True
                                    systems[sk]['minmax_basin_map_corrcoefs'][cb1k] = 0.
                                    minmax_basin_correlations[cb1k] = [[0.,rcbi2]]
                                    is_found = True
                                    break
                        if is_found:
                            break
                if is_found:
                    break
            
        
        for cb1k,cb1v in sv['minmax_basins'].items():
            if cb1k in minmax_basin_map:
                continue
            cor_list = []
            a_name = cb1k[:cb1k.find(":")]
            def_var1 = cb1k.split(' from ')[1]
            
            for cb2k,cb2v in systems[initial_sys]['minmax_basins'].items():
                if cb2k in minmax_basin_map.values():
                    continue
                def_var2 = cb2k.split(' from ')[1]
                if def_var2 == def_var1 and sv['atom_map'][a_name] == cb2k[:cb2k.find(":")] and (all(["Max " in k for k in [cb1k,cb2k]]) or  all(["Min " in k for k in [cb1k,cb2k]])):
                    lvals = [v for k,v in cb1v.items() if k in cb2v and any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                    rvals = [v for k,v in cb2v.items() if k in cb1v and any([v in k.lower() for v in BASIN_COMPARISON_VARLIST]) and all([v not in k.lower() for v in BASIN_COMPARISON_VARBLACKLIST])]
                    cor = compare_func(lvals, rvals)
                    cor_list.append([cor,cb2k])
            minmax_basin_correlations[cb1k] = sorted(cor_list, key = lambda x: x[0])
        
        all_r_names = sorted(list(minmax_basin_correlations.keys()))
        a_names = {col[:col.find(":")] for col in all_r_names}
        for a_name in a_names:
            for r_type in ["Min ","Max "]:
                col_strs = sorted([r for r in minmax_basin_correlations.keys() if a_name == r[:r.find(":")] and r_type in r and r not in minmax_basin_map])
                row_strs = sorted(list({v[1] for col in col_strs for v in minmax_basin_correlations[col] if sv['atom_map'][a_name] == v[1][:v[1].find(":")] and r_type in v[1]}))
                
                if len(col_strs) == 0 or len(row_strs) == 0:
                    continue
                
                cost_matrix = np.ndarray(shape=(len(row_strs),len(col_strs)))
                cost_matrix.fill(-1)
                for cj, col in enumerate(col_strs):
                    def_var1 = col.split(' from ')[1]
                    a_name = col[:col.find(":")]
                    for row in minmax_basin_correlations[col]:
                        def_var2 = row[1].split(' from ')[1]
                        if def_var2 == def_var1 and sv['atom_map'][a_name] == row[1][:row[1].find(":")] and (all(["Max " in k for k in [col,row[1]]]) or  all(["Min " in k for k in [col,row[1]]])):
                            ci = row_strs.index(row[1])
                            cost_matrix[ci,cj] = row[0]
                cost_max = np.max(cost_matrix)
                for cj in range(len(col_strs)):
                    for ci in range(len(row_strs)):
                        if cost_matrix[ci,cj] < 0.:
                            cost_matrix[ci,cj] = cost_max + 1e100
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for cj in range(len(col_strs)):
                    for ci in range(len(row_strs)):
                        if cost_matrix[ci,cj] > cost_max:
                            cost_matrix[ci,cj] = -1.
                for ci,cj in zip(row_ind,col_ind):
                    minmax_basin_correlations[col_strs[cj]] = [cost_matrix[ci,cj],row_strs[ci]]
                    minmax_basin_map[col_strs[cj]] = row_strs[ci]
                    systems[sk]['minmax_basin_map_corrcoefs'][col_strs[cj]] = cost_matrix[ci,cj]
        
        is_added = set(minmax_basin_map.values())
        systems[sk]['minmax_basin_map'] = minmax_basin_map
        systems[sk]['initial_sys_unmapped_minmax_basins'] = {k:v for k,v in systems[initial_sys]['minmax_basins'].items() if k not in is_added}
        
        is_added = {v for v in systems[sk]['region_map'].values()}
        systems[sk]['initial_sys_unmapped_regions'] = {k:v for k,v in systems[initial_sys]['regions'].items() if k not in is_added}
    
    
    # for sk,sv in systems.items():
    #     if sk == initial_sys:
    #         continue
    #
        # for rk,rv in sv['regions'].items():
        #     if rk in sv['region_map'] and 'minmax_basin_map' in rv and 'minmax_basin_map' in systems[initial_sys]['regions'][sv['region_map'][rk]]:
        #         for ck,cv in rv['minmax_basin_map'].items():
        #             systems[sk]['minmax_basin_map'][cv] = systems[initial_sys]['regions'][sv['region_map'][rk]]['minmax_basin_map'][sv['atom_map'][ck]]
        #             systems[sk]['minmax_basin_map_corrcoefs'][cv] = 0
        #
        # is_added = {v for v in systems[sk]['region_map'].values()}
        # systems[sk]['initial_sys_unmapped_minmax_basins'] = {k:v for k,v in systems[initial_sys]['minmax_basins'].items() if k not in is_added}
    
        # correct region maps, and minmax basin maps based on known correspondence using region minmax basin maps
    # for sk,sv in systems.items():
    #     if sk == initial_sys:
    #         continue
    #
    #     for rk,rv in sv['regions'].items():
    #         if len(rv) == 2:
    #             cb1,cb2 = list(rv['minmax_basin_map'].items())[0]
    #             for rki,rvi in systems[initial_sys]['regions'].items():
    #                 if len(rvi) > 2 and 'minmax_basin_map' in rvi and cb2 in list(rvi['minmax_basin_map'].values()):
    #                     del(systems[sk]['regions'][rk])
    #                     break
    #
    # for sk,sv in systems.items():
    #     if sk == initial_sys:
    #         continue
    #
    #     for rk,rv in sv['regions'].items():
    #         for rki,rvi in systems[initial_sys]['regions'].items():
    #             if 'minmax_basin_map' in rv and 'minmax_basin_map' in rvi \
    #                 and rk[:5] == rki[:5] \
    #                 and len(rv) > 2 and len(rvi) > 2 \
    #                 and all(sv['atom_map'][a1] in rvi for a1 in rv if '_map' not in a1):
    #
    #                 systems[sk]['region_map'][rk] = rki
    #                 systems[sk]['region_map_differences'][rk] = 0
    #                 # break
    #
    #     is_added = {v for v in systems[sk]['region_map'].values()}
    #     systems[sk]['initial_sys_unmapped_regions'] = {k:v for k,v in systems[initial_sys]['regions'].items() if k not in is_added}
    #
    #     # get min/max basins that don't take part in any special gradient bundles
    #     for ak in sv['atoms'].keys():
    #         lone_minmax_basins = set()
    #         for cbk in sv['minmax_basins'].keys():
    #             if cbk[:cbk.find(":")] == ak:
    #                 found = False
    #                 for rk, rv in sv['regions'].items():
    #                     found = (cbk in list(rv['minmax_basin_map'].values()))
    #                     if found:
    #                         break
    #                 if not found:
    #                     lone_minmax_basins.add(cbk)
    #         systems[sk]['atoms'][ak]['lone_minmax_basins'] = lone_minmax_basins
        
    
    # reverse map of initial system minmax basins pointing to final system minmax basins
    reverse_minmax_basin_map = {}
    for cb1k,cb1v in systems[initial_sys]['minmax_basins'].items():
        reverse_minmax_basin_map[cb1k] = []
        for s2k,s2v in systems.items():
            if s2k != initial_sys:
                for cb2f,cb2i in s2v['minmax_basin_map'].items():
                    if cb1k == cb2i:
                        reverse_minmax_basin_map[cb1k].append([s2k,cb2f])
    
    
    
    ##################################
    # GET LIST OF VARIABLES TO INCLUDE
    ################################## 
    
    # now get list of variables to look at the same way
    # get list of variables present in all systems
    var_list = systems[initial_sys]['atoms'][list(systems[initial_sys]['atoms'].keys())[0]]
    for k,v in systems.items():
        var_list = {k1:v1 for k1,v1 in var_list.items() if k1 in v['atoms'][list(v['atoms'].keys())[0]] and ":" in k1}
        
        
    # map min-min basins to bond bundles
    max_basin_to_bond_bundle_map = {}
    for rk,rv in systems[initial_sys]['regions'].items():
        if "Bond " in rk and len(rv) == 3:
            region_name = f"{'-'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in rv.keys() if '_map' not in k]) if '_map' not in k])}"
            for rk2 in rv['minmax_basin_map'].values():
                max_basin_to_bond_bundle_map[rk2] = region_name
    
            # get latest file as a starting point, if present
    # Get list of all files only in the given directory
    list_of_files = [f for f in glob.glob(os.path.join(os.getcwd(),'*')) if os.path.isfile(f) and "_Variables.txt" in f]
    if len(list_of_files) > 0 and use_old_data:
        # Sort list of files based on last modification time in ascending order
        list_of_files = sorted( list_of_files,
                                key = os.path.getmtime,
                                reverse = True)
        with open(list_of_files[0]) as f:
            str1 = f.read()
    else:
        str1 = '\n'.join([f"[{v};{v}] {v}" for v in var_list.keys()])
    
    file_name1 = f"{date_time_stamp}_Variables.txt"
    with open(file_name1, 'w') as f:
        f.write(str1)
    
    input(CLEAR + f"\n\nStep 2 of 3) TIME TO NAME SELECT AND ORDER VARIABLES TO BE COMPARED AND INCLUDED IN STATISTICAL ANALYSIS.\n\nPress enter to open the list of variable names present in all systems found that has been saved to \"{os.path.join(os.getcwd(),file_name1)}\"\n\nPlease order them how you'd like them to appear in the output and remove any that you don't want to analyze\n\n(If variables you wanted to look at are missing from the list, it means they weren't present in all systems so weren't included)")
    
    os.system(f"open {file_name1}")
    
    var_nickname = {}
    var_nickname_full = {}
    
    input("Once you've removed any unwanted variables, save the file, return to this window and then press enter. (note that you can also copy-paste the contents of a previous variable name file if you've already done this before for these systems)\n")
        
    num_errors = 1
    while num_errors > 0:        
        # Read the file back in to get the associations
        with open(file_name1, 'r') as f:
            str2 = f.read().strip()
        
        new_var_list = []
        for s in str2.split('\n'):
            if s.strip() == '':
                continue
            nickname_pair = s[s.find("[")+1:s.find("]")].split(';')
            nickname = nickname_pair[1].strip()
            nickname_full = nickname_pair[0].strip()
            name = s[s.find("]")+1:].strip()
            new_var_list.append(name)
            var_nickname[name] = nickname
            var_nickname_full[name] = nickname_full
        
        num_errors = 0
        for var in new_var_list:
            if var not in var_list:
                print(f"Variable {var} is invalid!")
        
        if num_errors:
            input(f"\n\nProblem(s) found with {num_errors} variable(s). Please fix, resave the file, and press enter to check again.")
    
    var_list = [k for k in new_var_list]
    
    
    ##################################
    # OUTPUT COMPARISONS AND GENERATE DIFFERENCE DATA FOR ML
    ################################## 
    
    # Output comparisons in a couple forms:
    # 1. A file for each system comparison that includes all variables
    # 2. A file for each variable that includes all systems, where each system gets it's value/difference/percent change grouped together
    # 3. Same as (2) but with all values, all differences, and all percent changes grouped together
    
    # Calculating changes in each condensed basin for each variable, and reporting it as a total and as a percent change.
    # Layout of first type of output csv file will be similar to Table 1 of  https://doi.org/10.33774/chemrxiv-2021-9tjtw-v3
    # One file per final perturbed system:
    #     1. First show changes in atomic basins, 
    #     2. Section for each class of special gradient bundle (i.e. first bonds, then rings...), each with a followup section showing the non-bonded (or non-ringed) regions, 
    #     3. Section for atoms again, but this time with their complete list of max then min basins and the SGBs to which they belong, 
    
    # output 1
    dir_name = f"{date_time_stamp}_output"
    os.mkdir(dir_name)
    header_str1 = "Region," + ",".join([f"{var_nickname[v]},,," for v in var_list])
    
    
    # get list of variables in use
    defining_var_list = set()
    for cbk in systems[initial_sys]['minmax_basins'].keys():
        v_name = cbk.split(" from ")[1]
        defining_var_list.add(v_name)
    
    system_diff_vals = {sk:sv for sk,sv in systems.items()}
    
    #ouput compison csv files
    if True:
        for sk,sv in systems.items():
            if sk == initial_sys:
                continue
            header_str2 = "Atomic basin decomposition," + ",".join([f"{systems[initial_sys]['nickname']},{sv['nickname']},∆,%∆" for var in var_list])
            
            with open(os.path.join(dir_name,f"{sanitize(sk)} minus {sanitize(initial_sys)}.csv"), "w", encoding='utf_8_sig') as f:
                
                f.write(header_str1 + "\n" + header_str2 + "\n")
                
                # atomic basin decomposition
                def output_atomic_basin_decomposition():
                    # write out a row for each atom in the final system
                    iatoms = sorted([(sv['atom_map'][k],k) for k in sv['atoms'].keys()], key=lambda x: systems[initial_sys]['atom_nicknames'][x[0]])
                    
                    for ik,ak in iatoms:
                        ik1 = systems[initial_sys]['atom_nicknames'][ik]
                        f.write(f"{ik1},") 
                        av = sv['atoms'][ak]
                        for var in var_list:
                            i_val = systems[initial_sys]['atoms'][sv['atom_map'][ak]][var]
                            f_val = av[var]
                            diff = f_val - i_val
                            percent_diff = (diff / i_val * 100.) if i_val != 0. else 0.
                            f.write(f"{i_val:.8f},{f_val:.8f},{diff:.8f},{percent_diff:.8f},")
                            system_diff_vals[sk]['atoms'][ak][diff_func(var)] = diff
                        f.write('\n')
                    # the total line
                    f.write("Total,")
                    for var in var_list:
                        i_tot = sum([systems[initial_sys]['atoms'][sv['atom_map'][ak]][var] for ak in sv['atoms'].keys()])
                        f_tot = sum([sv['atoms'][ak][var] for ak in sv['atoms'].keys()])
                        diff = f_tot - i_tot
                        percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                        f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                    f.write('\n')
                    
                output_atomic_basin_decomposition()
                
                # special gradient bundle decompositions
                def output_decomposition(r_type):
                    # r-type bundle decomposition (One of "Bond ", "Ring ", "Cage ") using bundles defined by one or more variables
                    for def_var in defining_var_list:
                    
                        # write out a row for each bond bundle in the final system, with it's component bond wedges
                        var_itotals = {k:0. for k in var_list}
                        var_ftotals = {k:0. for k in var_list}
                        num_regions = 0
                        
                        region_names = sorted([[rk, f"{' — '.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][sv['atom_map'][k]] for k in rv.keys() if '_map' not in k]) if '_map' not in k])}" + f" {r_type.lower()}bundle"] for rk,rv in sv['regions'].items() if def_var in rk and r_type in rk and rk in sv['region_map']], key = lambda x: x[1])
                        for rk,r_name in region_names:
                            rv = sv['regions'][rk]
                            if num_regions == 0:
                                f.write(f"{r_type}bundle decomposition according to {def_var}\n")
                            num_regions += 1
                            # region info
#                             r_name = f"{' — '.join([k for k in sorted(list(rv.keys())) if '_map' not in k])}" + f" {r_type.lower()}bundle"
                            f.write(f"{r_name},") 
                            
                            # region var totals
                            var_ivals = {k:0. for k in var_list}
                            var_fvals = {k:0. for k in var_list}
                            for var in var_list:
                                for ak, av in rv.items():
                                    if not "_map" in ak:
                                        try:
                                            # var_ivals[var] += systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][rv['minmax_basin_map'][ak]]][var]
                                            var_ivals[var] += systems[initial_sys]['regions'][sv['region_map'][rk]][sv['atom_map'][ak]][var]
                                            var_fvals[var] += av[var]
                                            var_itotals[var] += systems[initial_sys]['regions'][sv['region_map'][rk]][sv['atom_map'][ak]][var]
                                            # var_itotals[var] += systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][rv['minmax_basin_map'][ak]]][var]
                                            var_ftotals[var] += av[var]
                                        except Exception as e:
                                            print(f"Exception fetching value for total {r_type} value for {var} for {ak} in {rk} in {sk} defined by {def_var}: {str(e)}")
                                            exc_type, value, exc_traceback = exc_info()
                                            print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
                            for var in var_list:
                                diff = var_fvals[var] - var_ivals[var]
                                percent_diff = (diff / var_ivals[var] * 100.) if var_ivals[var] != 0. else 0.
                                f.write(f"{var_ivals[var]:.8f},{var_fvals[var]:.8f},{diff:.8f},{percent_diff:.8f},")
                            f.write('\n')
                            
                            # constituent condensed basin values
                            icbs = sorted([(systems[initial_sys]['atom_nicknames'][sv['atom_map'][k]],k) for k in rv.keys() if "_map" not in k], key=lambda x:x[0])
                            for icb,cb2k in icbs:
                                cbv = rv[cb2k]
                                if not "_map" in cb2k:
                                    f.write(f"      ↳ {icb} {r_type.lower()}wedge,")
                                    for var in var_list:
                                        try:
                                            # i_val = systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][rv['minmax_basin_map'][cb2k]]][var]
                                            i_val = systems[initial_sys]['regions'][sv['region_map'][rk]][sv['atom_map'][cb2k]][var]
                                            f_val = cbv[var]
                                            diff = f_val - i_val
                                            percent_diff = (diff / i_val * 100.) if i_val != 0. else 0.
                                            f.write(f"{i_val:.8f},{f_val:.8f},{diff:.8f},{percent_diff:.8f},")
                                            system_diff_vals[sk]['regions'][rk][cb2k][diff_func(var)] = diff
                                        except Exception as e:
                                            print(f"Exception fetching value for bundle constituent min/max basin for {var} for {ak} in {rk} in {sk} defined by {def_var}: {str(e)}")
                                            exc_type, value, exc_traceback = exc_info()
                                            print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
                                    f.write('\n')
                        
                        if num_regions == 0:
                            return
                                        
                        # any lone regions
                        # Because the max and min condensed basins both constitute a full partitioning of the system, can't just
                        # throw all the unused "lone" max/min basins in. Need to pick the type that completes whatever partitioning is being used.
                        # For bond bundle partitioning, it's _usually_ all max basins, so we can include all the non-bonded max basins to finish it off.
                        # Rather than assume, however, we'll check that the regions are all max or min condensed basins
                        ismin, ismax = [all([t == tt for r,t in sv['region_types'].items() if "Bond" in r]) for tt in [SGB_CONDENSED_BASIN_TYPE.MIN, SGB_CONDENSED_BASIN_TYPE.MAX]]
                        lone_regions = {}
                        if ismin:
                            for ak,av in sv['atoms'].items():
                                for cb2k in av['lone_minmax_basins']:
                                    if def_var in cb2k and "Min " in cb2k:
                                        lone_regions[cb2k] = sv['minmax_basins'][cb2k]
                        elif ismax:
                            for ak,av in sv['atoms'].items():
                                for cb2k in av['lone_minmax_basins']:
                                    if def_var in cb2k and "Max " in cb2k:
                                        lone_regions[cb2k] = sv['minmax_basins'][cb2k]
                        else:
                            f.write(f"\n{r_type.upper()}BUNDLE PARTITIONING INCLUDES CONTRIBUTIONS FROM BOTH MINIMUM AND MAXIMUM BASINS. THIS PREVENTS THE AUTOMATIC COMPLETION OF DECOMPOSITION WITH LONE REGIONS\n")
                        
                        if len(lone_regions) > 0:
                            unmapped = []
                            for lrk in sorted(list(lone_regions.keys())):
                                lrv = lone_regions[lrk]
                                if def_var in lrk and lrk in sv['minmax_basin_map']:
                                    f.write(f"Lone: {lrk.replace(' from ' + def_var,'')} → {sv['minmax_basin_map'][lrk].replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']},")
                                    for var in var_list:
                                        i_tot = systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][lrk]][var]
                                        var_itotals[var] += i_tot
                                        f_tot = lrv[var]
                                        var_ftotals[var] += f_tot
                                        diff = f_tot - i_tot
                                        percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                                        f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                                        system_diff_vals[sk]['minmax_basins'][lrk][diff_func(var)] = diff
                                    f.write('\n')
                                elif def_var in lrk:
                                    unmapped.append(lrk)
                            
                            if len(unmapped) > 0:
                                for lrk in unmapped:
                                    f.write(f"UNMAPPED lone: {lrk.replace(' from ' + def_var,'')},")
                                    for var in var_list:
                                        f_tot = lone_regions[lrk][var]
                                        var_ftotals[var] += f_tot
                                        f.write(f",{f_tot:.8f},,,")
                                        system_diff_vals[sk]['minmax_basins'][lrk][diff_func(var)] = 0.
                                    f.write('\n')
                            
                            if len(sv['initial_sys_unmapped_minmax_basins']) > 0:
                                for lrk in sorted(list(sv['initial_sys_unmapped_minmax_basins'].keys())):
                                    lrv = sv['initial_sys_unmapped_minmax_basins'][lrk]
                                    if def_var in lrk and (ismax and "Max " in lrk) or (ismin and "Min " in lrk):
                                        f.write(f"UNMAPPED lone: {lrk.replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']},")
                                        for var in var_list:
                                            i_tot = lrv[var]
                                            var_itotals[var] += i_tot
                                            f.write(f"{i_tot:.8f},,,,")
                                            system_diff_vals[initial_sys]['minmax_basins'][lrk][diff_func(var)] = 0.
                                        f.write('\n')
                                    
                        
                                
                                
                        # the total line
                        f.write("Total,")
                        for var in var_list:
                            i_tot = var_itotals[var]
                            f_tot = var_ftotals[var]
                            diff = f_tot - i_tot
                            percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                            f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                        f.write('\n')
                
                                # special gradient bundle decompositions
                def output_topological_cage_decomposition(r_type = "Cage"):
                    
                    # write out a row for each bond bundle in the final system, with it's component bond wedges
                    var_itotals = {k:0. for k in var_list}
                    var_ftotals = {k:0. for k in var_list}
                    num_regions = 0
                    
                    region_names = sorted([[rk, f"{' — '.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][sv['atom_map'][k]] for k in rv.keys() if '_map' not in k]) if '_map' not in k])}" + f" topological {rk}"] for rk,rv in sv['regions'].items()], key = lambda x: x[1])
                    for rk,r_name in region_names:
                        rv = sv['regions'][rk]
                        if r_type in rk:
                            if num_regions == 0:
                                f.write(f"Topological cage decomposition\n")
                            num_regions += 1
                            # region info
#                             r_name = f"{' — '.join([k for k in sorted(list(rv.keys())) if '_map' not in k])}" + f" {r_type.lower()}bundle"
                            f.write(f"{r_name},") 
                            
                            # region var totals
                            var_ivals = {k:0. for k in var_list}
                            var_fvals = {k:0. for k in var_list}
                            for var in var_list:
                                for ak, av in rv.items():
                                    if not "_map" in ak:
                                        try:
                                            var_ivals[var] += systems[initial_sys]['regions'][sv['region_map'][rk]][sv['atom_map'][ak]][var]
                                            var_fvals[var] += av[var]
                                            var_itotals[var] += systems[initial_sys]['regions'][sv['region_map'][rk]][sv['atom_map'][ak]][var]
                                            var_ftotals[var] += av[var]
                                        except Exception as e:
                                            print(f"Exception fetching value for total {r_type} value for {var} for {ak} in {rk} in {sk} defined by {def_var}: {str(e)}")
                                            exc_type, value, exc_traceback = exc_info()
                                            print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
                            for var in var_list:
                                diff = var_fvals[var] - var_ivals[var]
                                percent_diff = (diff / var_ivals[var] * 100.) if var_ivals[var] != 0. else 0.
                                f.write(f"{var_ivals[var]:.8f},{var_fvals[var]:.8f},{diff:.8f},{percent_diff:.8f},")
                            f.write('\n')
                            
                            # constituent condensed basin values
                            icbs = sorted([(systems[initial_sys]['atom_nicknames'][sv['atom_map'][k]],k) for k in rv.keys() if "_map" not in k], key=lambda x:x[0])
                            for icb,cb2k in icbs:
                                cbv = rv[cb2k]
                                if not "_map" in cb2k:
                                    f.write(f"      ↳ Wedge: {icb} {rk} mapped to {cb2k} {sv['region_map'][rk]} ({sv['region_map_differences'][rk]:.8f}),")
                                    for var in var_list:
                                        try:
                                            i_val = systems[initial_sys]['regions'][sv['region_map'][rk]][sv['atom_map'][cb2k]][var]
                                            f_val = cbv[var]
                                            diff = f_val - i_val
                                            percent_diff = (diff / i_val * 100.) if i_val != 0. else 0.
                                            f.write(f"{i_val:.8f},{f_val:.8f},{diff:.8f},{percent_diff:.8f},")
                                            system_diff_vals[sk]['regions'][rk][cb2k][diff_func(var)] = diff
                                        except Exception as e:
                                            print(f"Exception fetching value for topological cage for {var} for {ak} in {rk} in {sk}: {str(e)}")
                                            exc_type, value, exc_traceback = exc_info()
                                            print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
                                    f.write('\n')
                    
                    if num_regions == 0:
                        return
                            
                            
                    # the total line
                    f.write("Total,")
                    for var in var_list:
                        i_tot = var_itotals[var]
                        f_tot = var_ftotals[var]
                        diff = f_tot - i_tot
                        percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                        f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                    f.write('\n')
                
                output_decomposition("Bond ")
                output_decomposition("Ring ")
                output_topological_cage_decomposition()
                
                # max and min basin decomposition
                def output_minmax_basin_decompositions():
                    for def_var in defining_var_list:
                        for m in ['Max ',"Min "]:
                            f.write(f"\n{m}basin decomposition according to {def_var}\n")
                            unmapped = []
                            var_itotals = {k:0. for k in var_list}
                            var_ftotals = {k:0. for k in var_list}
                            
                            lines = []
                            for cb1k in sorted(list(sv['minmax_basins'].keys())):
                                cb1v = sv['minmax_basins'][cb1k]
                                if def_var not in cb1k or m not in cb1k:
                                    continue
                                if cb1k in sv['minmax_basin_map']:
                                    line = f"{sv['minmax_basin_map'][cb1k].replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']} ← {cb1k.replace(' from ' + def_var,'')} in {sv['nickname']} ({sv['minmax_basin_map_corrcoefs'][cb1k]:.8f}),"
                                    for var in var_list:
                                        i_tot = systems[initial_sys]['minmax_basins'][sv['minmax_basin_map'][cb1k]][var]
                                        var_itotals[var] += i_tot
                                        f_tot = cb1v[var]
                                        var_ftotals[var] += f_tot
                                        diff = f_tot - i_tot
                                        percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                                        line += f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},"
                                        if diff_func(var) not in system_diff_vals[sk]['minmax_basins'][cb1k]:
                                            system_diff_vals[sk]['minmax_basins'][cb1k][diff_func(var)] = diff
                                    lines.append(line)
                                else:
                                    unmapped.append(cb1k)
                            
                            for aline in sorted(lines, key = lambda x : x.split(",")[0]):
                                f.write(aline + "\n")
                                    
                            if len(sv['initial_sys_unmapped_minmax_basins']) > 0:
                                for cb1k in sorted(list(sv['initial_sys_unmapped_minmax_basins'].keys())):
                                    cb1v = sv['initial_sys_unmapped_minmax_basins'][cb1k]
                                    if def_var in cb1k and (m in cb1k):
                                        f.write(f"UNMAPPED: {cb1k.replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']},")
                                        for var in var_list:
                                            i_tot = cb1v[var]
                                            var_itotals[var] += i_tot
                                            f.write(f"{i_tot:.8f},,,,")
                                            if diff_func(var) not in system_diff_vals[initial_sys]['minmax_basins'][cb1k]:
                                                system_diff_vals[initial_sys]['minmax_basins'][cb1k][diff_func(var)] = 0.
                                        f.write('\n')
                            
                            if len(unmapped) > 0:
                                for cb1k in unmapped:
                                    if m not in cb1k:
                                        continue
                                    f.write(f"UNMAPPED: {cb1k.replace(' from ' + def_var,'')},")
                                    for var in var_list:
                                        f_tot = sv['minmax_basins'][cb1k][var]
                                        var_ftotals[var] += f_tot
                                        f.write(f",{f_tot:.8f},,,")
                                        if diff_func(var) not in system_diff_vals[sk]['minmax_basins'][cb1k]:
                                            system_diff_vals[sk]['minmax_basins'][cb1k][diff_func(var)] = 0.
                                    f.write('\n')
                                    
                            # the total line
                            f.write("Total,")
                            for var in var_list:
                                i_tot = var_itotals[var]
                                f_tot = var_ftotals[var]
                                diff = f_tot - i_tot
                                percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                                f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                                # max and min basin decomposition
                                
                def output_topological_cage_decompositions_junk():
                    for def_var in defining_var_list:
                        for m in ['Max ',"Min "]:
                            f.write(f"\nTopological cage decomposition according to {def_var}\n")
                            unmapped = []
                            var_itotals = {k:0. for k in var_list}
                            var_ftotals = {k:0. for k in var_list}
                            
                            lines = []
                            for cb1k in sorted(list(sv['regions'].keys())):
                                cb1v = sv['regions'][cb1k]
                                if "Cage " not in cb1k:
                                    continue
                                if cb1k in sv['region_map']:
                                    line = f"{sv['region_map'][cb1k].replace(' from ' + def_var,'')} in {systems[initial_sys]['nickname']} ← {cb1k.replace(' from ' + def_var,'')} in {sv['nickname']} ({sv['minmax_basin_map_corrcoefs'][cb1k]:.8f}),"
                                    for var in var_list:
                                        i_tot = systems[initial_sys]['regions'][sv['region_map'][cb1k]][var]
                                        var_itotals[var] += i_tot
                                        f_tot = cb1v[var]
                                        var_ftotals[var] += f_tot
                                        diff = f_tot - i_tot
                                        percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                                        line += f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},"
                                        if diff_func(var) not in system_diff_vals[sk]['regions'][cb1k]:
                                            system_diff_vals[sk]['regions'][cb1k][diff_func(var)] = diff
                                    lines.append(line)
                                else:
                                    unmapped.append(cb1k)
                            
                            for aline in sorted(lines, key = lambda x : x.split(",")[0]):
                                f.write(aline + "\n")
                                
                            unmapped_initial_sys_cages = {k:v for k,v in sv['initial_sys_unmapped_regions'] if "Cage " in k}
                                    
                            if len(unmapped_initial_sys_cages) > 0:
                                for cb1k in sorted(list(unmapped_initial_sys_cages.keys())):
                                    cb1v = unmapped_initial_sys_cages[cb1k]
                                    f.write(f"UNMAPPED: {cb1k} in {systems[initial_sys]['nickname']},")
                                    for var in var_list:
                                        i_tot = cb1v[var]
                                        var_itotals[var] += i_tot
                                        f.write(f"{i_tot:.8f},,,,")
                                        if diff_func(var) not in system_diff_vals[initial_sys]['regions'][cb1k]:
                                            system_diff_vals[initial_sys]['regions'][cb1k][diff_func(var)] = 0.
                                    f.write('\n')
                            
                            if len(unmapped) > 0:
                                for cb1k in unmapped:
                                    if m not in cb1k:
                                        continue
                                    f.write(f"UNMAPPED: {cb1k},")
                                    for var in var_list:
                                        f_tot = sv['minmax_basins'][cb1k][var]
                                        var_ftotals[var] += f_tot
                                        f.write(f",{f_tot:.8f},,,")
                                        if diff_func(var) not in system_diff_vals[sk]['regions'][cb1k]:
                                            system_diff_vals[sk]['regions'][cb1k][diff_func(var)] = 0.
                                    f.write('\n')
                                    
                            # the total line
                            f.write("Total,")
                            for var in var_list:
                                i_tot = var_itotals[var]
                                f_tot = var_ftotals[var]
                                diff = f_tot - i_tot
                                percent_diff = (diff / i_tot * 100.) if i_tot != 0. else 0.
                                f.write(f"{i_tot:.8f},{f_tot:.8f},{diff:.8f},{percent_diff:.8f},")
                                        
                output_minmax_basin_decompositions()
    
    
    old_var_list = var_list
    dir_name = os.path.join(os.getcwd(),dir_name,"plots")
    os.mkdir(dir_name)
    ##################################
    # STATS AND PLOTTING
    ################################## 
    
    diff_var_list = []
    for sk,sv in system_diff_vals.items():
        if sk == initial_sys:
            continue
        for v in list(list(sv['atoms'].values())[0]):
            if "∆" in v:
                diff_var_list.append(v)
        break

    diff_var_list = {k for k in diff_var_list}
    var_to_diff_var = {k1:k2 for k1 in var_list for k2 in diff_var_list if k1 in k2}
    diff_var_to_var = {k2:k1 for k1 in var_list for k2 in diff_var_list if k1 in k2}
    for v in diff_var_list:
        var_nickname[v] = var_nickname[diff_var_to_var[v]]
        var_nickname_full[v] = var_nickname_full[diff_var_to_var[v]]
    
    system_names_full = list({sv['nickname'] for ak,av in sv['atoms'].items() for sk,sv in systems.items()})# if sk != initial_sys}.keys())
    markers = ["s", "o" , "v" , "^" , "p", "D", ",", "<", ">", ".", "x"]
    
    plot_list = '''linear regression of system property vs atomic basin properties ---  all
linear regression of system property vs atomic basin properties ---  per element
linear regression of system property vs atomic basin properties ---  per atom
linear regression of system property vs bond bundle properties ---  per bond bundle
linear regression of system property vs bond bundle properties ---  per bond bundle type
linear regression of system property vs bond wedge properties
linear regression of system property vs condensed minimum basin properties
linear regression of system property vs topological cage properties
linear regression of system property vs atomic basin property difference in "final" vs "initial" systems
linear regression of system property vs bond bundle property difference in "final" vs "initial" systems
linear regression of system property vs bond wedge property difference in "final" vs "initial" systems
linear regression of system property vs condensed minimum basin property difference in "final" vs "initial" systems
linear regression of system property vs bond bundle bond wedge abs property differences
linear regression of system property vs atomic basin bond wedge property standard deviation
linear regression of system property vs atomic basin standard deviation of bond wedge property differences between "final" and "initial" systems
linear regression of system property vs atomic basin condensed minimum basin property standard deviation
linear regression of system property vs atomic basin standard deviation of condensed minimum basin property differences between "final" and "initial" systems
correlation diagrams of atomic basins ---  one diagram per property
correlation diagrams of bond bundles ---  one diagram per property
correlation diagrams of bond wedges ---  one diagram per property
correlation diagrams of condensed minimum basins ---  one diagram per property
correlation diagrams of topological cages ---  one diagram per property
correlation diagrams of atomic basin property differences between "final" and "initial" systems ---  one diagram per property
correlation diagrams of bond bundle property differences between "final" and "initial" systems ---  one diagram per property
correlation diagrams of bond wedge property differences between "final" and "initial" systems ---  one diagram per property
correlation diagrams of condensed minimum basin property differences between "final" and "initial" systems ---  one diagram per property
correlation diagrams of atomic basin bond wedge property standard deviations ---  one diagram per property
correlation diagrams of atomic basin condensed minimum basin property standard deviations ---  one diagram per property
correlation diagrams of atomic basin standard deviation of bond wedge property differences between "final" and "initial" systems ---  one diagram per property
correlation diagrams of atomic basin standard deviation of condensed minimum basin property differences between "final" and "initial" systems ---  one diagram per property
correlation diagrams of regional properties: one diagram per atomic basin
correlation diagrams of regional properties: one diagram per bond bundle
correlation diagrams of regional properties: one diagram per bond wedge
correlation diagrams of regional properties: one diagram per condensed minimum basin
correlation diagrams of regional properties: one diagram per topological cage
bar charts of atomic basin property correlations to system property ---  one chart per property
bar charts of bond bundle property correlations to system property ---  one chart per property
bar charts of bond wedge property correlations to system property ---  one chart per property
bar charts of condensed minimum basin property correlations to system property ---  one chart per property
bar charts of topological cage property correlations to system property ---  one chart per property
bar chart of atomic basin property correlations to system property ---  all properties on one chart
bar chart of bond bundle property correlations to system property ---  all properties on one chart
bar chart of bond wedge property correlations to system property ---  all properties on one chart
bar chart of condensed minimum basin property correlations to system property ---  all properties on one chart
bar chart of topological cage property correlations to system property ---  all properties on one chart
'''
    
    list_of_files = [f for f in glob.glob(os.path.join(os.getcwd(),'*')) if os.path.isfile(f) and "_PlotSelection.txt" in f]
    if len(list_of_files) > 0 and use_old_data:
        # Sort list of files based on last modification time in ascending order
        list_of_files = sorted( list_of_files,
                                key = os.path.getmtime,
                                reverse = True)
        with open(list_of_files[0]) as f:
            str1 = f.read()
    else:
        str1 = plot_list
    
    plot_set = set(plot_list.split('\n'))
    
    file_name1 = f"{date_time_stamp}_PlotSelection.txt"
    with open(file_name1, 'w') as f:
        f.write(str1)
    
    input(CLEAR + """Step 3 of 3)  TIME TO SELECT WHICH TYPES OF PLOTS TO GENERATE
    
    As before, press enter to open a text file, this time with a big list of all the types of plots that can be produced.
    
    Remove any plots you don't want to generate from the list by deleting the whole line, save the file, then return to this window and press enter to continue.""")
    
    os.system(f"open {file_name1}")
    
    input("\n\nAfter you've applied and saved any changes, return to this window and press enter to continue...")
    
    num_errors = 1
    while num_errors > 0:        
        # Read the file back in to get the associations
        with open(file_name1, 'r') as f:
            str2 = f.read().strip()
        
        num_errors = 0
        new_plot_list = []
        for s in str2.split('\n'):
            if s.strip() == '':
                continue
            if s not in plot_set:
                num_errors += 1
                print(f"{s} wasn't found in the list of plot types!")
                continue
            new_plot_list.append(s.strip())
            
        if num_errors:
            input(f"\n\nProblem(s) found with {num_errors} variable(s). Please fix, resave the file, and press enter to check again.\n\nHere's the original list for reference:\n\n{str1}")
    
    for plot_type in new_plot_list:
        #######  Simple regressions of different regional set properties vs system energy
            
        if plot_type == "linear regression of system property vs atomic basin properties ---  all":  # BAD all atomic basins      
            # Need to organize data on a per-sample basis: X and Y matrices.
            # X: (N x M) matrix: N=num_samples, M=num_independent_variables
            # Y: (N x F) matrix: N=num_samples, F=num_features (eg if fitting against just energy, Nx1, but if fitting against vibrational frequencies, F > 1.
            #
            
            # First atomic basins
            
            # first all atoms
            
            var_list = old_var_list
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for ak,av in sv['atoms'].items() for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            x = []
            y = []
            names = []
            atom_names = []
            colors = []
            sizes = []
            shapes = []
            for sk,sv in systems.items():
                for ak,av in sv['atoms'].items():
                    y.append(sv['energy'])
                    x.append([av[v] for v in var_list])
                    names.append(f"{sv['nickname']} : {systems[initial_sys]['atom_nicknames'][sv['atom_map'][ak]]}")
                    atom_names.append(''.join([i for i in ak if not i.isdigit()]))
                    colors.append(tuple(v/255. for v in hex_to_rgb(element(atom_names[-1]).cpk_color)) + tuple([0.7]))
                    sizes.append((element(atom_names[-1]).atomic_number-1)*10 + 30)
                    shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
            if len(y) > 2:
                Y = np.array(y)
                X = np.array(x)
                legend_atom_names = sorted(list({a for a in atom_names}), key=lambda x:element(x).atomic_number)
                legend_atom_colors = [tuple(v/255. for v in hex_to_rgb(element(a).cpk_color)) for a in legend_atom_names]
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    lc=list(zip(legend_atom_colors,legend_atom_names))
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr="atomic basins", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendcolors=lc, legendshapes=sc, filedir=sub_dir_name)
            
        elif plot_type == "linear regression of system property vs atomic basin properties ---  per element":  # BAD atomic basins per element basis
            var_list = old_var_list
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            legend_atom_names = sorted(list({''.join([i for i in ak[:ak.find(":")] if not i.isdigit()]) for ak in [ak for ak in sv['atoms'] for sv in systems.values()]}), key=lambda x:element(x).atomic_number)
            system_names = list({sv['nickname'] for ak,av in sv['atoms'].items() for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            
            for atom_name in legend_atom_names:
                x = []
                y = []
                names = []
                atom_names = []
                colors = []
                sizes = []
                shapes = []
                for sk,sv in systems.items():
                    for ak,av in sv['atoms'].items():
                        cur_atom = ''.join([i for i in ak[:ak.find(":")] if not i.isdigit()])
                        if cur_atom == atom_name:
                            y.append(sv['energy'])
                            x.append([av[v] for v in var_list])
                            names.append(f"{sv['nickname']} : {systems[initial_sys]['atom_nicknames'][sv['atom_map'][ak]]}")
                            atom_names.append(cur_atom)
                            colors.append(tuple(v/255. for v in hex_to_rgb(element(atom_names[-1]).cpk_color)) + tuple([0.7]))
                            sizes.append((element(atom_names[-1]).atomic_number-1)*10 + 30)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                legend_atom_names = sorted(list({a for a in atom_names}), key=lambda x:element(x).atomic_number)
                legend_atom_colors = [tuple(v/255. for v in hex_to_rgb(element(a).cpk_color)) for a in legend_atom_names]
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    lc=list(zip(legend_atom_colors,legend_atom_names))
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{atom_name} atomic basins", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        elif plot_type == "linear regression of system property vs atomic basin properties ---  per atom":  # MODERATE atomic basins    per atom basis
            var_list = old_var_list
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for ak,av in sv['atoms'].items() for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            
            
            for ak1 in systems[initial_sys]['atoms'].keys():
                
                x = []
                y = []
                atom_names = []
                colors = []
                sizes = []
                shapes = []
                cur_atom = systems[initial_sys]['atom_nicknames'][ak1]
                atom_name = ''.join([i for i in ak1[:ak1.find(":")] if not i.isdigit()])
                for sk,sv in systems.items():
                    for ak,av in sv['atoms'].items():
                        if (sk == initial_sys and ak == ak1) or (sk != initial_sys and sv['atom_map'][ak] == ak1):
                            y.append(sv['energy'])
                            x.append([av[v] for v in var_list])
                            atom_names.append(sv['nickname'])
                            colors.append(tuple(v/255. for v in hex_to_rgb(element(atom_name).cpk_color)) + tuple([0.7]))
                            sizes.append((element(atom_name).atomic_number-1)*10 + 30)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{cur_atom} atomic basins", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
    
        elif plot_type == "linear regression of system property vs bond bundle properties ---  per bond bundle":  # GOOD per individual bond bundle
            var_list = old_var_list
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            
            
            for rk1,rv1 in systems[initial_sys]['regions'].items():
                if "Bond " not in rk1 or len(rv1) < 3:
                    continue
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                region_name = f"{' — '.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in rv1.keys() if '_map' not in k]) if '_map' not in k])}"
                for sk,sv in systems.items():
                    for rk,rv in sv['regions'].items():
                        if (sk == initial_sys and rk == rk1) or (sk != initial_sys and rk in sv['region_map'] and sv['region_map'][rk] == rk1):
                            y.append(sv['energy'])
                            x.append([sum([av[v] for ak,av in rv.items() if "_map" not in ak]) for v in var_list])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} bond bundle", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
                    
        elif plot_type == "linear regression of system property vs bond bundle properties ---  per bond bundle type":  # BAD per bond bundle type
            var_list = old_var_list
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            
            
            # get list of bond types
            def simple_bond_name(bond_str):
                return ''.join([i for i in bond_str if not i.isdigit() and i != ' '])
            
            
            region_names1 = {f"{simple_bond_name('—'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in rv.keys() if '_map' not in k]) if '_map' not in k]))}" for rk,rv in systems[initial_sys]['regions'].items() if "_map" not in rk and "Bond " in rk and len(rv) == 3}
            
            for rk1 in region_names1:
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                for sk,sv in systems.items():
                    for rk,rv in sv['regions'].items():
                        region_name = simple_bond_name(f"{'—'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][sv['atom_map'][k]] for k in rv.keys() if '_map' not in k]) if '_map' not in k])}")
                        if region_name == rk1:
                            y.append(sv['energy'])
                            x.append([sum([av[v] for ak,av in rv.items() if "_map" not in ak]) for v in var_list])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{rk1} bond bundles", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
    
        elif plot_type == "linear regression of system property vs bond wedge properties":  # GOOD per individual bond wedge
            var_list = old_var_list
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            
            
            for rk1,rv1 in systems[initial_sys]['minmax_basins'].items():
                if "Max " not in rk1:
                    continue
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                
                a_name = systems[initial_sys]['atom_nicknames'][rk1[:rk1.find(':')]]
                if rk1 in max_basin_to_bond_bundle_map:
                    region_name = f"{a_name}: Max ({max_basin_to_bond_bundle_map[rk1]}) {rk1[rk1.find('('):]}"
                else:
                    region_name = f"{a_name}: Max {rk1[rk1.find('('):]}"
                for sk,sv in systems.items():
                    for rk,rv in sv['minmax_basins'].items():
                        if (sk == initial_sys and rk == rk1) or (sk != initial_sys and rk in sv['minmax_basin_map'] and sv['minmax_basin_map'][rk] == rk1):
                            y.append(sv['energy'])
                            x.append([rv[v] for v in var_list])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} bond wedge", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
    
        elif plot_type == "linear regression of system property vs condensed minimum basin properties":  # GOOD per individual condensed minimum basin
            var_list = old_var_list
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
            
            for rk1,rv1 in systems[initial_sys]['minmax_basins'].items():
                if "Min " not in rk1:
                    continue
                x = []
                y = []
                region_names = []
                region_markers = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                a_name = systems[initial_sys]['atom_nicknames'][rk1[:rk1.find(':')]]
                region_name = f"{a_name}: Min {rk1[rk1.find('('):]}"
                for sk,sv in systems.items():
                    for rk,rv in sv['minmax_basins'].items():
                        if (sk == initial_sys and rk == rk1) or (sk != initial_sys and rk in sv['minmax_basin_map'] and sv['minmax_basin_map'][rk] == rk1):
                            y.append(sv['energy'])
                            x.append([rv[v] for v in var_list])
                            if sk == initial_sys:
                                region_names.append(f"{sv['nickname']}")
                            else:
                                region_names.append(f"{sv['nickname']} ({sv['minmax_basin_map_corrcoefs'][rk]:.2f})")
                            region_markers.append(markers[system_names.index(sv['nickname']) % len(markers)])
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(region_markers,region_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} condensed minimum basin", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        elif plot_type == "linear regression of system property vs topological cage properties":  # GOOD per topological cage
            var_list = old_var_list
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            
            
            for rk1,rv1 in systems[initial_sys]['regions'].items():
                if "Cage " not in rk1:
                    continue
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                region_name = f"{' — '.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in rv1.keys() if '_map' not in k]) if '_map' not in k])}"
                for sk,sv in systems.items():
                    for rk,rv in sv['regions'].items():
                        if (sk == initial_sys and rk == rk1) or (sk != initial_sys and rk in sv['region_map'] and sv['region_map'][rk] == rk1):
                            y.append(sv['energy'])
                            x.append([sum([av[v] for ak,av in rv.items() if "_map" not in ak]) for v in var_list])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} Topological {rk1}", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
                    
        #######  Regressions of regional property differences between perturbed and original system
    
        elif plot_type == 'linear regression of system property vs atomic basin property difference in "final" vs "initial" systems':  # GOOD atomic basin differences to initial system
            # per atom basis
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items() if sk != initial_sys})
            
            systems = system_diff_vals
            var_list = diff_var_list
            
            for k,v in systems.items():
                var_list = {k1 for k1 in var_list if k == initial_sys or (k1 in v['atoms'][list(v['atoms'].keys())[0]] and ":" in k1)}
            
            legend_atom_names = sorted(list(systems[initial_sys]['atoms'].keys()), key=lambda x:element(''.join([i for i in x if not i.isdigit()])).atomic_number)
            system_names = list({sv['nickname'] for ak,av in sv['atoms'].items() for sk,sv in systems.items() if sk != initial_sys})# if sk != initial_sys}.keys())
            
            for atom_name in legend_atom_names:
                x = []
                y = []
                names = []
                atom_names = []
                colors = []
                sizes = []
                shapes = []
                atom_type = ''.join([i for i in atom_name if not i.isdigit()])
                for sk,sv in systems.items():
                    if sk == initial_sys:
                        continue
                    for ak,av in sv['atoms'].items():
                        cur_atom = ''.join([i for i in ak[:ak.find(":")]])
                        if sv['atom_map'][ak] == atom_name and all([v in av for v in var_list]):
                            y.append(sv['energy'])
                            x.append([av[v] for v in var_list])
                            names.append(f"{sv['nickname']} : {systems[initial_sys]['atom_nicknames'][sv['atom_map'][ak]]}")
                            atom_names.append(cur_atom)
                            
                            colors.append(tuple(v/255. for v in hex_to_rgb(element(atom_type).cpk_color)) + tuple([0.7]))
                            sizes.append((element(atom_type).atomic_number-1)*10 + 30)
                            shapes.append(markers[system_names_full.index(sv['nickname']) % len(markers)])
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                legend_atom_names = sorted(list({a for a in atom_names}), key=lambda x:element(''.join([i for i in x if not i.isdigit()])).atomic_number)
                legend_atom_colors = [tuple(v/255. for v in hex_to_rgb(element(atom_type).cpk_color)) for a in legend_atom_names]
                system_markers = [markers[system_names_full.index(i) % len(markers)] for i in system_names]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    lc=list(zip(legend_atom_colors,legend_atom_names))
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{systems[initial_sys]['atom_nicknames'][atom_name]} atomic basin property difference", xlabel=f"∆ {var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
            
        elif plot_type == 'linear regression of system property vs bond bundle property difference in "final" vs "initial" systems':  # VERY GOOD bond bundle differences to initial system
            # per atom basis
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items() if sk != initial_sys})
            
            systems = system_diff_vals
            
            
            var_list = diff_var_list
            
            for k,v in systems.items():
                var_list = {k1 for k1 in var_list if k == initial_sys or (k1 in v['atoms'][list(v['atoms'].keys())[0]] and ":" in k1)}
            
            system_names = list({sv['nickname'] for ak,av in sv['atoms'].items() for sk,sv in systems.items() if sk != initial_sys})# if sk != initial_sys}.keys())
            
            for rk1,rv1 in systems[initial_sys]['regions'].items():
                if "Bond " not in rk1 or len(rv1) < 3:
                    continue
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                region_name = f"{' — '.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in rv1.keys() if '_map' not in k]) if '_map' not in k])}"
                for sk,sv in systems.items():
                    for rk,rv in sv['regions'].items():
                        if (sk != initial_sys and rk in sv['region_map'] and sv['region_map'][rk] == rk1):
                            y.append(sv['energy'])
                            x.append([sum([av[v] for ak,av in rv.items() if "_map" not in ak]) for v in var_list])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names_full.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[system_names_full.index(i) % len(markers)] for i in system_names]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} bond bundle property difference", xlabel=f"∆ {var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        elif plot_type == 'linear regression of system property vs bond wedge property difference in "final" vs "initial" systems':  # VERY GOOD individual bond wedge differences
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items() if sk != initial_sys})
            
            var_list = diff_var_list
            
            for rk1,rv1 in systems[initial_sys]['minmax_basins'].items():
                if "Max " not in rk1:
                    continue
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                a_name = systems[initial_sys]['atom_nicknames'][rk1[:rk1.find(':')]]
                if rk1 in max_basin_to_bond_bundle_map:
                    region_name = f"{a_name}: Max ({max_basin_to_bond_bundle_map[rk1]}) {rk1[rk1.find('('):]}"
                else:
                    region_name = f"{a_name}: Max {rk1[rk1.find('('):]}"
                for sk,sv in systems.items():
                    for rk,rv in sv['minmax_basins'].items():
                        if (sk != initial_sys and rk in sv['minmax_basin_map'] and sv['minmax_basin_map'][rk] == rk1):
                            y.append(sv['energy'])
                            x.append([rv[v] for v in var_list])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names_full.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[system_names_full.index(i) % len(markers)] for i in system_names]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name}\nbond wedge property difference", xlabel=f"∆ {var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        elif plot_type == 'linear regression of system property vs condensed minimum basin property difference in "final" vs "initial" systems':  # GOOD individual minimum basin differences
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items() if sk != initial_sys})
            system_markers = [markers[system_names_full.index(i) % len(markers)] for i in system_names]
            
            var_list = diff_var_list
            
            for rk1,rv1 in systems[initial_sys]['minmax_basins'].items():
                if "Min " not in rk1:
                    continue
                x = []
                y = []
                region_names = []
                region_markers = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                a_name = systems[initial_sys]['atom_nicknames'][rk1[:rk1.find(':')]]
                region_name = f"{a_name}: Min {rk1[rk1.find('('):]}"
                for sk,sv in systems.items():
                    for rk,rv in sv['minmax_basins'].items():
                        if (sk != initial_sys and rk in sv['minmax_basin_map'] and sv['minmax_basin_map'][rk] == rk1):
                            y.append(sv['energy'])
                            x.append([rv[v] for v in var_list])
                            region_names.append(f"{sv['nickname']} ({sv['minmax_basin_map_corrcoefs'][rk]:.2f})")
                            region_markers.append(markers[system_names.index(sv['nickname']) % len(markers)])
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names_full.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(region_markers,region_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name}\ncondensed minimum basin property difference", xlabel=f"∆ {var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        
        #######  Try out some crazy stuff
    
        elif plot_type == 'linear regression of system property vs bond bundle bond wedge abs property differences':  # POOR bond bundle bond wedge abs difference
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            
            var_list = old_var_list
            
            
            for rk1,rv1 in systems[initial_sys]['regions'].items():
                if "Bond " not in rk1 or len(rv1) < 3:
                    continue
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                region_name = f"{' — '.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in rv1.keys() if '_map' not in k]) if '_map' not in k])}"
                for sk,sv in systems.items():
                    for rk,rv in sv['regions'].items():
                        if "Bond " not in rk or len(rv) < 3:
                            continue
                        if (sk == initial_sys and rk == rk1) or (sk != initial_sys and rk in sv['region_map'] and sv['region_map'][rk] == rk1):
                            vals = {v:[av[v] for ak,av in rv.items() if "_map" not in ak] for v in var_list}
                            do_run = True
                            for v in vals.values():
                                do_run &= len(v) == 2
                            if not do_run:
                                continue
                            y.append(sv['energy'])
                            x.append([abs(vals[v][0] - vals[v][1]) for v in var_list])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                            break
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} bond bundle abs bond wedge difference", xlabel="∆ " + selected_vars[i], ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
    
        elif plot_type == 'linear regression of system property vs atomic basin bond wedge property standard deviation':  # VERY GOOD atomic basin bond wedge property standard deviation
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            var_list = old_var_list
            
            for rk1,rv1 in systems[initial_sys]['atoms'].items():
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                region_name = systems[initial_sys]['atom_nicknames'][rk1]
                for sk,sv in systems.items():
                    for rk,rv in sv['atoms'].items():
                        if (sk == initial_sys and rk == rk1) or (sk != initial_sys and sv['atom_map'][rk] == rk1):
                            vals = [[av[v] for ak,av in sv['minmax_basins'].items() if rk in ak and "Max " in ak] for v in var_list]
                            if any([len(v)<2 for v in vals]):
                                continue
                            y.append(sv['energy'])
                            x.append([stdev(v) for v in vals])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} bond wedge standard deviation", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        elif plot_type == 'linear regression of system property vs atomic basin standard deviation of bond wedge property differences between "final" and "initial" systems':  # POOR atomic basin bond wedge property difference standard deviation
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items() if sk != initial_sys})
            system_markers = [markers[system_names_full.index(i) % len(markers)] for i in system_names]
            
            var_list = diff_var_list
            
            for rk1,rv1 in systems[initial_sys]['atoms'].items():
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                region_name = systems[initial_sys]['atom_nicknames'][rk1]
                for sk,sv in systems.items():
                    for rk,rv in sv['atoms'].items():
                        if (sk != initial_sys and sv['atom_map'][rk] == rk1):
                            vals = [[av[v] for ak,av in sv['minmax_basins'].items() if rk in ak and "Max " in ak] for v in var_list]
                            if any([len(v)<2 for v in vals]):
                                continue
                            y.append(sv['energy'])
                            x.append([stdev(v) for v in vals])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names_full.index(sv['nickname']) % len(markers)])
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} bond wedge difference standard deviation", xlabel=f"∆ {var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        elif plot_type == 'linear regression of system property vs atomic basin condensed minimum basin property standard deviation':  # GOOD atomic basin min basin property standard deviation
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items()})# if sk != initial_sys}.keys())
            
            
            var_list = old_var_list
            
            for rk1,rv1 in systems[initial_sys]['atoms'].items():
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                region_name = systems[initial_sys]['atom_nicknames'][rk1]
                for sk,sv in systems.items():
                    for rk,rv in sv['atoms'].items():
                        if (sk == initial_sys and rk == rk1) or (sk != initial_sys and sv['atom_map'][rk] == rk1):
                            vals = [[av[v] for ak,av in sv['minmax_basins'].items() if rk in ak and "Min " in ak] for v in var_list]
                            if any([len(v)<2 for v in vals]):
                                continue
                            y.append(sv['energy'])
                            x.append([stdev(v) for v in vals])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names.index(sv['nickname']) % len(markers)])
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} min basin standard deviation", xlabel=f"{var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        elif plot_type == 'linear regression of system property vs atomic basin standard deviation of condensed minimum basin property differences between "final" and "initial" systems':  # MODERATE atomic basin min basin property difference standard deviation
            
            sub_dir_name = os.path.join(os.getcwd(),dir_name,plot_type)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            system_names = list({sv['nickname'] for sk,sv in systems.items() if sk != initial_sys})
            system_markers = [markers[system_names_full.index(i) % len(markers)] for i in system_names]
            
            var_list = diff_var_list
            
            for rk1,rv1 in systems[initial_sys]['atoms'].items():
                x = []
                y = []
                region_names = []
                colors = []
                sizes = []
                shapes = []
                cur_region = rk1
                region_name = systems[initial_sys]['atom_nicknames'][rk1]
                for sk,sv in systems.items():
                    for rk,rv in sv['atoms'].items():
                        if (sk != initial_sys and sv['atom_map'][rk] == rk1):
                            vals = [[av[v] for ak,av in sv['minmax_basins'].items() if rk in ak and "Min " in ak] for v in var_list]
                            if any([len(v)<2 for v in vals]):
                                continue
                            y.append(sv['energy'])
                            x.append([stdev(v) for v in vals])
                            region_names.append(region_name)
                            colors.append((0,0,0,.7))
                            sizes.append(50)
                            shapes.append(markers[system_names_full.index(sv['nickname']) % len(markers)])
                if len(y) < 3:
                    continue
                Y = np.array(y)
                X = np.array(x)
                system_markers = [markers[i % len(markers)] for i in range(len(system_names))]
                # shapes = ['o' for i in shapes]
                # Y is of length ((num_systems x num_atoms_in_each) x 1)
                # X is ((num_systems x num_atoms_in_each) x num_vars)
                
                # Reject non-correlating variables
                selected_vars = list(var_list)
                # X, selected_features = BorutaReduceVars(X, Y)
                
                # Get names of variables that were selected
                # selected_vars = [var for i,var in enumerate(var_list) if selected_features.ranking_[i] == 1]
                
                # print(f"{len(var_list)} variables reduced down to {len(selected_vars)}\n\nOld list:")
                # [print(v) for v in var_list]
                # print("\n\nNew list:")
                # [print(v) for v in selected_vars]
                # regress each variable one at a time
                for i in range(len(selected_vars)):
                    Xi = X[:,i].reshape(-1, 1)
                    sc=list(zip(system_markers,system_names))                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    LinearRegression(Xi, Y, titlestr=f"{region_name} min basin difference standard deviation", xlabel=f"∆ {var_nickname_full[selected_vars[i]]} ({var_nickname[selected_vars[i]]})", ylabel=f"{energy_nickname_full} ({energy_nickname})", yunits = f" [{energy_units}]", colors=colors, shapes=shapes, sizes=sizes, legendshapes=sc, filedir=sub_dir_name)
        
        ####################################
        
        #######  Correlation matrices
        
        elif plot_type == 'correlation diagrams of atomic basins ---  one diagram per property':  # MODERATE all atomic basins      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1 in systems[initial_sys]['atoms'].keys():
                    a_name = systems[initial_sys]['atom_nicknames'][ak1]
                    xd[a_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                xd[a_name].append(av2[v])
                                break
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                                                
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Atomic basins\n{var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of bond bundles ---  one diagram per property':  # GOOD bond bundles      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1,av1 in systems[initial_sys]['regions'].items():
                    if "Bond " not in ak1 or len(av1) < 3:
                        continue
                    region_name = f"{'-'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])}"
                    xd[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[region_name].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                                                    
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Bond bundles\n{var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of bond wedges ---  one diagram per property':  # GOOD bond wedges      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                    if "Max " not in ak1:
                        continue
                    a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                    if ak1 in max_basin_to_bond_bundle_map:
                        region_name = f"{a_name} ({max_basin_to_bond_bundle_map[ak1]})"
                    else:
                        region_name = f"{a_name} {ak1[ak1.find('('):ak1.find(')')+1]}"
                    xd[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[region_name].append(av2[v])
                                break
                
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                                                    
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Bond wedges\n{var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of condensed minimum basins ---  one diagram per property':  # GOOD min basins      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                    if "Min " not in ak1:
                        continue
                    a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                    region_name = f"{a_name} {ak1[ak1.find('('):ak1.find(')')+1]}"
                    xd[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[region_name].append(av2[v])
                                break
                
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                                                    
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Condensed minimum basins\n{var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of topological cages ---  one diagram per property':  # GOOD topo cages      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1,av1 in systems[initial_sys]['regions'].items():
                    if "Cage " not in ak1:
                        continue
                    region_name = f"{'-'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])} {ak1}"
                    xd[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[region_name].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                                                    
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Topological cages\n{var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        #######  Difference correlation matrices
            
        elif plot_type == 'correlation diagrams of atomic basin property differences between "final" and "initial" systems ---  one diagram per property':  # MODERATE all atomic basins      
            var_list = diff_var_list
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1 in systems[initial_sys]['atoms'].keys():
                    a_name = systems[initial_sys]['atom_nicknames'][ak1]
                    xd[a_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                xd[a_name].append(av2[v])
                                break
                do_run = True
                for k1,v1 in xd.items():
                    for k2,v2 in xd.items():
                        if k1 != k2 and len(v1) != len(v2):
                            print(f"irregular input matrix for {v = } in {sub_dir_base} correlation matrix")
                            do_run = False
                    
                if len(xd) == 0:
                    continue
                if do_run:                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    x = np.array(list(xd.values())).transpose()
                    CorrelationMatrix1(x, titlestr=f"Atomic basin differences\n∆ {var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of bond bundle property differences between "final" and "initial" systems ---  one diagram per property':  # GOOD bond bundles      
            var_list = diff_var_list
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1,av1 in systems[initial_sys]['regions'].items():
                    if "Bond " not in ak1 or len(av1) != 3:
                        continue
                    region_name = f"{'-'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])}"
                    xd[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[region_name].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                break
                do_run = True
                for k1,v1 in xd.items():
                    for k2,v2 in xd.items():
                        if k1 != k2 and len(v1) != len(v2):
                            print(f"invalid input matrix for {v = } in {sub_dir_base} correlation matrix")
                            do_run = False
                            break
                    if not do_run:
                        break
                    
                if len(xd) == 0:
                    continue
                if do_run:
                                
                    try:
                        os.mkdir(sub_dir_name)
                    except:
                        pass
            
                    x = np.array(list(xd.values())).transpose()
                    CorrelationMatrix1(x, titlestr=f"Bond bundle differences\n∆ {var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of bond wedge property differences between "final" and "initial" systems ---  one diagram per property':  # GOOD bond wedges      
            var_list = diff_var_list
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                    if "Max " not in ak1:
                        continue
                    a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                    if ak1 in max_basin_to_bond_bundle_map:
                        region_name = f"{a_name} ({max_basin_to_bond_bundle_map[ak1]})"
                    else:
                        region_name = f"{a_name} {ak1[ak1.find('('):ak1.find(')')+1]}"
                    xd[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[region_name].append(av2[v])
                                break
                
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                                
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Bond wedge differences\n∆ {var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
    
        elif plot_type == 'correlation diagrams of condensed minimum basin property differences between "final" and "initial" systems ---  one diagram per property':  # GOOD min basins      
            var_list = diff_var_list
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                    if "Min " not in ak1:
                        continue
                    a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                    region_name = f"{a_name} {ak1[ak1.find('('):ak1.find(')')+1]}"
                    xd[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[region_name].append(av2[v])
                                break
                
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                                
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Condensed minimum basin differences\n∆ {var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        #######  Crazy stuff
        
        elif plot_type == 'correlation diagrams of atomic basin bond wedge property standard deviations ---  one diagram per property':  # GOOD atomic basin bond wedge stdev      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1 in systems[initial_sys]['atoms'].keys():
                    a_name = systems[initial_sys]['atom_nicknames'][ak1]
                    xd[a_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                vals = [av[v] for ak,av in sv['minmax_basins'].items() if ak2 in ak and "Max " in ak]
                                if len(vals)<2:
                                    continue
                                xd[a_name].append(stdev(vals))
                                break
                if len(xd) == 0:
                    continue
                mode_size = mode([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != mode_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Atomic basin bond wedge stdev\n{var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of atomic basin standard deviation of bond wedge property differences between "final" and "initial" systems ---  one diagram per property':  # GOOD atomic basin bond wedge difference stdev      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = diff_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1 in systems[initial_sys]['atoms'].keys():
                    a_name = systems[initial_sys]['atom_nicknames'][ak1]
                    xd[a_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                vals = [av[v] for ak,av in sv['minmax_basins'].items() if ak2 in ak and "Max " in ak]
                                if len(vals)<2:
                                    continue
                                xd[a_name].append(stdev(vals))
                                break
                if len(xd) == 0:
                    continue
                mode_size = mode([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != mode_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Atomic basin bond wedge difference stdev\n∆ {var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of atomic basin condensed minimum basin property standard deviations ---  one diagram per property':  # GOOD atomic basin min basin stdev      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1 in systems[initial_sys]['atoms'].keys():
                    a_name = systems[initial_sys]['atom_nicknames'][ak1]
                    xd[a_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                vals = [av[v] for ak,av in sv['minmax_basins'].items() if ak2 in ak and "Min " in ak]
                                if len(vals)<2:
                                    continue
                                xd[a_name].append(stdev(vals))
                                break
                if len(xd) == 0:
                    continue
                mode_size = mode([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != mode_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                
                x = np.array(list(xd.values())).transpose()            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                CorrelationMatrix1(x, titlestr=f"Atomic basin min basin stdev\n{var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of atomic basin standard deviation of condensed minimum basin property differences between "final" and "initial" systems ---  one diagram per property':
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = diff_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                for ak1 in systems[initial_sys]['atoms'].keys():
                    a_name = systems[initial_sys]['atom_nicknames'][ak1]
                    xd[a_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                vals = [av[v] for ak,av in sv['minmax_basins'].items() if ak2 in ak and "Min " in ak]
                                if len(vals)<2:
                                    continue
                                xd[a_name].append(stdev(vals))
                                break
                if len(xd) == 0:
                    continue
                mode_size = mode([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != mode_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                
                x = np.array(list(xd.values())).transpose()
            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                CorrelationMatrix1(x, titlestr=f"Atomic basin min basin difference stdev\n∆ {var_nickname_full[v]} ({var_nickname[v]}) correlation", labels=list(xd.keys()), filedir=sub_dir_name)
        
        

        ############################################
        
        #######  bar charts showing correlation of regions
        
            
        elif plot_type == 'bar charts of atomic basin property correlations to system property ---  one chart per property':  #  atomic basins (with property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1 in systems[initial_sys]['atoms'].keys():
                    a_name = systems[initial_sys]['atom_nicknames'][ak1]
                    xd[a_name] = []
                    xd_diff[a_name] = []
                    xe[a_name] = []
                    xe_diff[a_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                xd[a_name].append(av2[v])
                                xe[a_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[a_name].append(av2[var_to_diff_var[v]])
                                    xe_diff[a_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                if not INCLUDE_DIFF: xr_diff = {}
            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                try:
                    # CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Atomic basin condensed property correlation ∆\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=True)
                    CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Atomic basin condensed property correlation\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=False)
                except Exception as e:
                    print(e) 
        
        elif plot_type == 'bar charts of bond bundle property correlations to system property ---  one chart per property':  #  bond bundles (with property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1,av1 in systems[initial_sys]['regions'].items():
                    if "Bond " not in ak1 or len(av1) != 3:
                        continue
                    region_name = f"{'--'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])}"
                    xd[region_name] = []
                    xd_diff[region_name] = []
                    xe[region_name] = []
                    xe_diff[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[region_name].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                xe[region_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[region_name].append(sum([av[var_to_diff_var[v]] for ak,av in av2.items() if "_map" not in ak]))
                                    xe_diff[region_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                if not INCLUDE_DIFF: xr_diff = {}
            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                try:
                    # CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Bond bundle condensed property correlation ∆\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=True)
                    CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Bond bundle condensed property correlation\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=False)
                except Exception as e:
                    print(e) 
            
        elif plot_type == 'bar charts of bond wedge property correlations to system property ---  one chart per property':  #  bond wedges (with property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                    if "Max " not in ak1:
                        continue
                    a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                    if ak1 in max_basin_to_bond_bundle_map:
                        region_name = f"{a_name} ({max_basin_to_bond_bundle_map[ak1]})"
                    else:
                        region_name = f"{a_name} {ak1[ak1.find('('):ak1.find(')')+1]}"
                    xd[region_name] = []
                    xd_diff[region_name] = []
                    xe[region_name] = []
                    xe_diff[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[region_name].append(av2[v])
                                xe[region_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[region_name].append(av2[var_to_diff_var[v]])
                                    xe_diff[region_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                if not INCLUDE_DIFF: xr_diff = {}
            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                try:
                    # CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Bond wedge condensed property correlation ∆\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=True)
                    CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Bond wedge condensed property correlation\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=False)
                except Exception as e:
                    print(e) 
                
        elif plot_type == 'bar charts of condensed minimum basin property correlations to system property ---  one chart per property':  #  min basins (with property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                    if "Min " not in ak1:
                        continue
                    a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                    region_name = f"{a_name} {ak1[ak1.find('('):ak1.find(')')+1]}"
                    xd[region_name] = []
                    xd_diff[region_name] = []
                    xe[region_name] = []
                    xe_diff[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[region_name].append(av2[v])
                                xe[region_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[region_name].append(av2[var_to_diff_var[v]])
                                    xe_diff[region_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                if not INCLUDE_DIFF: xr_diff = {}
            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                try:
                    # CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Condensed minimum basin property correlation ∆\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=True)
                    CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Condensed minimum basin property correlation\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=False)
                except Exception as e:
                    print(e) 
                        
        
        elif plot_type == 'bar charts of topological cage property correlations to system property ---  one chart per property':  #  topo cages
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            
            for v in var_list:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1,av1 in systems[initial_sys]['regions'].items():
                    if "Cage " not in ak1:
                        continue
                    region_name = f"{'--'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])} {ak1}"
                    xd[region_name] = []
                    xd_diff[region_name] = []
                    xe[region_name] = []
                    xe_diff[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[region_name].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                xe[region_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[region_name].append(sum([av[var_to_diff_var[v]] for ak,av in av2.items() if "_map" not in ak]))
                                    xe_diff[region_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                if not INCLUDE_DIFF: xr_diff = {}
            
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
            
                try:
                    # CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Topological cage property correlation ∆\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=True)
                    CorrelationBarChart(list(xr.values()), x_over=list(xr_diff.values()), titlestr=f"Topological cage property correlation\n{energy_nickname_full} ({energy_nickname}) vs {var_nickname_full[v]} ({var_nickname[v]})", filedir=sub_dir_name, labels=list(xr.keys()), ylabel=energy_nickname, xlabel=var_nickname[v], x_over_suffix=" (∆)", x_over_sort=False)
                except Exception as e:
                    print(e)
            
        #######  Bar charts showing same but for one or more variables
        elif plot_type == 'bar chart of atomic basin property correlations to system property ---  all properties on one chart':  #  atomic basins (with ALL property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            var_vals = []
            var_over_vals = []
            rev_var = list(var_list)
            for v in rev_var:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1 in systems[initial_sys]['atoms'].keys():
                    a_name = systems[initial_sys]['atom_nicknames'][ak1]
                    xd[a_name] = []
                    xd_diff[a_name] = []
                    xe[a_name] = []
                    xe_diff[a_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                xd[a_name].append(av2[v])
                                xe[a_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[a_name].append(av2[var_to_diff_var[v]])
                                    xe_diff[a_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                var_vals.append(list(xr.values()))
                var_over_vals.append(list(xr_diff.values()))
            
            if not INCLUDE_DIFF: var_over_vals = None
            rev_var.reverse()
            
            try:
                os.mkdir(sub_dir_name)
            except:
                pass
            
            try:
                # CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Atomic basins: Condensed property correlations to {energy_nickname_full} ({energy_nickname}) ∆", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v1] for v1 in rev_var], x_over_suffix=" (∆)", x_over_sort=True)
                CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Atomic basins: Condensed property correlations to {energy_nickname_full} ({energy_nickname})", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v1] for v1 in rev_var], x_over_suffix=" (∆)", x_over_sort=False)
            except Exception as e:
                print(e)
        
    
        elif plot_type == 'bar chart of bond bundle property correlations to system property ---  all properties on one chart':  #  bond bundles (with ALL property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            var_vals = []
            var_over_vals = []
            rev_var = list(var_list)
            for v in rev_var:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1,av1 in systems[initial_sys]['regions'].items():
                    if "Bond " not in ak1 or len(av1) != 3:
                        continue
                    region_name = f"{'--'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])}"
                    
                    xd[region_name] = []
                    xd_diff[region_name] = []
                    xe[region_name] = []
                    xe_diff[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[region_name].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                xe[region_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[region_name].append(sum([av[var_to_diff_var[v]] for ak,av in av2.items() if "_map" not in ak]))
                                    xe_diff[region_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                var_vals.append(list(xr.values()))
                var_over_vals.append(list(xr_diff.values()))
                
            if not INCLUDE_DIFF: var_over_vals = None
            rev_var.reverse()
            
            try:
                os.mkdir(sub_dir_name)
            except:
                pass
            
            try:
                # CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Bond bundles: Condensed property correlations to {energy_nickname_full} ({energy_nickname}) ∆", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v] for v in rev_var], x_over_suffix=" (∆)", x_over_sort=True)
                CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Bond bundles: Condensed property correlations to {energy_nickname_full} ({energy_nickname})", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v] for v in rev_var], x_over_suffix=" (∆)", x_over_sort=False)
            except Exception as e:
                print(e)
        
        elif plot_type == 'bar chart of bond wedge property correlations to system property ---  all properties on one chart':  #  bond wedges (with ALL property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            var_vals = []
            var_over_vals = []
            rev_var = list(var_list)
            for v in rev_var:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                    if "Max " not in ak1:
                        continue
                    a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                    if ak1 in max_basin_to_bond_bundle_map:
                        region_name = f"{a_name} ({max_basin_to_bond_bundle_map[ak1]})"
                    else:
                        region_name = f"{a_name} {ak1[ak1.find('('):ak1.find(')')+1]}"
                    
                    xd[region_name] = []
                    xd_diff[region_name] = []
                    xe[region_name] = []
                    xe_diff[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[region_name].append(av2[v])
                                xe[region_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[region_name].append(av2[var_to_diff_var[v]])
                                    xe_diff[region_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                var_vals.append(list(xr.values()))
                var_over_vals.append(list(xr_diff.values()))
                
            if not INCLUDE_DIFF: var_over_vals = None
            rev_var.reverse()
            
            try:
                os.mkdir(sub_dir_name)
            except:
                pass
            
            try:
                # CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Bond wedges: Condensed property correlations to {energy_nickname_full} ({energy_nickname}) ∆", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v] for v in rev_var], x_over_suffix=" (∆)", x_over_sort=True)
                CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Bond wedges: Condensed property correlations to {energy_nickname_full} ({energy_nickname})", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v] for v in rev_var], x_over_suffix=" (∆)", x_over_sort=False)
            except Exception as e:
                print(e)
        
        elif plot_type == 'bar chart of condensed minimum basin property correlations to system property ---  all properties on one chart':  #  min basins (with ALL property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            var_vals = []
            var_over_vals = []
            rev_var = list(var_list)
            for v in rev_var:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                    if "Min " not in ak1:
                        continue
                    a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                    region_name = f"{a_name} {ak1[ak1.find('('):ak1.find(')')+1]}"
                    
                    xd[region_name] = []
                    xd_diff[region_name] = []
                    xe[region_name] = []
                    xe_diff[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[region_name].append(av2[v])
                                xe[region_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[region_name].append(av2[var_to_diff_var[v]])
                                    xe_diff[region_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                var_vals.append(list(xr.values()))
                var_over_vals.append(list(xr_diff.values()))
                
            if not INCLUDE_DIFF: var_over_vals = None
            rev_var.reverse()
            
            try:
                os.mkdir(sub_dir_name)
            except:
                pass
            
            
            try:
                # CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Condensed minimum basins: Condensed property correlations to {energy_nickname_full} ({energy_nickname}) ∆", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v] for v in rev_var], x_over_suffix=" (∆)", x_over_sort=True)
                CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Condensed minimum basins: Condensed property correlations to {energy_nickname_full} ({energy_nickname})", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v] for v in rev_var], x_over_suffix=" (∆)", x_over_sort=False)
            except Exception as e:
                print(e)
    
    
    
        ####################################################
        
        #######  Correlation matrices of variables over single "regions"
       
                
    
        elif plot_type == 'bar chart of topological cage property correlations to system property ---  all properties on one chart':  #  topo cages (with ALL property and property difference correlations)
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            var_vals = []
            var_over_vals = []
            rev_var = list(var_list)
            for v in rev_var:
                # collect v values for each atom in each system
                xd = {}
                xd_diff = {}
                xe = {}
                xe_diff = {}
                for ak1,av1 in systems[initial_sys]['regions'].items():
                    if "Cage " not in ak1:
                        continue
                    region_name = f"{'--'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])} {ak1}"
                    
                    xd[region_name] = []
                    xd_diff[region_name] = []
                    xe[region_name] = []
                    xe_diff[region_name] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[region_name].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                xe[region_name].append(sv['energy'])
                                if sk != initial_sys and v in var_to_diff_var:
                                    xd_diff[region_name].append(sum([av[var_to_diff_var[v]] for ak,av in av2.items() if "_map" not in ak]))
                                    xe_diff[region_name].append(sv['energy'])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                max_size = max([len(x) for x in xd_diff.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd_diff.items()}
                [(xd.pop(k),xe.pop(k)) for k,b in x_pop.items() if b]
                [(xd_diff.pop(k),xe_diff.pop(k)) for k,b in x_pop.items() if b and k in xd_diff]
                
                # get fits
                xr = {}
                for xk,xv in xd.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe[xk],y_pred)
                    coef = regr.coef_
                    xr[xk] = r2_func(xi,xe[xk]) * np.sign(coef[0])
                xr_diff = {}
                for xk,xv in xd_diff.items():
                    regr = linear_model.LinearRegression()
                    xi = np.array(xv).reshape(-1, 1)
                    regr.fit(xi,xe_diff[xk])
                    y_pred = regr.predict(xi)
                    rsqr = r2_score(xe_diff[xk],y_pred)
                    coef = regr.coef_
                    xr_diff[xk] = r2_func(xi,xe_diff[xk]) * np.sign(coef[0])
                
                var_vals.append(list(xr.values()))
                var_over_vals.append(list(xr_diff.values()))
                
            if not INCLUDE_DIFF: var_over_vals = None
            rev_var.reverse()
            
            try:
                os.mkdir(sub_dir_name)
            except:
                pass
            
            try:
                # CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Topological cages: Condensed property correlations to {energy_nickname_full} ({energy_nickname}) ∆", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v] for v in rev_var], x_over_suffix=" (∆)", x_over_sort=True)
                CorrelationMultiVarBarChart(var_vals, x_over=var_over_vals, titlestr=f"Topological cages: Condensed property correlations to {energy_nickname_full} ({energy_nickname})", filedir=sub_dir_name, labels_in=list(xr.keys()), ylabel=energy_nickname, xlabels=[var_nickname[v] for v in rev_var], x_over_suffix=" (∆)", x_over_sort=False)
            except Exception as e:
                print(e)
        
    
        
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = -30  # pad is in points...
        
    for plot_type in new_plot_list:
        
        # correlation diagrams among properties in regions
        
        if plot_type == 'correlation diagrams of regional properties: one diagram per atomic basin':  # atomic basins 
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            for ak1 in systems[initial_sys]['atoms'].keys():
                a_name = systems[initial_sys]['atom_nicknames'][ak1]
                # collect v values for each atom in each system
                xd = {}
                for v in var_list:
                    vn = var_nickname[v]
                    xd[vn] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['atoms'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and sv['atom_map'][ak2] == ak1):
                                xd[vn].append(av2[v])
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Variable correlation\n{a_name} atomic basins", labels=list(xd.keys()), filedir=sub_dir_name)
        
        
            
        elif plot_type == 'correlation diagrams of regional properties: one diagram per bond bundle':  # bond bundles      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            for ak1,av1 in systems[initial_sys]['regions'].items():
                if "Bond " not in ak1 or len(av1) != 3:
                    continue
                region_name = f"{'-'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])}"
                # collect v values for each atom in each system
                xd = {}
                for v in var_list:
                    vn = var_nickname[v]
                    xd[vn] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[vn].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Variable correlation\n{region_name} bond bundles", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of regional properties: one diagram per bond wedge':  # bond wedges      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                if "Max " not in ak1:
                    continue
                
                a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                if ak1 in max_basin_to_bond_bundle_map:
                    region_name = f"{a_name} Max ({max_basin_to_bond_bundle_map[ak1]})"
                else:
                    region_name = f"{a_name} Max {ak1[ak1.find('('):ak1.find(')')+1]}"
                # collect v values for each atom in each system
                xd = {}
                for v in var_list:
                    vn = var_nickname[v]
                    xd[vn] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[vn].append(av2[v])
                                break
                
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]                
                
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Variable correlation\n{region_name} bond wedges", labels=list(xd.keys()), filedir=sub_dir_name)
        
        elif plot_type == 'correlation diagrams of regional properties: one diagram per condensed minimum basin':  # min basins      
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            for ak1,av1 in systems[initial_sys]['minmax_basins'].items():
                if "Min " not in ak1:
                    continue
                a_name = systems[initial_sys]['atom_nicknames'][ak1[:ak1.find(':')]]
                region_name = f"{a_name} Min {ak1[ak1.find('('):ak1.find(')')+1]}"
            
                # collect v values for each atom in each system
                xd = {}
                for v in var_list:
                    vn = var_nickname[v]
                    xd[vn] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['minmax_basins'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['minmax_basin_map'] and sv['minmax_basin_map'][ak2] == ak1):
                                xd[vn].append(av2[v])
                                break
                
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Variable correlation\n{region_name} minimum basins", labels=list(xd.keys()), filedir=sub_dir_name)
        
                
            
        elif plot_type == 'correlation diagrams of regional properties: one diagram per topological cage':  # topo cages
            
            # first all atoms
            
            sub_dir_base = plot_type
            sub_dir_name = os.path.join(os.getcwd(), dir_name, sub_dir_base)
            
            print(f"\n\nGenerating plots in {sub_dir_name}...")
            
            # one correlation matrix per variable
            var_list = old_var_list
            
            for ak1,av1 in systems[initial_sys]['regions'].items():
                if "Cage " not in ak1:
                    continue
                region_name = f"{'-'.join([k for k in sorted([systems[initial_sys]['atom_nicknames'][k] for k in av1.keys() if '_map' not in k]) if '_map' not in k])}"
                # collect v values for each atom in each system
                xd = {}
                for v in var_list:
                    vn = var_nickname[v]
                    xd[vn] = []
                    for sk,sv in systems.items():
                        for ak2,av2 in sv['regions'].items():
                            if (sk == initial_sys and ak2 == ak1) or (sk != initial_sys and ak2 in sv['region_map'] and sv['region_map'][ak2] == ak1):
                                xd[vn].append(sum([av[v] for ak,av in av2.items() if "_map" not in ak]))
                                break
                if len(xd) == 0:
                    continue
                max_size = max([len(x) for x in xd.values()])
                x_pop = {xk:len(xv) != max_size for xk,xv in xd.items()}
                [xd.pop(k) for k,v in x_pop.items() if v]
                try:
                    os.mkdir(sub_dir_name)
                except:
                    pass
                    
                x = np.array(list(xd.values())).transpose()
                CorrelationMatrix1(x, titlestr=f"Variable correlation\n{region_name} Topological {ak1}", labels=list(xd.keys()), filedir=sub_dir_name)
        
        
    # max_num_reverse_minmax_basin_mappings = max([len(v) for v in reverse_minmax_basin_map.values()])
    
    print(f"\n\nFinished! You'll find your comparison files in {dir_name} next to the input files")
    
if __name__ == "__main__":
#     try:
    inpath = argv[1] if len(argv) > 1 else "/Users/haiiro/SynologyDrive/Eclipse/Python/School/GBA_PostProcessing/example"
    os.chdir(inpath)
    main()
#     except Exception as e:
#         print(f"Something went wrong. To use, run:\n\npython3 path/to/gbapost.py path/to/folder/of/gba/results\n\neg. python3 ~/Desktop/gpapost.py ~/Desktop/results\n\nerror: {e}")
#         exc_type, value, exc_traceback = exc_info()
#         print(f"{exc_type = }\n{value = }\n{exc_traceback = }")
