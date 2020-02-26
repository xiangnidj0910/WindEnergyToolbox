#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:28:42 2017

@author: dave
"""

import os

import numpy as np
import pandas as pd

from wetb.prepost import hawcstab2


class Campbell(object):
    """
    Base class that holds the plotting stuff
    """

    alpha_box = 0.5

    def __init__(self):
        super(Campbell, self).__init__()

    def _inplot_label_pos(self, nr_xpos, nr_series, xpos):
        """
        Generate sensible label positions
        """

        if xpos == 'random':
            pos = np.random.randint(1, nr_xpos-5, nr_series)
        elif xpos == 'centre':
            pos = np.zeros((nr_series,))
            pos[0:len(pos):2] = np.ceil(nr_xpos/4.0)
            pos[1:len(pos):2] = np.ceil(2.0*nr_xpos/4.0)
        elif xpos == 'borders':
            pos = np.zeros((nr_series,))
            pos[0:len(pos):2] = 2
            pos[2:len(pos):4] += 1
            pos[1:len(pos):2] = np.floor(3.0*nr_xpos/4.0)
            # and +1 alternating on the right
            pos[1:len(pos):4] += 1
            pos[3:len(pos):4] -= 1
        elif xpos == 'right':
            pos = np.zeros((nr_series,))
            pos[0:len(pos):2] = 2
            pos[1:len(pos):2] = np.ceil(1.0*nr_xpos/4.0)
        elif xpos == 'left':
            pos = np.zeros((nr_series,))
            pos[0:len(pos):2] = np.ceil(2.0*nr_xpos/4.0)
            pos[1:len(pos):2] = np.ceil(3.0*nr_xpos/4.0)

        return pos

    def plot_freq(self, ax, xpos='random', col='k', mark='^', ls='-',
                  modes='all'):

        if isinstance(modes, str) and modes == 'all':
            df_freq = self.df_freq
        elif isinstance(modes, int):
            df_freq = self.df_freq[:modes]
        else:
            df_freq = self.df_freq[modes]

        nr_winds = df_freq.shape[0]
        nr_modes = df_freq.shape[1]
        pos = self._inplot_label_pos(nr_winds, nr_modes, xpos)

        if isinstance(col, str):
            col = [col]
        if isinstance(mark, str):
            mark = [mark]
        if isinstance(ls, str):
            ls = [ls]
        # just to make sure we always have enough colors/marks,lss
        col = col*nr_modes
        mark = mark*nr_modes
        ls = ls*nr_modes

        wind = self.df_perf.wind.values
        for i, (name, row) in enumerate(df_freq.iteritems()):
            colmark = '%s%s%s' % (col[i], ls[i], mark[i])
            ax.plot(wind, row.values, colmark)#, mfc='w')
            x, y = wind[pos[i]], row.values[pos[i]]
            bbox = dict(boxstyle="round", alpha=self.alpha_box,
                        edgecolor=col[i], facecolor=col[i])
            ax.annotate(name, xy=(x, y), xycoords='data',
                        xytext=(-6, +20), textcoords='offset points',
                        fontsize=12, bbox=bbox, arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3,rad=.05"), color='w')

        self.nr_modes = nr_modes

        return ax

    def plot_damp(self, ax, xpos='random', col='r', mark='o' ,ls='--',
                  modes=14):

        # the first mode can be the drive-train free-free for some badly sorted
        # HawcStab2 results
#        if self.df_freq.iloc[0,0] < 0.25:
#            name_freq_rem = self.df_freq.columns[0]
#            names_damp = set(self.df_damp.columns)
#            names_damp.discard(name_freq_rem)
#            self.damp = self.df_damp[list(names_damp)]

        # reduce the number of modes we are going to plot
        if isinstance(modes, int):
            nr_modes = modes
            # sort the columns according to damping: lowest damped modes first
            # sort according to damping at lowest wind speed
            isort = self.df_damp[:1].values[0].argsort()
            modes_sort_reduced = self.df_damp.columns[isort][:nr_modes]
            df_damp = self.df_damp[modes_sort_reduced]
        else:
            df_damp = self.df_damp[modes]
            nr_modes = len(modes)

        wind = self.df_perf.wind.values
        # put the labels in sensible places
        nr_winds = df_damp.shape[0]
        nr_damps = df_damp.shape[1]
        pos = self._inplot_label_pos(nr_winds, nr_damps, xpos)
        bbox = dict(boxstyle="round", alpha=self.alpha_box, edgecolor=col,
                    facecolor=col,)
        for imode, (name, row) in enumerate(df_damp.iteritems()):
            colmark = '%s%s%s' % (col, ls, mark)
            ax.plot(wind, row.values, colmark, alpha=0.8)#, mfc='w')
            x, y = wind[pos[imode]], row.values[pos[imode]]
            ax.annotate(name, xy=(x, y), xycoords='data',
                        xytext=(-6, 20), textcoords='offset points',
                        fontsize=12, bbox=bbox, arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3,rad=.05"), color='w')

        self.nr_modes = nr_modes

        return ax

    def add_legend(self, ax, labels, on='freq'):
        if on == 'freq':
            i0s = self.istart_freqs
        else:
            i0s = self.istart_damps

        return ax.legend([ax.lines[k] for k in i0s[:-1]], labels, loc='best')


class Campbell2(object):
    """
    Base class that holds the plotting stuff
    """

    alpha_box = 0.5

    def __init__(self):
        super(Campbell2, self).__init__()

    def _inplot_label_pos(self, nr_xpos, nr_series, xpos):
        """
        Generate sensible label positions
        """

        if xpos == 'random':
            pos = np.random.randint(1, nr_xpos-5, nr_series)
        elif xpos == 'centre':
            pos = np.zeros((nr_series,))
            pos[0:len(pos):2] = np.ceil(nr_xpos/4.0)
            pos[1:len(pos):2] = np.ceil(2.0*nr_xpos/4.0)
        elif xpos == 'borders':
            pos = np.zeros((nr_series,))
            pos[0:len(pos):2] = 2
            pos[2:len(pos):4] += 1
            pos[1:len(pos):2] = np.floor(3.0*nr_xpos/4.0)
            # and +1 alternating on the right
            pos[1:len(pos):4] += 1
            pos[3:len(pos):4] -= 1
        elif xpos == 'right':
            pos = np.zeros((nr_series,))
            pos[0:len(pos):2] = 2
            pos[1:len(pos):2] = np.ceil(1.0*nr_xpos/4.0)
        elif xpos == 'left':
            pos = np.zeros((nr_series,))
            pos[0:len(pos):2] = np.ceil(2.0*nr_xpos/4.0)
            pos[1:len(pos):2] = np.ceil(3.0*nr_xpos/4.0)

        return pos

    def plot_freq(self, ax, dflabel, xpos='random', c='k', marker='^', ls='-',
                  modes='all', modelabels=None, alpha=1, label=None):

        bbox = dict(boxstyle="round", alpha=self.alpha_box, edgecolor=c,
                    facecolor=c)

        nr_winds = len(self.df['wind_ms'].unique())
        nr_modes = len(self.df['mode'].unique())
        pos = self._inplot_label_pos(nr_winds, nr_modes, xpos)

        for i, (modenr, gr_mode) in enumerate(self.df.groupby('mode')):
            name = str(modenr)
#            colmark = '%s%s%s' % (col, ls, mark)
            wind = gr_mode['wind_ms'].values
            freq = gr_mode[dflabel].values
            # only label on the first
            if i < 1:
                ax.plot(wind, freq, c=c, ls=ls, marker=marker, alpha=alpha,
                        label=label)
            else:
                ax.plot(wind, freq, c=c, ls=ls, marker=marker, alpha=alpha)
            x, y = wind[pos[i]], freq[pos[i]]
            if modelabels is not None:
                ax.annotate(name, xy=(x, y), xycoords='data',
                            xytext=(-6, +20), textcoords='offset points',
                            fontsize=12, bbox=bbox, arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=.05"), color='w')

        self.nr_modes = nr_modes

        return ax

    def plot_damp(self, ax, xpos='random', col='r', mark='o' ,ls='--',
                  modes=14):

        # the first mode can be the drive-train free-free for some badly sorted
        # HawcStab2 results
#        if self.df_freq.iloc[0,0] < 0.25:
#            name_freq_rem = self.df_freq.columns[0]
#            names_damp = set(self.df_damp.columns)
#            names_damp.discard(name_freq_rem)
#            self.damp = self.df_damp[list(names_damp)]

        # reduce the number of modes we are going to plot
        if isinstance(modes, int):
            nr_modes = modes
            # sort the columns according to damping: lowest damped modes first
            # sort according to damping at lowest wind speed
            isort = self.df_damp[:1].values[0].argsort()
            modes_sort_reduced = self.df_damp.columns[isort][:nr_modes]
            df_damp = self.df_damp[modes_sort_reduced]
        else:
            df_damp = self.df_damp[modes]
            nr_modes = len(modes)

        wind = self.df_perf.wind.values
        # put the labels in sensible places
        nr_winds = df_damp.shape[0]
        nr_damps = df_damp.shape[1]
        pos = self._inplot_label_pos(nr_winds, nr_damps, xpos)
        bbox = dict(boxstyle="round", alpha=self.alpha_box, edgecolor=col,
                    facecolor=col,)
        for imode, (name, row) in enumerate(df_damp.iteritems()):
            colmark = '%s%s%s' % (col, ls, mark)
            ax.plot(wind, row.values, colmark, alpha=0.8)#, mfc='w')
            x, y = wind[pos[imode]], row.values[pos[imode]]
            ax.annotate(name, xy=(x, y), xycoords='data',
                        xytext=(-6, 20), textcoords='offset points',
                        fontsize=12, bbox=bbox, arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3,rad=.05"), color='w')

        self.nr_modes = nr_modes

        return ax

    def add_legend(self, ax, labels, on='freq'):
        if on == 'freq':
            i0s = self.istart_freqs
        else:
            i0s = self.istart_damps

        return ax.legend([ax.lines[k] for k in i0s[:-1]], labels, loc='best')


class hs2_campbell(Campbell):

    def __init__(self, basename, f_cmb=False, f_modid=None):

        hs2 = hawcstab2.results()
        self.basename = basename

        # performance indicators
        f_pwr = basename + '.pwr'
        res = hs2.load_pwr(f_pwr)
        tmp = np.ndarray((len(res.wind), 3))
        tmp[:,0], tmp[:,1], tmp[:,2] = res.wind, res.pitch_deg, res.rpm
        self.df_perf = pd.DataFrame(tmp, columns=['wind', 'pitch', 'rpm'])

        # create DataFrames with the freq/damp on the column name
        if f_cmb is None:
            f_cmb = basename + '.cmb'
        wind, freqs, damps, real_eig = hs2.load_cmb(f_cmb)
        nr_modes = freqs.shape[1]

        # strip characters if there is a comment after the description
        if f_modid is None:
            f_modid = basename + '.modid'
        if not os.path.isfile(f_modid):
            modes_descr = ['{:02d}'.format(k+1) for k in range(nr_modes)]
        else:
            modes_descr = self.read_modid(f_modid)[:nr_modes]

#        # the first mode can be the drive-train free-free, remove if so
#        if freqs[0,0] < 0.25:
#            freqs = freqs[:,1:]
#            damps = damps[:,1:]

        self.df_freq = pd.DataFrame(freqs, columns=modes_descr)
        self.df_damp = pd.DataFrame(damps, columns=modes_descr)

        tmp = pd.DataFrame(freqs, columns=modes_descr, index=wind)
        tmp.index.name = 'windspeed'
        # add 1P, 3P and 6P columns
        tmp['1P'] =  res.rpm / 60
        tmp['3P'] =  3*res.rpm / 60
        tmp['6P'] =  6*res.rpm / 60
        # sort columns on mean frequeny over wind speeds
        icolsort = tmp.values.mean(axis=0).argsort()
        tmp = tmp[tmp.columns[icolsort]]
        tmp.to_excel(basename + '_freqs.xlsx')

        tmp = pd.DataFrame(damps, columns=modes_descr, index=wind)
        tmp.index.name = 'windspeed'
        tmp.to_excel(basename + '_damps.xlsx')

    def read_modid(self, fname):

        df_modes = pd.read_csv(fname, comment='#', delimiter=';',
                               header=None, names=['mode_nr', 'description'],
                               converters={0:lambda x:x.strip(),
                                           1:lambda x:x.strip()})
        return df_modes['description'].values.tolist()


def add_Ps(ax, wind, rpm, col='g', fmax=10, ps=None):
    # plot 1P, 2P, ..., 9P
    bbox = dict(boxstyle="round", alpha=0.4, edgecolor=col, facecolor=col,)
    if ps is None:
        pmax = int(60*fmax/rpm.mean())
        ps = list(range(1, pmax))
    for p in ps:
        if p%3 == 0:
            alpha=0.6
            ax.plot(wind, rpm*p/60, '%s--' % col, )
        else:
            alpha = 0.4
            ax.plot(wind, rpm*p/60, '%s-.' % col)#, alpha=alpha)
        x, y = wind[10], rpm[10]*p/60
        p_str = '%iP' % p
        bbox['alpha'] = alpha
        ax.text(x, y, p_str, fontsize=9, verticalalignment='bottom',
                horizontalalignment='center', bbox=bbox, color='w')

    return ax


if __name__ == '__main__':

    dummy = None
