import pandas as pd
import numpy as np
from scipy.stats import t


def D2D_conv(*args, **kwargs):
    "returns True if convergence criterion is reached"
    params = kwargs.get('params', None)
    perc_values = kwargs.get('perc', None)

    df = pd.DataFrame()
    df['t'] = perc_values
    df['t-1'] = df.t.shift(1)
    df = df.drop([0])
    df['rdiff'] = (df['t'] - df['t-1']) / df['t-1']
    df['stable'] = abs(df.rdiff) < params['conv_factor']
    df['conv_cum'] = df.stable.cumsum()
    df['conv_cum_early'] = df.conv_cum.shift(params['conv_day'])
    df['conv'] = (df.conv_cum - df.conv_cum_early == params['conv_day'])
    if df.conv.sum() == 0:
        t_conv = params['nD']-1
        perc_conv = perc_values[t_conv]
    else:
        t_conv = df.index.get_loc(df.index[df.conv == True][0])
        perc_conv = df.iloc[t_conv].t
    conv_res = [t_conv, perc_conv]

    return conv_res


def repl_num(*args, **kwargs):
    params = kwargs.get('params', None)
    perc_values = kwargs.get('perc', None)
    params['nReplications'] = len(perc_values)

    mean_perc = np.mean(perc_values)
    std_perc = np.std(perc_values)

    df = params['nReplications'] - 1
    crit_t = t.ppf(params['conv_signif'], df)
    num_needed = ((std_perc * crit_t) / (mean_perc * params['conv_max_error'])) ** 2

    return num_needed


def empty_df(**kwargs):
    params = kwargs.get('params', None)

    df = pd.DataFrame(index=range(0,params[0]['nD']),columns=range(0,params[0]['parallel']['nReplications']))
    df.index.name = 'day'

    return df


def join_scn_df(d2d, vals, kpi):
    df = pd.DataFrame([d2d[i][kpi]['mean'] for i in vals]).T
    df = df.set_axis(vals, axis=1, inplace=False)

    return df