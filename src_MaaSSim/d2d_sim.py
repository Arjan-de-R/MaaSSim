import pandas as pd
from dotmap import DotMap
import numpy as np


def d2d_summary_day(drivers_summary, travs_summary):
    "add stats of last day to d2d dataframe and prepare for saving to csv"

    # Demand
    indicators = ['informed', 'registered', 'requests', 'gets_offer', 'accepts_offer', 'init_perc_wait',
                           'xp_wait', 'corr_xp_wait', 'init_perc_ivt', 'xp_ivt', 'init_perc_km_fare', 'xp_km_fare', 'chosen_mode']
    occ_strings = [s for s in travs_summary.columns if s.startswith("time_occ")]
    indicators = indicators + occ_strings
    dem_df = travs_summary[indicators].copy()
    for col in dem_df:
        if isinstance(dem_df.head(1)[col].values[0], np.ndarray):
            new_col_list = ['{}_{}'.format(col, plf_id) for plf_id in range(len(dem_df.head(1)[col].values[0]))]
            dem_df[new_col_list] = np.stack(dem_df[col].values)
            dem_df = dem_df.drop(columns=[col])
    
    # Supply
    indicators = ['informed', 'registered', 'out', 'init_perc_inc', 'exp_inc', 'pickup_dist', 'repos_dist']
    occ_strings = [s for s in drivers_summary.columns if s.startswith("km_occ")]
    indicators = indicators + occ_strings
    sup_df = drivers_summary[indicators].copy()
    for col in sup_df:
        if isinstance(sup_df.head(1)[col].values[0], np.ndarray):
            new_col_list = ['{}_{}'.format(col, plf_id) for plf_id in range(len(sup_df.head(1)[col].values[0]))]
            sup_df[new_col_list] = np.stack(sup_df[col].values)
            sup_df = sup_df.drop(columns=[col])

    return dem_df, sup_df


def d2d_agg_statistics(evol_micro):
    # create aggregated d2d statistics based on day-to-day statistics of individual agents

    # d2d lists to dataframes
    for key in evol_micro.supply:
        evol_micro.supply[key] = pd.DataFrame(evol_micro.supply[key]).T
    for key in evol_micro.demand:
        evol_micro.demand[key] = pd.DataFrame(evol_micro.demand[key]).T

    # Create df with aggregated statistics (not on agent-level)
    evol_agg = DotMap()
    evol_agg.supply = pd.DataFrame({'inform': evol_micro.supply.inform.sum(), 'regist': evol_micro.supply.regist.sum(),
                                    'rejected_reg': evol_micro.supply.rejected_reg.sum(),
                                    'particip': evol_micro.supply.ptcp.sum(),
                                    'reject_particip': evol_micro.supply.rejected_ptcp.sum(),
                                    'mean_perc_inc': evol_micro.supply.perc_inc.mean(),
                                    'mean_perc_inc_ptcp': evol_micro.supply.perc_inc_ptcp.mean(),
                                    'mean_perc_inc_reg': evol_micro.supply.perc_inc_reg.mean(),
                                    'mean_exp_inc': evol_micro.supply.exp_inc.mean()})
    evol_agg.supply.index.name = 'day'
    evol_agg.demand = pd.DataFrame(
        {'inform': evol_micro.demand.inform.sum(), 'requests': evol_micro.demand.requests.sum(),
         'gets_offer': evol_micro.demand.gets_offer.sum(), 'accepts_offer': evol_micro.demand.accepts_offer.sum(),
         'mean_wait': evol_micro.demand.wait_time.mean(), 'corr_mean_wait': evol_micro.demand.corr_wait_time.mean(),
         'perc_wait': evol_micro.demand.perc_wait.mean(), 'perc_wait_req': evol_micro.demand.perc_wait_req.mean(),
         'bike': evol_micro.demand.bike.sum(), 'car': evol_micro.demand.car.sum(), 'pt': evol_micro.demand.pt.sum()})
    evol_agg.demand.index.name = 'day'

    return evol_micro, evol_agg


def D2D_stop_crit(*args, **kwargs):
    "returns True if simulation will be stopped, False otherwise"
    res = kwargs.get('d2d_res', None)
    params = kwargs.get('params', None)

    if len(res) < params.evol.min_it:
        return False
    ret = (res[len(res)-1].new_perc_inc - res[len(res)-1].init_perc_inc) / res[len(res)-1].init_perc_inc
    return bool(ret.abs().max() <= params.evol.conv)


def return_scn_params(_params, key, val):
    '''updates parameter files, including scenario-specific parameters'''
    if key in ['comm_rate', 'fare', 'base_fare', 'reg_cap', 'ptcp_cap']:
        _params.platforms[key] = val
    if key in ['comm_rate', 'fare', 'base_fare', 'reg_cap', 'ptcp_cap']:
            _params.platforms[key] = val
    if key == 'gini':
        _params.evol.drivers[key] = val
        _params.evol.travellers.mode_pref[key] = val
    if key == 'inf_dem':
        _params.evol.travellers.inform.beta = val
    if key == 'inf_sup':
        _params.evol.drivers.inform.beta = val
    if key == 'inf_start_both':
        _params.evol.travellers.inform.prob_start = val
        _params.evol.drivers.inform.prob_start = val
    if key == 'inf_start_dem':
        _params.evol.travellers.inform.prob_start = val
    if key == 'inf_start_sup':
        _params.evol.drivers.inform.prob_start = val
    if key == 'reg_start':
        _params.evol.drivers.regist.prob_start = val
    if key == 'init_inc_ratio':
        _params.evol.drivers[key] = val
    if key == 'start_wait':
        _params.evol.travellers.inform[key] = val
    if key in ['cost_comp', 'samp', 'min_days']:
        _params.evol.drivers.regist[key] = val
    if key == 'kappa':
        _params.evol.travellers[key] = val
    if key == 'beta_reg':
        _params.evol.drivers.regist.beta = val
    if key == 'beta_ptcp':
        _params.evol.drivers.particip.beta = val
    else:
        _params[key] = val

    return _params
