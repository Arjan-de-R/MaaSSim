import pandas as pd
from dotmap import DotMap
import numpy as np


def d2d_summary_day(evol_micro, drivers_summary, travs_summary, day):
    "add stats of last day to d2d dataframe"
    # d2d = kwargs.get('d2d', None)

    # Supply
    evol_micro.supply.inform.append(drivers_summary.informed.to_list())
    evol_micro.supply.regist.append(drivers_summary.registered.to_list())
    evol_micro.supply.rejected_reg.append(drivers_summary.rejected_reg.to_list())
    evol_micro.supply.ptcp.append((~drivers_summary.out).to_list())
    evol_micro.supply.rejected_ptcp.append(drivers_summary.forced_out.to_list())
    evol_micro.supply.perc_inc.append(drivers_summary.init_perc_inc.to_list())
    evol_micro.supply.exp_inc.append(drivers_summary.exp_inc.to_list())
    ptcp = (~drivers_summary.out).replace(False, np.nan)
    evol_micro.supply.perc_inc_ptcp.append((ptcp * drivers_summary.init_perc_inc).to_list())
    reg = drivers_summary.registered.replace(False, np.nan)
    evol_micro.supply.perc_inc_reg.append((reg * drivers_summary.init_perc_inc).to_list())

    # Demand
    evol_micro.demand.inform.append(travs_summary.informed.to_list())
    evol_micro.demand.requests.append(travs_summary.requests.to_list())
    evol_micro.demand.gets_offer.append(travs_summary.gets_offer.to_list())
    evol_micro.demand.accepts_offer.append(travs_summary.accepts_offer.to_list())
    evol_micro.demand.wait_time.append(travs_summary.xp_wait.to_list())
    evol_micro.demand.corr_wait_time.append(travs_summary.corr_xp_wait.to_list())
    evol_micro.demand.perc_wait.append(travs_summary.init_perc_wait.to_list())
    evol_micro.demand.chosen_mode.append(travs_summary.chosen_mode.to_list())
    perc_wait_req = travs_summary.requests.replace(False, np.nan)
    evol_micro.demand.perc_wait_req.append((perc_wait_req * travs_summary.init_perc_wait).to_list())
    evol_micro.demand.bike.append((travs_summary.chosen_mode == 'bike').to_list())
    evol_micro.demand.car.append((travs_summary.chosen_mode == 'car').to_list())
    evol_micro.demand.pt.append((travs_summary.chosen_mode == 'pt').to_list())

    return evol_micro


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


def init_d2d_dotmap():
    # create empty dotmap for adding daily statistics
    evol_micro = DotMap()
    evol_micro.supply = DotMap()
    evol_micro.supply.inform = []
    evol_micro.supply.regist = []
    evol_micro.supply.rejected_reg = []
    evol_micro.supply.ptcp = []
    evol_micro.supply.rejected_ptcp = []
    evol_micro.supply.perc_inc = []
    evol_micro.supply.perc_inc_ptcp = []
    evol_micro.supply.perc_inc_reg = []
    evol_micro.supply.exp_inc = []
    evol_micro.demand = DotMap()
    evol_micro.demand.inform = []
    evol_micro.demand.requests = []
    evol_micro.demand.gets_offer = []
    evol_micro.demand.accepts_offer = []
    evol_micro.demand.wait_time = []
    evol_micro.demand.corr_wait_time = []
    evol_micro.demand.perc_wait = []
    evol_micro.demand.perc_wait_req = []
    evol_micro.demand.chosen_mode = []
    evol_micro.demand.bike = []
    evol_micro.demand.car = []
    evol_micro.demand.pt = []

    return evol_micro


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
