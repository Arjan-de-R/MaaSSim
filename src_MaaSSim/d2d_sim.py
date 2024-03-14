import pandas as pd
from dotmap import DotMap
import numpy as np
import random
import os
import pickle

def d2d_summary_day(inData, drivers_summary, travs_summary):
    "add stats of last day to d2d dataframe and prepare for saving to csv"

    # Demand
    indicators_wd = ['requests', 'gets_offer', 'accepts_offer', 'xp_wait', 'corr_xp_wait', 'xp_ivt', 'xp_km_fare', 'chosen_mode']
    indicators_d2d = ['informed', 'registered', 'expected_wait', 'expected_ivt', 'expected_km_fare', 'days_since_reg']
    if 'tmc_balance' in inData.passengers.columns:
        indicators_d2d = indicators_d2d + ['tmc_balance','net_purchase','denied_order', 'money_balance']
    occ_strings = [s for s in travs_summary.columns if s.startswith("time_occ")]
    indicators_wd = indicators_wd + occ_strings

    dem_df = pd.concat([travs_summary[indicators_wd].copy(), inData.passengers[indicators_d2d].copy()], axis=1)
    for col in dem_df:
        if isinstance(dem_df.head(1)[col].values[0], np.ndarray):
            new_col_list = ['{}_{}'.format(col, plf_id) for plf_id in range(len(dem_df.head(1)[col].values[0]))]
            dem_df[new_col_list] = np.stack(dem_df[col].values)
            dem_df = dem_df.drop(columns=[col])
    
    # Supply
    indicators_wd = ['out', 'exp_inc', 'pickup_dist', 'repos_dist']
    indicators_d2d = ['informed', 'registered', 'expected_income','days_since_reg','work_exp']
    occ_strings = [s for s in drivers_summary.columns if s.startswith("km_occ")]
    indicators_wd = indicators_wd + occ_strings
    sup_df = pd.concat([drivers_summary[indicators_wd].copy(), inData.vehicles[indicators_d2d].copy()], axis=1)
    for col in sup_df:
        if isinstance(sup_df.head(1)[col].values[0], np.ndarray):
            new_col_list = ['{}_{}'.format(col, plf_id) for plf_id in range(len(sup_df.head(1)[col].values[0]))]
            sup_df[new_col_list] = np.stack(sup_df[col].values)
            sup_df = sup_df.drop(columns=[col])
    dem_df.index.name = 'pax'
    sup_df.index.name = 'veh'

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
    if key in ['comm_rate', 'fare', 'base_fare', 'reg_cap', 'ptcp_cap', 'service_types', 'pool_discount', 'start_reg_plf_share']:
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
    if key in ['init_inc_ratio', 'start_perc_inc_avg_ratio']:
        _params.evol.drivers[key] = val
    if key in ['start_wait', 'start_pool_detour']:
        _params.evol.travellers.inform[key] = val
    if key in ['cost_comp', 'min_days']:
        _params.evol.drivers.regist[key] = val
    if key == 'samp':
         _params.evol.drivers.regist[key] = val
         _params.evol.travellers.regist[key] = val
    if key == 'kappa':
        _params.evol.travellers[key] = val
    if key == 'beta_reg':
        _params.evol.drivers.regist.beta = val
    if key == 'beta_ptcp':
        _params.evol.drivers.particip.beta = val
    if key == 'util_multiplier':
        _params.evol.travellers.regist[key] = val
    if key == 'sup_util_multiplier':
        _params.evol.drivers.regist.util_multiplier = val
    if key == 'num_signals':
        _params.evol.travellers.inform[key] = val
        _params.evol.drivers.inform[key] = val
    else:
        _params[key] = val

    return _params


def determine_convergence(inData, d2d_conv, params, scn_name, day):
    if d2d_conv.shape[0] >= (params.convergence.get('first_moving_avg', 20) + params.convergence.get('second_moving_avg', 20) + params.convergence.get('req_steady_days', 10) + 1): # first day that convergence is possible
        if params.convergence.get('abs_ptcp_diff_dem', False) and params.convergence.get('abs_ptcp_diff_sup', False):
            rel_diff_ma_df = d2d_conv.rolling(params.convergence.get('first_moving_avg', 20)).mean().rolling(params.convergence.get('second_moving_avg', 20)).mean().diff().tail(params.convergence.get('req_steady_days', 10))
            rel_diff_ma_df['dem_mh_conv'] = rel_diff_ma_df.ptcp_dem_mh.abs() < params.convergence.abs_ptcp_diff_dem
            rel_diff_ma_df['dem_sh_0_conv'] = rel_diff_ma_df.ptcp_dem_sh_0.abs() < params.convergence.abs_ptcp_diff_dem
            rel_diff_ma_df['sup_mh_conv'] = rel_diff_ma_df.ptcp_sup_mh.abs() < params.convergence.abs_ptcp_diff_sup
            rel_diff_ma_df['sup_sh_0_conv'] = rel_diff_ma_df.ptcp_sup_sh_0.abs() < params.convergence.abs_ptcp_diff_sup
            if inData.platforms.shape[0] > 1:
                rel_diff_ma_df['dem_sh_1_conv'] = rel_diff_ma_df.ptcp_dem_sh_1.abs() < params.convergence.abs_ptcp_diff_dem
                rel_diff_ma_df['sup_sh_1_conv'] = rel_diff_ma_df.ptcp_sup_sh_1.abs() < params.convergence.abs_ptcp_diff_sup
                conv_per_indicator = rel_diff_ma_df[['dem_mh_conv','dem_sh_0_conv','dem_sh_1_conv','sup_mh_conv','sup_sh_0_conv','sup_sh_1_conv']].all()
            else:
                conv_per_indicator = rel_diff_ma_df[['dem_mh_conv','dem_sh_0_conv','sup_mh_conv','sup_sh_0_conv']].all()
        else:
            conv_factor = params.convergence.get('factor', 0.01)
            rel_diff_ma_df = d2d_conv.rolling(params.convergence.moving_avg).mean().tail(params.convergence.req_steady_days + 1).pct_change().tail(params.convergence.req_steady_days)
            conv_per_indicator = (rel_diff_ma_df.abs() < conv_factor).all()
        if conv_per_indicator.all():
            print('Scenario {} - All indicators have converged at end of day {}, day-to-day simulation is terminated.'.format(scn_name, day))
            converged = True
        else:
            print('Scenario {} - Not all indicators have converged at end of day {}, next day is initialised.'.format(scn_name, day))
            converged = False
    else:
        print('Scenario {} - Initialisation period, simulation can not yet converge at end of day {}, next day is initialised.'.format(scn_name, day))
        converged = False
        
    return converged


def save_random_states(result_path):
    # Record and store the random state
    random_state = random.getstate()
    with open(os.path.join(result_path,"random_state.pkl"), "wb") as file:
        pickle.dump(random_state, file)
    with open(os.path.join(result_path,"np_random_state.bin"), 'wb') as f:
        pickle.dump(np.random.get_state(), f)

    return 0


def set_random_states(result_path):
    # Load and set the state from a file using pickle
    with open(os.path.join(result_path,"random_state.pkl"), "rb") as file:
        random_state = pickle.load(file)
    random.setstate(random_state)
    with open(os.path.join(result_path,"np_random_state.bin"), 'rb') as f:
        np.random.set_state(pickle.load(f))

    return 0