from MaaSSim.src_MaaSSim.driver import driverEvent
from MaaSSim.src_MaaSSim.utils import generate_vehicles
from source.d2d.supply import set_multihoming_drivers, sh_init_reg
from source.d2d.utils import zero_to_nan
import pandas as pd
import numpy as np
# import math
import random
import math
from scipy.special import erfinv


# Generation
def generate_vehicles_d2d(_inData, _params=None):
    """
    generates vehicle database
    index is consecutive number if dataframe
    registered whether drivers have already made sign up decision
    position is random graph node
    event is set to STARTS_DAY
    """

    vehs = generate_vehicles(_inData, _params.nV)

    lognorm_std = 2 * erfinv(_params.evol.drivers.gini)
    lognorm_mean = np.log(_params.evol.drivers.res_wage.mean) - (lognorm_std ** 2) / 2
    vehs['res_wage'] = np.random.lognormal(lognorm_mean, lognorm_std, _params.nV) * _params.simTime

    # Set multi-homing behaviour
    vehs = set_multihoming_drivers(vehs, _params)
    
    # Determine which platforms job seekers are informed about
    rand_informed = np.random.rand(_params.nV) < _params.evol.drivers.inform.prob_start
    vehs['informed'] = rand_informed
    vehs['registered'] = (np.random.rand(_params.nV) < _params.evol.drivers.regist.prob_start) * rand_informed
    vehs['registered'] = vehs.apply(lambda row: (row.registered * np.full(len(_inData.platforms.index),True)) if row.multihoming else sh_init_reg(row.registered * np.full(len(_inData.platforms.index),True)), axis=1) # multi-homers, else single-homers
    if _params.evol.drivers.get('start_perc_inc_avg_ratio', False):
        vehs['expected_income'] = vehs.apply(lambda row: zero_to_nan(row.registered * _params.evol.drivers.start_perc_inc_avg_ratio * vehs.res_wage.mean()), axis=1)  # everyone expects (factor of) mean reservation wage in population
    else:
        vehs['expected_income'] = vehs.apply(lambda row: zero_to_nan(row.registered * _params.evol.drivers.init_inc_ratio * row.res_wage), axis=1) # expect (factor of) own res wage
    vehs['rejected_reg'] = False
    vehs['days_since_reg'] = vehs.apply(lambda row: _params.evol.drivers.regist.min_days if row.registered.sum() > 0 else np.nan, axis=1)
    vehs['work_exp'] = vehs.apply(lambda row: 0 if row.registered.sum() > 0 else np.nan, axis=1)

    return vehs


# Evaluation
def D2D_veh_exp(*args, **kwargs):
    # calculate vehicle KPIs (global and individual)
    sim = kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    vehindex = sim.inData.vehicles.index
    params = sim.params
    df = simrun['rides'].copy()  # results of previous simulation
    DECIDES_NOT_TO_DRIVE = df[df.event == driverEvent.DECIDES_NOT_TO_DRIVE.name].veh  # track drivers out
    NOT_ALLOWED_TO_DRIVE = df[df.event == driverEvent.NOT_ALLOWED_TO_DRIVE.name].veh  # track drivers rejected
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.veh == df.veh_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['veh', 'event'])['dt'].sum().unstack()  # aggregated by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex)  # update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(
        ['veh']).size().reindex(ret.index)
    ret['nREJECTED'] = df[df.event == driverEvent.IS_REJECTED_BY_TRAVELLER.name].groupby(
        ['veh']).size().reindex(ret.index)

    dfd = df.loc[df.event == 'DEPARTS_FROM_PICKUP']
    dfd['fare'] = (dfd.dt * (params.speeds.ride / 1000) * params.platforms.fare).add(params.platforms.base_fare)
    dfd['min_fare'] = params.platforms.min_fare
    dfd['revenue'] = (dfd[['fare', 'min_fare']].max(axis=1) * (1 - params.platforms.comm_rate))

    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuss
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    NOT_ALLOWED_TO_DRIVE.index = NOT_ALLOWED_TO_DRIVE.values
    ret['DECIDED_OUT'] = DECIDES_NOT_TO_DRIVE
    ret['FORCED_OUT'] = NOT_ALLOWED_TO_DRIVE
    ret['DECIDED_OUT'] = ~ret['DECIDED_OUT'].isnull()
    ret['FORCED_OUT'] = ~ret['FORCED_OUT'].isnull()
    ret['OUT'] = ret[['FORCED_OUT', 'DECIDED_OUT']].any(axis=1)
    ret = ret[['nRIDES', 'nREJECTED', 'OUT', 'FORCED_OUT'] + [_.name for _ in driverEvent]].fillna(0)  # nans become 0

    ret['DRIVING_TIME'] = ret.REJECTS_REQUEST + ret.IS_ACCEPTED_BY_TRAVELLER + ret.DEPARTS_FROM_PICKUP
    ret['DRIVING_DIST'] = ret['DRIVING_TIME'] * (params.speeds.ride / 1000)
    ret['REVENUE'] = dfd.groupby(['veh'])['revenue'].sum().reindex(ret.index).fillna(0)
    ret['COST'] = ret['DRIVING_DIST'] * params.drivers.fuel_costs
    ret['NET_INCOME'] = ret['REVENUE'] - ret['COST']
    ret = ret[
        ['nRIDES', 'nREJECTED', 'DRIVING_TIME', 'DRIVING_DIST', 'REVENUE', 'COST', 'NET_INCOME', 'OUT', 'FORCED_OUT'] + [_.name for _
                                                                                                           in
                                                                                                           driverEvent]]
    ret.index.name = 'veh'

    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    return {'veh_exp': ret, 'veh_kpi': kpi}


def update_d2d_drivers(*args, **kwargs):
    "updating drivers' day experience and determination of new perceived income"
    sim = kwargs.get('sim', None)
    params = kwargs.get('params', None)
    # run_id = len(sim.res)-1

    ret = pd.DataFrame()
    ret['veh'] = np.arange(1, params.nV + 1)
    ret['pos'] = sim.vehicles.pos.to_numpy()
    ret['informed'] = sim.vehicles.informed.to_numpy()
    ret['registered'] = sim.vehicles.registered.to_numpy()
    ret['out'] = sim.last_res.veh_exp.OUT.to_numpy()
    ret['init_perc_inc'] = sim.vehicles.expected_income.to_numpy()
    ret['exp_inc'] = sim.last_res.veh_exp.NET_INCOME.to_numpy()
    ret['forced_out'] = sim.last_res.veh_exp.FORCED_OUT.to_numpy()
    ret['rejected_reg'] = sim.vehicles.rejected_reg.to_numpy()
    new_perc_inc = learning_drivers(ret, params=params)
    ret['new_perc_inc'] = new_perc_inc.to_numpy()
    ret['pickup_dist'] = sim.last_res.veh_exp.pickup_dist.to_numpy()
    ret['repos_dist'] = sim.last_res.veh_exp.repos_dist.to_numpy()
    for col in sim.last_res.veh_exp.columns:
        if col.startswith("km occ"):
            ret[col.replace(" ","_")] = sim.last_res.veh_exp[col].to_numpy()
    cols = list(ret.columns)
    ret = ret[cols]
    ret = ret.set_index('veh')

    return ret


# Driver decisions
def wom_driver(inData, **kwargs):
    "determine which drivers are informed before the start of the new day"
    params = kwargs.get('params', None)

    total_particip = inData.vehicles.ptcp.apply(lambda row: row.sum() > 0).sum()

    # Probability to be informed
    prob_inf_plf = params.evol.drivers.inform.beta * total_particip / len(inData.vehicles.index)
    rand_draw = np.random.sample(len(inData.vehicles.index))
    informed = rand_draw < prob_inf_plf
    informed = informed + inData.vehicles.informed.values # any of these two conditions is satisfied, either previously or newly informed
    res_inf = pd.DataFrame(data={'informed': informed}, index=np.arange(1, params.nV + 1))

    return res_inf


def learning_unregist(inData, end_day, **kwargs):
    "determine new perceived income of informed, yet unregistered drivers, based on signal with noise"
    params = kwargs.get('params', None)
    exp_reg_drivers = end_day[end_day.registered]
    average_xp_income = exp_reg_drivers.exp_inc.mean()
    std_xp_income = exp_reg_drivers.exp_inc.std()

    cond_new_inf = inData.vehicles.informed & ~inData.vehicles.registered & end_day.new_perc_inc.isna()
    cond_prev_inf = inData.vehicles.informed & ~inData.vehicles.registered & ~end_day.new_perc_inc.isna()
    df = pd.DataFrame(data={'expected_income': end_day.new_perc_inc, 'cond_new_inf': cond_new_inf,
                            'cond_prev_inf': cond_prev_inf}, index=np.arange(1, params.nV + 1))

    if (~end_day.out).any(axis=0):  # at least a single participating driver
        if (~end_day.out).sum() == 1:
            std_xp_income = exp_reg_drivers.exp_inc.std(ddof=0)
        df['signal'] = np.random.normal(average_xp_income, params.evol.drivers.inform.std_fact * std_xp_income, len(inData.vehicles))
        df['perc_inc'] = end_day.new_perc_inc * (1 - params.evol.drivers.kappa) + df.signal * params.evol.drivers.kappa
    else:
        df['signal'] = np.ones(len(inData.vehicles)) * end_day.new_perc_inc.mean()
        df['perc_inc'] = end_day.new_perc_inc
    df['expected_income'] = df['expected_income'].where(~df.cond_new_inf, df.signal)
    df['expected_income'] = df['expected_income'].where(~df.cond_prev_inf, df['perc_inc'])

    return df.expected_income


def platform_regist_driver(inData, end_day, **kwargs):
    "determine probability of registering at platform overnight for all unregistered drivers"
    params = kwargs.get('params', None)
    
    def signal_mh(params, avg_perc_earnings_mh, std_perc_earnings_mh):
        rand_signal = np.random.normal(avg_perc_earnings_mh, params.evol.drivers.inform.std_fact * std_perc_earnings_mh, size=params.nV)
        # rand_signal = np.where(np.isnan(rand_signal), params.evol.drivers.start_perc_inc_avg_ratio * inData.vehicles.res_wage.mean(), rand_signal) # if no multihomer is registered, take average of res wage
        return rand_signal

    def signal_plf(params, avg_perc_earnings_plf, std_perc_earnings_plf):
        signal_list = []
        for plf in range(len(avg_perc_earnings_plf)):
            rand_signal = np.random.normal(avg_perc_earnings_plf[plf], params.evol.drivers.inform.std_fact * std_perc_earnings_plf[plf])
            signal_list.append(rand_signal)
        # signal_list = [params.evol.drivers.start_perc_inc_avg_ratio * inData.vehicles.res_wage.mean() if math.isnan(x) else x for x in signal_list]
        return signal_list
    
    def regist_plf(inData, row):
        '''returns boolean array with each item indicating whether you are registered with that platform after today'''
        reg_arr = row.prev_regist.copy() # if not making a registration decision
        if row.decis: # making a decision
            if not np.any(row.prev_regist): # not previously registered with any platform
                if row.satisfied: # satisfied with at least one platform - want to be registered
                    if row.multihoming: 
                        reg_arr = np.full(len(inData.platforms.index), True)
                    else:
                        reg_arr[row.chosen_plf_index] = True
            elif (row.work_exp >= params.evol.drivers.regist.min_work_exp) and (row.days_since_reg >= params.evol.drivers.regist.min_days): # enough experience to change their decision
                if not row.satisfied:
                    reg_arr = np.full(len(inData.platforms.index), False) # deregister with all
                elif not row.multihoming: # satisfied and singlehoming
                    reg_arr = np.full(len(inData.platforms.index), False)
                    reg_arr[row.chosen_plf_index] = True 
        return reg_arr
    
    def return_days(reg_outcome, days):
        if 1 in reg_outcome: # signed up today with at least one, number of days since reg is 0
            days = 0
        elif not np.any(reg_outcome): # all zeros (no change in registration), take previous
            days = days
        else: # deregistered with at least 1 today and did not register with another, returns np.nan
            days = np.nan
        return days
    
    def new_perc_after_communication_driver(row):
        # First replace NaN signals by previous expected income
        if row.multihoming:
            if math.isnan(row.relevant_signal):
                row.relevant_signal = row.expected_income[0]
        else:
            nan_mask = np.isnan(row.relevant_signal)
            row.relevant_signal = np.where(nan_mask, row.expected_income, row.relevant_signal)
        # Now determine new expected income
        if not row.decis or np.all(row.prev_regist): # not making a decision today or already registered with all platforms
            new_perc_inc = row.expected_income
        elif row.multihoming and not np.any(~np.isnan(row.expected_income)): # multihoming and no previous expectation
            new_perc_inc = row.relevant_signal * np.ones(len(row.prev_regist))
        else:
            if row.multihoming: # multihoming and you have a previous expectation
                kappa_comm = params.evol.drivers.kappa_comm
            else: # singlehoming
                kappa_comm = np.nan_to_num(np.isnan(row.expected_income) * ~row.prev_regist) * 1 + np.nan_to_num(params.evol.drivers.kappa_comm * ~np.isnan(row.expected_income) * ~row.prev_regist)
            new_perc_inc = row.relevant_signal * kappa_comm + np.nan_to_num(row.expected_income) * (1 - kappa_comm)
        return new_perc_inc
    

    if end_day.forced_out.sum() == 0:   # all willing drivers were allowed to participate on previous day
        prob_ptcp_rejected = 0
    else:
        prob_ptcp_rejected = end_day.forced_out.sum() / (end_day.forced_out.sum() + (~end_day.out).sum())

    regist_df = pd.DataFrame(data={'inform': inData.vehicles.informed, 'prev_regist': end_day.registered,
                                   'work_exp': inData.vehicles.work_exp,
                                   'days_since_reg': inData.vehicles.days_since_reg,
                                   'expected_income': end_day.new_perc_inc,
                                   'prob_ptcp_rejected': prob_ptcp_rejected, 'multihoming': inData.vehicles.multihoming},
                             index=np.arange(1, len(inData.vehicles) + 1))
    regist_df['days_since_reg'] = regist_df['days_since_reg'] + 1
    regist_df['decis'] = pd.Series(np.random.rand(params.nV) <= params.evol.drivers.regist.samp, index=np.arange(1, len(inData.vehicles) + 1))  # Sample of drivers making (de)registration decision
    
    # only informed agents can make regist decision
    regist_df.loc[~regist_df.inform, 'decis'] = False

    ### If a job seeker is currently unregistered, he seeks information about platform earnings, which he receives with noise
    ## Multi-homer: interested in multi-homing earnings (of reg. job seekers) only
    regist_df['reg_any'] = regist_df.apply(lambda row: row.prev_regist.sum() > 0, axis=1)
    regist_df['perc_inc_rel'] = regist_df.apply(lambda row: np.nanmean(row.expected_income) if row.reg_any else np.nan, axis=1) # relevant perc inc for learning
    avg_perc_earnings_mh = regist_df.loc[regist_df.multihoming].perc_inc_rel.mean()
    # print('driver mh perc kpi: {}'.format(avg_perc_earnings_mh))
    std_perc_earnings_mh = regist_df.loc[regist_df.multihoming].perc_inc_rel.std(ddof=0)
    regist_df['perc_inc_reg'] = regist_df.apply(lambda row: np.where(row.prev_regist == False, np.nan, row.prev_regist * row.expected_income), axis=1)
    if regist_df.multihoming.all(): # only multihomers
        avg_perc_earnings_plf = np.ones(inData.platforms.shape[0]) * np.nan
        std_perc_earnings_plf = np.ones(inData.platforms.shape[0]) * np.nan
    else:
        avg_perc_earnings_plf = np.nanmean(regist_df.loc[~regist_df.multihoming].perc_inc_reg.to_list(), axis=0) # list with average perceived earnings per platform
        # print('driver plf perc kpi: {}'.format(avg_perc_earnings_plf))
        std_perc_earnings_plf = np.nanstd(regist_df.loc[~regist_df.multihoming].perc_inc_reg.to_list(), axis=0, ddof=0)
    
    regist_df['signal_mh'] = signal_mh(params, avg_perc_earnings_mh, std_perc_earnings_mh)
    regist_df['signal_plf'] = regist_df.apply(lambda _: signal_plf(params, avg_perc_earnings_plf, std_perc_earnings_plf), axis=1)
    regist_df['relevant_signal'] = regist_df.apply(lambda row: row.signal_mh if row.multihoming else row.signal_plf, axis=1)
    regist_df['new_perc_inc'] = regist_df.apply(lambda row: new_perc_after_communication_driver(row), axis=1)
    # First: determine probability to participate (when registered) depending on expected earnings
    regist_df['res_wage'] = inData.vehicles.res_wage
    regist_df['util_ptcp'] = regist_df.apply(lambda row: params.evol.drivers.particip.beta * row.new_perc_inc, axis=1)
    regist_df['util_no_ptcp'] = regist_df.apply(lambda row: params.evol.drivers.particip.beta * row.res_wage, axis=1)
    regist_df['prob_d_ptcp'] = regist_df.apply(lambda row: np.exp(row.util_ptcp) / (np.exp(row.util_ptcp) + np.exp(row.util_no_ptcp)), axis=1)
    regist_df['util_reg_plf'] = regist_df.apply(lambda row: params.evol.drivers.regist.beta * row.new_perc_inc * row.prob_d_ptcp * (1-row.prob_ptcp_rejected), axis=1)
    
    regist_df['prob_plf'] = regist_df.apply(lambda row: np.array([(np.exp(row.util_reg_plf[plf]) / np.exp(row.util_reg_plf).sum()) for plf in inData.platforms.index]), axis=1)
    regist_df['chosen_plf_index'] = regist_df.apply(lambda row: np.random.choice(len(row.prob_plf), p=np.nan_to_num(row.prob_plf)) if (np.nan_to_num(row.prob_plf).sum() != 0) else 0, axis=1)

    # Now decide between ridesourcing or other activity
    regist_df['util_not_reg'] = regist_df.apply(lambda row: params.evol.drivers.regist.beta * (row.res_wage * row.prob_d_ptcp[row.chosen_plf_index] + params.evol.drivers.regist.cost_comp), axis=1)
    regist_df['prob_regist_util'] = regist_df.apply(lambda row: np.exp(row.util_reg_plf[row.chosen_plf_index]) / (np.exp(row.util_reg_plf[row.chosen_plf_index]) + np.exp(row.util_not_reg)), axis=1)
    regist_df['satisfied'] = np.random.rand(params.nV) < regist_df.prob_regist_util

    regist_df['regist_plf'] = regist_df.apply(lambda row: regist_plf(inData, row), axis=1)
    regist_df['reg_outcome'] = regist_df.apply(lambda row: np.where(row.regist_plf & ~row.prev_regist, 1, np.where(row.regist_plf & row.prev_regist, 0, np.where(~row.regist_plf & ~row.prev_regist, 0, -1))), axis=1) # 1: newly registered with plf, 0: same status as before, -1 deregistered with plf
    
    if params.platforms.reg_cap: # TODO: regist cap per platform
        max_entrants = params.platforms.reg_cap - still_regist.sum()
        regist_df['lot_ticket'] = np.random.rand(params.nV) * regist_df.regist_decision
        regist_df['lot_rank'] = regist_df.lot_ticket.rank(method='first', ascending=False)
        regist_df['entry_rejected'] = regist_df.lot_rank > max_entrants
        regist_df['result'] = regist_df.regist_decision * (~regist_df.entry_rejected)
        regist_df['rejected'] = regist_df[['regist_decision', 'entry_rejected']].all(axis=1)
    else:
        regist_df['rejected'] = False

    regist_df['work_exp'] = regist_df.apply(lambda row: return_days(row.reg_outcome, row.work_exp), axis=1)
    regist_df['days_since_reg'] = regist_df.apply(lambda row: return_days(row.reg_outcome, row.days_since_reg), axis=1)
    
    # return inData
    inData.vehicles.registered = regist_df.regist_plf
    inData.vehicles.work_exp = regist_df.work_exp
    inData.vehicles.days_since_reg = regist_df.days_since_reg
    inData.vehicles.expected_income = regist_df.new_perc_inc
    inData.vehicles.rejected_reg = regist_df.rejected

    return inData.vehicles


def D2D_driver_out(*args, **kwargs):
    """ returns True if driver decides not to drive, and False if he drives"""
    veh = kwargs.get('veh', None)

    perc_income = veh.veh.expected_income
    if ~veh.veh.registered:
        return True
    if veh.sim.params.evol.drivers.particip.probabilistic:
        util_d = veh.sim.params.evol.drivers.particip.beta * perc_income
        util_nd = veh.sim.params.evol.drivers.particip.beta * veh.veh.res_wage
        prob_d_reg = np.exp(util_d) / (np.exp(util_d) + np.exp(util_nd))
        prob_d_all = prob_d_reg
        return bool(prob_d_all < random.random())
    return bool(perc_income < veh.veh.res_wage)


def learning_drivers(df_exp, params):
    "returns new perceived income of group of drivers"
    df = df_exp[['init_perc_inc','out','exp_inc']]
    df['kappa'] = df.apply(lambda row: np.nan_to_num(~row.out * params.evol.drivers.kappa), axis=1)
    df['new_perc_inc'] = df.apply(lambda row: np.nan_to_num(row.kappa * row.exp_inc) + ((1-row.kappa) * row.init_perc_inc), axis=1)

    return df['new_perc_inc']


def update_work_exp(inData, end_day):
    df = inData.vehicles.copy()
    df['out'] = end_day.out
    df['work_exp'] = df.apply(lambda x: x.work_exp + 1 if (~x.out).sum() > 0 else x.work_exp, axis=1)

    return df


def learning_unregist(inData, end_day, **kwargs):
    "determine new perceived platform earnings of newly informed, yet unregistered drivers, based on signal with noise"
    params = kwargs.get('params', None)
    exp_reg_drivers = end_day[end_day.registered]
    average_xp_income = exp_reg_drivers.exp_inc.mean()
    std_xp_income = exp_reg_drivers.exp_inc.std()

    cond_new_inf = inData.vehicles.informed & ~inData.vehicles.registered & end_day.new_perc_inc.isna()
    cond_prev_inf = inData.vehicles.informed & ~inData.vehicles.registered & ~end_day.new_perc_inc.isna()
    df = pd.DataFrame(data={'expected_income': end_day.new_perc_inc, 'cond_new_inf': cond_new_inf,
                            'cond_prev_inf': cond_prev_inf}, index=np.arange(1, params.nV + 1))

    if (~end_day.out).any(axis=0):  # at least a single participating driver
        if (~end_day.out).sum() == 1:
            std_xp_income = exp_reg_drivers.exp_inc.std(ddof=0)
        df['signal'] = np.random.normal(average_xp_income, params.evol.drivers.inform.std_fact * std_xp_income, len(inData.vehicles))
        df['perc_inc'] = end_day.new_perc_inc * (1 - params.evol.drivers.kappa) + df.signal * params.evol.drivers.kappa
    else:
        df['signal'] = np.ones(len(inData.vehicles)) * end_day.new_perc_inc.mean()
        df['perc_inc'] = end_day.new_perc_inc
    df['expected_income'] = df['expected_income'].where(~df.cond_new_inf, df.signal)
    df['expected_income'] = df['expected_income'].where(~df.cond_prev_inf, df['perc_inc'])

    return df.expected_income