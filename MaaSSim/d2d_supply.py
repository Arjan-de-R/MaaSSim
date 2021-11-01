from MaaSSim.driver import driverEvent
from MaaSSim.utils import generate_vehicles
import pandas as pd
import numpy as np
import math
import random
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

    # np.random.seed(_params.repl_id)
    vehs = generate_vehicles(_inData, _params.nV)

    vehs.expected_income = np.nan
    lognorm_std = 2 * erfinv(_params.evol.drivers.gini)
    lognorm_mean = np.log(_params.evol.drivers.res_wage.mean) - (lognorm_std ** 2) / 2

    vehs['res_wage'] = np.random.lognormal(lognorm_mean, lognorm_std, _params.nV) * _params.simTime
    vehs['informed'] = (np.random.rand(_params.nV) < _params.evol.drivers.inform.prob_start)
    vehs['registered'] = (np.random.rand(_params.nV) < _params.evol.drivers.regist.prob_start) & vehs.informed
    vehs.loc[vehs.registered, "expected_income"] = _params.evol.drivers.init_inc_ratio * vehs.res_wage.mean()
    vehs['work_exp'] = np.nan
    vehs.loc[vehs.registered, "work_exp"] = 0

    return vehs

# Evaluation
def D2D_veh_exp(*args,**kwargs):
    #calculate vehicle KPIs (global and individual)
    sim =  kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    vehindex = sim.inData.vehicles.index
    params = sim.params
    df = simrun['rides'].copy() #results of previous simulation
    DECIDES_NOT_TO_DRIVE = df[df.event == driverEvent.DECIDES_NOT_TO_DRIVE.name].veh # track drivers out
    dfs = df.shift(-1) # to map time periods between events
    dfs.columns = [_+"_s" for _ in df.columns] #columns with _s are shifted
    df = pd.concat([df,dfs],axis=1) # now we have time periods
    df = df[df.veh == df.veh_s] #filter for the same vehicles only
    df=df[~(df.t == df.t_s)] # filter for positive time periods only
    df['dt'] = df.t_s - df.t # make time intervals
    ret = df.groupby(['veh','event'])['dt'].sum().unstack() #aggregated by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex) #update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(
        ['veh']).size().reindex(ret.index)
    ret['nREJECTED'] = df[df.event == driverEvent.IS_REJECTED_BY_TRAVELLER.name].groupby(
        ['veh']).size().reindex(ret.index)
    
    dfd = df.loc[df.event == 'DEPARTS_FROM_PICKUP']
    dfd['fare'] = (dfd.dt * (params.speeds.ride/1000) * params.platforms.fare).add(params.platforms.base_fare)
    dfd['min_fare'] = params.platforms.min_fare
    dfd['revenue'] = (dfd[['fare','min_fare']].max(axis=1) * (1-params.platforms.comm_rate))
    
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name]=0 #cover all statuss
    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret = ret[['nRIDES','nREJECTED','OUT']+[_.name for _ in driverEvent]].fillna(0) #nans become 0
    
    ret['DRIVING_TIME'] = ret.REJECTS_REQUEST + ret.IS_ACCEPTED_BY_TRAVELLER + ret.DEPARTS_FROM_PICKUP
    ret['DRIVING_DIST'] = ret['DRIVING_TIME'] * (params.speeds.ride/1000)
    ret['REVENUE'] = dfd.groupby(['veh'])['revenue'].sum().reindex(ret.index).fillna(0)
    ret['COST'] = ret['DRIVING_DIST'] * (params.drivers.fuel_costs)
    ret['NET_INCOME'] = ret['REVENUE'] - ret['COST']
    ret = ret[['nRIDES','nREJECTED', 'DRIVING_TIME', 'DRIVING_DIST', 'REVENUE', 'COST', 'NET_INCOME', 'OUT']+[_.name for _ in driverEvent]]
    ret.index.name = 'veh'
    
    #KPIs
    kpi = ret.agg(['sum','mean','std'])
    kpi['nV']=ret.shape[0]
    return {'veh_exp':ret,'veh_kpi':kpi}

def update_d2d_drivers(*args, **kwargs):
    "updating drivers' day experience and determination of new perceived income"
    sim = kwargs.get('sim',None)
    params = kwargs.get('params',None)
    # run_id = len(sim.res)-1

    ret = pd.DataFrame()
    ret['veh'] = np.arange(1,params.nV+1)
    ret['pos'] = sim.vehicles.pos.to_numpy()
    ret['informed'] = sim.vehicles.informed.to_numpy()
    ret['registered'] = sim.vehicles.registered.to_numpy()
    ret['out'] = sim.last_res.veh_exp.OUT.to_numpy()
    ret['init_perc_inc'] = sim.vehicles.expected_income.to_numpy()
    ret['exp_inc'] = sim.last_res.veh_exp.NET_INCOME.to_numpy()
    ret.loc[ret.out, 'exp_inc'] = np.nan
    new_perc_inc = learning_drivers(params = params, prev_perc = ret.init_perc_inc, exp = ret.exp_inc.fillna(0), out = ret.out)
    
    ret['new_perc_inc'] = new_perc_inc.to_numpy()
    cols = list(ret.columns)
    ret = ret[cols]
    ret = ret.set_index('veh')

    return  ret

# Driver decisions
def wom_driver(inData, **kwargs):
    "determine which drivers are informed before the start of the new day"
    params = kwargs.get('params', None)
    nV_inf = inData.vehicles.informed.sum()
    nV_uninf = params.nV - nV_inf
    if nV_uninf > 0:
        exp_inf_day = (params.evol.drivers.inform.beta * nV_inf * nV_uninf) / params.nV
        prob_inf = exp_inf_day / nV_uninf
    else:
        prob_inf = 0

    new_inf = np.random.rand(params.nV) < prob_inf
    prev_inf = inData.vehicles.informed.to_numpy()
    informed = (np.concatenate(([prev_inf],[new_inf]),axis=0).transpose()).any(axis=1)
    res_inf = pd.DataFrame(data={'informed': informed}, index=np.arange(1,params.nV+1))

    return res_inf


def learning_unregist(inData, end_day, **kwargs):
    "determine new perceived income of informed, yet unregistered drivers, based on signal with noise"
    params = kwargs.get('params', None)
    exp_reg_drivers = end_day[end_day.registered]
    average_xp_income = exp_reg_drivers.exp_inc.mean()
    signal = np.random.normal(average_xp_income,params.evol.drivers.inform.signal_rel_std * average_xp_income,len(inData.vehicles))
    cond_new_inf = inData.vehicles.informed & ~inData.vehicles.registered & end_day.new_perc_inc.isna()
    cond_prev_inf = inData.vehicles.informed & ~inData.vehicles.registered & ~end_day.new_perc_inc.isna()
    new_perc_inc = end_day.new_perc_inc * (1-params.evol.drivers.kappa) + signal * params.evol.drivers.kappa

    df = pd.DataFrame(data={'expected_income': end_day.new_perc_inc, 'signal': signal, 'perc_inc': new_perc_inc, 'cond_new_inf': cond_new_inf, 'cond_prev_inf': cond_prev_inf}, index=np.arange(1,params.nV+1))
    df['expected_income'] = df['expected_income'].where(~df.cond_new_inf, df['signal'])
    df['expected_income'] = df['expected_income'].where(~df.cond_prev_inf, df['perc_inc'])

    return df


def platform_regist(inData, end_day, **kwargs):
    "determine probability of registering at platform overnight for all unregistered drivers"
    params = kwargs.get('params', None)

    regist_df = pd.DataFrame(data={'inform': inData.vehicles.informed, 'prev_regist': end_day.registered, 'work_exp': inData.vehicles.work_exp, 'expected_income': end_day.new_perc_inc},index=np.arange(1,len(inData.vehicles)+1))
    regist_df['decis'] = pd.Series(np.random.rand(params.nV) <= params.evol.drivers.regist.samp) # Sample of drivers making (de)registration decision
    regist_df.loc[~regist_df.inform, 'decis'] = False

    # Probability to participate
    util_ptcp = params.evol.drivers.particip.beta * regist_df.expected_income.to_numpy()
    util_no_ptcp = params.evol.drivers.particip.beta * inData.vehicles.res_wage.to_numpy()
    prob_d_ptcp = np.exp(util_ptcp) / (np.exp(util_ptcp) + np.exp(util_no_ptcp))

    util_reg = params.evol.drivers.regist.beta * regist_df.expected_income.to_numpy() * prob_d_ptcp
    util_not_reg = params.evol.drivers.regist.beta * (
                inData.vehicles.res_wage.to_numpy() * prob_d_ptcp + params.evol.drivers.regist.cost_comp)
    prob_regist_util = np.exp(util_reg) / (np.exp(util_reg) + np.exp(util_not_reg))
    satisfied = np.random.rand(params.nV) < prob_regist_util
    regist_df['regist_decision'] = satisfied & regist_df.decis
    regist_df['deregist_decision'] = ~satisfied & regist_df.decis & (regist_df.work_exp >= 5)

    prev_regist = inData.vehicles.registered.to_numpy()
    still_regist = prev_regist * (~regist_df.deregist_decision)
    registered = (np.concatenate(([still_regist], [regist_df.regist_decision]), axis=0).transpose()).any(axis=1)

    regist_df['work_exp'][regist_df.deregist_decision] = np.nan
    regist_df['work_exp'][regist_df.regist_decision] = 0
    regist_res = pd.DataFrame(data={'registered': registered, 'work_exp': regist_df.work_exp}, index=np.arange(1,len(inData.vehicles)+1))
    
    return regist_res


def D2D_driver_out(*args, **kwargs):
    """ returns True if driver decides not to drive, and False if he drives"""
    veh = kwargs.get('veh',None)

    perc_income = veh.veh.expected_income
    if ~veh.veh.registered:
        return True
    if veh.sim.params.evol.drivers.particip.probabilistic:
        util_d = veh.sim.params.evol.drivers.particip.beta * perc_income
        util_nd = veh.sim.params.evol.drivers.particip.beta * veh.veh.res_wage
        prob_d_reg = math.exp(util_d) / (math.exp(util_d) + math.exp(util_nd))
        prob_d_all = prob_d_reg
        return bool(prob_d_all < random.random())
    return bool(perc_income < veh.veh.res_wage)


def learning_drivers(params, out, prev_perc, exp):
    "returns new perceived income of group of drivers"
    kappa = params.evol.drivers.kappa * (1 - out)
    new_perc = (1 - kappa) * prev_perc + kappa * exp
    
    return new_perc


def update_work_exp(inData, end_day):
    df = inData.vehicles.copy()
    df['out'] = end_day.out
    df.loc[~df.out, "work_exp"] = df.work_exp + 1

    return df
