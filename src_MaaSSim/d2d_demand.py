from src_MaaSSim.traveller import travellerEvent
import pandas as pd
import numpy as np
import math
import os
from scipy.special import erfinv
import osmnx as ox
from source.d2d.utils import *


def load_albatross_proc(_inData, _params, avg_speed=False):
    # loads the full csv of albatross for a given city
    # changes date for today
    # filters for simulation time (t0 hour + simTime)

    df = pd.read_csv(os.path.join(_params.paths.albatross,
                                  _params.city.split(",")[0] + "_requests_proc.csv"),
                     index_col= 'Unnamed: 0')
    df.rename(columns={'Unnamed: 0': 'pax_id'}, inplace=True)

    df['treq'] = pd.to_datetime(df['treq'])
    df['tarr'] = pd.to_datetime(df['tarr'])
    df['ttrav_alb'] = pd.to_timedelta(df.ttrav)
    df['ttrav'] = df.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    if avg_speed:
        df.ttrav = (pd.to_timedelta(df.ttrav) / _params.speeds.ride).dt.floor('1s')

    missing_col = list(set(_inData.requests.columns.values.tolist()).difference(df.columns.values.tolist()))
    df = df.reindex(columns=df.columns.tolist() + missing_col)
    df.pax_id = df.index
    df.schedule_id = df.index
    df.shareable = False

    _inData.requests = df

    _inData.passengers.pos = _inData.requests.origin
    _inData.passengers.event = travellerEvent.STARTS_DAY
    _inData.passengers.platforms = _inData.passengers.platforms.apply(lambda x: [0])

    return _inData


def load_OTP_result(params):
    # loads the attributes of the recommended PT initeraries

    df = pd.read_csv(params.paths.PT_trips, index_col='id')
    df.index.name = 'pax_id'

    # Determine distance for all PT legs
    # Split mode information in separate legs
    legs = df['modes'].str.replace(r'[','')
    legs = legs.str.split(']', expand=True)
    for column in range(len(legs.columns)):
        legs[column] = legs[column].str.lstrip(', ')  # Remove leading commas
        # Set PT distance for walk segments and empty segments to zero (not part of fare calculation), and extract distance for PT legs
        legs[column].fillna('', inplace=True)
        legs[column] = np.where(((legs[column].str.contains('WALK')==True) | (legs[column] == '')), 0, legs[column].str.split(',').str[2])
        legs[column] = legs[column].astype(int)
    # Calculate total PT distance
    legs['PTdistance'] = legs.sum(axis=1, skipna=True)
    df = df.merge(legs['PTdistance'], how='left', left_index=True, right_index=True)
    # Calculate fare
    df['PTfare'] = round((df['PTdistance'] * (1/1000) * params.alt_modes.pt.km_fare) + params.alt_modes.pt.base_fare,2)
    df.loc[(df['PTdistance'] == 0), 'PTfare'] = 9999  # If only walking is used, set PT fare to zero
    
    return df


def d2d_kpi_pax(*args ,**kwargs):
    # calculate passenger indicators (global and individual)

    sim = kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    paxindex = sim.inData.passengers.index

    df = simrun['trips'].copy()  # results of previous simulation
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.pax == df.pax_s]  # filter for the same vehicles only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['pax', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event

    ret.columns.name = None
    ret = ret.reindex(paxindex)  # update for vehicles with no record

    if 'PREFERS_OTHER_SERVICE' in ret.columns:
        ret['NO_REQUEST'] = ~ret.PREFERS_OTHER_SERVICE.isna()
    else:
        ret['NO_REQUEST'] = False

    if 'REJECTS_OFFER' in ret.columns:
        ret['OTHER_MODE'] = ~ret.REJECTS_OFFER.isna()
    else:
        ret['OTHER_MODE'] = False

    ret.index.name = 'pax'
    ret = ret.fillna(0)

    for status in travellerEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuses

    # meaningful names
    ret['TRAVEL'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['WAIT'] = ret['RECEIVES_OFFER'] + ret[
        'MEETS_DRIVER_AT_PICKUP']  # time waiting for traveller (by default zero)
    ret['OPERATIONS'] = ret['ACCEPTS_OFFER'] + ret['DEPARTS_FROM_PICKUP'] + ret['SETS_OFF_FOR_DEST']

    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nP'] = ret.shape[0]
    return {'pax_exp': ret, 'pax_kpi': kpi}


def offer_accepted(params, row):
    '''determine which platform the traveller accepted an offer from (possibly none), returning a boolean numpy array of length number of platforms'''
    array = np.full(len(params.platforms.service_types), False)
    if not math.isnan(row.platform):
        array[int(row.platform)] = True
    return array


def determine_corr_xp_wait(params, row):
    '''determine experienced waiting considering rejection penalty, and per platform'''
    if np.any(np.logical_and(row.requests[~np.isnan(row.requests)], True)) and not np.any(np.logical_and(row.gets_offer[~np.isnan(row.gets_offer)], True)): # requested but no offer
        corr_xp_wait = params.evol.travellers.reject_penalty * zero_to_nan(row.requests * np.ones(len(row.xp_wait)))
    else:
        corr_xp_wait = row.xp_wait * zero_to_nan(row.requests * np.ones(len(row.xp_wait)))
    return corr_xp_wait


def learning_new_kpi(params, perc, xp, requests, gets_offer):
    '''determine new perceived value for indicator (not waiting time) based on experience'''
    if not np.any(requests):
        kappa_plf = np.full(len(requests), 0)
    elif requests.sum() > 1:  # traveller is multi-homer (and participating in the market)
        kappa_plf = params.evol.travellers.kappa * np.full(len(requests), np.any(gets_offer))
    else: # single-homer traveller participating 
        kappa_plf = params.evol.travellers.kappa * np.nan_to_num(gets_offer)
    new_perc_kpi = (1 - kappa_plf) * perc + np.nan_to_num(kappa_plf * xp)
    return new_perc_kpi


def learning_new_wait(params, perc, xp, requests):
    '''update new perceived waiting time (which is non-zero when ride request was rejected)'''
    kappa_plf = requests * params.evol.travellers.kappa
    new_perc_kpi = (1 - kappa_plf) * perc + np.nan_to_num(kappa_plf * xp)
    return new_perc_kpi


def determine_xp_fare(sim, row):
    if row.accepts_offer.sum() > 0: # traveller was served by one of the platforms
        xp_fare = sim.inData.platforms[row.accepts_offer]['fare'].values * row.requests
    else:
        xp_fare = np.ones(len(row.requests)) * np.nan
    return xp_fare


def update_d2d_travellers(*args, **kwargs):
    "updating travellers' experience and updating new expected waiting time"
    sim = kwargs.get('sim', None)
    params = kwargs.get('params', None)
    pax = kwargs.get('pax', None)

    ret = pd.DataFrame()
    ret['pax'] = np.arange(0, len(sim.passengers))
    ret['orig'] = sim.inData.requests.origin.to_numpy()
    ret['dest'] = sim.inData.requests.destination.to_numpy()
    ret['t_req'] = sim.inData.requests.treq.to_numpy()
    ret['tt_min'] = sim.inData.requests.ttrav.to_numpy()
    ret['dist'] = sim.inData.requests.dist.to_numpy()
    ret['informed'] = sim.passengers.informed.to_numpy()
    ret['registered'] = pax.registered
    ret['requests'] = (pax.mode_day == 'rs').to_numpy()
    ret['requests'] =  ret.apply(lambda row: row.requests * row.registered, axis=1)
    ret['LOSES_PATIENCE'] = sim.last_res.pax_exp.LOSES_PATIENCE
    ret['gets_offer'] = ret.apply(lambda row: ~np.where(row.LOSES_PATIENCE == None, True, row.LOSES_PATIENCE).astype(bool), axis=1)
    ret['gets_offer'] = ret.apply(lambda row: row.gets_offer * row.requests, axis=1)
    ret['accepts_offer'] = sim.last_res.pax_exp.apply(lambda row: offer_accepted(params, row), axis=1)
    ret['xp_wait'] = sim.last_res.pax_exp.WAIT.to_numpy()
    ret['xp_ivt'] = sim.last_res.pax_exp.TRAVEL.to_numpy()
    ret['xp_ops'] = sim.last_res.pax_exp.OPERATIONS.to_numpy()
    ret['xp_tt_total'] = ret.xp_wait + ret.xp_ivt + ret.xp_ops
    ret['xp_ivt'] = ret.apply(lambda row: row.xp_ivt * zero_to_nan(np.ones(len(row.accepts_offer)) * row.requests), axis=1)
    ret['xp_wait'] = ret.apply(lambda row: row.xp_wait * zero_to_nan(np.ones(len(row.accepts_offer)) * row.requests), axis=1)
    ret['xp_ops'] = ret.apply(lambda row: row.xp_ops * zero_to_nan(np.ones(len(row.accepts_offer)) * row.requests), axis=1) 
    ret['xp_tt_total'] = ret.apply(lambda row: row.xp_tt_total * zero_to_nan(np.ones(len(row.accepts_offer)) * row.requests), axis=1)
    ret['xp_km_fare'] = ret.apply(lambda row: zero_to_nan(np.ones(len(row.accepts_offer)) * determine_xp_fare(sim, row)), axis=1)
    ret['init_perc_wait'] = sim.passengers.expected_wait.to_numpy()
    ret['init_perc_ivt'] = sim.passengers.expected_ivt.to_numpy()
    ret['init_perc_km_fare'] = sim.passengers.expected_km_fare.to_numpy()
    ret['corr_xp_wait'] = ret.xp_wait.copy()
    ret['corr_xp_wait'] = ret.apply(lambda row: determine_corr_xp_wait(params, row), axis=1)
    ret['new_perc_wait'] = ret.apply(lambda row: learning_new_wait(params, row.init_perc_wait, row.corr_xp_wait, row.requests), axis=1)
    ret['new_perc_ivt'] = ret.apply(lambda row: learning_new_kpi(params, row.init_perc_ivt, row.xp_ivt, row.requests, row.gets_offer), axis=1)
    ret['new_perc_fare'] = ret.apply(lambda row: learning_new_kpi(params, row.init_perc_km_fare, row.xp_km_fare, row.requests, row.gets_offer), axis=1)
    for col in sim.last_res.pax_exp:
        if col.startswith("time_occ"):
            ret[col] = sim.last_res.pax_exp[col].to_numpy()

    ret['chosen_mode'] = sim.passengers.mode_day.to_numpy()
    ret = ret.set_index('pax')
    ret = ret.drop(columns=['LOSES_PATIENCE'])

    return ret


def d2d_no_request(*args, **kwargs):
    " returns True if traveller does not make RS request, False if he does"
    traveller = kwargs.get('pax',None)

    trav_out = (traveller.pax.mode_day != "rs")

    return trav_out


def wom_trav(inData, end_day, **kwargs):
    params = kwargs.get('params', None)
    "determine which travellers are informed"
    total_particip = inData.passengers[inData.passengers.mode_day == 'rs'].shape[0]
    # Probability to be informed
    prob_inf_plf = params.evol.travellers.inform.beta * total_particip / params.nP
    rand_draw = np.random.sample(len(inData.passengers.index))
    informed = rand_draw < prob_inf_plf
    informed = informed + inData.passengers.informed.values # any of these two conditions is satisfied, either previously or newly informed
    res_inf = pd.DataFrame(data={'informed': informed}, index=np.arange(0, len(inData.passengers.index)))

    return res_inf


def prefs_travs(inData, params):
    "draw mode preferences for the group of travellers"
    prefs = params.evol.travellers.mode_pref
    passengers = inData.passengers

    ASCs, vot, utils = util_alt_modes(inData, params)
    passengers['ASC_bike'] = ASCs.bike
    passengers['ASC_car'] = ASCs.car
    passengers['ASC_pt'] = ASCs.pt

    passengers['VoT'] = vot
    passengers['U_bike'] = utils.bike
    passengers['U_car'] = utils.car
    passengers['U_pt'] = utils.pt
    
    passengers['ASC_rs'] = np.random.normal(prefs.ASC_rs, prefs.ASC_rs_sd,len(inData.passengers))
    passengers['ASC_pool'] = passengers.ASC_rs + np.random.uniform(prefs.min_wts_constant, 0, len(inData.passengers))

    return passengers


def mode_filter(inData, params):
    "mode choice based on no waiting time for RS, used to filter travellers with low probability of using RS"
    passengers = inData.passengers
    utils = pd.DataFrame({'bike': passengers.U_bike, 'car': passengers.U_car, 'pt': passengers.U_pt})
    rs_wait = 0
    rs_ivt = inData.requests.ttrav.dt.total_seconds()

    # The filter is based on the cheapest fare (i.e. the pooling fare), because highest probability under no waiting and detours
    rs_km_fare = params.platforms.fare * (1 - params.platforms.pool_discount)
    utils['rs'] = util_rs(inData, params, rs_wait, rs_ivt, rs_km_fare)

    probabilities = mode_probs(utils)
    probs_without_rs = probabilities.apply(lambda row: row / (1 - row.rs), axis=1)[['bike','car','pt']]
    cuml = probs_without_rs.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    probabilities.loc[probabilities.rs > params.evol.travellers.min_prob, "decis"] = 'day-to-day'
    passengers['mode_choice'] = probabilities.decis
    passengers['prob_rs'] = probabilities.rs
    
    return passengers


def perc_rs_indicators(row):
    '''determine utility of opting for ridesourcing today'''
    registered_anywhere = (row.registered.sum() > 0) # registered with at least one platform
    if row.multihoming and registered_anywhere:
        expected_wait = row.expected_wait[0]
        expected_ivt = row.expected_ivt[0]
        expected_fare = row.expected_km_fare[0]
    elif not row.multihoming and registered_anywhere:
        expected_wait = row.expected_wait[row.registered][0] # wait time of one you are registered with
        expected_ivt = row.expected_ivt[row.registered][0] # same but for ivt
        expected_fare = row.expected_km_fare[row.registered][0] # same but for ivt
    else:
        expected_wait = math.inf
        expected_ivt = math.inf
        expected_fare = math.inf
    return expected_wait, expected_ivt, expected_fare


def mode_preday(inData, params):
    "determine the mode at the start of a day for a pool of travellers"
    passengers = inData.passengers

    df = inData.passengers.copy()
    df[['wait', 'ivt', 'fare']] = df.apply(lambda row: perc_rs_indicators(row), axis=1, result_type='expand')
    df['U_rs'] = util_rs(inData, params, df.wait, df.ivt, df.fare)
    utils = pd.DataFrame({'bike': passengers.U_bike, 'car': passengers.U_car, 'pt': passengers.U_pt, 'rs': df.U_rs})

    probabilities = mode_probs(utils)
    cuml = probabilities.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    passengers['mode_day'] = probabilities.decis
    
    return passengers


def util_alt_modes(inData, params):
    "determine utility of alternative modes for group of travellers"
    requests = inData.requests
    prefs = params.evol.travellers.mode_pref
    props = params.alt_modes

    # Draw Value of Time and corresponding beta's for travellers
    # lognorm_std = 2 * erfinv(prefs.gini)
    # lognorm_mean = np.log(prefs.mean_vot) - (lognorm_std ** 2) / 2
    # vot = np.random.lognormal(lognorm_mean, lognorm_std, params.nP)  # euro/h
    vot = np.random.lognormal(mean=prefs.ivt_mean_lognorm, sigma=prefs.ivt_sigma_lognorm, size=params.nP) * (-1) / prefs.beta_cost * 60 # euro/h
    beta_ivt = vot * prefs.beta_cost / 3600
    beta_access = beta_ivt * prefs.access_multip
    beta_wait = beta_ivt * prefs.wait_multip
    beta_bike_time = beta_ivt * prefs.bike_multip

    # Draw mode preferences (ASCs) for travellers
    ASC_car = np.random.normal(prefs.ASC_car, prefs.ASC_car_sd, params.nP)
    ASC_bike = np.random.normal(0, prefs.ASC_bike_sd, params.nP)
    
    # Attributes of modes
    car_ivt = requests.ttrav.dt.total_seconds()  # assumed same as RS
    requests['car_park_cost'] = props.car.park_cost
    if props.car.diff_parking:
        requests['dest_center'] = requests.apply(lambda x: inData.nodes.center.loc[x.destination], axis=1)
        requests.loc[requests.dest_center, 'car_park_cost'] = props.car.park_cost_center
    car_cost = props.car.km_cost * car_ivt * (params.speeds.ride / 1000) + requests.car_park_cost
    bike_tt = requests.ttrav.dt.total_seconds() * (params.speeds.ride / params.speeds.bike)

    # Utilities
    U_bike = beta_bike_time * bike_tt + ASC_bike
    U_car = beta_access * props.car.access_time + beta_ivt * car_ivt + prefs.beta_cost * car_cost + ASC_car

    # PT alternative - if included in simulation
    if params.paths.get('PT_trips',False):
        ASC_pt = np.random.normal(prefs.ASC_pt, prefs.ASC_pt_sd, params.nP)
        pt_trans_pen = inData.requests.transfers * prefs.transfer_pen
        pt_ivt = inData.requests.transitTime + pt_trans_pen
        pt_fare = inData.requests.PTfare
        pt_wait = inData.requests.waitingTime
        pt_access = inData.requests.walkDistance / params.speeds.walk
        U_pt = beta_access * pt_access + beta_wait * pt_wait + beta_ivt * pt_ivt + prefs.beta_cost * pt_fare + ASC_pt
    else:
        U_pt = -99999 # extremely large penalty so that no one will opt for PT if it is not offered
    utils = pd.DataFrame({'bike': U_bike, 'car': U_car, 'pt': U_pt})
    ASCs = pd.DataFrame({'bike': ASC_bike, 'car': ASC_car, 'pt': ASC_pt})
    
    return ASCs, vot, utils


def mode_probs(utils):
    "determine probabilities of modes based on utilities"
    exp_sum = np.exp(utils.bike) + np.exp(utils.car) + np.exp(utils.pt) + np.exp(utils.rs)
    
    P_bike = np.exp(utils.bike) / exp_sum
    P_car = np.exp(utils.car) / exp_sum
    P_pt = np.exp(utils.pt) / exp_sum
    P_rs = 1 - P_bike - P_car - P_pt
    
    probabilities = pd.DataFrame({'bike': P_bike, 'car': P_car, 'pt': P_pt, 'rs': P_rs})
    
    return probabilities


def util_rs(inData, params, rs_wait, rs_ivt, rs_km_fare, trav_vot=False, trav_ASC=False):
    '''determine utility of ridesourcing, either aggregated (if no trav_vot is provided) or for an individual traveller'''
    passengers = inData.passengers
    prefs = params.evol.travellers.mode_pref
    
    if not trav_vot: # determine utility for all passengers
        rs_fare = np.ones(len(inData.passengers)) * params.platforms.base_fare + rs_km_fare * rs_ivt * (params.speeds.ride / 1000)
        rs_fare[rs_fare < params.platforms.min_fare] += params.platforms.min_fare # min fare for solo ride
        beta_ivt = passengers.VoT * prefs.beta_cost / 3600
        ASC_rs = passengers.ASC_rs
    else:  # only for an individual traveller
        rs_fare = params.platforms.base_fare + rs_km_fare * rs_ivt * (params.speeds.ride / 1000)
        rs_fare = max(rs_fare, params.platforms.min_fare)
        beta_ivt = trav_vot * prefs.beta_cost / 3600
        ASC_rs = trav_ASC

    beta_wait = beta_ivt * prefs.wait_multip
    U_rs = beta_wait * rs_wait + beta_ivt * rs_ivt + prefs.beta_cost * rs_fare + ASC_rs
    
    return U_rs

def util_plfs(inData, params, row):
    util_plf = []
    for plf in inData.platforms.index:
        if params.platforms.service_types[plf] == 'solo':
            ASC_rs = row.ASC_rs
        else:
            ASC_rs = row.ASC_pool
        util_plf = util_plf + [util_rs(inData, params, row.perc_wait[plf], row.perc_ivt[plf], row.perc_fare[plf], trav_vot=row.VoT, trav_ASC=ASC_rs)]
    return util_plf


def learning_travs(params, prev_perc, exp):
    "returns new perceived waiting time of group of travellers"
    kappa = params.evol.travellers.kappa
    new_perc = (1 - kappa) * prev_perc + kappa * exp
    
    return new_perc


def diff_parking(inData):
    ox.config(log_console=True, use_cache=True)
    Z = ox.graph_from_place('Centrum Amsterdam', network_type='drive')
    Z_nodelist = list(Z.nodes)
    inData.nodes['center'] = False
    inData.nodes.loc[inData.nodes.index.isin(Z_nodelist), 'center'] = True

    return inData

def sample_from_alba(inData, params):
    '''Samples nP from Albatross dataset, replicating requests if nP is larger than size of dataset'''
    if params.nP > inData.requests.shape[0]:
        factor = math.floor(params.nP / inData.requests.shape[0])
        mod = params.nP % inData.requests.shape[0]
        df_req = pd.concat(factor * [inData.requests])
        df_pax = pd.concat(factor * [inData.passengers])
        df_add_req = inData.requests.sample(mod, replace=False, random_state=1)
        df_add_pax = inData.passengers[inData.passengers.index.isin(df_add_req.index)]
        inData.requests = df_req.append(df_add_req)
        inData.passengers = df_pax.append(df_add_pax)
    else:
        inData.requests = inData.requests.sample(params.nP, replace=False, random_state=1)
        inData.passengers = inData.passengers[inData.passengers.index.isin(inData.requests.index)]
    inData.requests = inData.requests.sort_index()
    inData.passengers = inData.passengers.sort_index()
    inData.requests.reset_index(inplace=True, drop=True)
    inData.requests.pax_id = inData.requests.index
    inData.requests.schedule_id = inData.requests.index
    inData.passengers.reset_index(inplace=True, drop=True)

    return inData


def sample_from_alba_different_treq(inData, params):
    '''Samples nP from Albatross dataset, replicating OD-pairs - but different req time! - if nP is larger than size of dataset'''
    if params.nP > inData.requests.shape[0]:
        factor = math.floor(params.nP / inData.requests.shape[0])
        mod = params.nP % inData.requests.shape[0]
        df_req = inData.requests.copy()
        for repl in range(factor):
            copy_req = inData.requests.copy()
            copy_req['treq'] = np.random.permutation(copy_req.treq)
            if repl == (factor - 1):
                copy_req = copy_req.sample(mod, replace=False, random_state=1)
            df_req = df_req.append(copy_req)
        inData.requests = df_req.sort_values(by=['treq'])
    else:
        inData.requests = inData.requests.sample(params.nP, replace=False, random_state=1)
    inData.requests.reset_index(inplace=True, drop=True)
    inData.requests.pax_id = inData.requests.index
    inData.requests.schedule_id = inData.requests.index

    return inData.requests

def read_requests_csv(inData, path):
    # from src_MaaSSim.data_structures import structures
    inData.requests = pd.read_csv(path, index_col='pax_id')

    return inData 

def sample_from_database(inData, params):
    "samples nP from large preprocessed demand dataset"
    inData.requests = inData.requests.sample(params.nP, replace=False, random_state=1)
    inData.requests.treq = pd.to_datetime(inData.requests.treq)
    inData.requests.ttrav = pd.to_timedelta(inData.requests.ttrav)
    inData.requests = inData.requests.sort_values(by=['treq']).reset_index(drop=True)
    inData.requests.index.name = 'pax_id'
    inData.passengers = pd.DataFrame(index=inData.requests.index, columns=inData.passengers.columns)
    inData.passengers['pax_id'] = inData.passengers.index.copy()
    inData.passengers.pos = inData.requests.origin.copy()
    inData.passengers.platforms = inData.passengers.platforms.apply(lambda x: [0])
    inData.passengers = inData.passengers.set_index('pax_id')

    return inData


def platform_regist_trav(inData, end_day, **kwargs):
    '''determine which travellers are registered with which platform and what LoS they anticipate'''
    params = kwargs.get('params', None)
    
    def signal_mh(params, avg_perc_indicator_mh, std_perc_indicator_mh):
        rand_signal = np.random.normal(avg_perc_indicator_mh, params.evol.travellers.inform.std_fact * std_perc_indicator_mh, size=len(inData.passengers.index))
        rand_signal[rand_signal < 0] = 0
        return rand_signal

    def signal_plf(params, avg_perc_indicator_plf, std_perc_indicator_plf):
        signal_list = []
        for plf in range(len(avg_perc_indicator_plf)):
            rand_signal = np.random.normal(avg_perc_indicator_plf[plf], params.evol.travellers.inform.std_fact * std_perc_indicator_plf[plf])
            if rand_signal < 0:
                rand_signal = 0
            signal_list.append(rand_signal)
        return signal_list
    
    def new_perc_after_communication_trav(row):
        # First replace nan signals (when there are no sh/mh travs per plf) by previous expected kpi 
        if row.multihoming:
            if math.isnan(row.relevant_signal):
                row.relevant_signal = row.expected_kpi[0]
        else:
            nan_mask = np.isnan(row.relevant_signal)
            row.relevant_signal = np.where(nan_mask, row.expected_kpi, row.relevant_signal)
        # Now determine new expected kpi
        if not row.decis or np.all(row.prev_regist): # not making a decision today or already registered with all platforms
            new_perc_kpi = row.expected_kpi
        else:
            if row.multihoming: # multihoming and not registered with any platform
                if np.any(~np.isnan(row.expected_kpi)): # if you have a previous expectation
                    kappa_comm = params.evol.travellers.kappa_comm
                else:
                    kappa_comm = 1
            else:
                kappa_comm = np.nan_to_num(np.isnan(row.expected_kpi) * ~row.prev_regist) * 1 + np.nan_to_num(params.evol.travellers.kappa_comm * ~np.isnan(row.expected_kpi) * ~row.prev_regist)
            new_perc_kpi = row.relevant_signal * kappa_comm + np.nan_to_num(row.expected_kpi) * (1 - kappa_comm)
        return new_perc_kpi
    
    def new_perc_kpi(regist_df):
        regist_df['perc_kpi_rel'] = regist_df.apply(lambda row: np.nanmean(row.expected_kpi) if row.reg_any else np.nan, axis=1) # relevant perc kpi for learning
        avg_perc_kpi_mh = regist_df.loc[regist_df.multihoming].perc_kpi_rel.mean()
        # print('trav mh perc kpi: {}'.format(avg_perc_kpi_mh))
        std_perc_kpi_mh = regist_df.loc[regist_df.multihoming].perc_kpi_rel.std(ddof=0)
        regist_df['perc_kpi_reg'] = regist_df.apply(lambda row: np.where(row.prev_regist == False, np.nan, row.prev_regist * row.expected_kpi), axis=1)
        
        if regist_df.multihoming.all(): # only multihomers
            avg_perc_kpi_plf = np.ones(inData.platforms.shape[0]) * np.nan
            std_perc_kpi_plf = np.ones(inData.platforms.shape[0]) * np.nan
        else:
            avg_perc_kpi_plf = np.nanmean(regist_df.loc[~regist_df.multihoming].perc_kpi_reg.to_list(), axis=0) # list with average perceived indicator per platform
            # print('trav plf avg_perc_kpi: {}'.format(avg_perc_kpi_plf))
            std_perc_kpi_plf = np.nanstd(regist_df.loc[~regist_df.multihoming].perc_kpi_reg.to_list(), axis=0, ddof=0)
        regist_df['signal_mh'] = signal_mh(params, avg_perc_kpi_mh, std_perc_kpi_mh)
        regist_df['signal_plf'] = regist_df.apply(lambda _: signal_plf(params, avg_perc_kpi_plf, std_perc_kpi_plf), axis=1)
        regist_df['relevant_signal'] = regist_df.apply(lambda row: row.signal_mh if row.multihoming else row.signal_plf, axis=1)
        regist_df['new_perc_kpi'] = regist_df.apply(lambda row: new_perc_after_communication_trav(row), axis=1)
        return regist_df
    
    def regist_plf(inData, row):
        '''returns boolean array with each item indicating whether you are registered with that platform after today'''
        reg_arr = row.prev_regist.copy() # if not making a registration decision
        if row.decis:
            if (row.prev_regist.sum() == 0) or (row.days_since_reg >= params.evol.travellers.regist.min_days): # either not previously registered or sufficient days registered
                reg_arr = np.full(len(inData.platforms.index), False) # standard: don't want to be registered with any platform
                if row.satisfied: # want to be registered
                    if row.multihoming:
                        reg_arr = np.full(len(inData.platforms.index), True)
                    else:
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

    regist_df = pd.DataFrame(data={'inform': inData.passengers.informed, 'prev_regist': end_day.registered,
                                   'expected_wait': end_day.new_perc_wait, 'expected_ivt': end_day.new_perc_ivt,
                                   'expected_fare': end_day.new_perc_fare, 'multihoming': inData.passengers.multihoming,
                                   'VoT': inData.passengers.VoT, 'ASC_rs': inData.passengers.ASC_rs, 
                                   'ASC_pool': inData.passengers.ASC_pool, 'days_since_reg': inData.passengers.days_since_reg},
                             index=inData.passengers.index)
    regist_df['days_since_reg'] = regist_df['days_since_reg'] + 1
    regist_df['decis'] = pd.Series(np.random.rand(len(inData.passengers.index)) <= params.evol.travellers.regist.samp, index=np.arange(0, len(inData.passengers.index)))  # Sample of travellers making (de)registration decision
    # only informed agents can make regist decision
    regist_df.loc[~regist_df.inform, 'decis'] = False

    ### If a traveller is currently unregistered and considers registration, he seeks information about LoS indicators, which he receives with noise
    ## Multi-homer: interested in multi-homing LoS, i.e. platform-independent indicators, (of reg. travellers) only
    regist_df['reg_any'] = regist_df.apply(lambda row: row.prev_regist.sum() > 0, axis=1)

    kpis = ['wait', 'ivt', 'fare']
    for kpi in kpis:
        df = regist_df.copy()
        df = df.rename(columns={"expected_{}".format(kpi): "expected_kpi"})
        df = new_perc_kpi(df)
        regist_df['new_perc_{}'.format(kpi)] = df["new_perc_kpi"].copy()
    
    regist_df['util_reg_plf'] = regist_df.rename(columns={"new_perc_wait": "perc_wait", "new_perc_ivt": "perc_ivt", "new_perc_fare": "perc_fare"}).apply(lambda row: util_plfs(inData, params, row), axis=1)
    regist_df['prob_plf'] = regist_df.apply(lambda row: np.array([(np.exp(row.util_reg_plf[plf]) / np.exp(row.util_reg_plf).sum()) for plf in inData.platforms.index]), axis=1)
    regist_df['chosen_plf_index'] = regist_df.apply(lambda row: np.random.choice(len(row.prob_plf), p=np.nan_to_num(row.prob_plf)) if (np.nan_to_num(row.prob_plf).sum() != 0) else 0, axis=1)

    # # Now decide between ridesourcing or other activity
    regist_df['util_not_reg'] = -math.inf # no reason not sign up with at least one platform
    regist_df['prob_regist_util'] = regist_df.apply(lambda row: np.exp(row.util_reg_plf[row.chosen_plf_index]) / (np.exp(row.util_reg_plf[row.chosen_plf_index]) + np.exp(row.util_not_reg)), axis=1)
    regist_df['satisfied'] = np.random.rand(len(inData.passengers.index)) < regist_df.prob_regist_util
    regist_df['regist_plf'] = regist_df.apply(lambda row: regist_plf(inData, row), axis=1)
    regist_df['reg_outcome'] = regist_df.apply(lambda row: np.where(row.regist_plf & ~row.prev_regist, 1, np.where(row.regist_plf & row.prev_regist, 0, np.where(~row.regist_plf & ~row.prev_regist, 0, -1))), axis=1) # 1: newly registered with plf, 0: same status as before, -1 deregistered with plf
    regist_df['days_since_reg'] = regist_df.apply(lambda row: return_days(row.reg_outcome, row.days_since_reg), axis=1)

    # return inData
    inData.passengers.registered = regist_df.regist_plf
    inData.passengers.days_since_reg = regist_df.days_since_reg
    inData.passengers.expected_wait = regist_df.new_perc_wait
    inData.passengers.expected_ivt = regist_df.new_perc_ivt
    inData.passengers.expected_km_fare = regist_df.new_perc_fare

    return inData