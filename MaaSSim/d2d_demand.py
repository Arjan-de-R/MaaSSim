from MaaSSim.traveller import travellerEvent
import pandas as pd
import numpy as np
import math
import os
from scipy.special import erfinv
import osmnx as ox


def load_albatross_proc(_inData, _params, avg_speed=False):
    # loads the full csv of albatross for a given city
    # changes date for today
    # filters for simulation time (t0 hour + simTime)
    # samples the n

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


def load_OTP_result(_params):
    # loads the attributes of the recommended PT initeraries

    df = pd.read_csv(os.path.join(_params.paths.albatross,
                                    _params.city.split(",")[0]+"_requests_PT.csv"),
                 index_col = 'id')

    df['pax_id'] = df.index
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

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
    df['PTfare'] = round((df['PTdistance'] * (1/1000) * _params.alt_modes.pt.km_fare) + _params.alt_modes.pt.base_fare,2)
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


def update_d2d_travellers(*args, **kwargs):
    "updating travellers' experience and updating new expected waiting time"
    sim = kwargs.get('sim', None)
    params = kwargs.get('params', None)

    ret = pd.DataFrame()
    ret['pax'] = np.arange(0, len(sim.passengers))
    ret['orig'] = sim.inData.requests.origin.to_numpy()
    ret['dest'] = sim.inData.requests.destination.to_numpy()
    ret['t_req'] = sim.inData.requests.treq.to_numpy()
    ret['tt_min'] = sim.inData.requests.ttrav.to_numpy()
    ret['dist'] = sim.inData.requests.dist.to_numpy()
    ret['informed'] = sim.passengers.informed.to_numpy()
    ret['requests'] = ~sim.last_res.pax_exp['NO_REQUEST']
    ret['gets_offer'] = (sim.last_res.pax_exp['LOSES_PATIENCE'].apply(lambda x: True if x == 0 else False)
                         & ret['requests']).to_numpy()
    ret['accepts_offer'] = ~sim.last_res.pax_exp['OTHER_MODE'] & ret['gets_offer']
    ret['xp_wait'] = sim.last_res.pax_exp.WAIT.to_numpy()
    ret['xp_ivt'] = sim.last_res.pax_exp.TRAVEL.to_numpy()
    ret['xp_ops'] = sim.last_res.pax_exp.OPERATIONS.to_numpy()
    ret.loc[(ret.requests == False) | (ret.gets_offer == False) | (ret.accepts_offer == False), ['xp_wait', 'xp_ivt',
                                                                                                 'xp_ops']] = np.nan
    ret['xp_tt_total'] = ret.xp_wait + ret.xp_ivt + ret.xp_ops

    ret['init_perc_wait'] = sim.passengers.expected_wait.to_numpy()
    ret['corr_xp_wait'] = ret.xp_wait.copy()
    ret.loc[(ret.requests & (~ret.gets_offer)),['corr_xp_wait']] = params.evol.travellers.reject_penalty
    new_perc_wait = learning_travs(params = params, prev_perc = ret.init_perc_wait, exp = ret.corr_xp_wait)

    ret['new_perc_wait'] = new_perc_wait.to_numpy()
    ret.loc[ret.informed & (~ret.requests), 'new_perc_wait'] = ret.loc[ret.informed & (~ret.requests), 'init_perc_wait']
    ret['chosen_mode'] = sim.passengers.mode_day.to_numpy()

    ret = ret.set_index('pax')

    return ret


def d2d_no_request(*args, **kwargs):
    " returns True if traveller does not make RS request, False if he does"
    traveller = kwargs.get('pax',None)

    trav_out = (traveller.pax.mode_day != "rs")

    return trav_out


def wom_trav(inData, end_day, **kwargs):
    "determine which travellers are informed and the waiting time that is communicated before the start of the new day"
    params = kwargs.get('params', None)
    exp_inf_trav = end_day.loc[end_day.informed]
    average_xp_wait = exp_inf_trav.corr_xp_wait.mean() / 60
    signal = (np.random.lognormal(params.evol.travellers.inform.mu_log, np.sqrt(2 * (np.log(average_xp_wait) - params.evol.travellers.inform.mu_log)), len(inData.passengers))) * 60
    nP_inf = inData.passengers.informed.sum()
    nP_uninf = len(inData.passengers) - nP_inf

    if nP_uninf > 0:
        exp_inf_day = (params.evol.travellers.inform.beta * nP_inf * nP_uninf) / len(inData.passengers)
        prob_inf = exp_inf_day / nP_uninf
    else:
        prob_inf = 0

    new_inf = np.random.rand(len(inData.passengers)) < prob_inf
    prev_inf = inData.passengers.informed.to_numpy()
    informed = (np.concatenate(([prev_inf],[new_inf]),axis=0).transpose()).any(axis=1)
    res_inf = pd.DataFrame(data = {'informed': informed, 'perc_wait': end_day.new_perc_wait}, index=np.arange(0,len(inData.passengers)))
    res_inf['signal'] = signal
    res_inf['cond'] = res_inf.informed & (~end_day.informed)
    res_inf['perc_wait'] = res_inf['perc_wait'].where(~res_inf.cond, res_inf['signal'])
    res_inf.drop(['signal', 'cond'], axis=1)

    return res_inf


def prefs_travs(inData, params):
    "draw mode preferences for the group of travellers"
    prefs = params.evol.travellers.mode_pref
    passengers = inData.passengers

    vot, utils = util_alt_modes(inData, params)

    passengers['VoT'] = vot
    passengers['U_bike'] = utils.bike
    passengers['U_car'] = utils.car
    passengers['U_pt'] = utils.pt
    
    ASC_rs = np.random.normal(prefs.ASC_rs,prefs.ASC_rs_sd,len(inData.passengers))
    passengers['ASC_rs'] = ASC_rs
    
    return passengers


def mode_filter(inData, params):
    "mode choice based on no waiting time for RS, used to filter travellers with low probability of using RS"
    passengers = inData.passengers
    utils = pd.DataFrame({'bike': passengers.U_bike, 'car': passengers.U_car, 'pt': passengers.U_pt})
    rs_wait = 0
    utils['rs'] = util_rs(inData, params, rs_wait)

    probabilities = mode_probs(utils)
    cuml = probabilities.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    probabilities.loc[probabilities.rs > params.evol.travellers.min_prob, "decis"] = 'day-to-day'

    passengers['mode_choice'] = probabilities.decis
    passengers['prob_rs'] = probabilities.rs
    
    return passengers


def mode_preday(inData, params):
    "determine the mode at the start of a day for a pool of travellers"
    passengers = inData.passengers
    rs_wait = passengers.expected_wait
    
    U_rs = util_rs(inData, params, rs_wait)
    utils = pd.DataFrame({'bike': passengers.U_bike, 'car': passengers.U_car, 'pt': passengers.U_pt, 'rs': U_rs})
    utils.loc[~inData.passengers.informed, 'rs'] = -math.inf
    
    probabilities = mode_probs(utils)
    
    cuml = probabilities.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    passengers['mode_day'] = probabilities.decis
    
    return passengers


def util_alt_modes(inData, params):
    "determine utility of alternative modes for group of travellers"
    requests = inData.requests
    pt_itins = inData.pt_itinerary
    prefs = params.evol.travellers.mode_pref
    props = params.alt_modes

    # Draw Value of Time and corresponding beta's for travellers
    lognorm_std = 2 * erfinv(prefs.gini)
    lognorm_mean = np.log(prefs.mean_vot) - (lognorm_std ** 2) / 2
    vot = np.random.lognormal(lognorm_mean, lognorm_std, params.nP)  # euro/h
    beta_ivt = vot * prefs.beta_cost / 3600
    beta_access = beta_ivt * prefs.access_multip
    beta_wait = beta_ivt * prefs.wait_multip
    beta_bike_time = beta_ivt * prefs.bike_multip

    # Draw mode preferences (ASCs) for travellers
    ASC_car = np.random.normal(prefs.ASC_car, prefs.ASC_car_sd, params.nP)
    ASC_pt = np.random.normal(prefs.ASC_pt, prefs.ASC_pt_sd, params.nP)
    ASC_bike = np.random.normal(0, prefs.ASC_bike_sd, params.nP)
    
    # Attributes of modes
    car_ivt = requests.ttrav.dt.total_seconds()  # assumed same as RS
    requests['car_park_cost'] = props.car.park_cost
    if props.car.diff_parking:
        requests['dest_center'] = requests.apply(lambda x: inData.nodes.center.loc[x.destination], axis=1)
        requests.loc[requests.dest_center, 'car_park_cost'] = props.car.park_cost_center
    car_cost = props.car.km_cost * car_ivt * (params.speeds.ride / 1000) + requests.car_park_cost
    pt_trans_pen = pt_itins.transfers * prefs.transfer_pen
    pt_ivt = pt_itins.transitTime + pt_trans_pen
    pt_fare = pt_itins.PTfare
    pt_wait = pt_itins.waitingTime
    pt_access = pt_itins.walkDistance / params.speeds.walk
    bike_tt = requests.ttrav.dt.total_seconds() * (params.speeds.ride / params.speeds.bike)

    # Utilities
    U_bike = beta_bike_time * bike_tt + ASC_bike
    U_car = beta_access * props.car.access_time + beta_ivt * car_ivt + prefs.beta_cost * car_cost + ASC_car
    U_pt = beta_access * pt_access + beta_wait * pt_wait + beta_ivt * pt_ivt + prefs.beta_cost * pt_fare + ASC_pt
    utils = pd.DataFrame({'bike': U_bike, 'car': U_car, 'pt': U_pt})
    
    return vot, utils


def mode_probs(utils):
    "determine probabilities of modes based on utilities"
    exp_sum = np.exp(utils.bike) + np.exp(utils.car) + np.exp(utils.pt) + np.exp(utils.rs)
    
    P_bike = np.exp(utils.bike) / exp_sum
    P_car = np.exp(utils.car) / exp_sum
    P_pt = np.exp(utils.pt) / exp_sum
    P_rs = 1 - P_bike - P_car - P_pt
    
    probabilities = pd.DataFrame({'bike': P_bike, 'car': P_car, 'pt': P_pt, 'rs': P_rs})
    
    return probabilities


def util_rs(inData, params, rs_wait):
    passengers = inData.passengers
    requests = inData.requests
    prefs = params.evol.travellers.mode_pref
    
    rs_ivt = requests.ttrav.dt.total_seconds()
    rs_fare = np.ones(len(inData.passengers)) * params.platforms.base_fare + params.platforms.fare * rs_ivt * (params.speeds.ride / 1000)
    rs_fare[rs_fare < params.platforms.min_fare] += params.platforms.min_fare

    beta_ivt = passengers.VoT * prefs.beta_cost / 3600
    beta_wait = beta_ivt * prefs.wait_multip
    U_rs = beta_wait * rs_wait + beta_ivt * rs_ivt + prefs.beta_cost * rs_fare + passengers.ASC_rs
    
    return U_rs


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


def consist_OTP_alba(inData, params):
    # Check consistency between Albatross and OTP data
    inData.requests = inData.requests[inData.requests.index.isin(inData.pt_itinerary.index)]
    inData.passengers = inData.passengers[inData.passengers.index.isin(inData.pt_itinerary.index)]
    if params.nP > inData.requests.shape[0]:
        factor = math.floor(params.nP / inData.requests.shape[0])
        mod = params.nP % inData.requests.shape[0]
        df_req = pd.concat(factor * [inData.requests])
        df_pax = pd.concat(factor * [inData.passengers])
        df_pt = pd.concat(factor * [inData.pt_itinerary])
        df_add_req = inData.requests.sample(mod, replace=False, random_state=1)
        df_add_pax = inData.passengers[inData.passengers.index.isin(df_add_req.index)]
        df_add_pt = inData.pt_itinerary[inData.pt_itinerary.index.isin(df_add_req.index)]
        inData.requests = df_req.append(df_add_req)
        inData.passengers = df_pax.append(df_add_pax)
        inData.pt_itinerary = df_pt.append(df_add_pt)
    else:
        inData.requests = inData.requests.sample(params.nP, replace=False, random_state=1)
        inData.passengers = inData.passengers[inData.passengers.index.isin(inData.requests.index)]
        inData.pt_itinerary = inData.pt_itinerary[inData.pt_itinerary.index.isin(inData.requests.index)]
    inData.requests = inData.requests.sort_index()
    inData.passengers = inData.passengers.sort_index()
    inData.pt_itinerary = inData.pt_itinerary.sort_index()
    inData.requests.reset_index(inplace=True, drop=True)
    inData.requests.pax_id = inData.requests.index
    inData.requests.schedule_id = inData.requests.index
    inData.passengers.reset_index(inplace=True, drop=True)
    inData.pt_itinerary.reset_index(inplace=True, drop=True)
    inData.pt_itinerary.pax_id = inData.pt_itinerary.index

    return inData
