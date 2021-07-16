from MaaSSim.traveller import travellerEvent
import pandas as pd
import numpy as np
import random
import math


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
    run_id = len(sim.res) - 1
    hist = pd.concat([~sim.res[_]['pax_exp'].NO_REQUEST for _ in range(0, run_id + 1)], axis=1, ignore_index=True)
    days_with_exp = hist.sum(axis=1)

    ret = pd.DataFrame()
    ret['pax'] = np.arange(0, len(sim.passengers))
    ret['orig'] = sim.inData.requests.origin.to_numpy()
    ret['dest'] = sim.inData.requests.destination.to_numpy()
    ret['t_req'] = sim.inData.requests.treq.to_numpy()
    ret['tt_min'] = sim.inData.requests.ttrav.to_numpy()
    ret['dist'] = sim.inData.requests.dist.to_numpy()
    ret['informed'] = sim.passengers.informed.to_numpy()
    ret['requests'] = ~sim.res[run_id].pax_exp['NO_REQUEST']
    ret['gets_offer'] = (sim.res[run_id].pax_exp['LOSES_PATIENCE'].apply(lambda x: True if x == 0 else False)
                         & ret['requests']).to_numpy()
    ret['accepts_offer'] = ~sim.res[run_id].pax_exp['OTHER_MODE'] & ret['gets_offer']
    ret['xp_wait'] = sim.res[run_id].pax_exp.WAIT.to_numpy()
    ret['xp_ivt'] = sim.res[run_id].pax_exp.TRAVEL.to_numpy()
    ret['xp_ops'] = sim.res[run_id].pax_exp.OPERATIONS.to_numpy()
    ret.loc[(ret.requests == False) | (ret.gets_offer == False) | (ret.accepts_offer == False), ['xp_wait', 'xp_ivt',
                                                                                                 'xp_ops']] = np.nan
    ret['xp_tt_total'] = ret.xp_wait + ret.xp_ivt + ret.xp_ops

    ret['init_perc_wait'] = sim.passengers.expected_wait.to_numpy()
    ret['experience'] = days_with_exp.to_numpy()
    experienced_trav = (ret.experience >= params.evol.travellers.omega).astype(int)
    kappa = (experienced_trav / params.evol.travellers.omega + (1 - experienced_trav) / (ret.experience + 1)) * ret.requests.astype(int)
    ret['corr_xp_wait'] = ret.xp_wait.copy()
    ret.loc[(ret.requests & (~ret.gets_offer)),['corr_xp_wait']] = params.evol.travellers.reject_penalty
    new_perc_wait = (1 - kappa) * ret.init_perc_wait + kappa * ret.corr_xp_wait
    ret['new_perc_wait'] = new_perc_wait.to_numpy()
    ret.loc[ret.informed & (~ret.requests), 'new_perc_wait'] = ret.loc[ret.informed & (~ret.requests), 'init_perc_wait']
    ret['mode'] = sim.passengers.mode_day.to_numpy()

    cols = list(ret.columns)
    a, b = cols.index('new_perc_wait'), cols.index('experience')
    cols[b], cols[a] = cols[a], cols[b]
    ret = ret[cols]
    ret = ret.set_index('pax')

    return ret


def d2d_no_request(*args, **kwargs):
    " returns True if traveller does not make RS request, False if he does"
    traveller = kwargs.get('pax',None)
    sim = traveller.sim
    params = sim.params

    trav_out = (traveller.pax.mode_day != "rs")
#     rs_wait = traveller.pax.expected_wait

#     rs_choice = mode_choice(traveller=traveller, rs_wait=rs_wait)
#     trav_out = not rs_choice or not traveller.pax.informed

    return trav_out

def wom_trav(inData, end_day, **kwargs):
    "determine which travellers are informed before the start of the new day"
    params = kwargs.get('params', None)
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
    res_inf.loc[(res_inf.informed) & (~end_day.informed),'perc_wait'] = params.evol.travellers.inform.start_wait

    return res_inf


def d2d_accept_offer(*args, **kwargs):
    # returns boolean True if passenger decides not to use (private) ridesourcing for given day (i.e. low quality offer)
    traveller = kwargs.get('traveller', None)
    sim = traveller.sim

    platform_id, offer = list(traveller.offers.items())[0]
    rs_wait = sim.skims.ride.T[sim.vehs[offer['veh_id']].veh.pos][traveller.request.origin]

    rs_choice = mode_choice(traveller=traveller, rs_wait=rs_wait)

    return not rs_choice


# def mode_choice(**kwargs):
#     traveller = kwargs.get('traveller', None)
#     sim = traveller.sim
#     params = sim.params
#     mcp = params.mode_choice
#     mset = params.alt_modes
#     rs_wait = kwargs.get('rs_wait')

#     rs_ivt = traveller.request.ttrav.seconds
#     rs_fare = max(params.platforms.base_fare + params.platforms.fare * rs_ivt * (params.speeds.ride / 1000),
#                   params.platforms.min_fare)

#     # Attributes of alternative modes
# #     car_ivt = traveller.request.ttrav.seconds  # assumed same as RS
# #     car_cost = mset.car.km_cost * car_ivt * (params.speeds.ride / 1000) + mset.car.park_cost
# #     pt_ivt = traveller.request.ttrav.seconds  * (params.speeds.ride / params.speeds.pt)
# #     pt_fare = mset.pt.base_fare + mset.pt.km_fare * pt_ivt * (params.speeds.ride / 1000)

# #     bike_tt = sim.skims.ride.T[traveller.request.origin][traveller.request.destination] * (
# #                 params.speeds.ride / params.speeds.bike)

#     # Utilities
#     U_rs = mcp.beta_wait_rs * rs_wait + mcp.beta_time_moto * rs_ivt + mcp.beta_cost * rs_fare + mcp.ASC_rs
# #     U_car = mcp.beta_access * mset.car.access_time + mcp.beta_time_moto * car_ivt + mcp.beta_cost * car_cost + mcp.ASC_car
# #     U_pt = mcp.beta_access * mset.pt.access_time + mcp.beta_wait_pt * mset.pt.wait_time + mcp.beta_time_moto * pt_ivt + mcp.beta_cost * pt_fare + mcp.ASC_pt
# #     U_bike = mcp.beta_time_bike * bike_tt
#     U_car = traveller.car
#     U_pt = traveller.pt
#     U_bike = traveller.bike
#     U_list = [U_rs, U_car, U_pt, U_bike]
#     exp_sum = np.exp(U_rs) + np.exp(U_car) + np.exp(U_pt) + np.exp(U_bike)

#     # Decision
#     P_list = [np.exp(U_list[mode_id]) / exp_sum for mode_id in range(len(U_list))]
#     draw = np.cumsum(P_list) > random.random()
#     decis = np.full((len(U_list)), False, dtype=bool)
#     decis[np.argmax(draw)] = True

#     return decis[0]

def mode_filter(inData, params):
    "mode choice based on no waiting time for RS, used to filter travellers with low probability of using RS"
    prefs = params.mode_choice
    passengers = inData.passengers
    
    utils = util_alt_modes(inData, params)
    passengers['U_bike'] = utils.bike
    passengers['U_car'] = utils.car
    passengers['U_pt'] = utils.pt
    
    rs_wait = 0
    ASC_rs = np.random.normal(prefs.ASC_rs,prefs.ASC_rs_sd,len(inData.passengers))
    passengers['ASC_rs'] = ASC_rs
    utils.rs = util_rs(inData, params, rs_wait, ASC_rs)
    
    probabilities = mode_probs(utils)
    cuml = probabilities.cumsum(axis=1)
    draw = cuml.gt(np.random.random(len(passengers)),axis=0) * 1
    probabilities['decis'] = draw.idxmax(axis="columns")
    probabilities.loc[probabilities.rs > params.evol.travellers.min_prob, "decis"] = 'day-to-day'
    
    passengers['mode_choice'] = probabilities.decis
    
    return passengers
    
#     # Draw mode preferences (ASCs) for travellers
#     ASC_rs = np.random.normal(prefs.ASC_rs,prefs.ASC_rs_sd,params.nP)
#     passengers['ASC_rs'] = ASC_rs
#     ASC_car = np.random.normal(prefs.ASC_car,prefs.ASC_car_sd,params.nP)
#     ASC_pt = np.random.normal(prefs.ASC_pt,prefs.ASC_pt_sd,params.nP)
#     ASC_bike = np.random.normal(0,prefs.ASC_bike_sd,params.nP)
    
#     # Attributes of modes
#     car_ivt = requests.ttrav.dt.total_seconds()  # assumed same as RS
#     car_cost = props.car.km_cost * car_ivt * (params.speeds.ride / 1000) + props.car.park_cost
#     pt_ivt = requests.ttrav.dt.total_seconds()  * (params.speeds.ride / params.speeds.pt)
#     pt_fare = props.pt.base_fare + props.pt.km_fare * pt_ivt * (params.speeds.ride / 1000)
#     bike_tt = requests.ttrav.dt.total_seconds()  * (params.speeds.ride / params.speeds.bike)
#     rs_ivt = requests.ttrav.dt.total_seconds()
#     rs_fare = np.ones(params.nP) * params.platforms.base_fare + params.platforms.fare * rs_ivt * (params.speeds.ride / 1000)
#     rs_fare[rs_fare < params.platforms.min_fare] += params.platforms.min_fare
#     rs_wait = np.zeros(params.nP)
    
#     # Determine utilities of modes
#     passengers['U_bike'] = prefs.beta_time_bike * bike_tt + ASC_bike
#     passengers['U_car'] = prefs.beta_access * props.car.access_time + prefs.beta_time_moto * car_ivt + prefs.beta_cost * car_cost + ASC_car
#     passengers['U_pt'] = prefs.beta_access * props.pt.access_time + prefs.beta_wait_pt * props.pt.wait_time + prefs.beta_time_moto * pt_ivt + prefs.beta_cost * pt_fare + ASC_pt
#     U_rs = prefs.beta_wait_rs * rs_wait + prefs.beta_time_moto * rs_ivt + prefs.beta_cost * rs_fare + prefs.ASC_rs
#     exp_sum = np.exp(U_rs) + np.exp(passengers.U_car) + np.exp(passengers.U_pt) + np.exp(passengers.U_bike)
# #     P_rs = np.exp(U_rs) / exp_sum
#     prob_df = pd.DataFrame()
#     prob_df['bike'] = np.exp(passengers.U_bike) / exp_sum
#     prob_df['car'] = np.exp(passengers.U_car) / exp_sum
#     prob_df['pt'] = np.exp(passengers.U_pt) / exp_sum
#     prob_df['rs'] = 1 - prob_df.bike - prob_df.car - prob_df.pt
#     cuml = prob_df.cumsum(axis=1)
#     draw = cuml > random.random()
#     prob_df['decis'] = draw.idxmax(axis="columns")
#     prob_df.loc[prob_df.rs > params.evol.travellers.min_prob, "decis"] = 'day-to-day'
    
#     passengers['mode_choice'] = prob_df.decis
    
#     return inData.passengers

def mode_preday(inData, params):
    "determine the mode at the start of a day for a pool of travellers"
    passengers = inData.passengers
    rs_wait = passengers.expected_wait
    ASC_rs = passengers.ASC_rs
    
    U_rs = util_rs(inData, params, rs_wait, ASC_rs)
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
    passengers = inData.passengers
    requests = inData.requests
    prefs = params.mode_choice
    props = params.alt_modes
    
    # Draw mode preferences (ASCs) for travellers
    ASC_car = np.random.normal(prefs.ASC_car,prefs.ASC_car_sd,params.nP)
    ASC_pt = np.random.normal(prefs.ASC_pt,prefs.ASC_pt_sd,params.nP)
    ASC_bike = np.random.normal(0,prefs.ASC_bike_sd,params.nP)
    
    # Attributes of modes
    car_ivt = requests.ttrav.dt.total_seconds()  # assumed same as RS
    car_cost = props.car.km_cost * car_ivt * (params.speeds.ride / 1000) + props.car.park_cost
    pt_ivt = requests.ttrav.dt.total_seconds()  * (params.speeds.ride / params.speeds.pt)
    pt_fare = props.pt.base_fare + props.pt.km_fare * pt_ivt * (params.speeds.ride / 1000)
    bike_tt = requests.ttrav.dt.total_seconds()  * (params.speeds.ride / params.speeds.bike)

    # Utilities
    U_bike = prefs.beta_time_bike * bike_tt + ASC_bike
    U_car = prefs.beta_access * props.car.access_time + prefs.beta_time_moto * car_ivt + prefs.beta_cost * car_cost + ASC_car
    U_pt = prefs.beta_access * props.pt.access_time + prefs.beta_wait_pt * props.pt.wait_time + prefs.beta_time_moto * pt_ivt + prefs.beta_cost * pt_fare + ASC_pt
    utils = pd.DataFrame({'bike': U_bike, 'car': U_car, 'pt': U_pt})
    
    return utils


def mode_probs(utils):
    "determine probabilities of modes based on utilities"
    exp_sum = np.exp(utils.bike) + np.exp(utils.car) + np.exp(utils.pt) + np.exp(utils.rs)
    
    P_bike = np.exp(utils.bike) / exp_sum
    P_car = np.exp(utils.car) / exp_sum
    P_pt = np.exp(utils.pt) / exp_sum
    P_rs = 1 - P_bike - P_car - P_pt
    
    probabilities = pd.DataFrame({'bike': P_bike, 'car': P_car, 'pt': P_pt, 'rs': P_rs})
    
    return probabilities


def util_rs(inData, params, rs_wait, ASC_rs):
    passengers = inData.passengers
    requests = inData.requests
    prefs = params.mode_choice
    
    rs_ivt = requests.ttrav.dt.total_seconds()
    rs_fare = np.ones(len(inData.passengers)) * params.platforms.base_fare + params.platforms.fare * rs_ivt * (params.speeds.ride / 1000)
    rs_fare[rs_fare < params.platforms.min_fare] += params.platforms.min_fare
    
    U_rs = prefs.beta_wait_rs * rs_wait + prefs.beta_time_moto * rs_ivt + prefs.beta_cost * rs_fare + ASC_rs
    
    return U_rs