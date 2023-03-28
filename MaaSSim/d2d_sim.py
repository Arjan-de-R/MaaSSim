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


def d2d_agg_statistics(evol_micro, params):
    # create aggregated d2d statistics based on day-to-day statistics of individual agents
    
    # d2d lists to dataframes
    for key in evol_micro.supply:
        evol_micro.supply[key] = pd.DataFrame(evol_micro.supply[key]).T
    for key in evol_micro.demand:
        evol_micro.demand[key] = pd.DataFrame(evol_micro.demand[key]).T

    demand = evol_micro.demand
    supply = evol_micro.supply
    
    evol_agg = DotMap()
    
    evol_agg.supply = pd.DataFrame({'inform': supply.inform.sum(), 'regist': supply.regist.sum(),
                                    'rejected_reg': supply.rejected_reg.sum(),
                                    'particip': supply.ptcp.sum(),
                                    'reject_particip': supply.rejected_ptcp.sum(),
                                    'mean_perc_inc': supply.perc_inc.mean(),
                                    'mean_perc_inc_ptcp': supply.perc_inc_ptcp.mean(),
                                    'mean_perc_inc_reg': supply.perc_inc_reg.mean(),
                                    'mean_exp_inc': supply.exp_inc.mean()})
    evol_agg.supply.index.name = 'day'
    
    if params.shareability.get('offered', False):
        evol_agg.demand = pd.DataFrame(
        {'inform': demand.inform.sum(), 'requests': demand.requests.sum(), 'req_solo': demand.req_solo.sum(),
         'req_pool': demand.req_pool.sum(), 'gets_offer_solo': (demand.gets_offer * demand.req_solo).sum(), 
         'gets_offer_pooling': (demand.gets_offer * demand.req_pool).sum(), 'act_shared': demand.act_shared.sum(),
#          'accepts_offer': demand.accepts_offer.sum(),
         'mean_wait_solo': (demand.wait_time * demand.requests).mean(), 'corr_mean_wait_solo': (demand.corr_wait_time * demand.requests).mean(),
         'mean_wait_pooling': (demand.wait_time * demand.req_pool).mean(), 'corr_mean_wait_pooling': (demand.corr_wait_time * demand.req_pool).mean(),
         'perc_wait_solo': demand.perc_wait.mean(), 
         'perc_wait_req': demand.perc_wait_req.mean(),
         'mean_disc': demand.xp_disc.mean(), 'perc_disc': demand.perc_disc.mean(),
         'bike': demand.bike.sum(), 'car': demand.car.sum(), 'pt': demand.pt.sum()})
    else:
        
        ## What aggregated statistics we need in case pooling is offered:
    # private requests that get offer (served)
    # pooled rides that are actually shared
    # pooled rides that are not pooled but served
    # xp_wait of pooling
    # xp_wait of private
    # xp_ivt of pooling (i.e. opted for pooling)
    
    # And check existing stats if they need to be adjusted.
        
        
    # Create df with aggregated statistics (not on agent-level)

        evol_agg.demand = pd.DataFrame(
            {'inform': demand.inform.sum(), 'requests': demand.requests.sum(), 
             'gets_offer': demand.gets_offer.sum(), 'accepts_offer': demand.accepts_offer.sum(),
             'mean_wait': demand.wait_time.mean(), 'corr_mean_wait': demand.corr_wait_time.mean(),
             'perc_wait': demand.perc_wait.mean(), 'perc_wait_req': demand.perc_wait_req.mean(),
             'bike': demand.bike.sum(), 'car': demand.car.sum(), 'pt': demand.pt.sum()})
    
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


def init_d2d_dotmap(params):
    # create empty dotmap for adding daily statistics
    
    evol_micro = DotMap()
    evol_micro.supply = DotMap()
    evol_micro.demand = DotMap()
    
    sup_indics = ['inform','regist','rejected_reg','ptcp','rejected_ptcp','perc_inc','perc_inc_ptcp','perc_inc_reg','exp_inc']
    dem_indics = ['inform','requests','gets_offer','accepts_offer','wait_time','corr_wait_time','perc_wait','perc_wait_req','chosen_mode','bike','car','pt']
    if params.shareability.get('offered', False):
        dem_indics = dem_indics + ['req_solo','req_pool','act_shared','xp_disc','perc_disc']
    
    for indic in sup_indics:
        name = indic
        data = []
        setattr(evol_micro.supply, name, data)

    for indic in dem_indics:
        name = indic
        data = []
        setattr(evol_micro.demand, name, data)

    return evol_micro


def d2d_summ_pooling(evol_micro, travs_summary):
    evol_micro.demand.req_solo.append((travs_summary.chosen_mode == 'rs').to_list())
    evol_micro.demand.req_pool.append((travs_summary.chosen_mode == 'pool').to_list())
    evol_micro.demand.act_shared.append(travs_summary.act_shared.tolist())
    evol_micro.demand.xp_disc.append(travs_summary.xp_discount.tolist())
    evol_micro.demand.perc_disc.append(travs_summary.init_perc_disc.tolist())
    
    
    return evol_micro