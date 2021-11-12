import pandas as pd
import numpy as np
from scipy.stats import t
from dotmap import DotMap
import os, zipfile
import re
import json
import math

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

def add_indicators(micro, params, drivers, requests, res):
    for run_id in range(len(res.sup)):
        rel_perc_inc =  micro.sup.perc_inc[run_id].div(drivers[run_id].res_wage, axis=0)
        res.sup[run_id]['rel_perc_inc'] = rel_perc_inc.mean().values

        reg = micro.sup.regist[run_id].replace(False, np.nan)
        perc_inc_reg = reg * micro.sup.perc_inc[run_id]
        rel_perc_inc_reg =  perc_inc_reg.div(drivers[run_id].res_wage, axis=0)
        res.sup[run_id]['rel_perc_inc_reg'] = rel_perc_inc_reg.mean().values

        ptcp = micro.sup.ptcp[run_id].replace(False, np.nan)
        perc_inc_ptcp = ptcp * micro.sup.perc_inc[run_id]
        rel_perc_inc_ptcp = perc_inc_ptcp.div(drivers[run_id].res_wage, axis=0)
        res.sup[run_id]['rel_perc_inc_ptcp'] = rel_perc_inc_ptcp.mean().values

        res.sup[run_id]['std_perc_inc'] = micro.sup.perc_inc[run_id].std(axis=0).values
        res.sup[run_id]['std_exp_inc'] = micro.sup.exp_inc[run_id].std(axis=0).values
        res.sup[run_id]['std_perc_inc_reg'] = perc_inc_reg.std(axis=0).values
        res.sup[run_id]['std_perc_inc_ptcp'] = perc_inc_ptcp.std(axis=0).values

        inf = micro.sup.inform[run_id].replace(False, np.nan)
        res.sup[run_id]['res_wage_inf'] = inf.mul(drivers[run_id].res_wage, axis=0).mean().values
        res.sup[run_id]['res_wage_reg'] = reg.mul(drivers[run_id].res_wage, axis=0).mean().values
        res.sup[run_id]['res_wage_ptcp'] = ptcp.mul(drivers[run_id].res_wage, axis=0).mean().values

        # Regist and deregist
        init_day = micro.sup.regist[run_id].drop(micro.sup.regist[run_id].columns[-1], axis=1) * 1
        new_day = micro.sup.regist[run_id].drop(micro.sup.regist[run_id].columns[0], axis=1) * 1
        init_day.columns = list(map(str, np.arange(1,params[run_id]['nD'])))
        action = new_day - init_day
        action.insert(0, '0', np.zeros(len(action.index)))
        dereg = action < 0
        regi = action > 0
        res.sup[run_id]['new_regist'] = regi.sum(axis=0).values
        res.sup[run_id]['deregist'] = dereg.sum(axis=0).values

        res.dem[run_id].gets_offer = res.dem[run_id].gets_offer / res.dem[run_id].requests
        inform = (micro.dem.inform[run_id]).replace(False, np.nan)
        vot_inf =  inform.mul(requests[run_id].VoT, axis=0)
        res.dem[run_id]['vot_inf'] = vot_inf.mean().values

        req = (micro.dem.requests[run_id]).replace(False, np.nan)
        vot_req =  req.mul(requests[run_id].VoT, axis=0)
        res.dem[run_id]['vot_req'] = vot_req.mean().values

        res.dem[run_id]['std_perc_wait'] = micro.dem.perc_wait[run_id].std(axis=0).values
        res.dem[run_id]['std_exp_wait'] = micro.dem.wait_time[run_id].std(axis=0).values
        res.dem[run_id]['std_corr_wait'] = micro.dem.corr_wait_time[run_id].std(axis=0).values
        perc_wait_req = req * micro.dem.perc_wait[run_id]
        res.dem[run_id]['std_perc_wait_req'] = perc_wait_req.std(axis=0).values

        # Driver surplus
        ptcp_value = micro.sup.exp_inc[run_id].sub(drivers[run_id].res_wage.values, axis='rows').fillna(0)
        regist_costs = micro.sup.regist[run_id] * params[run_id]['evol']['drivers']['regist']['cost_comp']
        driver_surplus = ptcp_value - regist_costs
        res.sup[run_id]['surplus'] = driver_surplus.sum().values

        # Difference in Generalised Cost
        prefs = params[run_id]['evol']['travellers']['mode_pref']
        rs_ivt = requests[run_id].dist / params[run_id]['speeds']['ride']
        rs_fare = params[run_id]['platforms']['base_fare'] + params[run_id]['platforms']['fare'] * requests[run_id].dist / 1000
        rs_fare[rs_fare < params[run_id]['platforms']['min_fare']] = params[run_id]['platforms']['min_fare']
        beta_ivt = requests[run_id].VoT * prefs['beta_cost'] / 3600
        beta_wait = beta_ivt * prefs['wait_multip']
        U_rs = beta_ivt * rs_ivt + prefs['beta_cost'] * rs_fare + prefs['ASC_rs']
        U_wait = micro.dem.wait_time[run_id].mul(beta_wait, axis=0)
        U_rs = U_wait.add(U_rs, axis=0)
        U_best_alt = requests[run_id][['U_pt','U_car','U_bike']].max(axis=1)
        U_diff = U_rs.sub(U_best_alt, axis=0)
        res.dem[run_id]['GC_diff'] = (U_diff / -prefs['beta_cost']).sum().values

        # Platform profit
        rs_fare = requests[run_id].dist / 1000 * params[run_id]['platforms']['fare'] + params[run_id]['platforms']['base_fare']
        rs_fare = rs_fare.where(rs_fare > (params[run_id]['platforms']['min_fare']), params[run_id]['platforms']['min_fare'])
        res.plf[run_id] = micro.dem.requests[run_id].mul(rs_fare,axis=0).sum(axis=0) * params[run_id]['platforms']['comm_rate']
    
    return res


def determine_eql(res, params):
    res_list = list()
    for repl_id in range(len(res.sup)):
        params[repl_id]['conv_max_error'] = 0.1
        params[repl_id]['conv_signif'] = 0.05
        params[repl_id]['conv_day'] = 5 # X days in a row a change in perceived income below the convergence factor
        params[repl_id]['conv_factor'] = 0.01
        conv_sup = D2D_conv(perc = res.sup[repl_id].mean_perc_inc_reg, params = params[repl_id])
        conv_dem = D2D_conv(perc = res.dem[repl_id].perc_wait, params = params[repl_id])
        t_conv = max(conv_sup[0],conv_dem[0])
        conv_sys = conv_sup + conv_dem + [t_conv]                 
        res_list.append(conv_sys)
    conv_repl = pd.DataFrame(res_list,columns = ['t_sup*', 'perc_inc_reg*', 't_dem*', 'perc_wait*', 't*'])
    conv_repl['converged'] = (conv_repl['t*'] < params[0]['nD']-1)
    if not conv_repl.converged.all():
        print('WARNING: not all replications have converged')
    else:
        print('all replications have converged')
        
    return conv_repl


def find_vals(var_val, variable):
    scn = pd.DataFrame(var_val, columns = ['val'])
    vals = scn.val.unique()
    if variable["int"]:
        vals = list(map(int, vals))
    vals = sorted(vals)
    
    return vals, scn


def load_results(dir_name):
    extension = ".zip"
    var_val = []
    res = DotMap()
    params = DotMap()
    micro = DotMap()
    drivers = DotMap()
    ptcp = DotMap()
    requests = DotMap()
    perc_inc = DotMap()
    rs_choice = DotMap()
    run_id = 0
    for item in os.listdir(dir_name): # loop through items in dir
        if item.endswith(extension): # check for ".zip" extension
            zip_file_name = os.path.join(dir_name,item) # get full path of files
            zip_ref = zipfile.ZipFile(zip_file_name) # create zipfile object
            res.sup[run_id] = pd.read_csv(zip_ref.open('d2d_agg_supply.csv'),index_col=0)
            res.dem[run_id] = pd.read_csv(zip_ref.open('d2d_agg_demand.csv'),index_col=0)

            var_val.append(float(re.split('_|-', item)[2]))
    #         var_val.append(float(re.split('_|-', item)[2]))

            for filename in zip_ref.namelist():
                if filename.startswith("params"):
                    with zip_ref.open(filename) as f:
                        data = f.read()
                        params[run_id] = json.loads(data)

                if filename == "vehicles.csv":
                    drivers[run_id] = pd.read_csv(zip_ref.open(filename),index_col=0)
                    drivers[run_id]['veh'] = drivers[run_id].index
                    drivers[run_id].reset_index(drop=True)

                if filename == "requests.csv":
                    requests[run_id] = pd.read_csv(zip_ref.open(filename),index_col=0).set_index('pax_id')

                if filename.startswith("d2d_driver"):
                    key = filename.replace("d2d_driver_", "").replace(".csv","")
                    micro.sup[key][run_id] = pd.read_csv(zip_ref.open(filename),index_col=0)
                    micro.sup[key][run_id]['veh'] = micro.sup[key][run_id].index + 1
                    micro.sup[key][run_id] = micro.sup[key][run_id].set_index('veh')

                if filename.startswith("d2d_traveller"):
                    key = filename.replace("d2d_traveller_", "").replace(".csv","")
                    micro.dem[key][run_id] = pd.read_csv(zip_ref.open(filename),index_col=0)
                    micro.dem[key][run_id].index.names = ['pax_id']

            zip_ref.close() # close file
            run_id += 1
    
    return res, micro, params, drivers, requests, var_val


def runs_to_scenarios(res, params, vals, scn):
    d2d = DotMap()
    params_scn = DotMap()
    eql_runs = DotMap()
    eql_scn = DotMap()
    kpis_sup = res.sup[0].columns
    kpis_dem = res.dem[0].columns
    eql_scn.sup = pd.DataFrame(columns = kpis_sup)
    eql_scn.dem = pd.DataFrame(columns = kpis_dem)
    eql_scn.plf = pd.Series(dtype=int)

    for i in vals:
        # Determine which replications are part of the scenario and save params
        run_ids = scn[scn['val'] == i].index
        params_scn[i] = params[run_ids[0]]
        eql_runs.sup[i] = pd.DataFrame(columns = kpis_sup)
        eql_runs.dem[i] = pd.DataFrame(columns = kpis_dem)
        eql_runs.plf[i] = pd.Series(dtype=int)

        # For all runs in corresponding scenario, find indicators in equilibrium (last X days)
        for run_id in run_ids:
            df_sup = pd.DataFrame(columns = kpis_sup)
            df_dem = pd.DataFrame(columns = kpis_dem)
            series_plf = pd.Series(dtype=int)
            for day in range(params_scn[i]['nD'] - params_scn[i]['conv_day'], params_scn[i]['nD']):
                df_sup = df_sup.append(res.sup[run_id].loc[day], ignore_index=True)
                df_dem = df_dem.append(res.dem[run_id].loc[day], ignore_index=True)
                series_plf = series_plf.append(pd.Series([res.plf[run_id][day]]), ignore_index=True)
            eql_runs.sup[i] = eql_runs.sup[i].append(df_sup.mean(axis=0), ignore_index=True) 
            eql_runs.dem[i] = eql_runs.dem[i].append(df_dem.mean(axis=0), ignore_index=True)
            eql_runs.plf[i] = eql_runs.plf[i].append(pd.Series([series_plf.mean()]), ignore_index=True)

        conv_perc_inc = eql_runs.sup[i].mean_perc_inc_reg
        conv_perc_wait = eql_runs.dem[i].perc_wait

        repl_req_sup = repl_num(perc = conv_perc_inc, params = params_scn[i])
        repl_req_dem = repl_num(perc = conv_perc_wait, params = params_scn[i])
        repl_req_sys = max(repl_req_sup, repl_req_dem)

        if repl_req_sys > params_scn[i]['nReplications']:
            print("WARNING: insufficient number of replications: {}/{} for scenario {}".format(params_scn[i]['nReplications'],math.ceil(repl_req_sys),i))
        else:
            print("Sufficient number of replications: {}/{} for scenario {}".format(params_scn[i]['nReplications'],math.ceil(repl_req_sys),i))

        # Store equilibrium indicator values for this specific scenario
        eql_scn.sup = eql_scn.sup.append(eql_runs.sup[i].mean(axis=0), ignore_index=True)
        eql_scn.dem = eql_scn.dem.append(eql_runs.dem[i].mean(axis=0), ignore_index=True)
        eql_scn.plf = eql_scn.plf.append(pd.Series([eql_runs.plf[i].mean()]), ignore_index=True)

        # Create dataframes for each scenario
        for kpi in kpis_sup:
            d2d.sup[i][kpi] = empty_df(params = params)
        for kpi in kpis_dem:
            d2d.dem[i][kpi] = empty_df(params = params)
        d2d.plf[i] = empty_df(params = params)

        for k in range(len(run_ids)):
            # Replication within scenario
            run_id = run_ids[k]
            for kpi in kpis_sup:
                d2d.sup[i][kpi][k] = res.sup[run_id][kpi]
            for kpi in kpis_dem:
                d2d.dem[i][kpi][k] = res.dem[run_id][kpi]
            d2d.plf[i][k] = res.plf[run_id].values

        # Add mean value and st dev of different replications for a single scenario
        for kpi in kpis_sup:
            d2d.sup[i][kpi]['mean'] = d2d.sup[i][kpi].mean(axis=1)
            d2d.sup[i][kpi]['stdev'] = d2d.sup[i][kpi].loc[:, d2d.sup[i][kpi].columns != 'mean'].std(axis=1)
        for kpi in kpis_dem:
            d2d.dem[i][kpi]['mean'] = d2d.dem[i][kpi].mean(axis=1)
            d2d.dem[i][kpi]['stdev'] = d2d.dem[i][kpi].loc[:, d2d.dem[i][kpi].columns != 'mean'].std(axis=1)
        d2d.plf[i]['mean'] = d2d.plf[i].mean(axis=1)
        d2d.plf[i]['stdev'] = d2d.plf[i].loc[:, d2d.plf[i].columns != 'mean'].std(axis=1)
        
    # Reindex equilibrium database to use scenario values
    eql_scn.sup['scenario'] = vals
    eql_scn.dem['scenario'] = vals
    eql_scn.plf = pd.DataFrame(eql_scn.plf.values, index = vals, columns = ['profit'])
    eql_scn.sup = eql_scn.sup.set_index('scenario')
    eql_scn.dem = eql_scn.dem.set_index('scenario')
    
    return eql_runs, eql_scn, d2d, params_scn


def d2d_stats(d2d, vals, res):
    # Create new df with aggregated result for all scenarios
    res_scn = DotMap()
    kpis_sup = res.sup[0].columns
    kpis_dem = res.dem[0].columns
    for kpi in kpis_sup:
        res_scn.sup[kpi] = join_scn_df(d2d.sup,vals,kpi)
    for kpi in kpis_dem:
        res_scn.dem[kpi] = join_scn_df(d2d.dem,vals,kpi)
    res_scn.plf.profit = pd.DataFrame([d2d.plf[i]['mean'] for i in vals]).T
    res_scn.plf.profit = res_scn.plf.profit.set_axis(vals, axis=1, inplace=False)
    
    return res_scn
