################################################################################
# Module: runners.py
# Description: Wrappers to prepare and run simulations
# Rafal Kucharski @ TU Delft
################################################################################


from MaaSSim.maassim import Simulator
from MaaSSim.shared import prep_shared_rides
from MaaSSim.utils import get_config, load_G, generate_demand, generate_vehicles, initialize_df, empty_series, \
    slice_space, read_requests_csv, read_vehicle_positions
from scipy.optimize import brute
import logging
import re
from MaaSSim.d2d_sim import *
from MaaSSim.d2d_demand import *
from MaaSSim.d2d_supply import *
from MaaSSim.decisions import dummy_False
import os.path
import zipfile
import json


def single_pararun(one_slice, *args):
    # function to be used with optimize brute
    inData, params, search_space = args  # read static input
    _inData = inData.copy()
    _params = params.copy()
    stamp = dict()
    # parameterize
    for i, key in enumerate(search_space.keys()):
        val = search_space[key][int(one_slice[int(i)])]
        stamp[key] = val

        if key in ['comm_rate', 'fare', 'base_fare']:
            _params.platforms[key] = val
        if key == 'gini':
            _params.evol.travellers.drivers[key] = val
        else:
            _params[key] = val

        # if _params.get("decis_type",False) == "platforms":
        #     _params.platforms[key] = val
        # if _params.get("decis_type",False) == "drivers":
        #     _params.evol.travellers.drivers[key] = val
        # else:
        #     _params[key] = val

    stamp['dt'] = str(pd.Timestamp.now()).replace('-','').replace('.','').replace(' ','')

    filename = ''
    for key, value in stamp.items():
        filename += '-{}_{}'.format(key, value)
    filename = re.sub('[^-a-zA-Z0-9_.() ]+', '', filename)
    # _inData.passengers = initialize_df(_inData.passengers)
    # _inData.requests = initialize_df(_inData.requests)
    # _inData.vehicles = initialize_df(_inData.vehicles)

    sim = simulate(inData=_inData, params=_params, logger_level=logging.WARNING, filename = filename)
    # sim.dump(dump_id=filename, path = _params.paths.get('dumps', None))  # store results

    print(filename, pd.Timestamp.now(), 'end')
    return 0


def simulate_parallel(config="../data/config/parallel.json", inData=None, params=None, search_space=None, **kwargs):
    # from MaaSSim.data_structures import structures as inData
    # inData.passengers = pd.read_csv(os.getcwd() + '//scenarios//inData//passengers.csv', index_col=0)
    # inData.requests = pd.read_csv(os.getcwd() + '//scenarios//inData//requests.csv', index_col=0)
    # inData.requests.treq = pd.to_datetime(inData.requests.treq)
    # inData.requests.tarr = pd.to_datetime(inData.requests.tarr)
    # inData.requests.ttrav = pd.to_timedelta(inData.requests.ttrav)
    # inData.vehicles = pd.read_csv(os.getcwd() + '//scenarios//inData//vehicles.csv', index_col=0)
    # inData.platforms = pd.read_csv(os.getcwd() + '//scenarios//inData//platforms.csv', index_col=0)

    if inData is None:  # othwerwise we use what is passed
        from MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
        params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices
        if params.alt_modes.car.diff_parking:
            inData = diff_parking(inData) # determine which nodes are in center
    # if len(inData.passengers) == 0:  # only if no passengers in input
    #     inData = generate_demand(inData, params, avg_speed=True)
    # if len(inData.vehicles) == 0:  # only if no vehicles in input
    #     inData.vehicles = generate_vehicles(inData, params.nV)
    # if len(inData.platforms) == 0:  # only if no platforms in input
    #     inData.platforms = initialize_df(inData.platforms)
    #     inData.platforms.loc[0] = empty_series(inData.platforms)
    #     inData.platforms.fare = [1]
        # inData.vehicles.platform = 0
        # inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis=1)


    brute(func=single_pararun,
          ranges=slice_space(search_space, replications=params.parallel.get("nReplications",1)),
          args=(inData, params, search_space),
          full_output=True,
          finish=None,
          workers=params.parallel.get('nThread',1))


def simulate(config="data/config.json", inData=None, params=None, path = None, **kwargs):
    """
    main runner and wrapper
    loads or uses json config to prepare the data for simulation, run it and process the results
    :param config: .json file path
    :param inData: optional input data
    :param params: loaded json file
    :param kwargs: optional arguments
    :return: simulation object with results
    """

    if inData is None:  # otherwise we use what is passed
        from MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params is None:
            params = get_config(config, root_path = kwargs.get('root_path'))  # load from .json file
    if kwargs.get('make_main_path',False):
        from MaaSSim.utils import make_config_paths
        params = make_config_paths(params, main = kwargs.get('make_main_path',False), rel = True)

    if params.paths.get('requests', False):
        inData = read_requests_csv(inData, path=params.paths.requests)

    if params.paths.get('vehicles', False):
        inData = read_vehicle_positions(inData, path=params.paths.vehicles)

    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True)  # download graph for the 'params.city' and calc the skim matrices
    # if len(inData.passengers) == 0:  # only if no passengers in input
    #     inData = generate_demand(inData, params, avg_speed=True)
    # if len(inData.vehicles) == 0:  # only if no vehicles in input
    #     inData.vehicles = generate_vehicles(inData, params.nV)
    # if len(inData.platforms) == 0:  # only if no platforms in input
    #     inData.platforms = initialize_df(inData.platforms)
    #     inData.platforms.loc[0] = empty_series(inData.platforms)
    #     inData.platforms.fare = [1]

    # Set random seeds
    np.random.seed(params.repl_id)
    random.seed(params.repl_id)

    # Load processed Albatross file, the OTP result, and compute PT fares
    inData = load_albatross_proc(inData, params, avg_speed = True)
    inData.requests = inData.requests.drop(['orig_geo', 'dest_geo', 'origin_y', 'origin_x', 'destination_y', 'destination_x', 'time'], axis = 1)
    inData.pt_itinerary = load_OTP_result(params)
    inData = consist_OTP_alba(inData, params)
    # inData.vehicles = pd.read_csv(os.getcwd() + '//scenarios//inData//vehicles.csv', index_col=0)
    # inData.platforms = pd.read_csv(os.getcwd() + '//scenarios//inData//platforms.csv', index_col=0)

    inData.passengers = prefs_travs(inData, params)
    all_pax = mode_filter(inData, params)
    inData.passengers = all_pax[all_pax.mode_choice == "day-to-day"]
    inData.requests = inData.requests[inData.requests.pax_id.isin(inData.passengers.index)]
    inData.pt_itinerary = inData.pt_itinerary[inData.pt_itinerary.pax_id.isin(inData.passengers.index)]
    inData.passengers.reset_index(drop=True, inplace=True)
    inData.requests.reset_index(drop=True, inplace=True)
    inData.pt_itinerary.reset_index(drop=True, inplace=True)
    inData.requests['pax_id'] = inData.requests.index
    inData.pt_itinerary['pax_id'] = inData.pt_itinerary.index

    inData.passengers['informed'] = np.random.rand(len(inData.passengers)) < params.evol.travellers.inform.prob_start
    inData.passengers['expected_wait'] = params.evol.travellers.inform.start_wait
    fixed_supply = generate_vehicles_d2d(inData, params)
    inData.vehicles = fixed_supply.copy()
    inData.vehicles.platform = inData.vehicles.apply(lambda x: 0, axis = 1)
    inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis = 1)
    inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0], axis = 1)
    inData.platforms = pd.concat([inData.platforms,pd.DataFrame(columns=['base_fare','comm_rate','min_fare'])])
    inData.platforms = initialize_df(inData.platforms)
    inData.platforms.loc[0]=[params.platforms.fare,'Uber',30,params.platforms.base_fare,params.platforms.comm_rate,params.platforms.min_fare,]

    inData = prep_shared_rides(inData, params.shareability)  # prepare schedules


    # # inData = generate_demand(inData, params, avg_speed=True)
    # inData.passengers = inData.passengers.sample(n=params.nP,random_state=1)
    # inData.requests = inData.requests.sample(n=params.nP,random_state=1)
    # all_pax = mode_filter(inData, params)
    # inData.passengers = all_pax[all_pax.mode_choice == "day-to-day"]
    # inData.requests = inData.requests[inData.requests.pax_id.isin(inData.passengers.index)]
    # inData.passengers.reset_index(drop=True, inplace=True)
    # inData.requests.reset_index(drop=True, inplace=True)
    # inData.requests['pax_id'] = inData.requests.index

    # inData.passengers['informed'] = np.random.rand(len(inData.passengers)) < params.evol.travellers.inform.prob_start
    # inData.passengers['expected_wait'] = params.evol.travellers.inform.start_wait
    # inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis=1)
    # inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
    #                                                     axis=1)
    # fixed_supply = generate_vehicles_d2d(inData, params)
    # inData.vehicles = fixed_supply.copy()

    # inData.vehicles = inData.vehicles.sample(n=params.nV, random_state=2)
    # inData.vehicles.index = np.arange(1,len(inData.vehicles)+1)
    # inData.vehicles.platform = inData.vehicles.apply(lambda x: 0, axis=1)
    # fixed_supply = inData.vehicles.copy()

    # inData.platforms = pd.concat([inData.platforms, pd.DataFrame(columns=['base_fare', 'comm_rate', 'min_fare'])])
    # inData.platforms = initialize_df(inData.platforms)
    # inData.platforms.loc[0] = [params.platforms.fare, 'Uber', 30, params.platforms.base_fare,
    #                            params.platforms.comm_rate, params.platforms.min_fare, ]
    #
    # inData = prep_shared_rides(inData, params.shareability)  # prepare schedules

    # d2d = DotMap()
    # d2d.drivers = dict()
    # d2d.travs = dict()
    sim = Simulator(inData, params=params,
                    kpi_veh = D2D_veh_exp,
                    kpi_pax = d2d_kpi_pax,
                    f_driver_out = D2D_driver_out,
                    f_trav_out = d2d_no_request,
                    f_trav_mode = dummy_False, **kwargs)  # initialize

    filename = kwargs.get('filename')
    if path is None:
        path = os.getcwd()
    sim_zip = zipfile.ZipFile(os.path.join(path, '{}.zip'.format(filename)), 'w')
    params.t0 = str(params.t0)
    with open('params_{}.json'.format(filename), 'w') as file:
        json.dump(params, file)
    sim_zip.write('params_{}.json'.format(filename))
    os.remove('params_{}.json'.format(filename))
    df_req = inData.requests[['pax_id','origin','destination','treq','dist','haver_dist']]
    df_pax = inData.passengers[['VoT','U_car','U_pt','U_bike']]
    df = pd.concat([df_req, df_pax], axis=1)
    sim_zip.writestr("requests.csv", df.to_csv())
    sim_zip.writestr("PT_itineraries.csv", inData.pt_itinerary.to_csv())
    sim_zip.writestr("vehicles.csv", inData.vehicles[['pos','res_wage']].to_csv())
    sim_zip.writestr("platforms.csv", inData.platforms.to_csv())
    evol_micro = init_d2d_dotmap()

    for day in range(params.get('nD', 1)):  # run iterations
        inData.passengers = mode_preday(inData, params)
        sim.make_and_run(run_id=day)  # prepare and SIM
        sim.output()  # calc results
        sim.last_res = sim.res[day].copy()
        del sim.res[day]
        # sim_zip = sim.dump_d2d(dump_id=filename, day=day, csv_zip=sim_zip)  # store results

        # d2d.drivers[day] = update_d2d_drivers(sim=sim, params=params)
        # d2d.travs[day] = update_d2d_travellers(sim=sim, params=params)
        drivers_summary = update_d2d_drivers(sim=sim, params=params)
        travs_summary = update_d2d_travellers(sim=sim, params=params)
        exp_df = update_work_exp(inData, drivers_summary)
        inData.vehicles.work_exp = exp_df.work_exp
        res_inf_driver = wom_driver(inData, params=params)
        inData.vehicles.informed = res_inf_driver
        signal_df = learning_unregist(inData, drivers_summary, params = params)
        inData.vehicles.expected_income = signal_df.expected_income
        res_regist = platform_regist(inData, drivers_summary, params=params)
        inData.vehicles.registered = res_regist.registered
        inData.vehicles.work_exp = res_regist.work_exp
        # inData.vehicles.expected_income = res_regist.expected_income
        inData.vehicles.pos = fixed_supply.pos
        res_inf_trav = wom_trav(inData, travs_summary, params=params)
        inData.passengers.informed = res_inf_trav.informed
        inData.passengers.expected_wait = res_inf_trav.perc_wait

        evol_micro = d2d_summary_day(evol_micro, drivers_summary, travs_summary, day)

        if sim.functions.f_stop_crit(sim=sim):
            break
    evol_micro, evol_agg = d2d_agg_statistics(evol_micro)
    # evol_micro, evol_agg = D2D_summary(d2d=d2d)
    for data_sup in ['inform', 'regist', 'ptcp', 'perc_inc', 'exp_inc']:
        sim_zip.writestr("d2d_{}.csv".format(data_sup), evol_micro.supply.toDict()[data_sup].to_csv())
    for data_dem in ['inform', 'requests', 'wait_time', 'corr_wait_time', 'perc_wait', 'bike', 'car', 'pt']:
        sim_zip.writestr("d2d_{}.csv".format(data_dem), evol_micro.demand.toDict()[data_dem].to_csv())
    sim_zip.writestr("d2d_agg_supply.csv", evol_agg.supply.to_csv())
    sim_zip.writestr("d2d_agg_demand.csv", evol_agg.demand.to_csv())

    return sim


if __name__ == "__main__":
    # simulate(make_main_path='..')  # single run
    simulate()  # single run

    from MaaSSim.utils import test_space

    simulate_parallel(search_space = test_space())
