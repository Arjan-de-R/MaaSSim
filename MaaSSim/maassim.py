################################################################################
# Module: main.py
# Description: Simulator object
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################


from dotmap import DotMap
import pandas as pd
import math
import simpy
import time
import numpy as np
import os.path
import zipfile

from MaaSSim.traveller import PassengerAgent, travellerEvent
from MaaSSim.driver import VehicleAgent, f_repos
from MaaSSim.platform import f_match, PlatformAgent
from MaaSSim.performance import kpi_pax, kpi_veh
from MaaSSim.utils import initialize_df, dummy_False
import sys
import logging


class Simulator:
    """
    main class of MaaSSim

    used to prepare, populate, run simulations and analyze the results
    """

    def __init__(self, _inData, **kwargs):
        # STATICS and kwargs
        # list of functionalities
        # that may be filled with functions to represent desired behaviour
        self.FNAMES = ['print', 'params', 'f_match', 'f_trav_out',
                       'f_driver_learn', 'f_driver_out', 'f_trav_mode', 'f_driver_decline', 'f_platform_choice',
                       'f_driver_repos', 'f_timeout', 'kpi_pax', 'kpi_veh']

        self.default_args = {'print': False,
                          'params': None,
                          'f_match': f_match,
                          'f_trav_out': dummy_False,
                          'f_driver_learn': dummy_False,
                          'f_driver_out': dummy_False,
                          'f_trav_mode': dummy_False,
                          'f_driver_decline': dummy_False,
                          'f_driver_repos': f_repos,
                          'f_timeout': self.timeout,
                             'kpi_pax': kpi_pax,
                             'kpi_veh': kpi_veh,
                             'monitor': True}

        # input
        self.inData = _inData.copy()  # copy of data structure for simulations (copy needed for multithreading)
        self.functions = DotMap()
        self.vehicles = self.inData.vehicles  # input
        self.platforms = self.inData.platforms  # input

        self.myinit(**kwargs)  # part that is called every run
        # output
        self.run_ids = list()  # ids of consecutively executed runs
        self.runs = dict()  # simulation outputs (raw)
        self.res = dict()  # simulation results (processed)
        self.logger = self.init_log(**kwargs)
        self.logger.warning("""Setting up {}h simulation at {} for {} vehicles and {} passengers in {}"""
                            .format(self.params.simTime,
                                    self.t0, self.params.nV, self.params.nP,
                                    self.params.city))
        # self.logger.info(kwargs)

    ##########
    #  PREP  #
    ##########

    def myinit(self, **kwargs):
        # part of init that is repeated every run
        self.default_args.update(kwargs)
        self.params = self.default_args['params']  # json dict with parameters

        # populate functions
        for f in self.default_args.keys():
            if f in self.FNAMES:
                self.functions[f] = self.default_args[f]

        self.make_skims()
        self.set_variabilities()
        self.env = simpy.Environment()  # simulation environment init
        self.t0 = self.inData.requests.treq.min()  # start at the first request time
        self.t1 = 60 * 60 * (self.params.simTime + 2)

        self.trips = list()  # report of trips
        self.rides = list()  # report of rides
        self.passengers = self.inData.passengers.copy()
        self.requests = initialize_df(self.inData.requests)  # init requests
        self.reqQ = list()  # queue of requests (traveller ids)
        self.vehQ = list()  # queue of idle vehicles (driver ids)
        self.pax = dict()  # list of passengers
        self.vehs = dict()  # list of vehicles
        self.plats = dict()  # list of platforms
        self.sim_start = None

    def generate(self):
        # generate passengers and vehicles as agents in the simulation (inData stays intact)
        for platform_id in self.platforms.index:
            self.plats[platform_id] = PlatformAgent(self, platform_id)
        for pax_id in self.inData.passengers.index:
            self.pax[pax_id] = PassengerAgent(self, pax_id)
        for veh_id in self.vehicles.index:
            self.vehs[veh_id] = VehicleAgent(self, veh_id)

    #########
    #  RUN  #
    #########

    def simulate(self, run_id=None):
        # run
        self.sim_start = time.time()
        self.logger.info("-------------------\tStarting simulation\t-------------------")

        self.env.run(until=self.t1)  # main run sim time + cool_down
        self.sim_end = time.time()
        self.logger.info("-------------------\tSimulation over\t\t-------------------")
        if len(self.reqQ) >= 0:
            self.logger.info(f"queue of requests {len(self.reqQ)}")
        self.logger.warning(f"simulation time {round(self.sim_end - self.sim_start, 1)} s")
        self.make_res(run_id)
        if self.params.get('assert_me', True):
            self.assert_me()  # test consistency of results

    def make_and_run(self, run_id=None, **kwargs):
        # wrapper for the simulation routine
        self.myinit(**kwargs)
        self.generate()
        self.simulate(run_id=run_id)

    ############
    #  OUTPUT  #
    ############

    def make_res(self, run_id):
        # called at the end of simulation
        if run_id == None:
            if len(self.run_ids) > 0:
                run_id = self.run_ids[-1] + 1
            else:
                run_id = 0
        self.run_ids.append(run_id)
        trips = pd.concat([pd.DataFrame(self.pax[pax].rides) for pax in self.pax.keys()])
        outcomes = [self.pax[pax].rides[-1]['event'] for pax in self.pax.keys()]
        rides = pd.concat([pd.DataFrame(self.vehs[pax].myrides) for pax in self.vehs.keys()])
        queues = pd.concat([pd.DataFrame(self.plats[plat].Qs,
                                         columns=['t', 'platform', 'vehQ', 'reqQ'])
                            for plat in self.plats]).set_index('t')

        self.runs[run_id] = DotMap({'trips': trips, 'outcomes': outcomes, 'rides': rides, 'queues': queues})

    def output(self, run_id=None):
        # called after the run for refined results
        run_id = self.run_ids[-1] if run_id is None else run_id
        ret = self.functions.kpi_pax(simrun=self.runs[run_id])
        veh = self.functions.kpi_veh(simrun=self.runs[run_id], vehindex=self.inData.vehicles.index)
        ret.update(veh)
        self.res[run_id] = DotMap(ret)

    #########
    # UTILS #
    #########
    def init_log(self, **kwargs):
        logger = kwargs.get('logger', None)
        level = kwargs.get('logger_level', logging.INFO)
        if logger is None:
            logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s-%(message)s',
                                datefmt='%d-%m-%y %H:%M:%S', level=level)

            logger = logging.getLogger()
            logger.setLevel(level)
            return logging.getLogger(__name__)
        else:
            logger.setLevel(level)
            return logger

    def print_now(self):
        return self.t0 + pd.Timedelta(self.env.now, 's')

    def assert_me(self):
        #try:
        # basic checks for results consistency and correctness
        rides = self.runs[0].rides  # vehicles record
        trips = self.runs[0].trips  # travellers record
        for i in self.inData.passengers.sample(min(5, self.params.nP)).index.to_list():
            r = self.inData.requests[self.inData.requests.pax_id == i].iloc[0].squeeze()  # that is his request
            o, d = r['origin'], r['destination']  # his origin and destination
            trip = trips[trips.pax == i]  # his trip
            assert o in trip.pos.values  # was he at origin
            if travellerEvent.ARRIVES_AT_DEST.name in trip.event.values:
                # succesful trip
                assert d in trip.pos.values  # did he reach the destination
                veh = trip.veh_id.dropna().unique()  # did he travel with vehicle
                assert len(veh) == 1  # was there just one vehicle (should be)
                ride = rides[rides.veh == veh[0]]
                assert i in list(
                    set([item for sublist in ride.paxes.values for item in sublist]))  # was he assigned to a vehicle
                common_pos = list(set(list(ride.pos.values) + list(trip.pos.values)))
                assert len(common_pos) >= 2  # were there at least two points in common
                for pos in common_pos:
                    assert len(set(ride[ride.pos == pos].t.to_list() + trip[
                        trip.pos == pos].t.to_list())) > 0  # were they at the same time at the same place?
            else:
                # unsuccesful trip
                flag = False
                if travellerEvent.LOSES_PATIENCE.name in trip.event.values:
                    flag = True
                elif travellerEvent.IS_REJECTED_BY_VEHICLE.name in trip.event.values:
                    flag = True
                elif travellerEvent.REJECTS_OFFER.name in trip.event.values:
                    flag = True
                assert flag == True
        self.logger.warn('assertion tests for simulation results - passed')
        # except:
        #     self.logger.info('assertion tests for simulation results - failed')
        #     swwssw

    def dump(self, path=None, id="", inputs=True, results=True):
        """
        stores resulting files into .zip folder
        :param path:
        :param id: run id
        :param inputs: store input files (vehicles, passengers, platforms)
        :param results: stor output files (trips, rides, veh, pax KPIs)
        :return: zip file
        """
        if path is None:
            path = os.getcwd()

        with zipfile.ZipFile(os.path.join(path, 'res{}.zip'.format(id)), 'w') as csv_zip:
            if inputs:
                for data in ['vehicles', 'passengers', 'requests', 'platforms']:
                    csv_zip.writestr("{}.csv".format(data), self.inData[data].to_csv())
            if results:
                csv_zip.writestr("{}.csv".format('trips'), self.runs[0].trips.to_csv())
                csv_zip.writestr("{}.csv".format('rides'), self.runs[0].rides.to_csv())
                for key in self.res[0].keys():
                    csv_zip.writestr("{}.csv".format(key), self.res[0][key].to_csv())
        return csv_zip

    def make_skims(self):
        # uses distance skim in meters to populate 3 skims used in simulations
        self.skims = DotMap()
        self.skims.dist = self.inData.skim.copy()
        self.skims.ride = self.skims.dist.divide(self.params.speeds.ride).astype(int)  # <---- here we have travel time
        self.skims.walk = self.skims.dist.divide(self.params.speeds.walk).astype(int)  # <---- here we have travel time

    def timeout(self, n, variability=False):
        # overwrites sim timeout to add potential stochasticity
        if variability:
            # n = (random.random()-0.5)*perturb*n+n #uniform
            n = np.random.normal(n, math.sqrt(n * variability))  # normal
        return self.env.timeout(n)

    def set_variabilities(self):
        self.vars = DotMap()
        self.vars.walk = False
        self.vars.start = False
        self.vars.request = False
        self.vars.transaction = False
        self.vars.ride = False
        self.vars.pickup = False
        self.vars.dropoff = False
        self.vars.shift = False
        self.vars.pickup_patience = False

    def plot_trip(self, pax_id, run_id=None):
        import matplotlib.pyplot as plt
        import networkx as nx
        import osmnx as ox
        from .traveller import travellerEvent
        G = self.inData.G
        # space time
        if run_id is None:
            run_id = list(self.runs.keys())[-1]
        df = self.runs[run_id].trips
        df = df[df.pax == pax_id]
        df['status_num'] = df.apply(lambda x: travellerEvent[x.event].value, axis=1)

        fig, ax = plt.subplots()
        df.plot(x='t', y='status_num', ax=ax, drawstyle="steps-post")
        ax.yticks = plt.yticks(df.index, df.event)
        plt.show()

        # map
        routes = list()
        prev_node = df.pos.iloc[0]
        for node in df.pos[1:]:
            if prev_node != node:
                routes.append(nx.shortest_path(G, prev_node, node, weight='length'))
                routes.append(nx.shortest_path(G, prev_node, node, weight='length'))
            prev_node = node
        ox.plot_graph_routes(G, routes, node_size=0)