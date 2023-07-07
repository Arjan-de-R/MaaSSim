### Generating a large demand dataset - based on a spatiotemporal distribution or by replicating Albatross data
### Preparing this dataset for OpenTripPlanner query (repo: query_PT)

import os
import sys

MAIN_DIR = os.path.dirname(__file__)
ABS_MAIN_DIR = os.path.abspath(MAIN_DIR)
MAASSIM_DIR = os.path.join(ABS_MAIN_DIR, "..")
sys.path.append(MAASSIM_DIR)
from src_MaaSSim.d2d_demand import *
from src_MaaSSim.utils import get_config, load_G, generate_demand, read_requests_csv
import numpy as np
import random
import pandas as pd
from dotmap import DotMap

params = DotMap()
# Set main parameters
params.city = "Delft, Netherlands"
params = get_config('MaaSSim/data/config/Delft.json')
params.paths.albatross = 'MaaSSim/data/albatross'
params.repl_id = 0
params.albatross = False  # if False, demand is artificially generated
params.nP = 500 # travellers
params.dist_threshold_min = 2000 # min dist
# Start and sim time
params.t0 = pd.Timestamp(2023, 9, 19, 9) # YMDH
params.simTime = 8

# Set right paths
params.paths.G = 'MaaSSim/data/graphs/{}.graphml'.format(params.city.split(",")[0])
params.paths.skim = 'MaaSSim/data/graphs/{}.csv'.format(params.city.split(",")[0])
                                                        
def input_for_OTP(config="data/config.json", inData=None, params=None, path = None, **kwargs):
    """
    generate demand and convert it to input csv for OpenTripPlanner
    :param config: .json file path
    :param inData: optional input data
    :param params: loaded json file
    :param kwargs: optional arguments
    :return: simulation object with results
    """

    if inData is None:  # otherwise we use what is passed
        from src_MaaSSim.data_structures import structures
        inData = structures.copy()  # fresh data
    if params.paths.get('requests', False):
        inData = read_requests_csv(inData, path=params.paths.requests)
    if len(inData.G) == 0:  # only if no graph in input
        inData = load_G(inData, params, stats=True, set_t=False)  # download graph for the 'params.city' and calc the skim matrices

    # Set random seeds
    np.random.seed(params.repl_id)
    random.seed(params.repl_id)

    # Generate requests - either based on a distribution or taken from Albatross - and corresponding passenger data
    if not params.get('albatross', False):
        inData = generate_demand(inData, params, avg_speed = True)   
        # Convert data to right input for OpenTripPlanner - i.e. with coordinates of origins and destinations
        inData.requests['origin_x'] = inData.requests.apply(lambda row: inData.nodes.x[row.origin], axis=1)
        inData.requests['origin_y'] = inData.requests.apply(lambda row: inData.nodes.y[row.origin], axis=1)
        inData.requests['destination_x'] = inData.requests.apply(lambda row: inData.nodes.x[row.destination], axis=1)
        inData.requests['destination_y'] = inData.requests.apply(lambda row: inData.nodes.y[row.destination], axis=1)
    else:
        inData = load_albatross_proc(inData, params, avg_speed=True)
        inData.requests = sample_from_alba_different_treq(inData, params) # possibly replicating (with different time) if nP is larger than size of Albatross dataset
        inData.requests['treq'] = inData.requests.apply(lambda x: x.treq.replace(year=params.t0.year, month=params.t0.month, day=params.t0.day), axis=1)
        inData.requests['tarr'] = inData.requests.apply(lambda x: x.tarr.replace(year=params.t0.year, month=params.t0.month, day=params.t0.day), axis=1)

    # Save trip properties
    dem_type = 'albatross' if params.get('albatross',False) else 'distribution'
    inData.requests = inData.requests.set_index('pax_id')
    inData.requests.to_csv(os.path.join('MaaSSim','data','demand','{}'.format(params.city.split(",")[0]),'{}'.format(dem_type),'preprocessed.csv'))
    # Save (only) required input for OTP
    inData.requests[['treq','origin_x','origin_y','destination_x','destination_y']].to_csv(os.path.join('MaaSSim','preprocessing','OTP_input','{}'.format(params.city.split(",")[0]),'georequests_{}.csv'.format(dem_type)),index_label=['pax_id'])


input_for_OTP(params=params)