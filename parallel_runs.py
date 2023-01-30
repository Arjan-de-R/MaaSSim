import sys
from MaaSSim.simulators_d2d import simulate_parallel, simulate
from MaaSSim.utils import save_config, get_config, load_G, generate_demand
from MaaSSim.d2d_sim import *
sys.path.append('..')

params = get_config('data/config/ams_2B.json')
params.parallel.nThread = 25
params.parallel.nReplications = 5
params.paths.G = 'data/graphs/Amsterdam.graphml'
params.paths.skim = 'data/graphs/Amsterdam.csv'

# Model settings
params.nD = 200
params.nP = 20000
params.nV = 100

# Other day-to-day settings
params.evol.drivers.inform.prob_start = 0.05 # probability of being informed at start of sim
params.evol.drivers.inform.beta = 0.1 # information transmission rate
params.evol.drivers.regist.beta = 0.2 # registration choice model parameter
params.evol.drivers.regist.cost_comp = 20 # daily share of registration costs (euros)
params.evol.drivers.particip.beta = 0.1 # participation choice model parameter
params.evol.travellers.inform.prob_start = 0.05
params.evol.travellers.inform.beta = 0.1
params.evol.travellers.min_prob = 0.05 # filtering criterion, when probability is lower when waiting time is zero, never consider RS

# Financial settings
params.platforms.base_fare = 1.4 #euro
params.platforms.fare = 1.5 #euro/km
params.platforms.min_fare = 6 # euro
params.platforms.comm_rate = 0.25 #rate

def sample_space():
    # analysis of behavioural parameters
    space = DotMap()
    space.nP = [10000,20000,30000,40000,50000]
    return space


simulate_parallel(params=params, search_space=sample_space())
