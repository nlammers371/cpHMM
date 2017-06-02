import time
import sys
#import scipy # various algorithms
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from itertools import chain
from utilities.functions import generate_traces_gill, viterbi, viterbi_compound, viterbi_cp_init
from utilities.cpHMM_viterbi import cpEM_viterbi, cpEM_viterbi_full
from utilities.stack_decoder import decode_cp
import os
import csv

#------------------------------------------------Top Level Exp Specifications------------------------------------------#
###Project Params
project_folder = 'stack_validation'
test_name = 'test'
#test_name = 'basic_eve_10sec'
###Routine Params
#Num Independent Runs for final inference step
final_iters = 2
#Num Paths to Track for final inf
f_stack_size = 5
#num init runs
init_iters = 5
#size init stack
init_stack_size = 5
###Biological System Params
#num states
num_states = 2
#Time Resolution
dT = 10.0
#Type of rate matrix
exp_type = 'eve2'
#Routine Param Type
rType = 'basic'
#Set Core Num
cores = multiprocessing.cpu_count()
#Class of Routine param values to use for initial search to narrow range of possible values
class RPInitBase(object):
    def __init__(self, n_states, n_runs=init_iters, n_cores=cores, n_stack=init_stack_size):
        # ------------------------------------Routine Variable Definitions------------------------------------------------------#
        #num_states
        self.K = n_states
        #Max number of iterations permitted
        self.max_iter=1000
        #N Separate Inferences
        self.n_inf = n_runs
        # set num cores to use
        self.num_inf_cores = n_cores
        #Max num permitted paths in stack
        self.max_stack = n_stack
        #Estimate noise
        self.estimate_noise = 1
        # ------------------------------------------Inference Init Variables----------------------------------------------------#
        if n_states == 3:
            self.v_prior = np.array([0, 50.0, 50.0])
            self.A_prior = np.array([[.8, .1, .1],
                                [.1, .8, .1],
                                [.1, .1, .8]])
        elif n_states == 2:
            self.v_prior = [0, 50.0]
            self.A_prior = np.array([[.8, .2],
                                [.2, .8]])
            self.sigma_prior = self.v_prior[1]

        # Degree of flexibility to allow in param initiations (2 = +/- full variable value)
        self.A_temp = 2
        self.v_temp = 2
        self.sigma_temp = 2


class RPFinalBase(object):
    def __init__(self, n_runs=final_iters, n_cores=cores, n_stack=f_stack_size):
        # Max number of iterations permitted
        self.max_iter = 1000
        # N Separate Inferences
        self.n_inf = n_runs
        # set num cores to use
        self.num_inf_cores = n_cores
        # Max num permitted paths in stack
        self.max_stack = n_stack
        # Estimate noise
        self.estimate_noise = 0
        # Degree of flexibility to allow in param initiations (2 = +/- full variable value)
        self.A_temp = .25
        self.v_temp = .25
        self.sigma_temp = .25

#-------------------------------------"True" Variable Definitions------------------------------------------------------#
class Eve2Exp(object):
    def __init__(self, n_states, dt, n_traces=100, tr_len=200):
        # memory
        self.w = int(160 / dt)
        # Fix trace length for now
        self.T = tr_len
        # Number of traces per batch
        self.batch_size = n_traces
        # Set transition rate matrix for system
        if n_states == 3:
            self.R = np.array([[-.006, .009, .005], [.004, -.014, .02], [.002, .005, -.025]]) * dt
        elif n_states == 2:
            self.R = np.array([[-.004, .014], [.004, -.014]]) * dt
        # Set emission levels
        if n_states == 3:
            self.v = np.array([0.0, 25.0, 50.0])
        elif n_states == 2:
            self.v = np.array([0.0, 25.0])
        # noise
        self.sigma = .05 * self.v[1] * self.w
        # Initial stat pdf
        if n_states == 3:
            self.pi = [.8,.1,.1]
        elif n_states == 2:
            self.pi = [.8, .2]

class GenericExp(object):
    def __init__(self, n_states, dt, n_traces=100, tr_len=200):
        # memory
        self.w = int(160 / dt)
        # Fix trace length for now
        self.T = tr_len
        # Number of traces per batch
        self.batch_size = n_traces
        # Set transition rate matrix for system
        if n_states == 3:
            self.R = np.array([[-.0016, .008, .008], [.01, -.02, .01], [.012, .08, -.02]]) * dt
        elif n_states == 2:
            self.R = np.array([[-.012, .014], [.012, -.014]]) * dt
        # Set emission levels
        if n_states == 3:
            self.v = np.array([0.0, 20.0, 70.0])
        elif n_states == 2:
            self.v = np.array([0.0, 45.0])
        # noise
        self.sigma = .05 * self.v[1] * self.w
        # Initial stat pdf
        if n_states == 3:
            self.pi = [.8, .1, .1]
        elif n_states == 2:
            self.pi = [.8, .2]


#-----------------------------------------------Write Paths------------------------------------------------------------#
if exp_type == 'eve2':
    expClass = Eve2Exp(n_states=num_states, dt=dT)
else:
    expClass = GenericExp(n_states=num_states, dt=dT)
if rType == 'basic':
    RoutineParamsInit = RPInitBase(n_states=num_states)
    RoutineParamsFinal = RPFinalBase()

# Set test name
write_name = exp_type + '_' + str(num_states) + 'state_' + test_name
# Set writepath for results
outpath = '../results/' + project_folder + '/'

writepath = os.path.join(outpath, write_name)

#Define function to call viterbi fit in parallel
def runit(init_set, fluo,pi,est_noise):
    A_init = init_set[0]
    v_init = init_set[1]
    sigma_init = init_set[2]
    A_list, v_list, logL_list, sigma_list, iters, run_time = cpEM_viterbi_full(fluo=fluo,
                                                                               A_init=A_init,
                                                                               v_init=v_init,
                                                                               noise_init=sigma_init,
                                                                               pi0=pi,
                                                                               w=expClass.w,
                                                                               use_viterbi=0,
                                                                               estimate_noise=est_noise,
                                                                               n_groups=5,
                                                                               max_stack=RoutineParamsFinal.max_stack,
                                                                               max_iter=RoutineParamsFinal.max_iter,
                                                                               eps=10e-4)

    return np.exp(A_list[-1]), v_list[-1], logL_list[-1], sigma_list[-1], iters, run_time

if __name__ == "__main__":

    if not os.path.isdir(writepath):
        os.makedirs(writepath)
    else:
        print("Warning: preexisting directory detected. Appending datetime")
        write_name = write_name + '_' + str(time.time())
        writepath = os.path.join(outpath, write_name)
        os.makedirs(writepath)

    #Set number of inference routines
    print("Writing to: " + writepath + '...')
    #Write true param values
    with open(os.path.join(writepath, test_name + '_true_values.csv'), 'wb') as inf_out:
        writer = csv.writer(inf_out)
        R_flat = np.reshape(expClass.R, RoutineParamsInit.K ** 2).tolist()
        row = list(chain(*[R_flat, expClass.v.tolist(), [expClass.sigma], expClass.pi]))
        writer.writerow(row)
    #------------------------------------------Generate Traces---------------------------------------------------------#
    promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn = \
        generate_traces_gill(expClass.w, expClass.T, expClass.batch_size, r_mat=expClass.R, v=expClass.v, noise_level=expClass.sigma, alpha=0.0, pi0=expClass.pi)

    # -------------------------------------Conduct First Pass Initialization Values----------------------------------------------#

    init_list = []
    K_init = RoutineParamsInit.K
    for i in xrange(RoutineParamsInit.n_inf):
        deltaA = (np.random.rand(K_init,K_init) - .5) * RoutineParamsInit.A_prior * RoutineParamsInit.A_temp
        deltaV = (np.random.rand(RoutineParamsInit.K) - .5) * RoutineParamsInit.v_prior * RoutineParamsInit.v_temp
        v_init = RoutineParamsInit.v_prior + deltaV
        v_init[np.where(v_init < 0)[0]] = 0

        A_init = deltaA + RoutineParamsInit.A_prior
        A_init = A_init / np.tile(np.sum(A_init,axis=0),(K_init,1))

        sigma_init = RoutineParamsInit.sigma_prior + (np.random.rand()-.5)*RoutineParamsInit.sigma_prior*RoutineParamsInit.sigma_temp
        init_list.append([A_init, v_init, sigma_init])

    print("Running Initialization EM...")
    init_time = time.time()
    init_results = Parallel(n_jobs=RoutineParamsInit.num_inf_cores)(
        delayed(runit)(init_set=p0, fluo=fluo_states, pi=expClass.pi, est_noise=RoutineParamsInit.estimate_noise) for p0 in init_list)
    print("Runtime: " + str(time.time() - init_time))

    # -------------------------------------------Conduct Inference-----------------------------------------------------#
    inf_list = []
    #Use results of initial search to initialize final inference runs
    init_ids = len(init_results)
    init_weights = np.array([init_results[i][2] for i in xrange(len(init_results))])
    init_weights = init_weights / np.sum(init_weights)
    for i in xrange(RoutineParamsFinal.n_inf):
        init_id = np.random.choice(init_ids, p = init_weights)
        A_init = init_results[init_id][0]
        v_init = init_results[init_id][1]

        deltaA = (np.random.rand(K_init, K_init) - .5) * A_init * RoutineParamsFinal.A_temp
        deltaV = (np.random.rand(K_init) - .5) * v_init * RoutineParamsFinal.v_temp
        v_init += deltaV
        v_init[np.where(v_init < 0)[0]] = 0

        A_init += deltaA
        A_init[np.where(A_init < 0)[0]] = 0
        A_init = A_init / np.tile(np.sum(A_init, axis=0), (K_init, 1))

        sigma_init = init_results[init_id][3] + (np.random.rand() - .5) * init_results[init_id][3] * RoutineParamsFinal.sigma_temp
        inf_list.append([A_init, v_init, sigma_init])

    print("Running EM Optimization...")
    init_time = time.time()
    inf_results = Parallel(n_jobs=RoutineParamsFinal.num_inf_cores)(delayed(runit)(init_set=p0, fluo=fluo_states, pi=expClass.pi, est_noise=RoutineParamsFinal.estimate_noise) for p0 in inf_list)
    print("Runtime: " + str(time.time() - init_time))

    #Find routine with highest likelihood score
    logL_list = np.array([inf_results[i][2] for i in xrange(RoutineParamsFinal.n_inf)])
    max_id = np.argmax(logL_list)
    best_results = inf_results[max_id]
    print("Optimal Params: ")
    print("")
    print(best_results)

    # Write best param estimates to csv
    with open(os.path.join(writepath, 'best_results.csv'), 'wb') as inf_out:
        writer = csv.writer(inf_out)
        A_flat = np.reshape(best_results[0], RoutineParamsInit.K ** 2).tolist()
        v_best = best_results[1]
        row = list(chain(*[A_flat, v_best.tolist(), [best_results[2]], [best_results[3]], expClass.pi]))
        writer.writerow(row)

        # write full inference results to csv
        for n in xrange(RoutineParamsFinal.n_inf):
            if n == 0:
                write = 'wb'
            else:
                write = 'a'

            with open(os.path.join(writepath, 'full_results.csv'), write) as full_out:
                writer = csv.writer(full_out)
                results = inf_results[n]
                A_flat = np.reshape(results[0], RoutineParamsInit.K ** 2).tolist()
                row = list(chain(*[A_flat, inf_results[n][1].tolist(), [inf_results[n][2]], [inf_results[n][3]], expClass.pi, [inf_results[n][4]], [inf_results[n][5]]]))
                writer.writerow(row)

            with open(os.path.join(writepath, 'initializations.csv'), write) as init_out:
                writer = csv.writer(init_out)
                results = init_list[n]
                A_flat = np.reshape(results[0], RoutineParamsInit.K ** 2).tolist()
                row = list(chain(*[A_flat, inf_results[n][1].tolist(), [inf_results[n][2]], expClass.pi]))
                writer.writerow(row)

#Save Simulation Variables to File
initVars = vars(RoutineParamsInit)
finVars = vars(RoutineParamsFinal)
simVars = vars(expClass)
# {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
# now dump this in some way or another
initOut = open(os.path.join(writepath, "init_params.txt"), "a")
for item in initVars.items():
    initOut.write(str(item))
finOut = open(os.path.join(writepath, "final_params.txt"), "a")
for item in finVars.items():
    finOut.write(str(item))
simOut = open(os.path.join(writepath,"sim_params.txt"), "a")
for item in simVars.items():
    simOut.write(str(item))