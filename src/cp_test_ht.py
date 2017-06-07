from matplotlib import pyplot as plt
import time
import sys
import scipy as sp # various algorithms
from scipy import linalg, stats
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from itertools import chain
import math
from utilities.functions import generate_traces_gill, viterbi, viterbi_compound, viterbi_cp_init
from utilities.cpHMM_viterbi import cpEM_viterbi, cpEM_viterbi_full
from utilities.cpHMM_BW import cpEM_BW
from utilities.stack_decoder import decode_cp
import os
import csv
#------------------------------------------------Top Level Exp Specifications------------------------------------------#
###Project Params
project_folder = 'method_validation'
project_subfolder = 'ZZ_test_run'
test_name = 'bw_test'
#---------------------------------------Routine Params---------------------------------------#
#Specify whether to use truncated BW or Stack Decoder Viterbi
model = 'bw'
#Num Independent Runs for final inference step
final_iters = 5
#Num Paths to Track for final inf (Stack Decoder Only)
decoder_stack_size = 25
#Depth of Alpha and Beta Matrices (Truncated Bw only)
bw_stack_size = 20
#Estimate Noise in Final Sim?
est_sigma_final = 1
#Set prior regarding switching time scale (in seconds)
switch_scale = 60
#Set temp params that dictate degree of flexibility allowed for initialization
R_temp = 1
v_temp = 1
sigma_temp = 1

#-------------------------------------Experimental Sim Params-------------------------------------------#
#Fraction of total mem time steps needed to transcribe MS2
alpha_frac = 0
#Set Corr term between two promoters (1 = independence, >1 = pos correlation, <1 = negative correlation)
corr = 1
#num activity states
num_states = 3
#Time Resolution
dT = 10.1
#Number of Traces
n_traces = 20
#Trace Length (in time steps)
trace_length = 200
#set level of system noise (relative to w*v[1])
snr = .05
#Type of rate matrix
exp_type = 'eve2short'
#Routine Param Type
rType = 'basic'
#Set Core Num
cores = 20 #multiprocessing.cpu_count()
class RPFinalBase(object):
    def __init__(self):
        self.model = model
        self.eps = 10e-5
        # Max number of iterations permitted
        self.max_iter = 1000
        # N Separate Inferences
        self.n_inf = final_iters
        # set num cores to use
        self.num_inf_cores = cores
        # Max num permitted paths in stack
        self.max_decoder_stack = decoder_stack_size
        # Max num permitted rows to track in alpha/beta matrices
        self.max_bw_stack = bw_stack_size
        # Estimate noise
        self.estimate_noise = est_sigma_final
        # Degree of flexibility to allow in param initiations (2 = +/- full variable value)
        self.R_temp = R_temp
        self.v_temp = v_temp
        self.sigma_temp = sigma_temp

#-------------------------------------"True" Variable Definitions------------------------------------------------------#
class Eve2ExpRealistic(object):
    def __init__(self):
        #Degree of correlation btw two promoters (only relevant for 3+ state case)
        self.promoter_correlation = corr
        #Temporal Resolution of Experiment
        self.dt = dT
        #elongation time
        self.t_elong = 160
        # memory
        self.w = int(np.round(self.t_elong / dT))
        # alpha
        self.alpha = alpha_frac * self.w
        # Fix trace length for now
        self.T = trace_length
        # Number of traces per batch
        self.batch_size = n_traces
        # Set transition rate matrix for system
        if num_states == 3:
            self.R = np.array([[-.008, .009*corr, 0.0],
                               [.008, -.014*corr, .04],
                               [0.0,   .005*corr, -.04]]) * self.dt

        elif num_states == 2:
            self.R = np.array([[-.004, .014],
                               [.004, -.014]]) * self.dt
        # Set emission levels
        if num_states == 3:
            self.v = np.array([0.0, 25.0, 50.0])
        elif num_states == 2:
            self.v = np.array([0.0, 25.0])
        #snr
        self.snr = snr
        # noise
        self.sigma = snr * self.v[1] *self.w
        # Initial stat pdf
        if num_states == 3:
            self.pi = [.8,.1,.1]
        elif num_states == 2:
            self.pi = [.8, .2]

        self.K = num_states

#Class with assuming low mem but otherwise realistic param values
class Eve2ExpShort(object):
    def __init__(self):
        #Degree of correlation btw two promoters (only relevant for 3+ state case)
        self.promoter_correlation = corr
        #Temporal Resolution of Experiment
        self.dt = dT
        #elongation time
        self.t_elong = 50
        # memory
        self.w = int(np.round(self.t_elong / dT))
        # alpha
        self.alpha = alpha_frac * self.w
        # Fix trace length for now
        self.T = trace_length
        # Number of traces per batch
        self.batch_size = n_traces
        # Set transition rate matrix for system
        if num_states == 3:
            self.R = np.array([[-.008, .015*corr, 0.0],
                               [.008, -.019*corr, .03],
                               [0.0, .004*corr, -.03]]) * self.dt

        elif num_states == 2:
            self.R = np.array([[-.004, .014],
                               [.004, -.014]]) * self.dt
        # Set emission levels
        if num_states == 3:
            self.v = np.array([0.0, 25.0, 50.0])
        elif num_states == 2:
            self.v = np.array([0.0, 25.0])
        #snr
        self.snr = snr
        # noise
        self.sigma = snr * self.v[1] *self.w
        # Initial stat pdf
        if num_states == 3:
            self.pi = [.8,.1,.1]
        elif num_states == 2:
            self.pi = [.8, .2]

        self.K = num_states


class GenericExp(object):
    def __init__(self):
        # Temporal Resolution of Experiment
        self.dt = dT
        # elongation time
        self.t_elong = 160
        # memory
        self.w = int(self.t_elong / dT)
        # alpha
        self.alpha = alpha_frac * self.w
        # Fix trace length for now
        self.T = trace_length
        # Number of traces per batch
        self.batch_size = n_traces# Set transition rate matrix for system
        if num_states == 3:
            self.R = np.array([[-.014, .007, .007], [.007, -.014, .007], [.007, .007, -.014]]) * self.dt
        elif num_states == 2:
            self.R = np.array([[-.014, .014], [.014, -.014]]) * self.dt
        # Set emission levels
        if num_states == 3:
            self.v = np.array([0.0, 20.0, 70.0])
        elif num_states == 2:
            self.v = np.array([0.0, 45.0])
        # snr
        self.snr = snr
        # noise
        self.sigma = snr * self.v[1] * self.w  # Initial stat pdf
        if num_states == 3:
            self.pi = [.8, .1, .1]
        elif num_states == 2:
            self.pi = [.8,.2]
#-----------------------------------------------Write Paths------------------------------------------------------------#
if exp_type == 'eve2':
    expClass = Eve2ExpRealistic()
elif exp_type == 'eve2short':
    expClass = Eve2ExpShort()
elif exp_type == 'generic':
    expClass = GenericExp()

RoutineParamsFinal = RPFinalBase()

# Set test name
write_name = exp_type + '_' + str(num_states) + 'state_' + test_name
# Set writepath for results
outpath = '../results/' + project_folder + '/' + project_subfolder + '/'
writepath = os.path.join(outpath, write_name)
#Define function to call viterbi fit in parallel
def runit_viterbi(init_set, fluo,pi,est_noise):
    A_init = init_set[0]
    v_init = init_set[1]
    sigma_init = init_set[2]
    A_list, v_list, logL_list, sigma_list, iters, run_time = cpEM_viterbi_full(fluo=fluo,
                                                                               A_init=A_init,
                                                                               v_init=v_init,
                                                                               noise_init=sigma_init,
                                                                               pi0=expClass.pi,
                                                                               w=expClass.w,
                                                                               use_viterbi=0,
                                                                               estimate_noise=est_noise,
                                                                               n_groups=5,
                                                                               alpha=expClass.alpha,
                                                                               max_stack=RoutineParamsFinal.max_decoder_stack,
                                                                               max_iter=RoutineParamsFinal.max_iter,
                                                                               eps=RoutineParamsFinal.eps)

    return np.exp(A_list[-1]), v_list[-1], sigma_list[-1], logL_list[-1], iters, run_time

def runit_bw(init_set, fluo,pi,est_noise):

    A_init = init_set[0]
    v_init = init_set[1]

    sigma_init = init_set[2][0]

    A_list, v_list, logL_list, sigma_list, iters, run_time = cpEM_BW(  fluo=fluo,
                                                                       A_init=A_init,
                                                                       v_init=v_init,
                                                                       noise_init=sigma_init,
                                                                       pi0=expClass.pi,
                                                                       w=expClass.w,
                                                                       estimate_noise=est_noise,
                                                                       max_stack=RoutineParamsFinal.max_bw_stack,
                                                                       max_iter=RoutineParamsFinal.max_iter,
                                                                       eps=RoutineParamsFinal.eps,
                                                                       verbose=0)

    return np.exp(A_list[-1]), v_list[-1], sigma_list[-1], logL_list[-1], iters, run_time

#Convenience function for truncated random normal pdf
def t_norm(lower, upper, mu, sigma, N):
    samples = sp.stats.truncnorm.rvs(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
    return samples


if __name__ == "__main__":

    if not os.path.isdir(writepath):
        os.makedirs(writepath)
    else:
        print("Warning: preexisting directory detected. Appending datetime")
        write_name = write_name + '_' + str(np.round(time.time()))
        writepath = os.path.join(outpath, write_name)
        os.makedirs(writepath)

    # Set number of inference routines
    print("Writing to: " + writepath + '...')
    # Write true param values
    with open(os.path.join(writepath, 'true_values.csv'), 'wb') as inf_out:
        writer = csv.writer(inf_out)
        A_flat = np.reshape(sp.linalg.expm(expClass.R*dT) , expClass.K ** 2).tolist()
        R_flat = np.reshape(expClass.R / dT, expClass.K ** 2).tolist()
        row = list(chain(*[[1], A_flat, R_flat, expClass.v.tolist(), [expClass.sigma], expClass.pi]))
        writer.writerow(row)

    _, fluo_states, _, _ = \
        generate_traces_gill(expClass.w, expClass.T, expClass.batch_size, r_mat=expClass.R, v=expClass.v,
                             noise_level=expClass.sigma, alpha=expClass.alpha, pi0=expClass.pi)

    # -------------------------------------Conduct First Pass Initialization Values----------------------------------------------#
    # First Calculate Prior for v's based upon statistics of experimental data
    nine_five_p = np.percentile(list(chain(*fluo_states)), 95)
    if expClass.K == 2:
        v_prior = np.array([0,nine_five_p / expClass.w])
    elif expClass.K == 3:
        v_prior = np.array([0, nine_five_p / expClass.w, 2 * nine_five_p / expClass.w])
    else:
        print("Warning: Undefined or improper state num value")
        sys.exit(1)
    # Use rate prior to ensure that all transition prob matrices used for initialization
    # equate to a Rate matrix of proper form
    R_diag = 1.0 / switch_scale
    if expClass.K == 2:
        R_prior = np.array([[-R_diag,R_diag],
                            [R_diag,-R_diag]])
    if expClass.K == 3:
        R_prior = np.array([[-R_diag,.5*R_diag,.25*R_diag],
                   [.75*R_diag,-R_diag,.75*R_diag],
                   [.25 * R_diag, .5*R_diag, -R_diag]])
    init_list = []
    K_init = expClass.K

    for i in xrange(RoutineParamsFinal.n_inf):
        deltaR = np.reshape(t_norm(lower=0, upper=10, mu=1.0, sigma=R_temp, N=K_init**2),(K_init,K_init))
        deltaV = t_norm(lower=0, upper=10, mu=1.0, sigma=v_temp, N=K_init)
        deltaSig = t_norm(lower=0, upper=10, mu=1.0, sigma=sigma_temp, N=1)
        v_init = v_prior*deltaV

        R_init = R_prior*deltaR
        R_init = R_init - np.diag(np.diag(R_init)) - np.diag(np.sum(R_init - np.diag(np.diag(R_init)),axis=0))
        A_init = sp.linalg.expm(R_init*dT)
        sigma_init = snr*nine_five_p*deltaSig*(1.0 - (np.random.rand()-.5) / 2.0)
        init_list.append([A_init, v_init, sigma_init, R_init])

    # -------------------------------------------Conduct Inference-----------------------------------------------------#
    print("Running EM Optimization...")
    init_time = time.time()
    if model == 'viterbi':
        inf_results = Parallel(n_jobs=RoutineParamsFinal.num_inf_cores)(
                                                    delayed(runit_viterbi)(init_set=p0, fluo=fluo_states, pi=expClass.pi,
                                                    est_noise=RoutineParamsFinal.estimate_noise) for p0 in init_list)
    elif model == 'bw':
        inf_results = Parallel(n_jobs=RoutineParamsFinal.num_inf_cores)(
                                                    delayed(runit_bw)(init_set=p0, fluo=fluo_states, pi=expClass.pi,
                                                    est_noise=RoutineParamsFinal.estimate_noise) for p0 in init_list)
    print("Runtime: " + str(time.time() - init_time))

    # Find routine with highest likelihood score
    print(inf_results)
    logL_list = np.array([inf_results[i][3] for i in xrange(RoutineParamsFinal.n_inf)])
    logL_list = [value for value in logL_list if not math.isnan(value)]
    print(logL_list)
    max_id = np.argmax(logL_list)
    best_results = inf_results[max_id]
    print("Optimal Params: ")
    print("")
    print(best_results)
    # Write best param estimates to csv
    with open(os.path.join(writepath, 'best_results.csv'), 'wb') as inf_out:
        writer = csv.writer(inf_out)
        A_flat = np.reshape(best_results[0], expClass.K ** 2).tolist()
        try:
            R_flat = np.reshape(sp.linalg.logm(best_results[0]) / dT, expClass.K ** 2).tolist()
        except:
            R_flat = np.zeros(expClass.K**2)
        v_best = best_results[1]
        row = list(chain(*[[max_id],A_flat, R_flat, v_best.tolist(), [best_results[2]], [best_results[3]], expClass.pi,
                      [best_results[4]], [best_results[5]]]))
        writer.writerow(row)

        # write full inference results to csv
        with open(os.path.join(writepath, 'full_results.csv'), 'wb') as full_out:
            writer = csv.writer(full_out)
            for n in xrange(RoutineParamsFinal.n_inf):
                results = inf_results[n]
                A_flat = np.reshape(results[0], expClass.K ** 2).tolist()
                try:
                    R_flat = np.reshape(sp.linalg.logm(results[0]) / dT, expClass.K ** 2).tolist()
                except:
                    continue
                row = list(chain(
                    *[[n], A_flat, R_flat, inf_results[n][1].tolist(), [inf_results[n][2]], [inf_results[n][3]], expClass.pi,
                      [inf_results[n][4]], [inf_results[n][5]]]))
                writer.writerow(row)

        with open(os.path.join(writepath, 'initializations.csv'), 'wb') as init_out:
            writer = csv.writer(init_out)
            for n in xrange(RoutineParamsFinal.n_inf):
                results = init_list[n]
                A_flat = np.reshape(results[0], expClass.K ** 2).tolist()
                R_flat = np.reshape(results[-1], expClass.K ** 2).tolist()
                row = list(chain(*[[n], A_flat, R_flat, inf_results[n][1].tolist(), [inf_results[n][2]], expClass.pi]))
                writer.writerow(row)

    # Save Simulation Variables to File
    initVars = vars(RoutineParamsFinal)
    finVars = vars(RoutineParamsFinal)
    simVars = vars(expClass)
    # {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
    # now dump this in some way or another
    initOut = open(os.path.join(writepath, "init_params.txt"), "w")
    for item in initVars.items():
        initOut.write(str(item))
    initOut.close()
    finOut = open(os.path.join(writepath, "final_params.txt"), "w")
    for item in finVars.items():
        finOut.write(str(item))
    finOut.close()
    simOut = open(os.path.join(writepath, "sim_params.txt"), "w")
    for item in simVars.items():
        simOut.write(str(item))
    simOut.close()
