import time
import sys
#import scipy # various algorithms
from matplotlib import pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from itertools import chain
from utilities.functions import generate_traces_gill, viterbi, viterbi_compound, viterbi_cp_init
from utilities.cpHMM_viterbi import cpEM_viterbi, cpEM_viterbi_full
from utilities.stack_decoder import decode_cp
import os
import csv

#------------------------------------Routine Variable Definitions------------------------------------------------------#
#Nax number of iterations permitted
max_iter=1000
# Seconds per time step
dt = 5.1
n_inf = 500
# set num cores to use
num_inf_cores = 16 #multiprocessing.cpu_count()
# Set number of initialization routines
#n_init = 1
# set num cores to use
#num_init_cores = multiprocessing.cpu_count()
#Max num permitted paths in stack
max_stack = 250
#Estimate noise
estimate_noise = 0
#-------------------------------------"True" Variable Definitions------------------------------------------------------#
# noise
sigma = 25
# memory
w = 30
# Fix trace length for now
T = 200
#Num states
K = 3
# Number of traces per batch
batch_size = 100
# Set transition rate matrix for system
if K == 3:
    R = np.array([[-.006, .009, .005], [.004, -.014, .02], [.002, .005, -.025]]) * dt

elif K == 2:
    R = np.array([[-.004, .014], [.004, -.014]]) * dt

#A = scipy.linalg.expm(R, q=None)
#print(A)
# Set emission levels
if K == 3:
    v = np.array([0.0, 25.0, 50.0])
elif K == 2:
    v = np.array([0.0, 25.0])

# Initial stat pdf
if K == 3:
    pi = [.8,.1,.1]
elif K == 2:
    pi = [.8, .2]

#------------------------------------------Inference Init Variables----------------------------------------------------#
if K == 3:
    v_prior = np.array([   0,   20.0,  55.0])
    A_prior = np.array([[ .9,   .07,   .1],
                        [ .05,   .85,   .1],
                        [ .05,   .08,   .8]])
elif K == 2:
    v_prior = [0,35]
    A_prior = np.array([[.8, .2],
                        [.2, .8]])
sigma_prior = 35

#Degree of flexibility to allow in param initiations (2 = +/- full variable value)
A_temp = 2
v_temp = 1
sigma_temp = 1

#-----------------------------------------------Write Paths------------------------------------------------------------#
# Set test name
test_name = "mike_params_3state_fullstop_w30_t5_stack250"
# Set writepath for results
outpath = '../results/decode_validation/'
# Set project name (creates subfolder)
subfolder_name = test_name
writepath = os.path.join(outpath, subfolder_name)
if not os.path.isdir(outpath):
    os.makedirs(writepath)

if not os.path.isdir(os.path.join(outpath, subfolder_name, 'plots')):
    os.makedirs(os.path.join(outpath, subfolder_name, 'plots'))


#Define function to call viterbi fit in parallel
def runit(init_set, fluo,pi):
    A_init = init_set[0]
    v_init = init_set[1]
    sigma_init = init_set[2]
    A_list, v_list, logL_list, sigma_list = cpEM_viterbi_full(fluo=fluo, A_init=A_init, v_init=v_init, noise_init=sigma_init, pi0=pi, w=w, use_viterbi=0,estimate_noise=estimate_noise, n_groups=5, max_stack=max_stack, max_iter=max_iter, eps=10e-4)
    return np.exp(A_list[-1]), v_list[-1], logL_list[-1], sigma_list[-1]

if __name__ == "__main__":
    #Set number of inference routines

    print("Writing to: " + os.path.join(outpath, subfolder_name))

    #Write true param values
    with open(os.path.join(outpath, subfolder_name, test_name + '_true_values.csv'), 'wb') as inf_out:
        writer = csv.writer(inf_out)
        R_flat = np.reshape(R, K ** 2).tolist()
        v_best = v
        row = list(chain(*[R_flat, v_best.tolist(), [sigma], pi]))
        writer.writerow(row)
    #------------------------------------------Generate Traces---------------------------------------------------------#
    promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn = \
        generate_traces_gill(w, T, batch_size, r_mat=R, v=v, noise_level=sigma, alpha=0.0, pi0=pi)

    #Generate down-sampled versions for VT inference
    slow_traces = []
    ds_rate = 5
    slow_tp = ds_rate*np.arange(T/ds_rate)
    for fluo in fluo_states:
        fluo_arr = np.array(fluo)
        #f_slow =  np.interp(slow_tp, range(T), fluo, left=None, right=None, period=None)
        slow_traces.append(fluo_arr[slow_tp])

    # ----------------------------------Use VT fitting to Estimate Init Values-----------------------------------------#
    """
    v_raw = np.array([0,50,100])
    A_raw = A_prior = np.array([[.8,.1,.1],[.1,.8,.1],[.3,.2,.5]])
    sigma_raw = 25
    raw_init_list = []
    for i in xrange(n_init):
        deltaA = (np.random.rand(K, K) - .5) * A_raw * 2
        deltaV = np.random.rand(K)  * v_raw * 10
        v_init = v_raw + deltaV
        v_init[np.where(v_init < 0)[0]] = 0

        A_init = deltaA + A_prior
        A_init = A_init / np.tile(np.sum(A_init, axis=0), (K, 1))

        sigma_init = sigma_raw + (np.random.rand() - .5) * sigma_raw * 2
        raw_init_list.append([A_init, v_init, sigma_init])

    print("Running Initial VT Param Estimation...")
    init_time = time.time()
    init_results = Parallel(n_jobs=num_init_cores)(delayed(init)(raw_init_list[i]) for i in range(n_init))
    print("Runtime (unit): " + str(time.time() - init_time))
    init_logL_list = np.array([init_results[i][2] for i in xrange(n_inf)])
    init_max_id = np.where(init_logL_list == np.max(init_logL_list))[0]
    best_results = init_results[init_max_id]
    print(best_results)
    """
    # -------------------------------------Generate Initialization Values----------------------------------------------#

    init_list = []
    for i in xrange(n_inf):
        deltaA = (np.random.rand(K,K) - .5) * A_prior * A_temp
        deltaV = (np.random.rand(K) - .5) * v_prior * v_temp
        v_init = v_prior + deltaV
        v_init[np.where(v_init < 0)[0]] = 0

        A_init = deltaA + A_prior
        A_init = A_init / np.tile(np.sum(A_init,axis=0),(K,1))

        sigma_init = sigma_prior + (np.random.rand()-.5)*sigma_prior*sigma_temp
        init_list.append([A_init, v_init, sigma_init])

    # -------------------------------------------Conduct Inference-----------------------------------------------------#
    print("Running EM Optimization...")
    init_time = time.time()
    pi_list = [pi] * n_inf
    fluo_cp = [fluo_states] * n_inf
    inf_results = Parallel(n_jobs=num_inf_cores)(delayed(runit)(init_set=p0, fluo=fluo_states, pi=pi) for p0 in init_list)
    print("Runtime: " + str(time.time() - init_time))

    #Find routine with highest likelihood score
    logL_list = np.array([inf_results[i][2] for i in xrange(n_inf)])
    max_id = np.argmax(logL_list)
    best_results = inf_results[max_id]
    A_inf = best_results[0]
    v_inf = best_results[1]
    logL_inf = best_results[2]
    print(best_results)

    print("Running Stack Decoder to Fit Traces...")
    seq_out, f_out, v_out, logL_out = decode_cp(fluo_states, np.log(A_inf), np.log(pi), v_inf, w, sigma, stack_depth=max_stack)

    # Write best param estimates to csv
    with open(os.path.join(outpath, subfolder_name, test_name + '_best_results.csv'), 'wb') as inf_out:
        writer = csv.writer(inf_out)
        A_flat = np.reshape(best_results[0], K ** 2).tolist()
        v_best = best_results[1]
        row = list(chain(*[A_flat, v_best.tolist(), [best_results[2]], [best_results[3]], pi]))
        writer.writerow(row)

        # write full inference results to csv
        for tr in xrange(n_inf):
            if tr == 0:
                write = 'wb'
            else:
                write = 'a'

            with open(os.path.join(outpath, subfolder_name, test_name + '_full_results.csv'), write) as full_out:
                writer = csv.writer(full_out)
                results = inf_results[tr]
                A_flat = np.reshape(results[0], K ** 2).tolist()
                v_best = results[1]
                row = list(chain(*[A_flat, v_best.tolist(), [results[2]], [results[3]], pi]))
                writer.writerow(row)

            with open(os.path.join(outpath, subfolder_name, test_name + '_initializations.csv'), write) as init_out:
                writer = csv.writer(init_out)
                results = init_list[tr]
                A_flat = np.reshape(results[0], K ** 2).tolist()
                v_inf = results[1]
                row = list(chain(*[A_flat, v_inf.tolist(), [results[2]], pi]))
                writer.writerow(row)

    #save plots
    for tr in xrange(batch_size):
        fig_fluo = plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 1, 1)
        ax.plot(fluo_states[tr], c='g', alpha=0.4, label='Actual')
        ax.plot(f_out[tr], c='b', label='Predicted')

        # plt.legend()
        fig_fluo.savefig(os.path.join(outpath, subfolder_name, 'plots', 'tr' + "_" + str(tr) + "_fluo.png"))
        plt.close()

        fig_prom = plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 1, 1)
        ax.plot(promoter_states[tr], c='g', alpha=0.4, label='Actual')
        ax.plot(v_out[tr], c='b', label='Predicted')
        plt.ylim([0, 1.1 * np.max(v)])
        # plt.legend()
        fig_prom.savefig(os.path.join(outpath, subfolder_name, 'plots', 'tr' + "_" + str(tr) + "_promoter.png"))
        plt.close()



