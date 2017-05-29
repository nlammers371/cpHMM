import time
import sys
import scipy # various algorithms
from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import logsumexp
import math
from itertools import chain
import itertools
from functions import log_L_fluo, log_likelihood, viterbi_compound, viterbi, viterbi_cp_init
from stack_decoder import decode_cp

# Viterbi training approach that employes stack decoder for high complexity problems
def cpEM_viterbi(fluo, A_init, v_init, noise, pi0, w=1, use_viterbi=1, max_stack=100, max_iter=1000, eps=10e-4):
    """
    :param fluo: time series of fluorescent intensities (list of lists)
    :param A_init: Initial guess at the system's transition probability matrix (KxK)
    :param v_init: Initial guess at emissions vector (K)
    :param noise: Standard Deviation of fluo emissions (Taken as given at this stage)
    :param pi0: Initial state PDF (Taken as given)
    :param w: memory of system in time steps
    :param max_stack: if stack decoder is used for fitting, sets max stack size
    :param max_iter: Maximum number of iterations permitted
    :param eps: Termination criteria--minimum permissible percent change in estimated params
    :return: Infers A and v for set of sequences
    """
    min_val = 10e-10
    pi0_log = np.log(pi0)
    K = len(v_init)
    A_list = [np.log(A_init)]
    v_list = [v_init]
    logL_list = [-np.Inf]
    # Initialize variable to track percent change in likelihood
    delta = 1
    iter = 1
    while iter < max_iter:
        v_curr = v_list[iter - 1]
        A_log = A_list[iter - 1]
        # --------------------------Fwd Bkwd Algorithm for Each Sequence----------------------------------------------------#
        # store likelihood of each sequence given current parameter estimates
        seq_log_probs = []
        # store fits for each sequence for update calculations
        if use_viterbi:
            if w > 3:
                print("Warning: Attempting viterbi fits for high complexity data. Stack decoder recommended")
            _, _, seq_out, v_out, logL_out = viterbi_compound(fluo, A_log, v_curr, noise, pi0_log, w=w)
        else:
            seq_out, f_out, v_out, logL_out = decode_cp(fluo, A_log, pi0_log, v_curr, w, noise, stack_depth=max_stack)
        # ---------------------------------------Calculate Updated A and v--------------------------------------------------#

        # Update A
        A_new = np.zeros_like(A_log) + min_val
        for f, fluo_vec in enumerate(fluo):
            t_vec = range(1,len(fluo_vec))
            seq_fit = seq_out[f]
            #Count relevant transition events
            for k in xrange(K):
                for l in xrange(K):
                    rm_vec = []
                    for t in t_vec:
                        if seq_fit[t-1]==k and seq_fit[t]==l:
                            A_new[l,k] += 1
                            rm_vec.append(t)
                    #Attempt to cut down on redundant iterations
                    t_vec = [i for i in t_vec if i  not in rm_vec]

        A_new = A_new / np.tile(np.sum(A_new, axis=0), (K, 1))

        # Update v
        v_new = np.zeros_like(v_curr)
        b = []
        F_full = []
        for f, fluo_vec in enumerate(fluo):
            b.append(fluo_vec)
            s_lookup = list(chain(*[[0]*(w-1), seq_out[f]]))
            s_lookup = np.array(s_lookup)
            for t in xrange(len(fluo_vec)):
                ct = [0]*K
                for k in xrange(K):
                    ct[k] = len(np.where(s_lookup[t:t+w]==k)[0])
                F_full.append(ct)
        #Allocate arrays to store final values for v_new calc
        F_full = np.array(F_full)
        b = np.array(list(chain(*b)))
        F_square = np.zeros((K,K))
        b_vec = np.zeros(K)
        for k in xrange(K):
            F_square[k,:] = np.sum(F_full * F_full[:,k][:,np.newaxis], axis=0)
            b_vec[k] = np.sum(b * F_full[:,k])

        v_new = np.linalg.solve(F_square,b_vec)
        v_list.append(v_new)
        A_list.append(np.log(A_new))

        # Calculate log likelihood using param estimates from previous step
        logL_list.append(np.sum(logL_out))
        # Check % improvement in logL
        if iter > 1:
            print(np.sum(logL_out))
            print(logL_list[iter-1])
            print(v_new)
            delta = (logL_list[iter-1] - np.sum(logL_out)) / logL_list[iter-1]
            if delta < 0:
                print("Warning: Non-monotonic behavior in likelihood")
            #sys.exit(1)
        if iter % 10 == 0:
            print(np.sum(logL_out))
            print(v_new)
            print(A_new)
            print(delta)
        iter += 1

    return (A_list, v_list, logL_list)


# Add some bells and whistles to base model
def cpEM_viterbi_full(fluo, A_init, v_init, noise_init, pi0, n_groups=1, estimate_noise=0, w=1, use_viterbi=1, max_stack=100, max_iter=1000, eps=10e-4, verbose=False):
    """
    :param fluo: time series of fluorescent intensities (list of lists)
    :param A_init: Initial guess at the system's transition probability matrix (KxK)
    :param v_init: Initial guess at emissions vector (K)
    :param noise: Standard Deviation of fluo emissions (Taken as given at this stage)
    :param pi0: Initial state PDF (Taken as given)
    :param w: memory of system in time steps
    :param max_stack: if stack decoder is used for fitting, sets max stack size
    :param max_iter: Maximum number of iterations permitted
    :param eps: Termination criteria--minimum permissible percent change in estimated params
    :return: Infers A and v for set of sequences
    """
    min_val = 10e-10
    pi0_log = np.log(pi0)
    K = len(v_init)
    A_list = [np.log(A_init)]
    v_list = [v_init]
    logL_list = [-np.Inf]
    noise_list = [noise_init]
    #determine number of traces to hold out from training each iteration
    n_traces = len(fluo)
    trace_ids = range(n_traces)

    #If using viterbi alg, preallocate lookup tables
    if use_viterbi:
        cp_array, to_from, cp_init = viterbi_cp_init(K, w)
    # Initialize variable to track percent change in likelihood
    delta = 1
    iter = 1
    while iter < max_iter and abs(delta) > eps:
        fluo_ids = np.random.choice(trace_ids, n_traces - n_traces/n_groups, replace=False)
        fluo_samp = np.array(fluo)[fluo_ids]
        v_curr = v_list[iter - 1]
        A_log = A_list[iter - 1]
        sigma = noise_list[iter-1]
        # --------------------------Fwd Bkwd Algorithm for Each Sequence----------------------------------------------------#

        # store fits for each sequence for update calculations
        if use_viterbi:
            if w > 3:
                print("Warning: Attempting viterbi fits for high complexity data. Stack decoder recommended")
            _, _, seq_out, v_out, logL_out = viterbi_compound(fluo_samp, A_log, v_curr, sigma, pi0_log, w=w, cp_array=cp_array, to_from=to_from, cp_init=cp_init)
        else:
            seq_out, f_out, v_out, logL_out = decode_cp(fluo_samp, A_log, pi0_log, v_curr, w, sigma, stack_depth=max_stack)

        # -------------------------------Update Transition Probability Estimates----------------------------------------
        A_new = np.zeros_like(A_log) + min_val
        for f, fluo_vec in enumerate(fluo_samp):
            t_vec = range(1,len(fluo_vec))
            seq_fit = seq_out[f]
            #Count relevant transition events
            for k in xrange(K):
                for l in xrange(K):
                    rm_vec = []
                    for t in t_vec:
                        if seq_fit[t-1]==k and seq_fit[t]==l:
                            A_new[l,k] += 1
                            rm_vec.append(t)
                    #Attempt to cut down on redundant iterations
                    t_vec = [i for i in t_vec if i  not in rm_vec]

        A_new = A_new / np.tile(np.sum(A_new, axis=0), (K, 1))

        #--------------------------------------Update Emission Estimates-----------------------------------------------#
        b = []
        F_full = []
        sigmas = []
        for f, fluo_vec in enumerate(fluo_samp):
            b.append(fluo_vec)
            s_lookup = list(chain(*[[0]*(w-1), seq_out[f]]))
            s_lookup = np.array(s_lookup)
            v_lookup = list(chain(*[[0]*w, v_out[f]]))
            v_lookup = np.array(v_lookup)
            if estimate_noise:
                sig = np.square(np.array(fluo_vec) - (np.cumsum(v_lookup[w:]) - np.cumsum(v_lookup[:-w])))
                sigmas.append(sig)
            for t in xrange(len(fluo_vec)):
                ct = [0]*K
                for k in xrange(K):
                    ct[k] = len(np.where(s_lookup[t:t+w]==k)[0])
                F_full.append(ct)
        if estimate_noise:
            #Find mean sigma
            sigma_new = min(np.sqrt(np.mean(list(chain(*sigmas)))),noise_init)
        #Allocate arrays to store final values for v_new calc
        F_full = np.array(F_full)
        b = np.array(list(chain(*b)))
        F_square = np.zeros((K,K))
        b_vec = np.zeros(K)
        for k in xrange(K):
            F_square[k,:] = np.sum(F_full * F_full[:,k][:,np.newaxis], axis=0)
            b_vec[k] = np.sum(b * F_full[:,k])

        try:
            v_new = np.linalg.solve(F_square,b_vec)
        except:
            if verbose:
                print("Warning: Singular Matrix encountered. Using LSQ fitting instead")
            v_new, residuals, rank, singular_vals = np.linalg.lstsq(F_square,b_vec)

        #-------------------------------------Update Noise Estimate----------------------------------------------------$

        v_list.append(v_new)
        A_list.append(np.log(A_new))
        if estimate_noise:
            noise_list.append(sigma_new)
        else:
            noise_list.append(noise_init)
        # Calculate log likelihood using param estimates from previous step
        logL_list.append(np.sum(logL_out))
        # Check % improvement in logL
        if iter > 1:
            logL_comp_new = np.mean(logL_list[max(iter - n_groups, 1):])
            logL_comp_old = np.mean(logL_list[max(iter - n_groups - 1, 1):iter])
            delta = (logL_comp_old  - logL_comp_new ) / logL_comp_old
            if delta < 0:
                if verbose:
                    print("Warning: Non-monotonic behavior in likelihood")
            #sys.exit(1)
        if iter % 10 == 0 and verbose:
            print(np.sum(logL_out))
            print(v_new)
            print(A_new)
            print(delta)
        iter += 1

    return (A_list, v_list, logL_list, noise_list)