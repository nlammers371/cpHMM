import time
import sys
import scipy # various algorithms
from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import logsumexp
import math
from itertools import chain
from utilities.functions import generate_traces_gill, viterbi, viterbi_compound, viterbi_cp_init
from utilities.cpHMM_viterbi import cpEM_viterbi, cpEM_viterbi_full
from utilities.stack_decoder import decode_cp




if __name__ == "__main__":
    #noise
    sigma = 30
    # memory
    w = 3
    # Fix trace length for now
    T = 200
    # Number of traces per batch
    batch_size = 50
    R = np.array([[-.007, .007, .004], [.004, -.01, .003], [.003, .003, -.007]]) * 10
    A = scipy.linalg.expm(R, q=None)
    print(A)
    v = np.array([0.0, 50.0, 100.0])
    K = len(v)
    pi = [.8, .1, .1]


    promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn = \
        generate_traces_gill(w, T, batch_size, r_mat=R, v=v, noise_level=sigma, alpha=0.0, pi0=pi)


    print("Running EM Optimization...")
    t_init = time.time()
    A_rand = np.random.rand(len(v),len(v))
    #A_rand = A_rand / np.tile(np.sum(A_rand,axis=0),(3,1))
    #print(A_rand)
    A_init = [[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]

    A_list, v_list, logL_list, sigma_list = cpEM_viterbi_full(fluo_states, A_init, [0,25.0,120.0], sigma, pi, w=w, use_viterbi=0, n_groups=5, max_stack=100, max_iter=1000, eps=10e-4)
    #print(A_list[-1])
    print("Running Viterbi Fit...")
    cp_array, to_from, cp_init = viterbi_cp_init(K, w)
    cp_state_fits, cp_fluo_fits, state_fits, fluo_fits, logs = viterbi_compound([fluo_states[0]], A_list[-1], v_list[-1], sigma_list[-1],[.3,.3,.4],w=w,cp_array=cp_array,to_from=to_from,cp_init=cp_init)
    print(np.exp(A_list[-1]))
    print(v_list[-1])
    #seq_out, f_out, v_out, logL_out = decode_cp(fluo_states, A, np.log(pi), v, w, sigma, stack_depth=100)
    print(time.time()-t_init)
    plt.plot(np.array(fluo_states_nn[0]))
    #plt.plot(np.array(f_out[0]))
    plt.plot(np.array(cp_fluo_fits[0]))
    plt.show()
    #print(promoter_states)