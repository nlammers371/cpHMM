import time
import sys
import scipy # various algorithms
from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import logsumexp
import math
from itertools import chain
import itertools
from functions import log_L_fluo, generate_traces_gill
from stack_decoder import decode_cp


def alpha_alg_cp(fluo_vec, A_log, v, w, noise, pi0_log, max_stack):
    """

    :param fluo_vec: a single time series of fluorescence values
    :param A: current estimate of A
    :param v: current estimate of v
    :param w: system memory
    :param noise: system noise
    :param pi0: initial state PDF
    :param max_stack: max num states tracked per step
    :return: K x T vector of  log probabilities
    """
    K = len(v)
    T = len(fluo_vec)
    #List to Store Transition Pointers (len T-1). Points to positional ID NOT state ID
    p_list = []
    #List to Store most recently added state (len T)
    s_list = []
    #List to store cp fluo values (for convenience, could be calculated from above arrays) (len T)
    cf_list = []
    #Truncated alpha array
    alpha_array = np.zeros((max_stack, T), dtype=float) - np.Inf
    #Stack list to track most recent N active states
    Stack = []
    # Iteration
    for t in xrange(0, T):
        if t == 0:
            s_list.append(range(K))
            cf_list.append([v[s] for s in xrange(K)])
            alpha_array[:K,0] = pi0_log + np.array([log_L_fluo(fluo_vec[t], v[s], noise) for s in xrange(K)])
            Stack = [ [v[s]] + [0]*(w-1)  for s in xrange(K)]
        else:
            new_probs = []
            new_pointers = []
            new_states = []
            new_fluo = []
            new_stack = []
            for k, state in  enumerate(s_list[t-1]):
                for l in xrange(K):
                    sn = [l] + Stack[k][:-1]
                    dp = -1
                    for s, stak in enumerate(new_stack):
                        if np.array_equal(sn,stak):
                            dp = s
                            break
                    if dp > -1:
                        new_probs[dp] = logsumexp([log_L_fluo(fluo_vec[t], new_fluo[dp], noise) + A_log[l, state] + alpha_array[k,t-1], new_probs[dp]])
                        new_pointers[dp] = [pointer for pointer in new_pointers[dp]] + [k]
                    else:
                        if t <= w:
                            new_fluo.append(cf_list[t-1][k]+v[l])
                        else:
                            new_fluo.append(cf_list[t-1][k] + v[l] - Stack[k][-1])
                        new_stack.append([l] + Stack[k][:-1])
                        # get logL of transitions into state l from each of the previous states
                        new_states.append(l)
                        new_pointers.append([k])
                        # scipy has a built in function for handling logsums
                        new_probs.append(log_L_fluo(fluo_vec[t], new_fluo[-1], noise) + A_log[l,state] + alpha_array[k,t-1])
            #Sort resulting probs

            new_args = np.argsort(new_probs)[-max_stack:]
            n_arg = len(new_args)
            #Update lists
            alpha_array[0:n_arg, t] = [new_probs[a] for a in new_args]
            p_list.append([new_pointers[a] for a in new_args])
            s_list.append([new_states[a] for a in new_args])
            cf_list.append([new_fluo[a] for a in new_args])
            Stack = [new_stack[a] for a in new_args]

    return (alpha_array,s_list,p_list,cf_list,Stack)

def beta_alg_cp(fluo_vec, A_log, v, w, noise, pi0_log, pointers, alpha_states, cp_fluo, alpha_stack):
    """

    :param fluo_vec: a single time series of fluorescence values
    :param A: current estimate of A
    :param v: current estimate of v
    :param w: system memory
    :param noise: system noise
    :param pi0: initial state PDF
    :param pointers: transition pointers from alpha calculation
    :alpha_states: list of states visited by alpha alg
    :cp_fluo: fluorescence corresponding to each state
    :alpha_stack: list of final state vectors from alpha calc...starting point for beta
    :return: K x T vector of  log probabilities
    """
    T = len(fluo_vec)
    K = len(v)
    max_stack = len(alpha_stack)
    # Truncated beta array
    beta_array = np.zeros((max_stack, T), dtype=float) - np.Inf
    # initialize--We basically ignore this step
    beta_array[:, -1] = np.log(1.0)
    #Initialize Stack
    Stack = alpha_stack
    # Iteration
    steps = np.arange(T - 1)
    steps = steps[::-1]
    for t in steps:
        for l, state in enumerate(alpha_states[t+1]):
            for point in pointers[t]:
                for p in point:
                    beta_array[p, t] = beta_array[l,t+1] + log_L_fluo(fluo_vec[t], cp_fluo[t+1][l], noise) + A_log[state, alpha_states[t][p]]

    return beta_array


# Function to calculate likelhood of data given estimated parameters
def log_likelihood(fluo, A_log, v, noise, pi0_log, alpha, beta):
    """

    :param fluo: List of fluo time series
    :param A_log: Log of transition matrix
    :param v: Emission States
    :param noise: Noise (stddev of signal)
    :param pi0_log: Log of initial state PDF
    :param alpha: Forward matrix
    :param beta: Backward matrix
    :return: Log Probability
    """
    l_score = 0
    K = len(v)
    for f, fluo_vec in enumerate(fluo):
        # Get log likelihood of sequence
        p_x = logsumexp(alpha[f][:, -1])
        for t in xrange(len(fluo_vec)):
            for k in xrange(K):
                # Likelihood of observing F(t)
                l_score += math.exp(alpha[f][k, t] + beta[f][k, t] - p_x) * log_L_fluo(fluo_vec[t], v[k], noise)
            if t == 0:
                # Likelihood of sequencce starting with k
                for k in xrange(K):
                    l_score += math.exp(alpha[f][k, t] + beta[f][k, t] - p_x) * (
                    pi0_log[k] + alpha[f][k, t] + beta[f][k, t])
            else:
                # Likelihood of transition TO l FROM k
                for k in xrange(K):
                    for l in xrange(K):
                        l_score += math.exp(
                            alpha[f][l, t] + beta[f][l, t] + alpha[f][k, t - 1] + beta[f][k, t - 1] - p_x) * A_log[l, k]

    return l_score

def viterbi_cp_init(K, w):
    """

    :param K: n states
    :param w: mem size
    :return:
    """
    # Allocate Array containing all possible histories
    cp_array = np.array(list(itertools.product(range(K), repeat=w)))
    cp_array.astype('int')
    # Create Array to store permitted transition info
    to_from = np.zeros((K ** w, K), dtype='int')
    for l in xrange(K ** w):
        col = 0
        for k in xrange(K ** w):
            if np.all(cp_array[k, :][0:-1] == cp_array[l, :][1:]):
                to_from[l, col] = k
                col += 1

    # Generate Array to store permitted init states
    if w > 1:
        cp_init = np.where(np.max(cp_array[:, 1:], axis=1) == 0)[0]
    else:
        cp_init = np.arange(K)

    return(cp_array, to_from, cp_init)

def viterbi_compound(fluo, A_log, v, noise, pi0_log, w, cp_array, to_from, cp_init):
    """

    :param fluo: list of fluorescence time series
    :param A_log: Log of transition matrix
    :param v: State emission vector
    :param noise: system noise
    :param pi0_log: log of initation PDF
    :param w: memory in time steps
    :return: most likely series of promoter states and compound emissions for each fluo vector
    """
    #Get state count
    K = len(v)

    # Calculate fluo values for each state
    cp_e_vec = np.zeros(K ** w)
    for s in xrange(K ** w):
        cp_e_vec[s] = np.sum(np.array([v[cp_array[s, i]] for i in xrange(w)]))

    #Initialize list to store fits
    cp_state_fits = []
    cp_fluo_fits = []
    state_fits = []
    fluo_fits = []
    logL_list = []
    for f, fluo_vec in enumerate(fluo):
        T = len(fluo_vec)
        #Intilize array to store state likelihoods for each step
        v_array = np.zeros((K**w,T)) - np.Inf
        #Initialize array to store pointers
        p_array = np.zeros((K**w,T-1), dtype='int')
        for t in xrange(T):
            if t == 0:
                for l in cp_init:
                    v_array[l,t] = log_L_fluo(fluo=fluo_vec[t], fluo_est=cp_e_vec[l], noise=noise) + pi0_log[cp_array[l,0]]
            else:
                for l in xrange(K**w):
                    #Convenience lookup vector to simplify indexing
                    lk = to_from[l,:]
                    #most recent state in current cp state
                    rc = cp_array[l,0]
                    lookback =  [v_array[k,t-1] + A_log[rc,cp_array[k,0]] for k in lk]
                    e_prob = log_L_fluo(fluo=fluo_vec[t], fluo_est=cp_e_vec[l], noise=noise)
                    #Get probs for present time point
                    v_array[l,t] = np.max(e_prob + lookback)
                    #Get pointer to most likely previous state
                    p_array[l,t-1] = lk[np.argmax(np.array(lookback))]

        #Backtrack to find optimal path
        #Arrays to store compound sequences
        cp_v_fits = np.zeros(T, dtype='int')
        cp_f_fits = np.zeros(T)
        #Arrays to store simple sequences
        v_fits = np.zeros(T, dtype='int')
        f_fits = np.zeros(T)

        cp_v_fits[T-1] = np.argmax(v_array[:,T-1])
        cp_f_fits[T-1] = cp_e_vec[cp_v_fits[T-1]]
        v_fits[T-1] = cp_array[cp_v_fits[T-1],0]
        f_fits[T-1] = v[v_fits[T-1]]
        prev = cp_v_fits[T-1]

        for t in xrange(T-1):
            cp_v_fits[T-2-t] = p_array[prev,T-2-t]
            cp_f_fits[T-2-t] = cp_e_vec[cp_v_fits[T-2-t]]
            v_fits[T-2-t] = cp_array[cp_v_fits[T-2-t],0]
            f_fits[T-2-t] = v[v_fits[T-2-t]]
            prev = cp_v_fits[T-2-t]

        cp_state_fits.append(cp_v_fits)
        cp_fluo_fits.append(cp_f_fits)
        state_fits.append(v_fits)
        fluo_fits.append(f_fits)
        logL_list.append(np.max(v_array[:,T-1]))
    return (cp_state_fits, cp_fluo_fits, state_fits, fluo_fits, logL_list)

#Approximate compound BW for inferring HMM parameteris in high-memory systems
def cpEM(fluo, A_init, v_init, noise, pi0, max_stack=100, max_iter=1000, eps=10e-4):
    """
    :param fluo: time series of fluorescent intensities (list of lists)
    :param A_init: Initial guess at the system's transition probability matrix (KxK)
    :param v_init: Initial guess at emissions vector (K)
    :param noise: Standard Deviation of fluo emissions (Taken as given at this stage)
    :param pi0: Initial state PDF (Taken as given)
    :param max_stack: Max num states tracked per time step (<= K^w)
    :param max_iter: Maximum number of iterations permitted
    :param eps: Termination criteria--minimum permissible percent change in estimated params
    :return: Infers A and v for set of sequences
    """
    min_val = 10e-10
    pi0_log = np.log(pi0)
    K = len(v_init)
    A_list = [np.log(A_init)]
    v_list = [v_init]
    logL_list = []
    #Initialize variable to track percent change in likelihood
    delta = 1
    iter = -1
    while iter  < max_iter and delta < eps:
        v_curr = v_list[iter-1]
        A_log= A_list[iter-1]
        #--------------------------Fwd Bkwd Algorithm for Each Sequence----------------------------------------------------#
        #store likelihood of each sequence given current parameter estimates
        seq_log_probs = []
        #store alpha and beta matrices
        alpha_arrays = []
        beta_arrays = []
        for f, fluo_vec in enumerate(fluo):
            alpha = fwd_algorithm(fluo_vec=fluo_vec,A_log=A_log,v=v_curr,noise=noise,pi0_log=pi0_log)
            beta,bcp = bkwd_algorithm(fluo_vec=fluo_vec, A_log=A_log, v=v_curr, noise=noise, pi0_log=pi0_log)
            #use last values of alpha array to calculate series probability
            p_seq = logsumexp(alpha[:,-1])
            #Store Results
            alpha_arrays.append(alpha)
            beta_arrays.append(beta)
            seq_log_probs.append(p_seq)

        #Calculate log likelihood using param estimates from previous step
        logL_list.append(log_likelihood(fluo, A_log=A_log, v=v_init, noise=noise, pi0_log=pi0_log, alpha=alpha_arrays, beta=beta_arrays))
        #---------------------------------------Calculate Updated A and v--------------------------------------------------#

        #Update A
        A_log_new = np.zeros_like(A_log) - np.Inf
        for k in xrange(K):
            for l in xrange(K):
                #store log probs for each sequence
                a_probs = []
                #current transition prob from k to l
                akl = A_log[l,k]
                for f, fluo_vec in enumerate(fluo):
                    T = len(fluo_vec)
                    a = alpha_arrays[f]
                    b = beta_arrays[f]
                    event_list = [a[k,t] + b[l,t+1] + akl + log_L_fluo(fluo=fluo_vec[t+1], v=v_curr, state=l, noise=noise) for t in xrange(T-1)]
                    a_probs.append(logsumexp(event_list) - seq_log_probs[f])
                A_log_new[l,k] = logsumexp(a_probs)

        A_log_new = A_log_new - np.tile(logsumexp(A_log_new, axis=0),(K,1))
        #Update v
        v_new = np.zeros_like(v_curr)
        for i in xrange(K):
            v_num_list = []
            v_denom_list = []
            for f, fluo_vec in enumerate(fluo):
                #to avoid div 0 errors
                fluo_vec = np.array(fluo_vec)+min_val
                #Dot product in log space
                num = np.log(fluo_vec)+alpha_arrays[f][i,:]+beta_arrays[f][i,:]-alpha_arrays[f][i,-1]
                denom = alpha_arrays[f][i,:]+beta_arrays[f][i,:]-alpha_arrays[f][i,-1]
                v_num_list.append(num.tolist())
                v_denom_list.append(denom.tolist())
            #Flatten lists
            v_num_list = list(chain(*v_num_list))
            v_denom_list = list(chain(*v_denom_list))
            #Take average across sequences
            v_new[i] = np.exp(logsumexp(v_num_list)-logsumexp(v_denom_list))

        v_list.append(v_new)
        A_list.append(A_log_new)

        logL = log_likelihood(fluo, A_log_new, v_new, noise, pi0_log, alpha_arrays, beta_arrays)
        #Check % improvement in logL
        delta = (logL_list[iter-1]-logL) / logL_list[iter-1]
        if delta < 0:
            print("Error: Non-monotonic behavior in likelihood")
            sys.exit(1)
        if iter % 10 == 0:
            print(logL)
            print(v_new)
            print(np.exp(A_log_new))

    return(A_list, v_list, logL_list)

if __name__ == '__main__':
    # memory
    w = 5
    # Fix trace length for now
    T = 20
    # Number of traces per batch
    batch_size = 2
    R = np.array([[-.007, .007, .004], [.004, -.01, .003], [.003, .003, -.007]]) * 10.2
    A = scipy.linalg.expm(R, q=None)
    print(A)
    v = np.array([0.0, 50.0, 100.0])
    pi = [.2, .3, .5]
    promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn = \
        generate_traces_gill(w, T, batch_size, r_mat=R, v=v, noise_level=10, alpha=0.0, pi0=pi)

    t_init = time.time()
    alpha_array, s_list, p_list, cf_list, Stack = alpha_alg_cp(fluo_vec=fluo_states[0], A_log=np.log(A), v=v, w=w, noise=10, pi0_log=np.log(pi), max_stack=5)


    print(time.time() - t_init)
    #print(Stack)
    #print(s_list)
    #print(cf_list[-3:])

    beta_array = beta_alg_cp(fluo_vec=fluo_states[0], A_log=np.log(A), v=v, w=w, noise=10, pi0_log=np.log(pi), pointers=p_list, alpha_states=s_list, cp_fluo=cf_list, alpha_stack=Stack)
    print(alpha_array[:, 0])
    print(beta_array[:,0])
    print(s_list[0:2])
    print(p_list[0])
    plt.plot(np.array(promoter_states[0]))
    #plt.plot(np.array(fluo_states_nn[0]))
    #plt.show()