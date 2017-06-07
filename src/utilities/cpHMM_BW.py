import time
import sys
import scipy # various algorithms
#from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import logsumexp
import math
from itertools import chain
import itertools
from functions import log_L_fluo, generate_traces_gill, fwd_algorithm, bkwd_algorithm
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
    #List to Store Transition Pointers (len T-1). Points to positional id NOT state ID
    p_list = []
    #List to Store most recently added state (len T)
    s_list = []
    #List to store cp fluo values (for convenience, could be calculated from above arrays) (len T)
    cf_list = []
    #Truncated alpha array
    alpha_array = np.zeros((max_stack, T), dtype=float) - np.Inf
    #Stack list to track most recent N active states
    Stack = []
    #List of Lists to track state counte
    F_list = []
    # Iteration
    dp_total = 0
    update_total = 0
    sort_total = 0
    for t in xrange(0, T):
        if t == 0:
            s_list.append(range(K))
            cf = [v[s] for s in xrange(K)]
            #Pad wit zeros to easy maximization computations. Will be given p weight of zero, so value irrelevant
            cf += [0]*(max_stack-len(cf))
            cf_list.append(cf)
            alpha_array[:K,0] = pi0_log + np.array([log_L_fluo(fluo_vec[t], v[s], noise).tolist() for s in xrange(K)])
            Stack = [[s] + [0]*(w-1)  for s in xrange(K)]
            # [s_{t}, s_{t-1}, s_{t-2},...s_{t-w+1}
        else:
            #Iterate through previous list of states to determine updates
            new_probs = []
            new_pointers = []
            new_states = []
            new_fluo = []
            new_stack = []

            for l in xrange(K):
                for k, state in enumerate(s_list[t - 1]):
                    dp_start = time.time()
                    sn = [l] + Stack[k][:-1]
                    if k > 0 and sn == new_stack[-1]:
                        new_probs[-1] = logsumexp([log_L_fluo(fluo_vec[t], new_fluo[-1], noise) + A_log[
                            l, state] + alpha_array[k, t - 1], new_probs[-1]])
                        new_pointers[-1] = [pointer for pointer in new_pointers[-1]] + [k]
                        dp_total += time.time() - dp_start

                    else:
                        update_start = time.time()
                        if t < w:
                            new_fluo.append(cf_list[t-1][k]+v[l])
                        else:
                            new_fluo.append(cf_list[t-1][k] + v[l] - v[Stack[k][-1]])
                        new_stack.append([l] + Stack[k][:-1])
                        # get logL of transitions into state l from each of the previous states
                        new_states.append(l)
                        new_pointers.append([k])
                        # scipy has a built in function for handling logsums
                        new_probs.append(log_L_fluo(fluo_vec[t], new_fluo[-1], noise) + A_log[l,state] + alpha_array[k,t-1])
                        update_total += time.time() - update_start
            #Sort resulting probs
            sort_time = time.time()
            new_args = np.argsort(new_probs)[-max_stack:]
            n_arg = len(new_args)
            #Now that we have ranked entries by prob, sort so that most similar stack entries are adjacent
            #will be used in next iteration to (somewhat) efficiently look for new duplicate entries
            stack_strings = []
            for arg in new_args:
                stack_strings.append(''.join(map(str, new_stack[arg][:-1])))
            sort_args = np.argsort(stack_strings)
            sorted_new_args = [new_args[i] for i in sort_args]

            #Update lists
            alpha_array[0:n_arg, t] = [new_probs[a] for a in sorted_new_args]
            p_list.append([new_pointers[a] for a in sorted_new_args])
            s_list.append([new_states[a] for a in sorted_new_args])
            cf = [new_fluo[a] for a in sorted_new_args]
            cf += [0]*(max_stack - len(cf))
            cf_list.append(cf)
            Stack = [new_stack[a] for a in sorted_new_args]
            sort_total += time.time() - sort_time

        f = [[0]*K]*max_stack
        for i, stk in enumerate(Stack):
            ct = [len(np.where(np.array(stk,dtype='int')==k)[0]) for k in xrange(K)]
            f[i] = ct
        F_list.append(f)
    return (alpha_array,s_list,p_list,cf_list,Stack, F_list)

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
            for p in pointers[t][l]:
                beta_array[p,t] = logsumexp([beta_array[p,t], beta_array[l,t+1] + log_L_fluo(fluo_vec[t+1], cp_fluo[t+1][l], noise) + A_log[state, alpha_states[t][p]]])
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
def cpEM_BW(fluo, A_init, v_init, noise_init, pi0, w, estimate_noise=1, max_stack=100, max_iter=1000, eps=10e-4, verbose=0):
    """
    :param fluo: time series of fluorescent intensities (list of lists)
    :param A_init: Initial guess at the system's transition probability matrix (KxK)
    :param v_init: Initial guess at emissions vector (K)
    :param noise: Standard Deviation of fluo emissions (Taken as given at this stage)
    :param pi0: Initial state PDF (Taken as given)
    :param w: memory of system
    :param max_stack: Max num states tracked per time step (<= K^w)
    :param max_iter: Maximum number of iterations permitted
    :param eps: Termination criteria--minimum permissible percent change in estimated params
    :return: Infers A and v for set of sequences
    """
    pi0_log = np.log(pi0)
    K = len(v_init)
    A_list = [np.log(A_init)]
    v_list = [v_init]
    sigma_list = [noise_init]
    logL_list = [-10e7]
    stack_depth = min(K ** w, max_stack)
    #Initialize variable to track percent change in likelihood
    delta = 1
    iter = 1
    total_time = 0
    while iter  < max_iter and abs(delta) > eps:
        loop_start_time = time.time()
        v_curr = v_list[iter-1]
        A_log= A_list[iter-1]
        noise = sigma_list[-1]

        #--------------------------Fwd Bkwd Algorithm for Each Sequence----------------------------------------------------#
        #store likelihood of each sequence given current parameter estimates
        seq_log_probs = []
        #store alpha and beta matrices
        alpha_arrays = []
        beta_arrays = []
        #track recent state and transition pointer info
        pointer_list = []
        state_list = []
        cp_fluo_list = []
        cp_state_list = []
        for f, fluo_vec in enumerate(fluo):
            alpha_array, s_list, p_list, cf_list, Stack, F_list = alpha_alg_cp(fluo_vec=fluo_vec,
                                                                       A_log=A_log,
                                                                       v=v_curr,
                                                                       w=w,
                                                                       noise=noise,
                                                                       pi0_log=pi0_log,
                                                                       max_stack=stack_depth)

            beta_array = beta_alg_cp(fluo_vec=fluo_vec,
                                     A_log=A_log,
                                     v=v_curr,
                                     w=w,
                                     noise=noise,
                                     pi0_log=pi0_log,
                                     pointers=p_list,
                                     alpha_states=s_list,
                                     cp_fluo=cf_list,
                                     alpha_stack=Stack)

            #use last values of alpha array to calculate series probability
            p_seq = logsumexp(alpha_array[:,-1])
            #Store Results
            alpha_arrays.append(alpha_array)
            beta_arrays.append(beta_array)
            seq_log_probs.append(p_seq)
            pointer_list.append(p_list)
            state_list.append(s_list)
            cp_fluo_list.append(cf_list)
            cp_state_list.append(F_list)
        #---------------------------------------Calculate Updated A and v--------------------------------------------------#
        #Update A
        #List of Lists to store transition events
        #Index scheme: K*row + col
        event_list = []
        event_id = []
        for f, fluo_vec in enumerate(fluo):
            T = len(fluo_vec)
            a = alpha_arrays[f]
            b = beta_arrays[f]
            p = pointer_list[f]
            s = state_list[f]
            i = cp_fluo_list[f]
            sp = seq_log_probs[f]
            for t in xrange(0,T-1):
                for row in xrange(len(p[t])):
                    for r in p[t][row]:
                        from_state = s[t][r]
                        to_state = s[t+1][row]
                        event = a[r,t] + b[row,t+1] + A_log[to_state,from_state] + log_L_fluo(fluo=fluo_vec[t+1],
                                                fluo_est=i[t+1][row], noise=noise)
                        event_list.append(event-sp)
                        event_id.append(K*to_state+from_state)
        event_list = np.array(event_list)
        event_id = np.array(event_id)
        A_log_new = np.zeros((K,K))
        for k in xrange(K**2):
            A_log_new[k / K, k%K] = logsumexp(event_list[np.where(event_id == k)[0]])
        A_log_new = A_log_new - np.tile(logsumexp(A_log_new, axis=0),(K,1))
        A_new = np.exp(A_log_new)

        #Update v
        wt_full = []
        if estimate_noise:
            cp_full = []
        for f, fluo_vec in enumerate(fluo):
            #Transpose alpha beta arrays to keep format compatible with F counts
            wt_full += np.reshape(np.transpose(alpha_arrays[f] + beta_arrays[f] - seq_log_probs[f]).tolist(),(len(fluo_vec)*stack_depth)).tolist()
            if estimate_noise:
                cp_full += list(chain(*cp_fluo_list[f]))
        #Convert to arrays
        F_full = np.array(list(chain(*chain(*cp_state_list))))
        wt_full = np.exp(np.array(wt_full))
        b_full = np.repeat(list(chain(*fluo)),stack_depth)
        F_square = np.zeros((K,K))
        b_vec = np.zeros(K)
        for k in xrange(K):
            F_square[k,:] = np.sum(F_full * F_full[:, k][:, np.newaxis] * wt_full[:, np.newaxis], axis=0)
            b_vec[k] =  np.sum(F_full[:,k] * wt_full * b_full)
        try:
            v_new = np.linalg.solve(F_square,b_vec)
        except:
            if verbose:
                print("Warning: Singular Matrix encountered. Using LSQ fitting instead")
            v_new, residuals, rank, singular_vals = np.linalg.lstsq(F_square,b_vec)
        # Update sigma estimate
        if estimate_noise:
            wt_resid = np.sum(wt_full*np.square(b_full-cp_full))
            sigma_new = np.sqrt(wt_resid / np.sum(wt_full))
            sigma_list.append(sigma_new)
        v_list.append(v_new)
        A_list.append(A_log_new)
        logL = np.sum(seq_log_probs)
        logL_list.append(logL)
        #Check % improvement in logL
        delta = (logL_list[iter-1]-logL) / logL_list[iter-1]
        loop_time = time.time() - loop_start_time
        total_time += loop_time
        if delta < 0:
            if verbose:
                print("Warning: Non-monotonic behavior in likelihood")
            #sys.exit(1)
        if iter % 1 == 0:
            if verbose:
                print(logL)
                print(abs(delta))
                print(v_new)
                print(A_new)
                if estimate_noise:
                    print(sigma_new)
                print(loop_time)

        iter += 1

    return(A_list, v_list, sigma_list, logL_list, iter, total_time)

if __name__ == '__main__':
    # memory
    w = 15
    # Fix 5race length for now
    T = 1000
    # Number of traces per batch
    batch_size = 2
    R = np.array([[-.008, .009, .00], [.008, -.014, .035], [.0, .005, -.035]]) * 10.2
    #R = np.array([[-.004, .014], [.004, -.014]]) * 10.2
    A = scipy.linalg.expm(R, q=None)
    #print(A)
    A_init = np.array([[.8,.2,.2],[.2,.8,.2],[.2,.2,.8]])
    v = np.array([0.0, 50.0, 100.0])
    #v = np.array([0.0, 50.0])
    pi = [.2, .3, .5]
    #pi = [.7, .3]
    K = len(v)
    max_stack = 50
    sigma = 25
    promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn = \
        generate_traces_gill(w, T, batch_size, r_mat=R, v=v, noise_level=sigma, alpha=0.0, pi0=pi)

    t_init = time.time()

    A_list, v_list, noise_list, logL_list, iter, total_time\
        = cpEM_BW(fluo_states, A_init=A_init, v_init=np.array([15.0,45.0,90.0]), noise_init=sigma*1,
                  pi0=pi, w=w, max_stack=max_stack, estimate_noise=1, max_iter=1000, eps=10e-6, verbose=1)
