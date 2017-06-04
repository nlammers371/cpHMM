import time
import sys
#import scipy # various algorithm
import numpy as np
from scipy.misc import logsumexp
import math
from itertools import chain
import itertools

def generate_traces_gill(memory, length, batch_size, noise_level, num_steps=1, r_mat=np.array([]), v=np.array([]), pi0=np.array([]), alpha=1.0):
    #define convolution kernel
    if alpha > 0:
        alpha_vec = [(float(i + 1) / alpha + (float(i) / alpha)) / 2.0 * (i < alpha) * ((i + 1) <= alpha)
                     + ((alpha - i)*(1 + float(i) / alpha) / 2.0 + i + 1 - alpha) * (i < alpha) * (i + 1 > alpha)
                     + 1 * (i >= alpha) for i in xrange(memory)]

    else:

        alpha_vec = np.array([1.0]*memory)
    kernel = np.ones(memory)*alpha_vec

    for step in xrange(num_steps):
        promoter_states_full = []
        fluo_states = []
        fluo_states_nn = []
        promoter_states_discrete = []

        for b in xrange(batch_size):
            #Determine number of states for trace
            v_size = len(v)
            v_choices = v
            R = r_mat

            promoter_states = []
            #Generate promoter trajectory
            T_float = 0.0
            transitions = [0.0]
            p_curr = np.random.choice(v_size, p=pi0)
            promoter_states.append(v_choices[p_curr])
            #Choose transition times and corresponding promoter states
            while T_float < length:
                #time step
                r = np.random.random()
                tau = 1 / -R[p_curr, p_curr]
                t = tau * math.log(1.0 / r)
                transitions.append(T_float + t)
                p_probs = R[:,p_curr] / -R[p_curr, p_curr]
                p_probs[p_curr] = 0
                p_curr = np.random.choice(v_size,p=p_probs)
                promoter_states.append(v_choices[p_curr])
                T_float += t

            tr_array = np.array(transitions)
            promoter_states = promoter_states[:-1]
            promoter_grid = np.zeros(length)
            promoter_grid_discrete = np.zeros(length)
            #allocate activity to appropriate discrete time window
            for e in xrange(1,length):
                #Find transitions that occurred within preceding time step
                if e==1:
                    tr_prev=0
                else:
                    tr_prev = np.max(np.where(tr_array < e-1)[0])
                tr_post = np.min(np.where(tr_array >= e)[0])
                tr = transitions[tr_prev:tr_post+1]
                tr[0] = e-1
                tr[-1] = e
                tr_diffs = np.diff(tr)
                p_states = promoter_states[tr_prev:tr_post]
                promoter_grid[e] = np.sum(tr_diffs*p_states)

                tr_proximal = np.max(np.where(tr_array < e)[0])
                promoter_grid_discrete[e] = promoter_states[tr_proximal]

            #promoter_grid = promoter_grid.astype('int')

            #Convolve with kernel to generate compound signal
            F_series = np.convolve(kernel,promoter_grid,mode='full')
            F_series = F_series[0:length]
            #Apply noise
            noise_vec = np.random.randn(length)*noise_level

            F_noised = F_series + noise_vec
            F_noised[F_noised < 0] = 0

            promoter_states_full.append(promoter_grid.tolist())
            promoter_states_discrete.append(promoter_grid_discrete.tolist())
            fluo_states.append(F_noised.tolist())
            fluo_states_nn.append(F_series.tolist())

        #seq_lengths = [length]*batch_size
        return(promoter_states_full, fluo_states, promoter_states_discrete, fluo_states_nn)

    # Helper function to calculate log likelihood of a proposed state
def log_L_fluo(fluo, fluo_est, noise):
    """
    :param fluo: Fluorescence value
    :param fluo_est: putative fluorescence state
    :param noise: standard deviation of emissions
    :return: log likelihood associated with fluorescence
    """
    noise_lambda = noise ** -2
    logL = 0.5 * math.log(noise_lambda) - 0.5 * np.log(2 * np.pi) - 0.5 * noise_lambda * (fluo - fluo_est) ** 2
    return logL

def fwd_algorithm(fluo_vec, A_log, v, noise, pi0_log):
    """

    :param fluo_vec: a single time series of fluorescence values
    :param A: current estimate of A
    :param v: current estimate of v
    :param noise: system noise
    :param pi0: initial state PDF
    :return: K x T vector of  log probabilities
    """
    K = len(v)
    T = len(fluo_vec)
    # Allocate alpha array to store log probs
    alpha_array = np.zeros((K, T), dtype=float) - np.Inf
    # Iteration
    for t in xrange(0, T):
        if t == 0:
            prev = pi0_log
        else:
            prev = alpha_array[:, t - 1]
        for l in xrange(K):
            # get logL of transitions into state l from each of the previous states
            a_sums = A_log[l, :] + prev
            # scipy has a built in function for handling logsums
            alpha_array[l, t] = log_L_fluo(fluo_vec[t], v[l], noise) + logsumexp(a_sums)

    return alpha_array

def bkwd_algorithm(fluo_vec, A_log, v, noise, pi0_log):
    """

    :param fluo_vec: a single time series of fluorescence values
    :param A: current estimate of A
    :param v: current estimate of v
    :param noise: system noise
    :param pi0: initial state PDF
    :return: K x T vector of  log probabilities
    """
    K = len(v)
    T = len(fluo_vec)
    # Allocate alpha array to store log probs
    beta_array = np.zeros((K, T), dtype=float) - np.Inf
    # initialize--We basically ignore this step
    beta_array[:, -1] = np.log(1.0)
    # Iteration
    steps = np.arange(T - 1)
    steps = steps[::-1]
    for t in steps:
        post = beta_array[:, t + 1]
        for k in xrange(K):
            # get logL of transitions into state k from each of the previous states
            b_sums = [A_log[l, k] + post[l] + log_L_fluo(fluo_vec[t + 1], v[l], noise) for l in xrange(K)]
            # scipy has a built in function for handling logsums
            beta_array[k, t] = logsumexp(b_sums)
        if t == 0:
            close_probs = [beta_array[l, 0] + pi0_log[l] + log_L_fluo(fluo_vec[t], v[l], noise) for l in xrange(K)]
    return beta_array, logsumexp(close_probs)


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
                            alpha[f][l, t] + beta[f][l, t] + alpha[f][k, t - 1] + beta[f][k, t - 1] - p_x) * A_log[
                                       l, k]

    return l_score


def viterbi(fluo, A_log, v, noise, pi0_log):
    #Get state count
    K = len(v)
    #Initialize list to store fits
    state_fits = []
    fluo_fits = []
    for f, fluo_vec in enumerate(fluo):
        T = len(fluo_vec)
        #Intilize array to store state likelihoods for each step
        v_array = np.zeros((K,T))
        #Initialize array to store pointers
        p_array = np.zeros((K,T-1), dtype='int')
        for t in xrange(T):
            if t == 0:
                v_array[:,t] = [log_L_fluo(fluo=fluo_vec[t], fluo_est=v[l], noise=noise) + pi0_log[l] for l in xrange(K)]
            else:
                for l in xrange(K):
                    lookback =  v_array[:,t-1] + A_log[l,:]
                    e_prob = log_L_fluo(fluo=fluo_vec[t], fluo_est=v[l], noise=noise)
                    #Get probs for present time point
                    v_array[l,t] = np.max(e_prob + lookback)
                    #Get pointer to most likely previous state
                    p_array[l,t-1] = np.argmax(np.array(lookback))

        #Backtrack to find optimal path
        v_fits = np.zeros(T, dtype='int')
        f_fits = np.zeros(T)
        v_fits[T-1] = np.argmax(v_array[:,T-1])
        prev = v_fits[T-1]
        f_fits[T-1] = v[prev]
        for t in xrange(T-1):
            v_fits[T-2-t] = p_array[prev,T-2-t]
            f_fits[T-2-t] = v[v_fits[T - 2 - t]]
            prev = v_fits[T-2-t]

        state_fits.append(v_fits)
        fluo_fits.append(f_fits)
    return (state_fits, fluo_fits)

#Create lookup tables referenced by viterbi compound
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

def HMMbasic(fluo, A_init, v_init, noise, pi0, max_iter=1000, eps=10e-4):
        """
        :param fluo: time series of fluorescent intensities (list of lists)
        :param A_init: Initial guess at the system's transition probability matrix (KxK)
        :param v_init: Initial guess at emissions vector (K)
        :param noise: Standard Deviation of fluo emissions (Taken as given at this stage)
        :param pi0: Initial state PDF (Taken as given)
        :param max_iter: Maximum number of iterations permitted
        :param eps: Termination criteria--minimum permissible percent change in estimated params
        :return: Infers A and v for set of sequences
        """
        min_val = 10e-10
        pi0_log = np.log(pi0)
        K = len(v_init)
        A_list = [np.log(A_init)]
        v_list = [v_init]

        for iter in xrange(1, max_iter):
            v_curr = v_list[iter - 1]
            A_log = A_list[iter - 1]
            # A_log = np.log(A_curr)
            # --------------------------Fwd Bkwd Algorithm for Each Sequence----------------------------------------------------#
            # store likelihood of each sequence given current parameter estimates
            seq_log_probs = []
            # store alpha and beta matrices
            alpha_arrays = []
            beta_arrays = []
            for f, fluo_vec in enumerate(fluo):
                alpha = fwd_algorithm(fluo_vec=fluo_vec, A_log=A_log, v=v_curr, noise=noise, pi0_log=pi0_log)
                beta, bcp = bkwd_algorithm(fluo_vec=fluo_vec, A_log=A_log, v=v_curr, noise=noise, pi0_log=pi0_log)
                # use last values of alpha array to calculate series probability
                p_seq = logsumexp(alpha[:, -1])
                # Store Results
                alpha_arrays.append(alpha)
                beta_arrays.append(beta)
                seq_log_probs.append(p_seq)

            # ---------------------------------------Calculate Updated A and v--------------------------------------------------#

            # Update A
            A_log_new = np.zeros_like(A_log) - np.Inf
            for k in xrange(K):
                for l in xrange(K):
                    # store log probs for each sequence
                    a_probs = []
                    # current transition prob from k to l
                    akl = A_log[l, k]
                    for f, fluo_vec in enumerate(fluo):
                        T = len(fluo_vec)
                        a = alpha_arrays[f]
                        b = beta_arrays[f]
                        event_list = [a[k, t] + b[l, t + 1] + akl + log_L_fluo(fluo=fluo_vec[t + 1], v=v_curr, state=l,
                                                                               noise=noise) for t in xrange(T - 1)]
                        a_probs.append(logsumexp(event_list) - seq_log_probs[f])
                    A_log_new[l, k] = logsumexp(a_probs)

            A_log_new = A_log_new - np.tile(logsumexp(A_log_new, axis=0), (K, 1))
            # Update v
            v_new = np.zeros_like(v_curr)
            for i in xrange(K):
                v_num_list = []
                v_denom_list = []
                for f, fluo_vec in enumerate(fluo):
                    # to avoid div 0 errors
                    fluo_vec = np.array(fluo_vec) + min_val
                    # Dot product in log space
                    num = np.log(fluo_vec) + alpha_arrays[f][i, :] + beta_arrays[f][i, :] - alpha_arrays[f][i, -1]
                    denom = alpha_arrays[f][i, :] + beta_arrays[f][i, :] - alpha_arrays[f][i, -1]
                    v_num_list.append(num.tolist())
                    v_denom_list.append(denom.tolist())
                # Flatten lists
                v_num_list = list(chain(*v_num_list))
                v_denom_list = list(chain(*v_denom_list))
                # Take average across sequences
                v_new[i] = np.exp(logsumexp(v_num_list) - logsumexp(v_denom_list))

            v_list.append(v_new)
            A_list.append(A_log_new)

            logL = log_likelihood(fluo, A_log_new, v_new, noise, pi0_log, alpha_arrays, beta_arrays)
            # print(logL)

if __name__ == "__main__":
    # memory
    w = 1
    # Fix trace length for now
    T = 5000
    # Number of traces per batch
    batch_size = 1
    R = np.array([[-.007, .007, .004], [.004, -.01, .003], [.003, .003, -.007]]) * 10
    A = scipy.linalg.expm(R, q=None)
    print(A)
    v = np.array([0.0, 50.0, 100.0])
    pi = [.2, .3, .5]
    promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn = \
        generate_traces_gill(w, T, batch_size, r_mat=R, v=v, noise_level=10, alpha=0.0, pi0=pi)

    plt.plot(np.array(fluo_states[0]))
    plt.plot(np.array(fluo_states_nn[0]))
    # plt.show()
    # print(promoter_states)
    HMMbasic(fluo=fluo_states, A_init=A, v_init=[25.0, 25.0, 25.0], noise=10, pi0=pi, max_iter=1000, eps=10e-4)

