import sys
import scipy # various algorithms
from matplotlib import pyplot as plt
import numpy as np
from scipy.misc import logsumexp
import math
from itertools import chain
from utilities.functions import generate_traces_gill
#Helper function to calculate log likelihood of a proposed state
def log_L_fluo(fluo, v, state, noise):
    """
    :param fluo: Fluorescence value
    :param v: emission vector
    :param state: state
    :param noise: standard deviation of emissions
    :return: log likelihood associated with fluorescence
    """
    noise_lambda = noise ** -2
    logL = 0.5 * math.log(noise_lambda) - 0.5 * np.log(2*np.pi) - 0.5 * noise_lambda * (fluo - v[state])**2
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
    #Allocate alpha array to store log probs
    alpha_array = np.zeros((K,T), dtype=float) - np.Inf
    #Iteration
    for t in xrange(0,T):
        if t == 0:
            prev = pi0_log
        else:
            prev = alpha_array[:,t-1]
        for l in xrange(K):
            #get logL of transitions into state l from each of the previous states
            a_sums = A_log[l,:] + prev
            #scipy has a built in function for handling logsums
            alpha_array[l,t] = log_L_fluo(fluo_vec[t],v,l,noise) + logsumexp(a_sums)

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
    steps = np.arange(T-1)
    steps = steps[::-1]
    for t in steps:
        post = beta_array[:, t + 1]
        for k in xrange(K):
            # get logL of transitions into state k from each of the previous states
            b_sums = [A_log[l, k] + post[l] + log_L_fluo(fluo_vec[t+1],v,l,noise) for l in xrange(K)]
            # scipy has a built in function for handling logsums
            beta_array[k, t] = logsumexp(b_sums)
        if t == 0:
            close_probs = [beta_array[l,0] + pi0_log[l] + log_L_fluo(fluo_vec[t],v,l,noise) for l in xrange(K)]
    return beta_array, logsumexp(close_probs)

#Function to calculate likelhood of data given estimated parameters
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
        p_x = logsumexp(alpha[f][:,-1])
        for t in xrange(len(fluo_vec)):
            for k in xrange(K):
                #Likelihood of observing F(t)
                l_score += math.exp(alpha[f][k,t] + beta[f][k,t] - p_x) * log_L_fluo(fluo_vec[t], v, k, noise)
            if t == 0:
                #Likelihood of sequencce starting with k
                for k in xrange(K):
                    l_score += math.exp(alpha[f][k,t] + beta[f][k,t] - p_x) * (pi0_log[k] + alpha[f][k,t] + beta[f][k,t])
            else:
                #Likelihood of transition TO l FROM k
                for k in xrange(K):
                    for l in xrange(K):
                        l_score += math.exp(alpha[f][l,t] + beta[f][l,t] + alpha[f][k,t-1] + beta[f][k,t-1] - p_x) * A_log[l,k]

    return l_score
#Basic Implementation of BM for naive trajectories with gaussian emission characteristics
def cpEM(fluo, A_init, v_init, noise, pi0, max_iter=1000, eps=10e-4):
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

    for iter in xrange(1,max_iter):
        v_curr = v_list[iter-1]
        A_log= A_list[iter-1]
        #A_log = np.log(A_curr)
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
        #print(logL)


if __name__ == "__main__":

    # memory
    w = 1
    # Fix trace length for now
    T = 5000
    # Number of traces per batch
    batch_size = 1
    R = np.array([[-.007,.007,.004],[.004,-.01,.003],[.003,.003,-.007]])*10
    A = scipy.linalg.expm(R, q=None)
    print(A)
    v = np.array([0.0,50.0,100.0])
    pi = [.2,.3,.5]
    promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn =\
        generate_traces_gill(w,T,batch_size, r_mat=R, v=v, noise_level =10, alpha=0.0, pi0=pi)

    plt.plot(np.array(fluo_states[0]))
    plt.plot(np.array(fluo_states_nn[0]))
    #plt.show()
    #print(promoter_states)
    HMMbasic(fluo=fluo_states, A_init=A, v_init=[25.0,25.0,25.0], noise=10, pi0=pi, max_iter=1000, eps=10e-4)
