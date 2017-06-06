import numpy as np
import sys
from operator import itemgetter
import numpy
import random
import bisect
import itertools
from math import log
from functions import log_L_fluo

def decode_cp(fluo, A_log, pi0_log, v, w, noise, stack_depth, alpha=0):
    """

    :param fluo: list of fluorescence vectors
    :param A_log: log of transition probability matrix
    :param pi0_log: log of initial state PDF
    :param v: emission state vector
    :param w: memory of system
    :param noise: estimated system noise
    :param stack_depth: max num active hypotheses to keep in stack
    :param alpha: fractional num time steps needed to transcribe MS2 loops
    :return: Best guess at optimal path for each input fluo vec
    """
    # Get state count
    K = len(v)
    seq_out = []
    f_out = []
    v_out = []
    logL_out = []
    # Calculate convolution kernel to apply to fluorescence
    if alpha > 0:
        alpha_vec = [(float(i + 1) / alpha + (float(i) / alpha)) / 2.0 * (i < alpha) * ((i + 1) <= alpha)
                     + ((alpha - i)*(1 + float(i) / alpha) / 2.0 + i + 1 - alpha) * (i < alpha) * (i + 1 > alpha)
                     + 1 * (i >= alpha) for i in xrange(w)]

    else:

        alpha_vec = np.array([1.0]*w)
    kernel = np.ones(w)*alpha_vec
    kernel = kernel[::-1]
    for f, fluo_vec in enumerate(fluo):
        Stack = []
        T = len(fluo_vec)
        # assume that fluo vectors are off prior to first obs...
        for l in range(K):
            #Assume initial condition is OFF
            ns = [0]*(w)
            ns.append(l)
            ns = np.array(ns)
            #z_{t-w+1} ... z_{t-1}, z_{t}
            f = np.sum(ns[1:]*kernel)
            ns_score = log_L_fluo(fluo=fluo_vec[0], fluo_est=f, noise=noise) + pi0_log[l]
            Stack.append([ns_score, ns, f])
        Stack.sort(key=itemgetter(0))
        while (len(Stack[-1][1])) < T + w:
            #Get current best hypothesis
            hypothesis = Stack.pop()
            h_seq= hypothesis[1]
            prev = h_seq[-1]
            t = len(h_seq) - w
            #Check possible extensions
            for l in range(K):
                new_seq = [0]*(len(h_seq) + 1)
                new_seq[:-1] = h_seq
                new_seq[-1] = l
                f1 = np.array([v[z] for z in new_seq])
                f1 = np.sum(kernel*f1[-w:])
                ns_score = (hypothesis[0]*float(t) + log_L_fluo(fluo=fluo_vec[t], fluo_est=f1, noise=noise) + A_log[l,prev]) / float(t+1)
                bisect.insort(Stack, [ns_score, new_seq, f1])
            #Enforce max stack length
            while (len(Stack) > stack_depth):
                Stack.pop(0)

        logL_out.append(T*Stack[-1][0])
        seq_out.append(Stack[-1][1][w:])
        v_out.append([v[i] for i in Stack[-1][1][w:]])
        emissions = [v[t] for t in Stack[-1][1]]
        f_cp = np.convolve(kernel[::-1], emissions, mode='full')
        f_cp = f_cp[w:-w+1]
        f_out.append(f_cp)
    return(seq_out, f_out, v_out, logL_out)