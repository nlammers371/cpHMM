import numpy as np
import sys
from operator import itemgetter
import numpy
import random
import bisect
import itertools
from math import log
from functions import log_L_fluo

def decode_cp(fluo, A_log, pi0_log, v, w, noise, stack_depth):
    """

    :param fluo: list of fluorescence vectors
    :param A_log: log of transition probability matrix
    :param pi0_log: log of initial state PDF
    :param v: emission state vector
    :param w: memory of system
    :param noise: estimated system noise
    :param stack_depth: max num active hypotheses to keep in stack
    :return: Best guess at optimal path for each input fluo vec
    """
    # Get state count
    K = len(v)
    seq_out = []
    f_out = []
    v_out = []
    logL_out = []
    for f, fluo_vec in enumerate(fluo):
        Stack = []
        T = len(fluo_vec)
        # assume that fluo vectors are off prior to first obs...
        for l in range(K):
            #Assume initial condition is OFF
            ns = [0]*(w)
            ns.append(l)
            f = v[l]
            ns_score = log_L_fluo(fluo=fluo_vec[0], fluo_est=f, noise=noise) + pi0_log[l]
            Stack.append([ns_score, ns, f])
        Stack.sort(key=itemgetter(0))
        while (len(Stack[-1][1])) < T + w:
            #Get current best hypothesis
            hypothesis = Stack.pop()
            h_seq= hypothesis[1]
            prev = h_seq[-1]
            t = len(h_seq) - w
            f0 = hypothesis[2]
            #Check possible extensions
            for l in range(K):
                new_seq = [0]*(len(h_seq) + 1)
                new_seq[:-1] = h_seq
                new_seq[-1] = l
                f1 = f0 - v[h_seq[-w]] + v[l]
                ns_score = (hypothesis[0]*float(t) + log_L_fluo(fluo=fluo_vec[t], fluo_est=f1, noise=noise) + A_log[l,prev]) / float(t+1)
                bisect.insort(Stack, [ns_score, new_seq, f1])
            #Enforce max stack length
            while (len(Stack) > stack_depth):
                Stack.pop(0)

        logL_out.append(T*Stack[-1][0])
        seq_out.append(Stack[-1][1][w:])
        v_out.append([v[i] for i in Stack[-1][1][w:]])
        emissions = [v[t] for t in Stack[-1][1]]
        f_cp = np.cumsum(emissions[w:]) - np.cumsum(emissions[:-w])
        f_out.append(f_cp)
    return(seq_out, f_out, v_out, logL_out)