import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib as plt
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
from utilities.functions import generate_traces_gill
from utilities.stack_decoder import decode_cp
from matplotlib import pyplot as plt
from matplotlib import animation
from utilities.cpHMM_BW import cpEM_BW
import time

#Set Paths
root = "../results"
folder = "method_validation"
subfolder = "BW_PDF_Animation"

outpath = os.path.join(root, folder, subfolder)

plt.rcParams['animation.ffmpeg_path'] ='C:\\Users\\Nicholas\\Downloads\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = animation.FFMpegWriter()

out_name = 'bw_test'

# memory
w = 4
# Fix 5race length for now
T = 200
# Number of traces per batch
batch_size = 20
R = np.array([[-.008, .009, .01], [.006, -.014, .025], [.002, .005, -.035]]) * 10.2
A = sp.linalg.expm(R, q=None)
print(A)
pi = [.2, .3, .5]
v = np.array([0.0, 25.0, 100.0])


v_init = np.array([0.0, 23.0, 83.0])
A_init = [[.6,.2,.2],[.2,.6,.2],[.2,.2,.8]]
K = len(v)
max_stack = K**w
sigma = 25
promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn = \
    generate_traces_gill(w, T, batch_size, r_mat=R, v=v, noise_level=sigma, alpha=0.0, pi0=pi)

t_init = time.time()
A_list, v_list, sigma_list, logL_list, iter, total_time, full_seq_probs = \
    cpEM_BW(fluo_states, A_init=A_init, v_init=v, noise_init= sigma*2, estimate_noise=1, pi0=pi, w=w, max_stack=max_stack, keep_probs=1, verbose=1, max_iter=1000, eps=10e-6)


# First set up the figure, the axis, and the plot element we want to animate
# create the figure
fig = plt.figure()
ax = fig.add_subplot(111)


def init():
    im = ax.imshow(np.array(full_seq_probs[0]), interpolation='none')
    return im,

plt.show(block=False)


def animate(i):
    time.sleep(1)
    im = ax.imshow(np.array(full_seq_probs[i]), interpolation='none')
    return im,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(full_seq_probs), interval=1, blit=True)

if not os.path.exists(outpath):
    os.makedirs(outpath)
if os.path.exists(os.path.join(outpath,'bw_animation_' + out_name + '.mp4')):
    print("Warning: preexisting file detected. Appending datetime")
    out_name = out_name + '_' + str(np.round(time.time()))

anim.save(outpath + '/bw_animation_' + out_name + '.mp4', writer=FFwriter, fps=30)

plt.show()