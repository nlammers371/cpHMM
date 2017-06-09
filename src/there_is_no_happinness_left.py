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

# Set Paths
root = "../results"
folder = "method_validation"
subfolder = "A_base_case"
test_name = "eve2short_3state_bw_test"

readpath = os.path.join(root, folder, subfolder, test_name)

from utilities.functions import generate_traces_gill
from utilities.stack_decoder import decode_cp

plt.rcParams['animation.ffmpeg_path'] ='C:\\Users\\Nicholas\\Downloads\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = animation.FFMpegWriter()


# Basic Test Parameters
K = 3
dt = 10.0
# Read in true param values
true_params = pd.read_csv(os.path.join(readpath, "true_values.csv"), header=None)
R_start = K ** 2 + 1
v_start = R_start + K ** 2
pi_start = -K

# Recover proper order for params using v values
for i in xrange(len(true_params.index)):
    v = true_params.iloc[i, v_start:v_start + K].values
    arg_order = np.argsort(v)
    true_params.iloc[i, v_start:v_start + K] = v[arg_order]
    R = np.reshape(true_params.iloc[i, R_start:R_start + K ** 2], (K, K))
    R = R[arg_order, :]
    R = R[:, arg_order]
    true_params.iloc[i, R_start:R_start + K ** 2] = np.reshape(R, K ** 2)

# Read in best estimated param values
inf_params = pd.read_csv(os.path.join(readpath, "best_results.csv"), header=None)
for i in xrange(len(true_params.index)):
    v = inf_params.iloc[i, v_start:v_start + K].values
    arg_order = np.argsort(v)
    inf_params.iloc[i, v_start:v_start + K] = v[arg_order]
    R = np.reshape(inf_params.iloc[i, R_start:R_start + K ** 2], (K, K))
    R = R[arg_order, :]
    R = R[:, arg_order]
    inf_params.iloc[i, R_start:R_start + K ** 2] = np.reshape(R, K ** 2)

noise_true_mean = np.mean(true_params.iloc[:, v_start + K].values)
R_true_mean = np.reshape(np.mean(true_params.iloc[:, R_start:v_start].values, axis=0), (K, K))
pi_true_mean = np.mean(true_params.iloc[:, pi_start:].values, axis=0)
v_true_mean = np.mean(true_params.iloc[:, v_start:v_start + K].values, axis=0)

noise_inf_mean = np.mean(inf_params.iloc[:, v_start + K].values)
R_inf_mean = np.reshape(np.mean(inf_params.iloc[:, R_start:v_start].values, axis=0), (K, K))
pi_inf_mean = np.mean(inf_params.iloc[:, pi_start - 2:pi_start - 2 + K].values, axis=0)
v_inf_mean = np.mean(inf_params.iloc[:, v_start:v_start + K].values, axis=0)



#-------------------------------------Generate Example Trace(s)--------------------------------------------------------#
n_traces = 1
n_animations = 2
trace_length = 300
stack_depth = 100
w = 15
alpha = 0.0

if alpha > 0:
    alpha_vec = [(float(i + 1) / alpha + (float(i) / alpha)) / 2.0 * (i < alpha) * ((i + 1) <= alpha)
                 + ((alpha - i) * (1 + float(i) / alpha) / 2.0 + i + 1 - alpha) * (i < alpha) * (i + 1 > alpha)
                 + 1 * (i >= alpha) for i in xrange(w)]

else:

    alpha_vec = np.array([1.0] * w)
kernel = np.ones(w) * alpha_vec
kernel = kernel[::-1]
for tr in xrange(n_animations):
    #Generate Traces
    promoter_states, fluo_states, promoter_states_discrete, fluo_states_nn = \
            generate_traces_gill(w, trace_length, n_traces, r_mat=R_true_mean*dt, v=v_true_mean,  \
                                 noise_level=noise_true_mean, alpha=alpha, pi0=pi_true_mean)

    #Decode
    seq_out, f_out, v_out, logL_out, paths = decode_cp(fluo_states, np.log(sp.linalg.expm(R_inf_mean*dt)), np.log(pi_inf_mean), \
                                                 v_inf_mean, w, noise_inf_mean, stack_depth=stack_depth, log_stack=1,alpha=alpha)


    fluo_paths = []
    for i in xrange(len(paths)):
        emissions = [v_inf_mean[t] for t in paths[i]]
        f_cp = np.convolve(kernel[::-1], emissions, mode='full')
        f_cp = f_cp[w:-w + 1]
        fluo_paths.append(f_cp)

    #plt.plot(np.array(fluo_paths[-1]))
    #plt.plot(fluo_states[0])
    #plt.show()

    #--------------------------------------Generate Animation--------------------------------------------------------------#
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, len(fluo_states[0])), ylim=(0, np.max(fluo_states[0])))
    ax.plot(range(len(fluo_states[0])),fluo_states[0])
    line,  = ax.plot([], [], lw=4)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        y = np.array(fluo_paths[i])
        x = np.arange(0,len(y))
        line.set_data(x, y)
        return line,
    print(len(fluo_paths))
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(fluo_paths), interval=20, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    #anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    if len(fluo_states[0]) < 1000:
        writepath = os.path.join(readpath,'figs')
        if not os.path.isdir(writepath):
            os.makedirs(writepath)
        anim.save(readpath + '//figs' + '//stack_animation_' + str(tr) + '.mp4', writer = FFwriter, fps=30)

#plt.show()
