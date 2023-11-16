import numpy as np
import importlib
import iLQR
importlib.reload(iLQR)
from iLQR import iLQR
import quad_sim

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import PillowWriter



# Setup the iLQR problem
N = 200
dt = 0.02
x_goal = np.array([-0.5, 1.8, 0, 0, 0, 0])

Q = np.eye(6)
Qf = 1e2 * np.eye(6)
R = 1e-3 * np.eye(2)

ilqr = iLQR(x_goal, N, dt, Q, R, Qf)

# Initial state at rest at the origin
x0 = np.zeros((6,))

# initial guess for the input is hovering in place
u_guess = [0.5 * 9.81 * ilqr.m * np.ones((2,))] * (N-1) 
 

x_sol, u_sol, K_sol, x_list = ilqr.calculate_optimal_trajectory(x0, u_guess)



##############################################
############## PLOT PLANNER ##################
##############################################
"""
def update(i):
    plt.cla()  # Clear the current axis
    xx = np.array(x_list[i])

    ax.plot(x0[0], x0[1], marker='x', color='blue')
    ax.plot(x_goal[0], x_goal[1], marker='x', color='orange')
    ax.plot(xx[:,0], xx[:,1], linestyle='--', color='black')

    ax.set_title('iLQR Trajectory Iteration: {}'.format(i))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(['Start', 'Goal', 'Trajectory'])


fig = plt.figure()
ax = fig.add_subplot()

animation = FuncAnimation(fig, update, frames=len(x_list), interval=200, repeat=True)

writer = FFMpegWriter(fps=15)
animation.save('demo/iLQR_traj.mp4', writer=writer)

writer = PillowWriter(fps=15)
animation.save('demo/iLQR_traj.gif', writer=writer)

plt.show()
"""


##############################################
############## PLOT CONTROLLER ###############
##############################################
"""
def update(frame):
    plt.cla()  # Clear the current axis

    ax.plot(x0[0], x0[1], marker='x', color='blue')
    ax.plot(xx[-1, 0], xx[-1, 1], marker='x', color='orange')
    ax.plot(xx[:,0], xx[:,1], linestyle='--', color='black')

    plt.plot(xtraj[:frame, 0], xtraj[:frame, 1], color='red')  # Plot trajectory up to the current frame
    plt.scatter(xtraj[frame, 0], xtraj[frame, 1], color='red')  # Mark the current position in red
    plt.title('Time Frame: {}'.format(frame))


    ax.set_title('iLQR Trajectory Solution')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(['Start', 'Goal', 'Desired Trajectory', 'Actual Trajectory'])



xtraj = [np.zeros((4, ))] * N
utraj = [np.zeros((2, ))] * (N - 1)

x_init = np.zeros((6,)) + 0.1 * np.ones((6,))
xtraj[0] = x_init

for k in np.arange(0, N-1):
    utraj[k] = u_sol[k] + K_sol[k] @ (xtraj[k] - x_sol[k])
    xtraj[k+1] =  quad_sim.F(xtraj[k], utraj[k], dt)


fig = plt.figure()
ax = fig.add_subplot()
xx = np.array(x_list[-1])
xtraj = np.array(xtraj)


animation = FuncAnimation(fig, update, frames=N, interval=100, repeat=False)

# writer = FFMpegWriter(fps=30)
# animation.save('demo/iLQR_cont.mp4', writer=writer)

writer = PillowWriter(fps=30)
animation.save('demo/iLQR_cont.gif', writer=writer)

plt.show()
"""

