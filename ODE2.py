# E19 FINAL PROJECT: Brachistochrone curve generator and simulator
# Description: Using the shooting method for two point boundary ODEs, solve for the path of least time then simulate
# Author: Lindsey Turner adapted from Matt Zucker's lab5.py
# Last Edit: 2022/11/29
######################################################################

import numpy as np
import numpy.polynomial.polynomial as nppoly

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import sys
import math
import scipy.optimize

######################################################################

# solve cubic spline by solving diagonal matrix
def fit_cubic_spline(xy_values):

    n = xy_values.shape[0] - 1

    assert xy_values.shape == (n+1, 2)

    A = np.zeros((n+1, n+1))
    b = np.zeros_like(xy_values)

    idx = np.arange(n+1)
    idx1  = idx[:-1]

    A[idx, idx] = 4
    A[idx1, idx1+1] = 1
    A[idx1+1, idx1] = 1
    A[0, 0] = 2
    A[n, n] = 2

    b[0] = 3*(xy_values[1] - xy_values[0])
    b[1:-1] = 3*(xy_values[2:] - xy_values[:-2])
    b[n] = 3*(xy_values[n] - xy_values[n-1])

    D_values = np.linalg.solve(A, b)

    yi = xy_values[:-1]
    yi1 = xy_values[1:]

    Di = D_values[:-1]
    Di1 = D_values[1:]

    # all of these are shape (n, 2)
    ai = yi
    bi = Di
    ci = 3*(yi1 - yi) - 2*Di - Di1
    di = 2*(yi-yi1) + Di + Di1

    coeffs = np.stack((ai, bi, ci, di), axis=0)

    assert coeffs.shape == (4, n, 2)

    return coeffs

######################################################################
# evaluates a 2d spline at one or more u values

def eval_spline(u, coeffs):

    u_is_scalar = np.isscalar(u)

    if u_is_scalar:
        u = np.array([u])

    # degree-(d-1) splines, n of them, with k dimensions
    d, n, k = coeffs.shape

    assert k == 2
    
    # which of the n splines to use?
    floor_u = np.clip(np.floor(u).astype(int), 0, n-1)

    # where in the [0,1] interval to evaluate the spline?
    fract_u = u - floor_u

    # we will do l*k polynomial evaluations where
    # l = len(u)
    #
    # j indexes dimensions in range(k)
    # i indexes entries in u
    #
    # both have size l, k
    j, i = np.meshgrid(np.arange(k), floor_u)

    assert j.shape == (len(u), k)
    assert i.shape == j.shape
    
    cij = coeffs[:, i, j]
    assert cij.shape == (d, len(u), k)

    # evaluate the polynomial at the given u points
    rval = nppoly.polyval(fract_u.reshape(-1, 1), cij, tensor=False)

    if u_is_scalar:
        rval = rval.reshape(k)

    return rval

######################################################################
# evaluate the spline and its derivatives at one more u values

def eval_spline_and_derivs(u, coeffs_and_deriv_coeffs):

    rvals = []

    for c in coeffs_and_deriv_coeffs:
        rvals.append(eval_spline(u, c))

    return rvals

######################################################################
# numerically solves an ordinary differential equation using either
# Euler's method, the midpoint method, or the 4th-order Runge-Kutta
# method. 

def approximate_ode(f, q0, dx, method, end_cond):

    q = q0.copy()

    q_values = [ q.copy() ]

    count = 0

    # repeat until some condition is met
    while not end_cond(q_values):

        if method == 'euler':

            dq = f(q)
            q += dq * dx

        elif method == 'midpoint':
            k1 = f(q)
            k2 = f(q + k1 * dx/2)
            q += k2 * dx
        else:

            assert method == 'rk4'
            k1 = f(q)
            k2 = f(q + k1 * dx/2)
            k3 = f(q + k2 * dx/2)
            k4 = f(q + k3 * dx)
            q += (k1+2*k2+2*k3+k4)* dx/6

        # keep track of iterations
        count += 1
        q_values.append(q.copy())

    # since repetition relies on q_values, remove the last value which met the end condition
    count -= 1
    q_values.pop()

    return np.array(q_values), count

######################################################################

def main():

    # ODE function - evaluate dy/dx at the given q
    def f(q):

        # state contains generalized position & velocity
        y, dy = q

        # avoid divide by zero errors
        if y == 0:
            return np.array([dy, -100000])

        # update function calculated by applying Euler-Lagrange to minimize functionel describing total time
        ddy = -(1+dy*dy)/(2*y)

        return np.array([dy, ddy])

    #####################################################################
    # generate curves

    # using RK4 method
    method = 'rk4'

    # step size
    sim_dx = 0.01

    # x and y between points A and B
    total_x = 10.0
    total_y = 3.0

    # end the brachistochrone curve when it reaches x = 0 or y = 0
    def brac_condition(q_values):
        return len(q_values)*sim_dx >= total_x or q_values[-1][0] >= 0

    def error(y0):
        # initial state for the ODE that starts at bottom of curve with some unknown slope
        q0 = np.array([-total_y, y0[0]])
        
        # approximate curves
        brac_approx, count = approximate_ode(f, q0, sim_dx, method, brac_condition)

        print("End y for",y0,":",brac_approx[-1,0])

        # return the error as either the final y-value if x=0 or the x-value if y=0
        return max(abs(brac_approx[-1,0]), total_x-count*sim_dx)
    
    # initial guess for y0, experimentally found to work best
    y0_guess = -2.0

    # implement shooting method to find best y0
    optimization = scipy.optimize.least_squares(error, [y0_guess], method='trf')

    print("Optimal y0:", optimization.x[0])

    # initial state for the optimized curve
    q0 = np.array([-total_y, optimization.x[0]])

    # approximate curves
    brac_approx, count = approximate_ode(f, q0, sim_dx, method, brac_condition)

    print("finished brac generation")

    # flip approximations to work from top to bottom
    by = np.flip(brac_approx[:, 0],0)
    bx = np.linspace(0, total_x, len(by))

    # make plots
    fig, ax = plt.subplots()

    ax.plot(bx,by, 'r', label=f'Curve of least time between [0.0,0.0] and [{total_x},{-total_y}]')
    ax.plot([0, total_x], [0,-total_y], 'b', label=f'Curve of least distance between [0.0,0.0] and [{total_x},{-total_y}]')

    ax.set_title('Curves')
    ax.legend()
    ax.axis('equal')
    ax.set_xlim(0.0,np.max(bx))
    ax.set_ylim(np.min(by),0.0)

    # prepare curves in [[x,y],...] form to be accepted by the spline generator
    curves = [np.stack((bx,by), axis=1),np.array([[0.0, 0.0], [total_x, -total_y]])]

    ######################################################################
    # run physics simulation

    # acceleration due to gravity 
    g = 9.8

    # frames for each curve being simulated
    xy_frames = []

    for c in curves:

        n = len(c) - 1

        # fit cubic spline in x & y to guarentee twice differentiability for simulated curves
        coeffs = fit_cubic_spline(c)

        dcoeffs = nppoly.polyder(coeffs)
        ddcoeffs = nppoly.polyder(dcoeffs)

        coeffs_and_deriv_coeffs = (coeffs, dcoeffs, ddcoeffs)


        # ODE function - evaluate dq/dt at the given q
        def f_sim(q):
            # state contains generalized position & velocity
            u, du = q

            (x, y), (xu, yu), (xuu, yuu) = eval_spline_and_derivs(
                u, coeffs_and_deriv_coeffs)

            # dynamics from Euler-Lagrange equation that we derived during lab 5
            ddu = (-du*du*(xu*xuu + yu*yuu) - g*yu) / (xu*xu + yu*yu)

            return np.array([du, ddu])

        # using Euler's method
        method = 'euler'

        # approximate the total (slowest) simulation time to be that of the linear path 
        # formula courtesy of Oliver
        total_time_est = math.sqrt(2*(total_x*total_x+total_y*total_y)/g/total_y)

        # step size
        sim_dt = 0.01

        # stop animating when the ball reaches the end point
        def sim_end(q_values):
            return eval_spline(q_values[-1][0], coeffs)[0] > total_x

        print("starting simulation")

        # start parameterized simulation at the beginning of the curve with zero velocity
        q0 = np.array([0.0,0.0])
        q_sim, count = approximate_ode(f_sim, q0, sim_dt, method, sim_end)

        print("Time taken:", count*sim_dt)

        u_sim = q_sim[:, 0]

        # check how much faster the simulation was from the estimated time
        sim_length_dif = round(total_time_est/sim_dt) - count

        # to make all simulations the same length, pad faster simulations with the end coordinate for the last frames
        xy_frame = eval_spline(u_sim, coeffs)
        for i in range(sim_length_dif):
            xy_frame = np.append(xy_frame, np.array([[total_x,-total_y]]), axis=0)

        xy_frames.append(xy_frame)

    
    xy_frames = np.array(xy_frames)
    num_frames = np.shape(xy_frames)[1]

    handle, = ax.plot([], [], 'ko')
    time_text = ax.text(1, 0, '', fontsize=15)

    def init_func():
        return handle,

    def update_func(frame):
        # show time
        time_text.set_text(f"Time : {frame/num_frames*total_time_est}")

        # get points on all curves
        xy = xy_frames[:,frame]
            
        handle.set_data(xy[:,0], xy[:,1])
        return handle,

    # animate
    ani = FuncAnimation(fig, update_func, frames=num_frames,init_func=init_func, blit=True,interval=100)
    filename = 'curve.gif'

    # save file
    ani.save(filename, writer=PillowWriter(fps=25))
    print('wrote', filename)

    #plt.show()
    ##################################################

main()
