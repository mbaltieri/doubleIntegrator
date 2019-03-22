#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:17:16 2018

Implementation of a standard double integrator system, \ddot{q} = u.

@author: manuelbaltieri
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

#np.random.seed(42)

### define font size for plots ###
#
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)            # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)       # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)       # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)     # fontsize of the figure title
#

plt.close('all')
dt = .01


variables = 2

def sigmoid(x):
#    return 1 / (1 + np.exp(-x))
    return np.tanh(x)


def doubleInt(simulation, iterations):    
    # real dynamics
    x = np.zeros((variables, 1))
    x_dot = np.zeros((variables, 1))
    y = np.zeros((variables, 1))
    u = np.zeros((variables, 1))                # all inputs (motor actions and external forces, u = a + I)
    a = np.zeros((variables, 1))                # motor actions
    I = np.zeros((variables, 1))                # external forces
    
    A = np.array([[0, 1], [0, 0]])               # state transition matrix
    B = np.array([[0, 0], [0, 1]])                     # input matrix
    C = np.exp(-1) * np.array([[0, 0], [0, 1]])               # dynamics noise matrix
    D = np.exp(0) * np.array([[1, 0], [0, 1]])               # observations noise matrix
    H = np.array([[1, 0], [0, 1]])               # measurement matrix
    
    w = np.random.randn(iterations, variables)
    z = np.random.randn(iterations, variables)
    
    # controller
    Q = 1*np.array([[1, 0], [0, 1]])
    R = 4*np.array([[1, 0], [0, 1]])
    
    x_hat = np.zeros((variables, 1))
    x_dot_hat = np.zeros((variables, 1))
    
    V = np.zeros((iterations, variables, variables))
    V_dot = np.zeros((variables, variables))
    L = np.zeros((variables, variables))
    
    # estimator
#    Z = .1*np.array([[1, 0], [0, 1]])
#    W = np.array([[1, 0], [0, 1]])
    
    P = np.zeros((iterations, variables, variables))
    P_dot = np.zeros((variables, variables))
    K = np.zeros((variables, variables))
    
    # history
    x_history = np.zeros((iterations, variables, 1))
    x_dot_history = np.zeros((iterations, variables, 1))
    y_history = np.zeros((iterations, variables, 1))
    u_history = np.zeros((iterations, variables, 1))
    a_history = np.zeros((iterations, variables, 1))
    I_history = np.zeros((iterations, variables, 1))
    
    x_hat_history = np.zeros((iterations, variables, 1))
    x_dot_hat_history = np.zeros((iterations, variables, 1))
    
    # initialise state
    #x = np.random.randn(variables, 1)
    x = 300 * np.random.rand(variables, 1) - 150
    x_hat = x + .1 * np.random.rand(variables, 1)
    
    # use Riccati equations solver in scipy, assuming the system is LTI
    # P = scipy.linalg.solve_continuous_are(A.transpose(), H.transpose(), np.dot(C, C.transpose()), np.dot(D, D.transpose()))
    # V = scipy.linalg.solve_continuous_are(A, B, Q, R)
    
    # use this method to avoid bad approximations due to random initialization of V
    if simulation == 2:
        V = scipy.linalg.solve_continuous_are(A, B, Q, R)
    else:
        for i in range(iterations-1, 1, -1):
            V_dot = np.dot(A.transpose(), V[i, :, :]) + np.dot(V[i, :, :], A) + Q - np.dot(L.transpose(), np.dot(R, L))
            V[i-1, :, :] = V[i, :, :] + dt * V_dot
            L = np.dot(np.linalg.inv(R), np.dot(B.transpose(), V[i, :, :]))
    
    for i in range(iterations-1):
        # simulate real dynamics
        if simulation == 2 and i >= iterations/2:
            I[1,0] = 50
            
        u = a + I
        x_dot = np.dot(A, x) + np.dot(B, u) + np.dot(C, w[[i], :].transpose())
        x += dt * x_dot
        
        y = np.dot(H, x) + np.dot(D, z[[i], :].transpose())
        
        # control dynamics
        # (estimate state for output feedback)
        if simulation == 0:
            x_dot_hat = np.dot(A, x_hat) + np.dot(K, (y - np.dot(H, x_hat))) + np.dot(B, u)
        elif simulation == 1:
            x_dot_hat = np.dot(A, x_hat) + np.dot(K, (y - np.dot(H, x_hat)))
        elif simulation == 2:
            x_dot_hat = np.dot(A, x_hat) + np.dot(K, (y - np.dot(H, x_hat))) + np.dot(B, a)
        
        P_dot = np.dot(A, P[i, :, :]) + np.dot(P[i, :, :], A.transpose()) + np.dot(C, C.transpose()) - np.dot(K, np.dot(np.dot(D, D.transpose()), K.transpose()))
        K = np.dot(P[i, :, :], np.dot(H.transpose(), np.linalg.inv(np.dot(D, D.transpose()))))
        
        x_hat += dt * x_dot_hat
        P[i+1, :, :] = P[i, :, :] + dt * P_dot
        
        # (create controller)
        if simulation == 2:
            L = np.dot(np.linalg.inv(R), np.dot(B.transpose(), V))
        else:
            L = np.dot(np.linalg.inv(R), np.dot(B.transpose(), V[i, :, :]))
        
        a = - np.dot(L, x_hat)    
        
        # save history
        x_history[i,:,:] = x
        x_dot_history[i,:,:] = x_dot
        y_history[i,:,:] = y
        u_history[i,:] = u
        a_history[i,:] = a
        I_history[i,:] = I
        
        x_hat_history[i,:,:] = x_hat
        x_dot_hat_history[i,:,:] = x_dot_hat
        
    return y_history, x_hat_history, u_history, a_history, I_history

simulation = 0
# 0: all inputs u available to Kalman filter
# 1: no inputs u available to Kalman filter
# 2: only motor actions a available to Kalman filter


T = 15
iterations = int(T / dt)

simulations_n = 5
y_history = np.zeros((simulations_n, iterations, variables, 1))
x_hat_history = np.zeros((simulations_n, iterations, variables, 1))
u_history = np.zeros((simulations_n, iterations, variables, 1))
a_history = np.zeros((simulations_n, iterations, variables, 1))
I_history = np.zeros((simulations_n, iterations, variables, 1))

plt.figure(figsize=(9, 6))
if simulation == 0:
    plt.title('Double integrator - LQG')
elif simulation == 1:
    plt.title('Double integrator - LQG, no input $a$ in KBF')
elif simulation == 2:
    plt.title('Double integrator - LQG, no external force in KBF')
plt.xlabel('Position ($m$)')
plt.ylabel('Velocity ($m/s$)')
for k in range(simulations_n):
    y_history[k,:,:,:], x_hat_history[k,:,:,:], u_history[k,:,:,:], a_history[k,:,:,:], I_history[k,:,:,:] = doubleInt(simulation, iterations)
    plt.plot(y_history[k,:-1, 0, 0], y_history[k,:-1, 1, 0], 'b')
    plt.plot(x_hat_history[k, :-1, 0, 0], x_hat_history[k, :-1, 1, 0], 'r')
    plt.plot(y_history[k, 0, 0, 0], y_history[k, 0, 1, 0], 'o', markersize = 15, label='Agent ' + str(k+1))
if simulation == 0:
    plt.legend(loc=1)
elif simulation == 2:
    plt.legend(loc=4)
else:
    plt.legend(loc=2)
    

plt.figure(figsize=(9, 6))
plt.title('Action of double integrator - LQG')
plt.xlabel('Time ($s$)')
plt.ylabel('Action, $a$ ($m/s^2$)')
for k in range(simulations_n):
    plt.plot(np.arange(0, T-dt, dt), a_history[k,:-1,1,0], label='Agent ' + str(k+1))
plt.plot(np.arange(0, T-dt, dt), I_history[2,:-1,1,0], 'k', label='Ext. force')
plt.xlim(0, T)
plt.ylim(-250, 500)
plt.xticks(np.arange(0, T+1, 1))
if simulation == 0:
    plt.legend(loc=1)
elif simulation == 2:
    plt.legend(loc=1)
else:
    plt.legend(loc=2)


