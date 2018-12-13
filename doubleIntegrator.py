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

plt.close('all')
dt = .01
T = 15
iterations = int(T / dt)

variables = 2

def sigmoid(x):
#    return 1 / (1 + np.exp(-x))
    return np.tanh(x)


def doubleInt():
    # real dynamics
    x = np.zeros((variables, 1))
    x_dot = np.zeros((variables, 1))
    y = np.zeros((variables, 1))
    u = 0
    
    A = np.array([[0, 1], [0, 0]])               # state transition matrix
    B = np.array([[0], [1]])                     # input matrix
    C = np.array([[0, 0], [0, 1]])               # dynamics noise matrix
    D = np.array([[1, 0], [0, 1]])               # observations noise matrix
    H = np.array([[1, 0], [0, 1]])               # measurement matrix
    
    w = 1*np.random.randn(iterations, variables)
    z = .01*np.random.randn(iterations, variables)
    
    # controller
    Q = np.array([[1, 0], [0, 1]])
    R = 2
    
    x_hat = np.zeros((variables, 1))
    x_dot_hat = np.zeros((variables, 1))
    
    A_c = np.array([[1, 0], [0, 1]])               # state transition matrix
    B_c = np.array([[1, 0], [0, 1]])               # input matrix
    H_c = np.array([[-1, -1]])             # measurement matrix
    
    V = np.zeros((iterations, variables, variables))
    V_dot = np.zeros((variables, variables))
    L = np.zeros((variables, variables))
    
    # estimator
    Z = .1*np.array([[1, 0], [0, 1]])
    W = np.array([[1, 0], [0, 1]])
    
    P = np.zeros((iterations, variables, variables))
    P_dot = np.zeros((variables, variables))
    K = np.zeros((variables, variables))
    
    # history
    x_history = np.zeros((iterations, variables, 1))
    x_dot_history = np.zeros((iterations, variables, 1))
    y_history = np.zeros((iterations, variables, 1))
    u_history = np.zeros((iterations, 1))
    
    x_hat_history = np.zeros((iterations, variables, 1))
    x_dot_hat_history = np.zeros((iterations, variables, 1))
    
    # initialise state
    #x = np.random.randn(variables, 1)
    x = 10 * np.random.rand(variables, 1) - 5
    x_hat = 10 * np.random.rand(variables, 1) - 5
    x_hat = x + .1 * np.random.rand(variables, 1)
    
    # use Riccati equations solver in scipy
    #P = scipy.linalg.solve_continuous_are(A.transpose(), H.transpose(), np.dot(W, W.transpose()), np.dot(Z, Z.transpose()))
    #V = scipy.linalg.solve_continuous_are(A, B, Q, R)
    
    for i in range(iterations-1, 1, -1):
        V_dot = np.dot(A.transpose(), V[i, :, :]) + np.dot(V[i, :, :], A) + Q - np.dot(L.transpose(), np.dot(R, L))
        V[i-1, :, :] = V[i, :, :] + dt * V_dot
        L = np.dot(1/R, np.dot(B.transpose(), V[i, :, :]))
    
    for i in range(iterations-1):
        # simulate real dynamics
        x_dot = np.dot(A, x) + np.dot(B, sigmoid(u)) + np.dot(C, w[[i], :].transpose())
        x += dt * x_dot
        
        y = np.dot(H, x) + np.dot(D, z[[i], :].transpose())
        
        # control dynamics
        # (estimate state for output feedback)
        x_dot_hat = np.dot(A, x_hat) + np.dot(B, u) + np.dot(K, (y - np.dot(H, x_hat)))
        P_dot = np.dot(A, P[i, :, :]) + np.dot(P[i, :, :], A.transpose()) + W - np.dot(K, np.dot(Z, K.transpose()))
        K = np.dot(P[i, :, :], np.dot(H.transpose(), np.linalg.inv(np.dot(Z, Z.transpose()))))
        
    #    K[0, 1] = 0.
    #    K[1, 0] = 0.
        
        x_hat += dt * x_dot_hat
        P[i+1, :, :] = P[i, :, :] + dt * P_dot
        
        # (create controller)
        L = np.dot(1/R, np.dot(B.transpose(), V[i, :, :]))
        
        u = - np.dot(L, x_hat)    
        
        # save history
        x_history[i,:,:] = x
        x_dot_history[i,:,:] = x_dot
        y_history[i,:,:] = y
        u_history[i,:] = u
        
        x_hat_history[i,:,:] = x_hat
        x_dot_hat_history[i,:,:] = x_dot_hat
        
    return y_history, x_hat_history


simulations_n = 5
y_history = np.zeros((simulations_n, iterations, variables, 1))
x_hat_history = np.zeros((simulations_n, iterations, variables, 1))

plt.figure(figsize=(9, 6))
plt.xlabel('Position ($m$)')
plt.ylabel('Velocity ($m/s$)')
for k in range(simulations_n):
    y_history[k,:,:,:], x_hat_history[k,:,:,:] = doubleInt()
    plt.plot(y_history[k,:-1, 0, 0], y_history[k,:-1, 1, 0], 'b')
    plt.plot(x_hat_history[k, :-1, 0, 0], x_hat_history[k, :-1, 1, 0], 'r')
    plt.plot(y_history[k, 0, 0, 0], y_history[k, 0, 1, 0], 'o')

#plt.figure()
#plt.plot(u_history[:-1, :])
