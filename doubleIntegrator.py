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

# real dynamics

x = np.zeros((variables, 1))
x_dot = np.zeros((variables, 1))
y = np.zeros((variables, 1))
u = 0

A = np.array([[0, 1], [0, 0]])               # state transition matrix
B = np.array([[0], [1]])                     # input matrix
C = np.array([[0, 0], [0, 1]])               # noise dynamics matrix
H = np.array([[1, 0], [0, 1]])               # measurement matrix

w = 1*np.random.randn(iterations, variables)

# controller
Q = np.array([[1, 0], [0, 1]])
R = 2

x_c = np.zeros((variables, 1))
x_dot_c = np.zeros((variables, 1))

A_c = np.array([[1, 0], [0, 1]])               # state transition matrix
B_c = np.array([[1, 0], [0, 1]])               # input matrix
H_c = np.array([[-1, -1]])             # measurement matrix

S = np.zeros((variables, variables))
S_dot = np.zeros((variables, variables))
L = np.zeros((variables, variables))

# estimator
Z = 1
W = 50 * np.array([[1, 0], [0, 1]])

P = np.zeros((variables, variables))
P_dot = np.zeros((variables, variables))
K = np.zeros((variables, variables))

# history
x_history = np.zeros((iterations, variables, 1))
x_dot_history = np.zeros((iterations, variables, 1))
y_history = np.zeros((iterations, variables, 1))
u_history = np.zeros((iterations, 1))

x_c_history = np.zeros((iterations, variables, 1))
x_dot_c_history = np.zeros((iterations, variables, 1))

# initialise state
#x = np.random.randn(variables, 1)
x = 10 * np.random.rand(variables, 1) - 5
x_c = 10 * np.random.rand(variables, 1) - 5
x_c = x + .1 * np.random.rand(variables, 1)

P = scipy.linalg.solve_continuous_are(A, B, W, Z)
S = scipy.linalg.solve_continuous_are(A, B, Q, R)

for i in range(iterations-1):
    # simulate real dynamics
    x_dot = np.dot(A, x) + np.dot(B, sigmoid(u)) + np.dot(C, w[[i], :].transpose())
    x += dt * x_dot
    
    y = np.dot(H, x)
    
    # control dynamics
    # (estimate state for output feedback)
    #x_dot_c = np.dot(A_c, x_c) + np.dot(B_c, y)
    x_dot_c = np.dot(A, x_c) + np.dot(B, u) + np.dot(K, (y - np.dot(H, x_c)))
#    P_dot = np.dot(A, P) + np.dot(P, A.transpose()) + W - np.dot(K, np.dot(Z, K.transpose()))
    K = np.dot(P, np.dot(H.transpose(), 1/Z))
    
    x_c += dt * x_dot_c
#    P += dt * P_dot
    
    # (create controller)
#    S_dot = np.dot(A.transpose(), S) + np.dot(S, A) + Q - np.dot(L.transpose(), np.dot(R, L))               # not negative because here I'm not really solving a Riccati equation, could solve it outside this loop for LTI systems and for T -> infnty
    L = np.dot(1/R, np.dot(B.transpose(), S))
    
#    S += dt * S_dot
#    u = np.dot(H_c, x_c)
    u = - np.dot(L, y)    
    
    # save history
    x_history[i,:,:] = x
    x_dot_history[i,:,:] = x_dot
    y_history[i,:,:] = y
    u_history[i,:] = u
    
    x_c_history[i,:,:] = x_c
    x_dot_c_history[i,:,:] = x_dot_c


plt.figure()
plt.plot(y_history[:-1, 0, 0], y_history[:-1, 1, 0], 'b')
plt.plot(x_c_history[:-1, 0, 0], x_c_history[:-1, 1, 0], 'r')
plt.plot(y_history[0, 0, 0], y_history[0, 1, 0], 'o')

#plt.figure()
#plt.plot(u_history[:-1, :])
