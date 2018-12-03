#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:04:15 2018

@author: mb540
"""

import autograd.numpy as np
import matplotlib.pyplot as plt

dt = .01
T = 55
iterations = int(T / dt)
plt.close('all')

obs_states = 2
hidden_states = 2                                           # x, in Friston's work
hidden_causes = 2                                           # v, in Friston's work
states = obs_states + hidden_states
temp_orders_states = 2                                      # generalised coordinates for hidden states x, but only using n-1
temp_orders_causes = 2                                      # generalised coordinates for hidden causes v (or \eta in biorxiv manuscript), but only using n-1

A = np.array([[0, 1], [0, 0]])               # state transition matrix
B = np.array([[0, 0], [0, 1]])                     # input matrix
C = .2*np.array([[1, 0], [0, 1]])               # noise dynamics matrix
D = .15*np.array([[1, 0], [0, 1]])
H = np.array([[1, 0], [0, 1]])               # measurement matrix

x = np.zeros((hidden_states, temp_orders_states))           # position
y = np.zeros((obs_states, temp_orders_states-1))           # position reading
a = np.zeros((hidden_states, temp_orders_states - 1))

w = np.random.randn(iterations, hidden_states, temp_orders_states)
z = np.random.randn(iterations, hidden_states, temp_orders_states)

y_history = np.zeros((iterations, obs_states, temp_orders_states-1))           # position reading

x_hat = np.zeros((hidden_states, temp_orders_states))           # position
eta = np.zeros((hidden_states, temp_orders_states - 1))

x_hat_history = np.zeros((iterations, hidden_states, temp_orders_states))           # position reading
x_hat_dot_history = np.zeros((iterations, hidden_states, temp_orders_states))           # position reading

for i in range(iterations):
    x[:, 1:] = np.dot(A, x[:, :-1]) + np.dot(B, a) + np.dot(C, w[i, :, 1:])
    x[:, 0] += dt*x[:, 1]
    
    y = np.dot(H, x[:, :-1]) + np.dot(D, z[i, :, :-1])
    
    x_hat_dot = - np.dot(H.transpose(), np.dot(np.linalg.inv(np.dot(D, D.transpose())), (y - np.dot(H, x_hat[:, :-1])))) \
                    - np.dot(A.transpose(), np.dot(np.linalg.inv(np.dot(C, C.transpose())), (x_hat[:, 1:] - np.dot(A, x_hat[:, :-1]) + np.dot(B, eta)))) 
        
    x_prime_hat_dot = np.dot(np.linalg.inv(np.dot(C, C.transpose())), (x_hat[:, 1:] - np.dot(A, x_hat[:, :-1]) + np.dot(B, eta)))
    
    x_hat[:, 1:] += dt * - x_prime_hat_dot
    x_hat[:, :-1] += dt * (x_hat[:, 1:] - x_hat_dot)
    
#    bb = np.append(x_hat[:, 1:], np.zeros((2,1)), axis=1)
#    x_hat += dt * (bb - x_hat_dot)
    
#    x_hat[:, 1:] = np.dot(A, x_hat[:, :-1]) + np.dot(B, eta)
    
#    x_hat[0,0] += dt* (x_hat[0,1] - D[0,0]*(y[0,0] - x_hat[0,0]))
#    x_hat[1,0] += dt* (x_hat[1,1] - D[1,0]*(y[0,0] - x_hat[0,0]))
#    
    x_hat_dot_history[i, :, :] = x_hat_dot
    x_hat_history[i, :, :] = x_hat
    y_history[i, :, :] = y

plt.figure()
plt.plot(y_history[:,0,0], 'b')
plt.plot(x_hat_history[:,0,0], 'r')

plt.figure()
plt.plot(y_history[:,1,0], 'b')
plt.plot(x_hat_history[:,1,0], 'r')

plt.figure()
plt.plot(y_history[:,1,0], 'b')
plt.plot(x_hat_history[:,0,1], 'r')

plt.figure()
plt.plot(x_hat_history[:,1,1], 'r')
