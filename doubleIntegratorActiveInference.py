#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:16:51 2018

Implementation of a standard double integrator system, \ddot{q} = u.

@author: manuelbaltieri
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as splin
from autograd import grad, jacobian

dt = .01
T = 25
T_switch = int(T/3)
iterations = int(T / dt)
alpha = np.exp(12)                                              # drift in Generative Model
gamma = 1                                                   # drift in OU process
plt.close('all')
small_value = np.exp(-50)

# at the moment these values need to be equal
obs_states = 2
hidden_states = 2                                           # x, in Friston's work
hidden_causes = 2                                           # v, in Friston's work
temp_orders_observations = 2
temp_orders_hidden_states = 3                                      # generalised coordinates for hidden states x, but only using n-1
temp_orders_hidden_causes = 3                                      # generalised coordinates for hidden causes v (or \eta in biorxiv manuscript), but only using n-1



### cruise control problem from Astrom and Murray (2010), pp 65-69

# environment parameters
x = np.zeros((hidden_states, temp_orders_observations))           # position

A = np.array([[0, 1], [0, 0]])               # state transition matrix
B = np.array([[0, 0], [0, 1]])                     # input matrix
C = np.array([[0, 0], [0, 1]])               # noise dynamics matrix
D = np.array([[1, 0], [0, 1]])
H = np.array([[1, 0], [0, 1]])               # measurement matrix

v = np.zeros((hidden_causes, temp_orders_hidden_causes - 1))
y = np.zeros((obs_states, temp_orders_observations))
eta = np.zeros((hidden_causes, temp_orders_hidden_causes))

eta[0, 0] = 0




### free energy variables
A_gm = np.array([[1, 0], [0, 1]])               # state transition matrix
B_gm = -np.array([[1, 0], [0, 1]])                     # input matrix
C_gm = np.array([[1, 0], [0, 1]])               # noise dynamics matrix
D_gm = np.array([[1, 0], [0, 1]])
H_gm = np.array([[1, 0], [0, 1]])               # measurement matrix


a = np.zeros((hidden_states, temp_orders_observations - 1))

mu_x = np.random.randn(hidden_states, temp_orders_hidden_states)
#mu_x = np.zeros((hidden_states, temp_orders_hidden_states))
mu_v = np.random.randn(hidden_causes, temp_orders_hidden_states)
mu_v = np.zeros((hidden_causes, temp_orders_hidden_states))

# minimisation variables and parameters
dFdmu_x = np.zeros((hidden_states, temp_orders_hidden_states))
dFdmu_v = np.zeros((hidden_causes, temp_orders_hidden_states))
dFdmu_gamma_z = np.zeros((hidden_causes, temp_orders_hidden_states))
Dmu_x = np.zeros((hidden_states, temp_orders_hidden_states))
Dmu_v = np.zeros((hidden_causes, temp_orders_hidden_states))
k_mu_x = 1                                                  # learning rate perception
k_a = 1000                                                     # learning rate action

# noise on sensory input (world - generative process)
gamma_z = 6 * np.ones((obs_states, temp_orders_hidden_states))    # log-precisions
#gamma_z[:,1] = gamma_z[:,0] - np.log(2 * gamma)
pi_z = np.zeros((obs_states, obs_states))
np.fill_diagonal(pi_z, np.exp(gamma_z))

#pi_z = np.exp(gamma_z) * np.identity(obs_states)      # number of obs_states = temp_orders_hidden_states otherwise where does info on generalised observations come from?

#pi_z[0, 1] = pi_z[0, 0] / (2 * gamma)
sigma_z = np.linalg.inv(splin.sqrtm(pi_z))
z = np.random.randn(iterations, obs_states)

# noise on motion of hidden states (world - generative process)
gamma_w = 4                                                  # log-precision
pi_w = np.zeros((hidden_states, hidden_states))
np.fill_diagonal(pi_w, np.exp(gamma_w))


#pi_w = np.exp(gamma_w) * np.identity(hidden_states)      # number of hidden_states = temp_orders_hidden_states
#pi_w[0, 1] = pi_w[0, 0] / (2 * gamma)
sigma_w = np.linalg.inv(splin.sqrtm(pi_w))
w = np.random.randn(iterations, hidden_states)


# agent's estimates of the noise (agent - generative model)
mu_gamma_z = -6 * np.ones((obs_states, temp_orders_hidden_states))    # log-precisions
#mu_gamma_z[0, 1] = mu_gamma_z[0, 0] - np.log(2 * gamma)
mu_pi_z = np.exp(mu_gamma_z) * np.ones((obs_states, temp_orders_hidden_states))
mu_pi_z = np.diag(np.diag(mu_pi_z))

mu_gamma_w = -20 * np.ones((obs_states, temp_orders_hidden_states))   # log-precision
#mu_gamma_w[0, 1] = mu_gamma_w[0, 0] - np.log(2)
mu_pi_w = np.exp(mu_gamma_w) * np.ones((hidden_states, temp_orders_hidden_states))
mu_pi_w = np.diag(np.diag(mu_pi_w))


# history
x_history = np.zeros((iterations, hidden_states, temp_orders_hidden_states))
y_history = np.zeros((iterations, obs_states, temp_orders_observations))
v_history = np.zeros((iterations, hidden_causes, temp_orders_hidden_states - 1))
psi_history = np.zeros((iterations, obs_states, temp_orders_hidden_states - 1))
mu_x_history = np.zeros((iterations, hidden_states, temp_orders_hidden_states))
eta_history = np.zeros((iterations, hidden_causes, temp_orders_hidden_states))
a_history = np.zeros((iterations, obs_states, temp_orders_hidden_states-1))
mu_gamma_z_history = np.zeros((iterations, temp_orders_hidden_states-1))
mu_gamma_w_history = np.zeros((iterations, temp_orders_hidden_states-1))
mu_pi_z_history = np.zeros((iterations, obs_states, obs_states))
mu_pi_w_history = np.zeros((iterations, hidden_states, hidden_states))
dFdmu_x_history = np.zeros((iterations, hidden_states, temp_orders_hidden_states-1))
dFdmu_x_prime_history = np.zeros((iterations, hidden_states, temp_orders_hidden_states-1))
dFdmu_x_history2 = np.zeros((iterations, hidden_states, temp_orders_hidden_states+1))


xi_z_history = np.zeros((iterations, obs_states, temp_orders_hidden_states - 1))
xi_w_history = np.zeros((iterations, hidden_states, temp_orders_hidden_states - 1))

FE_history = np.zeros((iterations,))




### FUNCTIONS ###

## bounded control ##

def sigmoid(x):
#    return x
    return 1*np.tanh(x/5)

def dsigmoid(x):
#    return 1
    return 1 - sigmoid(x)**2


## free energy functions ##
# generative process
def g(x, v):
    return np.dot(H, x)

def f(x, v, a):
    abc = np.dot(B, a)
    return np.dot(A, x) + np.dot(B, sigmoid(a))# + np.dot(B, v)

# generative model
def g_gm(x, v):
    return g(x, v)

def f_gm(x, v):
    # no action in generative model, a = 0.0
    return np.dot(A_gm, x) + np.dot(B_gm, v)

def getObservation(x, v, a, w):
    x[:, 1:] = f(x[:, :-1], v, a) #+ np.dot(C, w[:, None])
    x[:, 0] += dt * x[:, 1]
    return g(x, v)
#    return g(x[:, :-1], v)

#def F(psi, mu_x, eta, mu_pi_z, mu_pi_w):
#    return .5 * np.dot(np.dot((psi - np.dot(H_gm, mu_x[:, :-2])).transpose(), mu_pi_z), (psi - np.dot(H_gm, mu_x[:, :-2]))) + \
#                np.dot(np.dot((mu_x[:, 1:-1] - f_gm(mu_x[:, :-2], eta)).transpose(), mu_pi_w), (mu_x[:, 1:-1] - f_gm(mu_x[:, :-2], eta))) - \
#                np.trace(np.log(mu_pi_z * mu_pi_w))
                
def F(psi, mu_x, eta, mu_pi_z, mu_pi_w):
    return .5 * np.dot(np.dot((psi - np.dot(H_gm, mu_x[:, 1:-1])).transpose(), mu_pi_z), (psi - np.dot(H_gm, mu_x[:, 1:-1]))) + \
                np.dot(np.dot((mu_x[:, 2:] - alpha * f_gm(mu_x[:, 1:-1], eta[:, 1:-1])).transpose(), mu_pi_w), (mu_x[:, 2:] - alpha * f_gm(mu_x[:, 1:-1], eta[:, 1:-1]))) - \
                np.trace(np.log(mu_pi_z * mu_pi_w))
    
def mode_path(mu_x):
    return np.dot(mu_x, np.eye(temp_orders_hidden_states, k=-1))

x[0, 0] = 1.
#x[0, 1] = 23.
#x[1, 1] = 21.

a[1, 0] = -8.

#mu_x[0, 0] = x[0, 0] + .1*np.random.randn()
#mu_x[1, 0] = x[0, 1] + .1*np.random.randn()

# automatic differentiation
abbb = np.dot(np.dot((psi - np.dot(H_gm, mu_x[:, 1:-1])).transpose(), mu_pi_z), (psi - np.dot(H_gm, mu_x[:, 1:-1])))
abb = mu_x[:, 2:] - f_gm(mu_x[:, 1:-1], eta[:, 1:-1])
ab = np.dot((mu_x[:, 2:] - f_gm(mu_x[:, 1:-1], eta[:, 1:-1])).transpose(), mu_pi_w)
Fe = F(psi, mu_x, eta, mu_pi_z, mu_pi_w) 
dFdmu_states = grad(F, 1)
#dyda_f = grad(sigmoid, 0)

for i in range(iterations - 1):
#    print(i)    
    
    # Analytical noise, for one extra level of generalised cooordinates, this is equivalent to an ornstein-uhlenbeck process
#    dw = - gamma * w[i, 0, 0] + w[i, 0, 1] / np.sqrt(dt)
#    dz = - gamma * z[i, 0, 0] + z[i, 0, 1] / np.sqrt(dt)
#    
#    w[i+1, 0, 0] = w[i, 0, 0] + dt * dw                               # noise in dynamics, at the moment not used in generative process
#    z[i+1, 0, 0] = z[i, 0, 0] + dt * dz
    
#    y = getObservation(x, v, a, w[i, :])
    
    x[:, 1:] = np.dot(A, x[:, :-1]) + np.dot(B, sigmoid(a)) + np.dot(sigma_w, w[i, :, None])
    x[:, 0] += dt*x[:, 1]
    
    
    
    y = np.dot(H, x) #+ np.dot(D, np.diagonal(z[[i]]))
    
    psi = y[:, 1:] + np.dot(sigma_z, z[i, :, None])
    
    ### minimise free energy ###
    # perception
    dFdmu_x = dFdmu_states(psi, mu_x, eta, mu_pi_z, mu_pi_w)

#    dFdmu_x = - np.dot(H.transpose(), np.dot(mu_pi_z, (psi - np.dot(H, mu_x[:, :-2])))) \
#                    - np.dot(A.transpose(), np.dot(mu_pi_w, (mu_x[:, 1:-1] - np.dot(A, mu_x[:, :-2]) + np.dot(B, eta)))) 
#    dFdmu_x_prime = np.dot(mu_pi_w, (mu_x[:, 1:-1] - np.dot(A, mu_x[:, :-2]) + np.dot(B, eta)))

    
    Dmu_x = mode_path(mu_x)
    
    # action
#    dFda = np.array([0., np.sum(mu_pi_z * (psi - mu_x[0, :-1])), 0.])
    dFdy = np.dot(mu_pi_z, (psi - mu_x[:, 1:-1]))
    dFdy = np.dot(mu_pi_z, (y[:, 1:] - eta[:, 1:-1]))
    dyda = np.ones((obs_states, 1))
    
    
    # update equations
#    mu_x[:, 1:] += dt * k_mu_x * (Dmu_x[:,1, None] - dFdmu_x_prime)
#    mu_x[:, 0:-1] += dt * k_mu_x * (Dmu_x[:,0, None] - dFdmu_x)
#    mu_x[:, :-1] += dt * k_mu_x * (Dmu_x - dFdmu_x)
    mu_x += dt * k_mu_x * (Dmu_x - dFdmu_x)
    a[1, 0] += dt * - k_a * dyda.transpose().dot(dFdy)
    
    # save history
    y_history[i, :] = y
    psi_history[i, :] = psi
    mu_x_history[i, :, :] = mu_x
    mu_pi_z_history[i, :, :] = mu_pi_z
    eta_history[i] = eta
    a_history[i] = a
    v_history[i] = v
    
#    dFdmu_x_history[i,:] = dFdmu_x
#    dFdmu_x_history2[i,:,:] = dFdmu_x2
#    dFdmu_x_prime_history[i,:] = dFdmu_x_prime
    
#    FE_history[i] = F(psi, mu_x, eta, mu_gamma_z, mu_pi_w)
    

plt.figure()
plt.plot(y_history[:-1, 0, 0], y_history[:-1, 1, 0], 'b')
#plt.plot(mu_x_history[:-1, 0, 0], mu_x_history[:-1, 1, 0], 'r')
plt.plot(psi_history[0, 0, 0], psi_history[0, 1, 0], 'o')



    
plt.figure()
plt.plot(np.arange(0, T-dt, dt), y_history[:-1,0,0], 'b', label='Measured position')
#plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,0], 'r', label='Estimated position')
#plt.plot(np.arange(0, T-dt, dt), eta_history[:-1,0,0], 'g', label='Desired position')
plt.title('Position')
plt.xlabel('Time (s)')
#plt.ylabel('Velocity (km/h)')
plt.legend()

#plt.figure()
#plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,1,0], 'b', label='Measured velocity')
#plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,1], 'r', label='Estimated velocity')
##plt.plot(np.arange(0, T-dt, dt), eta_history[:-1,0,1], 'g', label='Desired velocity')
#plt.title('Velocity')
#plt.xlabel('Time (s)')
##plt.ylabel('Velocity (km/h)')
#plt.legend()

plt.figure()
plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,0,0], 'b', label='Measured position')
plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,0,1], 'r', label='Estimated position')
#plt.plot(np.arange(0, T-dt, dt), eta_history[:-1,1,0], 'g', label='Desired position')
plt.title('Velocity')
plt.xlabel('Time (s)')
#plt.ylabel('Velocity (km/h)')
plt.legend()

plt.figure()
plt.plot(np.arange(0, T-dt, dt), psi_history[:-1,1,0], 'b', label='Measured acceleration')
plt.plot(np.arange(0, T-dt, dt), mu_x_history[:-1,1,1], 'r', label='Estimated acceleration')
#plt.plot(np.arange(0, T-dt, dt), eta_history[:-1,1,1], 'g', label='Desired acceleration')
plt.title('Acceleration')
plt.xlabel('Time (s)')
#plt.ylabel('Velocity (km/h)')
plt.legend()


plt.figure()
plt.title('Action')
plt.plot(range(iterations-1), a_history[:-1,1,1])
#


#plt.figure()
#plt.plot(dFdmu_x_history[:,0])
#plt.plot(dFdmu_x_history2[:,0,0])
#
#plt.figure()
#plt.plot(dFdmu_x_history[:,1])
#plt.plot(dFdmu_x_history2[:,1,0])
#
#plt.figure()
#plt.plot(dFdmu_x_prime_history[:,0])
#plt.plot(dFdmu_x_history2[:,0,1])
#
#plt.figure()
#plt.plot(dFdmu_x_history[:,1])
#plt.plot(dFdmu_x_history2[:,1,1])
