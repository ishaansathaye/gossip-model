import numpy as np
import matplotlib.pyplot as plt

def RK4(fun, y0, times, args=None):
    h = times[1]-times[0]
    y = np.zeros((len(y0), len(times)))
    ly = len(y0)
    y[:, 0] = y0.reshape((ly,))
    for i in range(len(times)-1):
        k1 = h * fun(y[:, i], args)
        k2 = h * fun(y[:, i] + k1/2, args)
        k3 = h * fun(y[:, i] + k2/2, args)
        k4 = h * fun(y[:, i] + k3, args)
        y[:, i+1] = (y[:, i] + k1/6 + k2/3 + k3/3 + k4/6).reshape((ly,))
    return y.T

delta = 0.01

def gossip(Y, args):
    #Y is a state vector, starting with the defined initial conditions.
    #args is an array of the parameters necessary for the model
    #the value returned, dYdt is given from the system of ODEs defined above.
    beta, gamma, p, alpha = args
    S, I, R = Y
    dYdt = np.array([-beta * S * I,\
                     p * beta * S * I - gamma * I + alpha * R,\
                     gamma * I - alpha * R + (1-p) * beta * S * I])
    return dYdt

def parameters(gossiper, b=None, g=None, p=None, a=None):
    #this function is an easy way to access the anthropomorphized sensitivity analyses
    #the string gossiper is required. If the string is not one defined below, values
    #b, g, p, and a must also be sent to the function.
    #the returned values are the parameter values for beta, gamma, rho, and alpha.
    if gossiper == 'Regina George':
        beta = 0.03
        gamma = 0.1
        p = 0.2
        alpha = 0.
    elif gossiper == 'Dr. Neverheardofher':
        beta = 0.0001
        gamma = 0.00001
        p = 0.99
        alpha = 0.
    elif gossiper == 'the Conwoman':
        beta = 0.003
        gamma = 0.001
        p = 0.7
        alpha = 0.009
    elif gossiper == 'Test':
        beta = 0.002
        gamma = 0.01
        p = 0.6
        alpha = 0
    else:
        beta = b
        gamma = g
        p = p
        alpha = a
    return beta, gamma, p, alpha

TotalPopulation = 1000
RumorStarters = 1
IC = np.array([TotalPopulation-RumorStarters, RumorStarters, 0.])

Days = 31.
times = np.arange(0., Days+delta, delta)

gossiper = 'Test'
Rumor = RK4(gossip, IC, times, parameters(gossiper))

with plt.rc_context({'figure.figsize':(9,7)}):
    plt.plot(times, Rumor[:, 0], lw='5', c = 'b', label= 'S')
    plt.plot(times, Rumor[:, 1], lw='5', c = 'r', label = 'I')
    plt.plot(times, Rumor[:, 2], lw='5', c = 'y', label = 'R')
    leg = plt.legend(loc='upper right',fontsize = 16)
    plt.xlabel('Days', fontsize = 14)
    plt.ylabel('Population', fontsize = 14)
    plt.title('The spread of '+gossiper+'\'s lie', fontsize = 18)
    plt.show()