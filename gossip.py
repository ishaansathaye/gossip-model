import numpy as np
import matplotlib.pyplot as plt
#https://github.com/izabelaguiar/The-Mathematics-of-Gossip/blob/master/The%20Mathematics%20of%20Gossip.ipynb
#https://scholarship.claremont.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=1048&context=codee

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

#Change Rates
def gossip(Y, args):
    #Y is a state vector, starting with the defined initial conditions.
    #args is an array of the parameters necessary for the model
    #the value returned, dYdt is given from the system of ODEs defined above.
    r_og, r_oi, p1, r_gb, r_gi, p2, r_ik, p3 = args 
    #p1 - prop. of believes who beleive, p2 - prop. of non beleivers who dont beleive, p3 - prop. of gossipers,
    #rest are rates from one group to another
    O, G, B, I, K = Y
    dYdt = np.array([-r_og * O * G - r_oi * O * I, \
                    p1 * r_og * O * G - r_gb * G - r_gi * G * I, \
                    (1-p1) * r_og * O * G + r_gb * G, \
                    p2 * r_oi * O * I + p3 * r_gi * G * I, \
                    (1 - p2) * r_oi * O * I + (1-p3) * r_gi * G * I + r_ik * I])
    return dYdt

def parameters(gossiper, o_g=None, o_i=None, prop1=None, g_b=None, g_i=None, prop2=None, i_k=None, prop3=None):
    #this function is an easy way to access the anthropomorphized sensitivity analyses
    #the string gossiper is required. If the string is not one defined below, values
    #b, g, p, and a must also be sent to the function.
    #the returned values are the parameter values for beta, gamma, rho, and alpha.
    if gossiper == 'Gossiper Name':
        r_og = 0.3
        r_oi = 0.1
        p1 = 300.
        r_gb = 0.05
        r_gi = 0.02
        p2 = 300.
        r_ik = 0.06
        p3 = 400.
    else:
        r_og = o_g
        r_oi = o_i
        p1 = prop1
        r_gb = g_b
        r_gi = g_i
        p2 = prop2
        r_ik = i_k
        p3 = prop3
    return r_og, r_oi, p1, r_gb, r_gi, p2, r_ik, p3

TotalPopulation = 1000
RumorStarters = 1
IC = np.array([TotalPopulation-RumorStarters, RumorStarters, 0., 0., 0.])

# Days = 31.
# times = np.arange(0., Days+delta, delta)

# gossiper = 'Gossiper Name'
# Rumor = RK4(gossip, IC, times, parameters(gossiper))

# with plt.rc_context({'figure.figsize':(9,7)}):
#     plt.plot(times, Rumor[:, 0], lw='5', c = 'b', label= 'O')
#     plt.plot(times, Rumor[:, 1], lw='5', c = 'r', label = 'G')
#     plt.plot(times, Rumor[:, 2], lw='5', c = 'y', label = 'B')
#     plt.plot(times, Rumor[:, 3], lw='5', c = 'k', label = 'I')
#     plt.plot(times, Rumor[:, 4], lw='5', c = 'g', label = 'K')
#     leg = plt.legend(loc='upper right',fontsize = 16)
#     plt.xlabel('Days', fontsize = 14)
#     plt.ylabel('Population', fontsize = 14)
#     plt.title('The spread of '+gossiper+'\'s lie', fontsize = 18)
#     plt.show()
