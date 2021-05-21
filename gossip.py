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
    rObliv_Goss, rObliv_Inform, p1, rGoss_Believ, rGoss_Inform, p2, rInform_Knowl, p3 = args 
    #p1 - prop. of believes who beleive, p2 - prop. of non beleivers who dont beleive, p3 - prop. of gossipers,
    #rest are rates from one group to another
    O, G, B, I, K = Y
    dYdt = np.array([-rObliv_Goss * O * G - rObliv_Inform * O * I, \
                    p1 * rObliv_Goss * O * G - rGoss_Believ * G - rGoss_Inform * G * I, \
                    (1-p1) * rObliv_Goss * O * G + rGoss_Believ * G, \
                    p2 * rObliv_Inform * O * I + p3 * rGoss_Inform * G * I, \
                    (1 - p2) * rObliv_Inform * O * I + (1-p3) * rGoss_Inform * G * I + rInform_Knowl * I])
    return dYdt

def parameters(gossiper, o_g=None, o_i=None, prop1=None, g_b=None, g_i=None, prop2=None, i_k=None, prop3=None):
    #this function is an easy way to access the anthropomorphized sensitivity analyses
    #the string gossiper is required. If the string is not one defined below, values
    #b, g, p, and a must also be sent to the function.
    #the returned values are the parameter values for beta, gamma, rho, and alpha.
    if gossiper == 'Gossiper Name':
        rObliv_Goss = 0.002
        rObliv_Inform = 0.01
        p1 = 0.3
        rGoss_Believ = 0.05
        rGoss_Inform = 0.02
        p2 = 0.3
        rInform_Knowl = 0.008
        p3 = 0.4
    else:
        rObliv_Goss = o_g
        rObliv_Inform = o_i
        p1 = prop1
        rGoss_Believ = g_b
        rGoss_Inform = g_i
        p2 = prop2
        rInform_Knowl = i_k
        p3 = prop3
    return rObliv_Goss, rObliv_Inform, p1, rGoss_Believ, rGoss_Inform, p2, rInform_Knowl, p3

TotalPopulation = 1000
RumorStarters = 1
IC = np.array([TotalPopulation-RumorStarters, RumorStarters, 0., 0., 0.])

Days = 100.
times = np.arange(0., Days+delta, delta)

gossiper = 'Gossiper Name'
Rumor = RK4(gossip, IC, times, parameters(gossiper))

with plt.rc_context({'figure.figsize':(9,7)}):
    plt.plot(times, Rumor[:, 0], lw='5', c = 'b', label= 'O')
    plt.plot(times, Rumor[:, 1], lw='5', c = 'r', label = 'G')
    plt.plot(times, Rumor[:, 2], lw='5', c = 'y', label = 'B')
    plt.plot(times, Rumor[:, 3], lw='5', c = 'k', label = 'I')
    plt.plot(times, Rumor[:, 4], lw='5', c = 'g', label = 'K')
    leg = plt.legend(loc='upper right',fontsize = 16)
    plt.xlabel('Days', fontsize = 14)
    plt.ylabel('Population', fontsize = 14)
    plt.title('The Spread of '+gossiper+'\'s lie', fontsize = 18)
    plt.show()
