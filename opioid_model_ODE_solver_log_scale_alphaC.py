
'''
This modeule specifies and solves the opioids ODE model where the community
     has additional influnce on itself in the interaction terms and there 
     there is both an alpha_C and alpha_H parameter.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from collections import OrderedDict
import math
import pandas as pd

# default start year
start_year = 2016

# default initial and final times in terms of years from start_year
t0 = 0
tf = 4

perc_hurt = 0.20 #default amount to put in the hurt category for community model

# default initial population values
global_init_vals = OrderedDict()
global_init_vals['PG0'] = 0.0432
global_init_vals['AG0'] = 0.00246 
global_init_vals['FG0'] = 0.000339
global_init_vals['RG0'] = 0.000508 
global_init_vals['SG0'] = 1 - global_init_vals['PG0'] - global_init_vals['AG0'] \
    - global_init_vals['FG0'] - global_init_vals['RG0']

global_init_vals['SH0'] = perc_hurt*global_init_vals['SG0']
global_init_vals['PC0'] = global_init_vals['PG0']
global_init_vals['AC0'] = global_init_vals['AG0']
global_init_vals['FC0'] = global_init_vals['FG0']
global_init_vals['RC0'] = global_init_vals['RG0']
global_init_vals['SC0'] = 1 - global_init_vals['SH0'] - global_init_vals['PC0'] \
    - global_init_vals['AC0'] - global_init_vals['FC0'] - global_init_vals['RC0']

# reorder the intial_vals OrderedDict
initial_vals_order = ['SG0', 'PG0', 'AG0', 'FG0', 'RG0', \
                    'SC0', 'SH0', 'PC0', 'AC0', 'FC0', 'RC0']
initial_vals_reordered = OrderedDict()
for key in initial_vals_order:
    initial_vals_reordered[key] = global_init_vals[key]
global_init_vals = initial_vals_reordered

# default parameter values
global_params = OrderedDict()
global_params['tilde_m'] = -0.0288
global_params['tilde_b'] = 0.332
global_params['log_beta_GA'] = -3.0
global_params['log_beta_GP'] = -4.0
global_params['log_theta_SG'] = -2.953
global_params['log_theta_P'] = -1.995
global_params['theta_A'] = 48.916
global_params['epsilon_G'] = 7.004
global_params['log_gamma'] = -4.937
global_params['zeta'] = 0.0536
global_params['log_nu'] = -2.496
global_params['sigma'] =  0.970
global_params['lambda_A'] = 0.00633
global_params['lambda_F'] = 1.0

global_params['omega'] = 10e-10
global_params['mu'] = 0.0143
global_params['mu_A'] = 0.0471
global_params['mu_F'] = 0.471 

# alpha_H = tilde_m_H*t+tilde_b_H
perc = 1+perc_hurt
global_params['tilde_m_H']  = global_params['tilde_m']*perc_hurt
global_params['tilde_b_H']  = global_params['tilde_b']*perc
global_params['tilde_m_C']  = global_params['tilde_m']*(perc_hurt+1)
global_params['tilde_b_C']  = global_params['tilde_b']*(((1-perc*perc_hurt)/(1-perc_hurt)))
global_params['rho_C']      = 0.1
global_params['rho_H']      = global_params['rho_C']*((1-perc_hurt)/perc_hurt)
global_params['log_beta_CA']    = global_params['log_beta_GA'] + math.log10(1-perc_hurt)
global_params['log_beta_HA']    = global_params['log_beta_GA'] + math.log10(perc_hurt)
global_params['log_beta_CP']    = global_params['log_beta_GP'] + math.log10(1-perc_hurt)
global_params['log_beta_HP']    = global_params['log_beta_GP'] + math.log10(perc_hurt)
global_params['epsilon_C']  = global_params['epsilon_G']*(1-perc_hurt)
global_params['epsilon_H']  = global_params['epsilon_G']*perc_hurt
global_params['log_theta_SH']   = global_params['log_theta_SG'] + math.log10(perc_hurt)
global_params['log_theta_SC']   = global_params['log_theta_SG'] + math.log10(1-perc_hurt)
global_params['k']          = 0.0

base_case = True #if false, then alter some of the parameters for other cases
if base_case == False:
    # global_params['k'] = 1.0
    # global_params['rho_H'] =  2*global_params['rho_H']
    # global_params['tilde_b_H'] = global_params['tilde_b_H']
    global_init_vals['AC0'] = global_init_vals['AC0']/2
        

def update_params(new_params):
    '''Update the global_params OrderedDict with any values listed in the new_params OrderedDict'''

    # call global_params from the global scope (i.e. outside this def)
    global global_params

    for key,val in new_params.items():
        if key in global_params.keys():
            global_params[key] = val
        else:
            print('Parameter {} does not exist.'.format(key))


def update_initial_vals(new_IC):
    
    '''Update the global_init_vals OrderedDict with any values listed in the 
        new_IC OrderedDict'''

    # call params from the global scope (i.e. outside this def)
    global global_init_vals

    for key,val in new_IC.items():
        if key in global_init_vals.keys():
            global_init_vals[key] = val
        else:
            print('Initial value {} does not exist.'.format(key))

    global_init_vals['SG0'] = 1 - global_init_vals['PG0'] - global_init_vals['AG0'] \
        - global_init_vals['FG0'] - global_init_vals['RG0']

    global_init_vals['SC0'] = 1 - global_init_vals['SH0'] - global_init_vals['PC0'] \
        - global_init_vals['AC0'] - global_init_vals['FC0'] - global_init_vals['RC0']
    

def ode_system(t, X, params):
    '''Define our general population and community model as a system of ODEs.
    Arguments:
        - t: time 
        - X: values of SG, PG, AG, FG, RG, SC, SH, PC, AC, FC, and RC
        - params: OrderedDict of parameters
    Returns:
        - Y: solution of SG, PG, AG, FG, RG, SC, SH, PC, AC, FC, and RC at 
            time t'''

    SG, PG, AG, FG, RG, SC, SH, PC, AC, FC, RC = X

    alpha_G = params['tilde_m']*t + params['tilde_b']
    alpha_H = params['tilde_m_H']*t + params['tilde_b_H']
    alpha_C = params['tilde_m_C']*t + params['tilde_b_C']

    beta_GP = 10**params['log_beta_GP']
    beta_GA = 10**params['log_beta_GA']
    theta_SG = 10**params['log_theta_SG']
    theta_P = 10**params['log_theta_P']
    gamma = 10**params['log_gamma']
    nu = 10**params['log_nu']

    beta_HP = 10**params['log_beta_HP']
    beta_HA = 10**params['log_beta_HA']
    beta_CA = 10**params['log_beta_CA']
    beta_CP = 10**params['log_beta_CP']
    theta_SH = 10**params['log_theta_SH']
    theta_SC = 10**params['log_theta_SC']

    # General Population Model
    dSGdt = - alpha_G*SG - beta_GA*SG*AG \
        - beta_GP*SG*PG - theta_SG*SG*FG \
        + params['epsilon_G']*PG + params['mu']*(PG+AG+FG+RG) \
        + params['mu_A']*AG + params['mu_F']*FG
    dPGdt = - params['epsilon_G']*PG - params['mu']*PG - gamma*PG \
        - theta_P*PG*FG  + alpha_G*SG
    dAGdt = - params['zeta']*AG - params['theta_A']*AG*FG \
        - (params['mu']+params['mu_A'])*AG + beta_GA*SG*AG \
        + beta_GP*SG*PG + gamma*PG \
        + params['sigma']*RG*((params['lambda_A']*AG \
        + (1-params['lambda_F'])*FG)/(AG+FG+params['omega']))  
    dFGdt = (-nu - (params['mu']+params['mu_F']) \
        + theta_SG*SG + theta_P*PG \
        + params['theta_A']*AG)*FG \
        + params['sigma']*RG*(((1-params['lambda_A'])*AG \
        + params['lambda_F']*FG)/(AG+FG+params['omega'])) 
    dRGdt = - params['sigma']*RG*((params['lambda_A']*AG \
            + (1-params['lambda_F'])*FG)/(AG+FG+params['omega'])) \
            - params['sigma']*RG*(((1-params['lambda_A'])*AG \
            + params['lambda_F']*FG)/(AG+FG+params['omega'])) \
            - params['mu']*RG + params['zeta']*AG + nu*FG

    # Community Model
    dSCdt = - (params['k']*FG+(1-params['k'])*FC)*theta_SC*SC \
        - params['rho_C']*SC - (params['k']*AG+(1-params['k'])*AC)*beta_CA*SC \
        - (params['k']*PG+(1-params['k'])*PC)*beta_CP*SC \
        + params['rho_H']*SH + params['epsilon_C']*PC \
        + params['mu']*(SH+PC+AC+FC+RC) + params['mu_A']*AC + params['mu_F']*FC - alpha_C*SC 
    dSHdt = - (params['k']*AG+(1-params['k'])*AC)*beta_HA*SH \
        - (params['k']*PG+(1-params['k'])*PC)*beta_HP*SH \
        - params['rho_H']*SH - alpha_H*SH \
        - (params['k']*FG+(1-params['k'])*FC)*theta_SH*SH \
        - params['mu']*SH + params['rho_C']*SC + params['epsilon_H']*PC
    dPCdt = - params['epsilon_H']*PC - params['epsilon_C']*PC \
        - gamma*PC - (params['k']*FG+(1-params['k'])*FC)*theta_P*PC \
        - params['mu']*PC + alpha_H*SH + alpha_C*SC 
    dACdt = - (params['k']*FG+(1-params['k'])*FC)*params['theta_A']*AC \
        - params['zeta']*AC - (params['mu']+params['mu_A'])*AC \
        + (params['k']*PG+(1-params['k'])*PC)*beta_HP*SH \
        + (params['k']*AG+(1-params['k'])*AC)*beta_HA*SH \
        + (params['k']*AG+(1-params['k'])*AC)*beta_CA*SC \
        + (params['k']*PG+(1-params['k'])*PC)*beta_CP*SC \
        + gamma*PC + params['sigma']*RC*((params['lambda_A']*AG \
        + (1-params['lambda_F'])*FG)/(AG+FG+params['omega'])) 
    dFCdt = - nu*FC - (params['mu']+params['mu_F'])*FC \
        + (params['k']*FG+(1-params['k'])*FC)*theta_SC*SC \
        + (params['k']*FG+(1-params['k'])*FC)*theta_SH*SH \
        + (params['k']*FG+(1-params['k'])*FC)*theta_P*PC \
        + (params['k']*FG+(1-params['k'])*FC)*params['theta_A']*AC \
        + params['sigma']*RC*(((1-params['lambda_A'])*AG \
        + params['lambda_F']*FG)/(AG+FG+params['omega'])) 
    dRCdt = - params['sigma']*RC*((params['lambda_A']*AG \
            + (1-params['lambda_F'])*FG)/(AG+FG+params['omega'])) \
            - params['sigma']*RC*(((1-params['lambda_A'])*AG \
            + params['lambda_F']*FG)/(AG+FG+params['omega'])) \
            - params['mu']*RC + params['zeta']*AC + nu*FC
    
    # print('t = {}'.format(t))

    Y = [dSGdt, dPGdt, dAGdt, dFGdt, dRGdt, dSCdt, dSHdt, dPCdt, dACdt, \
         dFCdt, dRCdt]

    return Y


def solve_odes(initial_vals=global_init_vals, time_range=[t0, tf], time_step = 1/100, params=global_params):
    '''
    Solve ode_system for the given initial conditions, time range, 
    and params using RK45.
    '''
    # update the parameters and ICs
    update_params(params)
    update_initial_vals(initial_vals)

    sol = solve_ivp(ode_system, 
                    time_range, 
                    list(global_init_vals.values()), 
                    t_eval = np.arange(time_range[0], time_range[1]+time_step, time_step), 
                    args=[global_params],
                    rtol=1e-4,
                    atol=1e-10)
    
    # Note to get, for example, S_G(4) we would run the following:
    # sol = solve_odes()
    # SG_sol, PG_sol, AG_sol, FG_sol, RG_sol, SC_sol, SH_sol, PC_sol, AC_sol, \
    #     FC_sol, RC_sol = sol.y
    # SG_sol[int(1/time_step*4)]
    
    return sol


def plot_genpop_or_comm(sol, start_year=start_year, time_range=[t0,tf], show=True, plot_genpop=True):
    SG_sol, PG_sol, AG_sol, FG_sol, RG_sol, SC_sol, SH_sol, PC_sol, AC_sol, \
    FC_sol, RC_sol = sol.y

    t0 = time_range[0]
    tf = time_range[1]

    if plot_genpop:
        # plot general population
        labels = ['SG','PG','AG','FG','RG']
        sols = [SG_sol, PG_sol, AG_sol, FG_sol, RG_sol]
        fig, ax = plt.subplots(3,2,figsize=(7,7))
        kk = 0
        I = 3
        J = 2
        for ii in range(I):
            for jj in range(J):
                if ii == I-1 and jj == J-1:
                    kk+= 1
                else:
                    axis = ax[ii,jj]
                    axis.plot(sol.t + start_year, sols[kk].T, color = 'b')
                    axis.set_xlabel('Year')
                    axis.set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
                    axis.set_ylabel('${}_G$'.format(labels[kk][0]))
                    # axis.legend(['$S_G$'], shadow=True)

                    kk += 1
        fig.delaxes(ax[I-1,J-1])
    else:
        # plot the commmunity 
        labels = ['SC','SH','PC','AC','FC','RC']
        sols = [SC_sol, SH_sol, PC_sol, AC_sol, FC_sol, RC_sol]
        fig, ax = plt.subplots(3,2,figsize=(7,7))
        kk = 0
        I = 3
        J = 2
        for ii in range(I):
            for jj in range(J):
                axis = ax[ii,jj]
                axis.plot(sol.t + start_year, sols[kk].T, color = 'b')
                axis.set_xlabel('Year')
                axis.set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
                axis.set_ylabel('${}_{}$'.format(labels[kk][0],labels[kk][1]))
                # axis.legend(['$S_G$'], shadow=True)

                kk += 1

    plt.tight_layout()
    if show:
        plt.show()
    else:
        if plot_genpop:
           plt.savefig('genpop_solutions.png')
        else:
            plt.savefig('comm_solutions.png')


def plot_solutions(sol, start_year=start_year, time_range=[t0,tf], show=True):
    '''Plot a solution set and either show it or return the plot object'''

    SG_sol, PG_sol, AG_sol, FG_sol, RG_sol, SC_sol, SH_sol, PC_sol, AC_sol, \
    FC_sol, RC_sol = sol.y

    t0 = time_range[0]
    tf = time_range[1]

    # General Population model results
    plt.subplot(2, 3, 1)
    plt.plot(sol.t + start_year, SG_sol.T, color = 'k')
    plt.xlabel('$t$')
    plt.xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    plt.ylabel('Proportion of Population')
    plt.legend(['$S_G$'], shadow=True)

    plt.subplot(2, 3, 2)
    plt.plot(sol.t + start_year, PG_sol.T, color = 'b')
    plt.xlabel('$t$')
    plt.xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    plt.legend(['$P_G$'], shadow=True)

    plt.subplot(2, 3, 3)
    plt.plot(sol.t + start_year, AG_sol.T, color = 'r')
    plt.plot(sol.t + start_year, FG_sol.T, color = 'g')
    plt.plot(sol.t + start_year, RG_sol.T, color = 'm')
    plt.xlabel('$t$')
    plt.xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    plt.legend(['$A_G$', '$F_G$', '$R_G$'], shadow=True)

    # Ceteran model results 
    plt.subplot(2, 3, 4)
    plt.plot(sol.t + start_year, SC_sol.T, color = 'k')
    plt.plot(sol.t + start_year, SH_sol.T, color = 'tab:orange')
    plt.xlabel('$t$')
    plt.xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    plt.ylabel('Proportion of Population')
    plt.legend(['$S_C$', '$S_H$'], shadow=True)

    plt.subplot(2, 3, 5)
    plt.plot(sol.t + start_year, PC_sol.T, color = 'b')
    plt.xlabel('$t$')
    plt.xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    plt.legend(['$P_C$'], shadow=True)

    plt.subplot(2, 3, 6)
    plt.plot(sol.t + start_year, AC_sol.T, color = 'r')
    plt.plot(sol.t + start_year, FC_sol.T, color = 'g')
    plt.plot(sol.t + start_year, RC_sol.T, color = 'm')
    plt.xlabel('$t$')
    plt.xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    plt.legend(['$A_C$', '$F_C$', '$R_C$'], shadow=True)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('genpop_and_comm_solutions.png')
        

def plot_combined_community(sol, start_year=start_year, time_range=[t0,tf], show=True, horizontal=True):
    '''Plot the solution set with the S_C and S_H combined and either show it or return the plot object'''

    SG_sol, PG_sol, AG_sol, FG_sol, RG_sol, SC_sol, SH_sol, PC_sol, AC_sol, \
        FC_sol, RC_sol = sol.y
    
    t0 = time_range[0]
    tf = time_range[1]

    if horizontal == True:
        fig,ax = plt.subplots(nrows=2, ncols=3, figsize=(13, 6))

        for ii in range(2):
            for jj in range(3):
                ax[ii,jj].set_xlabel('Year')

        # General Population model results
        ax[0,0].plot(sol.t + start_year, SG_sol.T, color = 'purple')
        ax[0,0].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        ax[0,0].set_ylabel('Proportion of Population')
        ax[0,0].legend(['$S_G$'], shadow=True)

        ax[0,1].plot(sol.t + start_year, PG_sol.T, color = 'darkorange')
        ax[0,1].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        ax[0,1].legend(['$P_G$'], shadow=True)
        ax[0,1].set_ylabel('Proportion of Population')

        ax[0,2].plot(sol.t + start_year, AG_sol.T, color = 'c')
        ax[0,2].plot(sol.t + start_year, FG_sol.T, color = 'm')
        ax[0,2].plot(sol.t + start_year, RG_sol.T, color = 'g')
        ax[0,2].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        ax[0,2].legend(['$A_G$', '$F_G$', '$R_G$'], shadow=True)
        ax[0,2].set_ylabel('Proportion of Population')

        # Ceteran model results 
        line1 = ax[1,0].plot(sol.t + start_year, SC_sol.T, 
            color = 'r', label='$S_C$')
        line2 = ax[1,0].plot(sol.t + start_year, SH_sol.T, 
            color = 'b', label='$S_H$')
        ax[1,0].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        ax[1,0].set_ylabel('Proportion of Population')
        #create a second axis for SC+SH
        ax2 = ax[1,0].twinx() 
        line3 = ax2.plot(sol.t + start_year, SC_sol.T + SH_sol.T, 
            color = 'purple', linestyle='--', label='$S_C + S_H$')
        ax2.set_ylabel('$S_C + S_H$', color='purple')
        ax2.tick_params(axis='y', colors='purple')
        # create a legend for all three 
        lns = line1+line2+line3
        labs = [l.get_label() for l in lns]
        ax[1,0].legend(lns, labs, shadow=True, loc='center left')

        ax[1,1].plot(sol.t + start_year, PC_sol.T, color = 'darkorange')
        ax[1,1].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        ax[1,1].legend(['$P_C$'], shadow=True)
        ax[1,1].set_ylabel('Proportion of Population')

        ax[1,2].plot(sol.t + start_year, AC_sol.T, color = 'c')
        ax[1,2].plot(sol.t + start_year, FC_sol.T, color = 'm')
        ax[1,2].plot(sol.t + start_year, RC_sol.T, color = 'g')
        ax[1,2].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        ax[1,2].legend(['$A_C$', '$F_C$', '$R_C$'], shadow=True)
        ax[1,2].set_ylabel('Proportion of Population')
    else:
        fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(8,7))

        for ii in range(3):
            for jj in range(2):
                ax[ii,jj].set_xlabel('Year')
                ax[ii,jj].set_ylabel('Proportion of Population')

        # General Population model results
        a = ax[0,0]
        a.plot(sol.t + start_year, SG_sol.T, color = 'purple')
        a.set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        a.legend(['$S_G$'], shadow=True)

        a = ax[1,0]
        a.plot(sol.t + start_year, PG_sol.T, color = 'darkorange')
        a.set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        a.legend(['$P_G$'], shadow=True)

        a = ax[2,0]
        a.plot(sol.t + start_year, AG_sol.T, color = 'c')
        a.plot(sol.t + start_year, FG_sol.T, color = 'm')
        a.plot(sol.t + start_year, RG_sol.T, color = 'g')
        a.set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        a.legend(['$A_G$', '$F_G$', '$R_G$'], shadow=True)

        # Ceteran model results 
        a = ax[0,1]
        line1 = a.plot(sol.t + start_year, SC_sol.T, 
            color = 'r', label='$S_C$')
        line2 = a.plot(sol.t + start_year, SH_sol.T, 
            color = 'b', label='$S_H$')
        a.set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        #create a second axis for SC+SH
        ax2 = a.twinx() 
        line3 = ax2.plot(sol.t + start_year, SC_sol.T + SH_sol.T, 
            color = 'purple', linestyle='--', label='$S_C + S_H$')
        ax2.set_ylabel('$S_C + S_H$', color='purple')
        ax2.tick_params(axis='y', colors='purple')
        # create a legend for all three 
        lns = line1+line2+line3
        labs = [l.get_label() for l in lns]
        a.legend(lns, labs, shadow=True, loc='center left')

        a = ax[1,1]
        a.plot(sol.t + start_year, PC_sol.T, color = 'darkorange')
        a.set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        a.legend(['$P_C$'], shadow=True)

        a = ax[2,1]
        a.plot(sol.t + start_year, AC_sol.T, color = 'c')
        a.plot(sol.t + start_year, FC_sol.T, color = 'm')
        a.plot(sol.t + start_year, RC_sol.T, color = 'g')
        a.set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
        a.legend(['$A_C$', '$F_C$', '$R_C$'], shadow=True)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig('opioid_model.png')


def plot_S_classes(sol, start_year=start_year, time_range=[t0,tf]):

    SG_sol, PG_sol, AG_sol, FG_sol, RG_sol, SC_sol, SH_sol, PC_sol, AC_sol, \
        FC_sol, RC_sol = sol.y
    
    t0 = time_range[0]
    tf = time_range[1]

    fig,ax = plt.subplots(nrows=2, ncols=2)

    ax[0,0].plot(sol.t + start_year, SG_sol.T, color = 'purple')
    ax[0,0].set_xlabel('$t$')
    ax[0,0].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    ax[0,0].set_ylabel('Proportion of Population')
    ax[0,0].legend(['$S_G$'], shadow=True)

    ax[0,1].plot(sol.t + start_year, SC_sol.T + SH_sol.T)
    ax[0,1].set_xlabel('$t$')
    ax[0,1].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    ax[0,1].set_ylabel('Proportion of Population')
    ax[0,1].legend(['$S_H + S_C$'], shadow=True)
    
    ax[1,0].plot(sol.t + start_year, SC_sol.T)
    ax[1,0].set_xlabel('$t$')
    ax[1,0].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    ax[1,0].set_ylabel('Proportion of Population')
    ax[1,0].legend(['$S_C$'], shadow=True)

    ax[1,1].plot(sol.t + start_year, SH_sol.T)
    ax[1,1].set_xlabel('$t$')
    ax[1,1].set_xticks(np.arange(t0+start_year, tf+start_year+1, step=1))
    ax[1,1].set_ylabel('Proportion of Population')
    ax[1,1].legend(['$S_H$'], shadow=True)

    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.tight_layout()
    plt.show()


def print_sol_vals(time_step=1.0):

    sol = solve_odes(time_step=time_step)
    
    data = {'time': sol.t+start_year}

    labels = ['SG','PG','AG','FG','RG','SC','SH','PC','AC','FC','RC']
    for ii in range(len(labels)):
        data[labels[ii]] = sol.y[ii]

    df = pd.DataFrame(data)

    print(df) 


if __name__ == "__main__":

    # run model with default values and plot

    sol = solve_odes()

    # print_sol_vals(time_step=1.0)
    
    plot_combined_community(sol,horizontal=True)

    # plot_S_classes(sol)

    # plot_solutions(sol)

    # print(global_params['tilde_m_H'], global_params['tilde_b_H'], global_params['tilde_m_C'],global_params['tilde_b_C'])

    # plot_genpop_or_comm(sol,plot_genpop=False)