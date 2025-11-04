

'''
This module generates tables for comparing hypothetical community cases for the
     parameters zeta_C, log_nu_C, sigma_C, mu_F_C, mu_A_C', and rho_H'.
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

perc_hurt = 0.20 #default amount to put in the hurt category for subcomm model

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

global_init_vals['JC0'] = 0
global_init_vals['KC0'] = 0

# reorder the intial_vals OrderedDict
initial_vals_order = ['SG0', 'PG0', 'AG0', 'FG0', 'RG0', \
                    'SC0', 'SH0', 'PC0', 'AC0', 'FC0', 'RC0', 'JC0', 'KC0']
initial_vals_reordered = OrderedDict()
for key in initial_vals_order:
    initial_vals_reordered[key] = global_init_vals[key]
global_init_vals = initial_vals_reordered

# default parameter values
global_params = OrderedDict()
global_params['tilde_m_G'] = -0.0288
global_params['tilde_b_G'] = 0.332
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
global_params['tilde_m_H']  = global_params['tilde_m_G']*perc_hurt
global_params['tilde_b_H']  = global_params['tilde_b_G']*perc
global_params['tilde_m_C']  = global_params['tilde_m_G']*(perc_hurt+1)
global_params['tilde_b_C']  = global_params['tilde_b_G']*(((1-perc*perc_hurt)/(1-perc_hurt)))
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
global_params['k']          = 1.0

# new subcomm params
global_params['mu_F_C'] = global_params['mu_F'] 
global_params['mu_A_C'] = global_params['mu_A'] 
global_params['sigma_C'] = global_params['sigma']
global_params['zeta_C'] = global_params['zeta']
global_params['log_nu_C'] = global_params['log_nu']


def update_params(new_params):
    '''Update the global_params OrderedDict with any values listed in the new_params OrderedDict
    
    Argumnets:
        - new_params: OrderedDict of new param values'''

    # call global_params from the global scope (i.e. outside this def)
    global global_params

    for key,val in new_params.items():
        if key in global_params.keys():
            global_params[key] = val
        else:
            print('Parameter {} does not exist.'.format(key))


def update_initial_vals(new_IV):
    
    '''Update the global_init_vals OrderedDict with any values listed in the new_IV OrderedDict'''

    # call params from the global scope (i.e. outside this def)
    global global_init_vals

    for key,val in new_IV.items():
        if key in global_init_vals.keys():
            global_init_vals[key] = val
        else:
            print('Initial value {} does not exist.'.format(key))

    global_init_vals['SG0'] = 1 - global_init_vals['PG0'] - global_init_vals['AG0'] \
        - global_init_vals['FG0'] - global_init_vals['RG0']

    global_init_vals['SC0'] = 1 - global_init_vals['SH0'] - global_init_vals['PC0'] \
        - global_init_vals['AC0'] - global_init_vals['FC0'] - global_init_vals['RC0']
    

def ode_system(t, X, params):
    '''Define our community and subcommunity model as a system of ODEs.
    Arguments:
        - t: time 
        - X: values of SG, PG, AG, FG, RG, SC, SH, PC, AC, FC, RC, JC, and KC
        - params: OrderedDict of parameters
    Returns:
        - Y: solution of SG, PG, AG, FG, RG, SC, SH, PC, AC, FC, RC, JC, 
            and KC at time t'''

    SG, PG, AG, FG, RG, SC, SH, PC, AC, FC, RC, JC, KC = X

    alpha_G = params['tilde_m_G']*t + params['tilde_b_G']
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
    nu_C = 10**params['log_nu_C']

    # Gommunity Model
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

    # Subcommunity Model
    dSCdt = - (params['k']*FG+(1-params['k'])*FC)*theta_SC*SC \
        - params['rho_C']*SC - (params['k']*AG+(1-params['k'])*AC)*beta_CA*SC \
        - (params['k']*PG+(1-params['k'])*PC)*beta_CP*SC \
        + params['rho_H']*SH + params['epsilon_C']*PC \
        + params['mu']*(SH+PC+AC+FC+RC) + params['mu_A_C']*AC + params['mu_F_C']*FC - alpha_C*SC 
    dSHdt = - (params['k']*AG+(1-params['k'])*AC)*beta_HA*SH \
        - (params['k']*PG+(1-params['k'])*PC)*beta_HP*SH \
        - params['rho_H']*SH - alpha_H*SH \
        - (params['k']*FG+(1-params['k'])*FC)*theta_SH*SH \
        - params['mu']*SH + params['rho_C']*SC + params['epsilon_H']*PC
    dPCdt = - params['epsilon_H']*PC - params['epsilon_C']*PC \
        - gamma*PC - (params['k']*FG+(1-params['k'])*FC)*theta_P*PC \
        - params['mu']*PC + alpha_H*SH + alpha_C*SC 
    dACdt = - (params['k']*FG+(1-params['k'])*FC)*params['theta_A']*AC \
        - params['zeta_C']*AC - (params['mu']+params['mu_A_C'])*AC \
        + (params['k']*PG+(1-params['k'])*PC)*beta_HP*SH \
        + (params['k']*AG+(1-params['k'])*AC)*beta_HA*SH \
        + (params['k']*AG+(1-params['k'])*AC)*beta_CA*SC \
        + (params['k']*PG+(1-params['k'])*PC)*beta_CP*SC \
        + gamma*PC + params['sigma_C']*RC*((params['lambda_A']*AG \
        + (1-params['lambda_F'])*FG)/(AG+FG+params['omega'])) 
    dFCdt = - nu_C*FC - (params['mu']+params['mu_F_C'])*FC \
        + (params['k']*FG+(1-params['k'])*FC)*theta_SC*SC \
        + (params['k']*FG+(1-params['k'])*FC)*theta_SH*SH \
        + (params['k']*FG+(1-params['k'])*FC)*theta_P*PC \
        + (params['k']*FG+(1-params['k'])*FC)*params['theta_A']*AC \
        + params['sigma_C']*RC*(((1-params['lambda_A'])*AG \
        + params['lambda_F']*FG)/(AG+FG+params['omega'])) 
    dRCdt = - params['sigma_C']*RC*((params['lambda_A']*AG \
            + (1-params['lambda_F'])*FG)/(AG+FG+params['omega'])) \
            - params['sigma_C']*RC*(((1-params['lambda_A'])*AG \
            + params['lambda_F']*FG)/(AG+FG+params['omega'])) \
            - params['mu']*RC + params['zeta_C']*AC + nu_C*FC
    dJCdt = params['mu_A_C']*AC 
    dKCdt = params['mu_F_C']*FC
    
    # print('t = {}'.format(t))

    Y = [dSGdt, dPGdt, dAGdt, dFGdt, dRGdt, dSCdt, dSHdt, dPCdt, dACdt, \
         dFCdt, dRCdt, dJCdt, dKCdt]

    return Y


def solve_odes(initial_vals=global_init_vals, time_range=[t0, tf], time_step = 1/100, params=global_params):
    '''
    Solve community_veteran_odes for the given initial conditions, time range, 
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
    
    return sol

def print_sol_vals(time_step=1.0):

    sol = solve_odes(time_step=time_step)
    
    data = {'time': sol.t+start_year}

    labels = ['SG','PG','AG','FG','RG','SC','SH','PC','AC','FC','RC']
    for ii in range(len(labels)):
        data[labels[ii]] = sol.y[ii]

    df = pd.DataFrame(data)

    print(df) 


def subcomm_cases(param_str_list, to_Excel=False):
    ''' 
    Prints a table that shows how the A_C, F_C, and R_C classes and the 
        overdoses change at the final time if we alter the specified 
        parameter to be +-25% and +-50%.

    Arguments:
        - param_str_list: list of string of parameter name to change in subcomm to be +-25% and +-50%
        - to_Excel: if true output an Excel file, else print a latex formatted table
    '''

    # throw an error and kill the program (with raise) if param_str not valid
    for param_str in param_str_list:
        try:
            assert param_str in global_params.keys(), 'param_str not a valid parameter string'
        except AssertionError as e:
            e.args += ('param_str = {}'.format(param_str),)
            raise

    orignial_param_val_list = []
    for param in param_str_list:
        orignial_param_val_list.append(global_params[param])

    # Create a dataframe and name columns
    if param_str == 'rho_H':
        cols = param_str_list + ['Change in Param', 
                                    'Change in $S_C$', 
                                    'Change in $S_H$', 
                                    'Change in $P_C$',
                                    'Change in $A_C$', 
                                    'Change in $F_C$', 
                                    'Change in $R_C$',
                                    'Change in $A_C$ ODs',
                                    'Change in $F_C$ ODs',
                                    'Change in Total ODs']
    else:
        cols = param_str_list + ['Change in Param', 
                                    'Change in $A_C$', 
                                    'Change in $F_C$', 
                                    'Change in $R_C$',
                                    'Change in $A_C$ ODs',
                                    'Change in $F_C$ ODs',
                                    'Change in Total ODs']
    df = pd.DataFrame(columns=cols)

    # determine base case solutions
    base_sol = solve_odes(time_step=1.0)
    SG_base_sol, PG_base_sol, AG_base_sol, FG_base_sol, RG_base_sol, \
        SC_base_sol, SH_base_sol, PC_base_sol, AC_base_sol, \
        FC_base_sol, RC_base_sol, JC_base_sol, KC_base_sol = base_sol.y

    # calculate solutions for differente perc changes in params
    perc_vals = [-0.5, -0.25, 0.0, 0.25, 0.5]
    if param_str == 'k':
        perc_vals = [-1.0, -0.5, 0.0]
    if param_str == 'rho_H':
        perc_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    for perc in perc_vals:
        if perc == 0.0:
            if param_str == 'rho_H':
                df.loc[len(df)] = orignial_param_val_list + [100*perc, 
                                                        None, 
                                                        None, 
                                                        None,
                                                        None, 
                                                        None, 
                                                        None, 
                                                        None,
                                                        None,
                                                        None]
            else:
                df.loc[len(df)] = orignial_param_val_list + [100*perc, 
                                                        None, 
                                                        None, 
                                                        None, 
                                                        None,
                                                        None,
                                                        None]
        else:
            # update the param vals with the percent times the param value
            new_params_dict = OrderedDict()
            for ii in range(len(param_str_list)):
                if param_str_list[ii][:3] == 'log':
                    # want to only vary perc on orginal param, not the 
                    #   log params
                    new_params_dict[param_str_list[ii]] = \
                        math.log10(1+perc)+orignial_param_val_list[ii]
                else:
                    new_params_dict[param_str_list[ii]] = \
                        (1+perc)*orignial_param_val_list[ii]
            update_params(new_params_dict)

            # calculate the new solution
            sol = solve_odes(time_step=1.0)
            SG_sol, PG_sol, AG_sol, FG_sol, RG_sol, SC_sol, SH_sol, PC_sol, \
                AC_sol, FC_sol, RC_sol, JC_sol, KC_sol = sol.y

            # calcualte total ODs
            base_ODs = JC_base_sol[-1] + KC_base_sol[-1]
            new_ODs = JC_sol[-1] + KC_sol[-1]

            # create the new row
            new_row = []

            #collect the param values
            for param in param_str_list:
                new_row.append(global_params[param])
            # collec the rest of the row
            if param_str == 'rho_H':
                new_row = new_row + [100*perc, 
                                    100*(SC_sol[-1]-SC_base_sol[-1])/SC_base_sol[-1], 
                                    100*(SH_sol[-1]-SH_base_sol[-1])/SH_base_sol[-1],
                                    100*(PC_sol[-1]-PC_base_sol[-1])/PC_base_sol[-1],
                                    100*(AC_sol[-1]-AC_base_sol[-1])/AC_base_sol[-1], 
                                    100*(FC_sol[-1]-FC_base_sol[-1])/FC_base_sol[-1],
                                    100*(RC_sol[-1]-RC_base_sol[-1])/RC_base_sol[-1],
                                    100*(JC_sol[-1]-JC_base_sol[-1])/JC_base_sol[-1],
                                    100*(KC_sol[-1]-KC_base_sol[-1])/KC_base_sol[-1],
                                    100*(new_ODs-base_ODs)/base_ODs]
            else:
                new_row = new_row + [100*perc, 
                                    100*(AC_sol[-1]-AC_base_sol[-1])/AC_base_sol[-1], 
                                    100*(FC_sol[-1]-FC_base_sol[-1])/FC_base_sol[-1],
                                    100*(RC_sol[-1]-RC_base_sol[-1])/RC_base_sol[-1],
                                    100*(JC_sol[-1]-JC_base_sol[-1])/JC_base_sol[-1],
                                    100*(KC_sol[-1]-KC_base_sol[-1])/KC_base_sol[-1],
                                    100*(new_ODs-base_ODs)/base_ODs]

            # add on the new row
            df.loc[len(df)] = new_row
    
    # change all of the log params to their orginal param
    for col in df.columns[:len(param_str_list)]:
        if col[:3] == 'log':
            # remove the logs with 10**
            for ii in range(len(df[col])):
                df[col][ii] = 10**df[col][ii]

            #change the column name to be without log
            df = df.rename(columns={col:col[4:]})

    if to_Excel:
        file_name = 'subcomm_cases'
        for param_str in param_str_list:
            file_name = '{}_{}'.format(file_name, param_str)

        df.to_excel(file_name+'.xlsx', index=False)
    else:
        # turn the table entries into formatted strings for the latex table
        for col in df.columns[len(param_str_list):]:
            # format the percent columns to have + or -
            for ii in range(len(df[col])):
                if df[col][ii] > 0.0:
                    df[col][ii] = '$+%s\\%%$' % float('%.3g' % df[col][ii])
                elif df[col][ii] <= 0.0:
                    df[col][ii] =  '$%s\\%%$' % float('%.3g' % df[col][ii])

                # change any small/large values with e to be 10^
                if type(df[col][ii])==str and 'e' in df[col][ii]:
                    e_index = df[col][ii].index('e')
                    df[col][ii] = '{}*10^{{{}}}\\%$'.format(df[col][ii][:e_index],
                                                        df[col][ii][e_index+1:-3])

        df_no_nan = df.fillna('---')
        final_print = df_no_nan.to_latex(header=True,
                                        escape=False,
                                        index=False)
        final_print = final_print.replace('\\\\\n', '\\\\ \\hline\n')
        print(final_print)


if __name__ == "__main__":

    # run model with default values and plot

    # param_str = ['zeta_C', 'log_nu_C']
    param_str = ['sigma_C']
    # param_str = ['mu_F_C', 'mu_A_C']
    # param_str = ['rho_H']
    subcomm_cases(param_str, to_Excel=False)