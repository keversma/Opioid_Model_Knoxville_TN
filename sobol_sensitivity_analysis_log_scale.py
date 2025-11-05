
'''
Run sensitivity analysis on inital opioiod model paramters.
'''

import sys, os, time
import pickle # use to temporarily save the results
import argparse
from multiprocessing import Pool
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import opioid_model_ODE_solver_log_scale_alphaC as ode_model
import least_squares_parameter_estimation_log_scale as ls

# determine the number of CPUs in the system
default_n = os.cpu_count()

# allows you to run this from the command line with the given arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=2**5,
                    help="obtain N*(2D+2) samples from parameter space, N must be 2 to some power")
parser.add_argument("-n", "--ncores", type=int,
                    help="number of cores, defaults to {}".format(default_n))
parser.add_argument("-o", "--filename", type=str, 
                    help="filename to write output to, no extension",
                    default='analysis')
parser.add_argument("-p", "--params_file", type=str, default='raw_output\least_squares_sol_AC2016_False_x0_1000_assumemuAmuF_False.pickle',
                    help="name of file where final_params is stored from least squares output to use for param ranges")

def run_full_model(tilde_m,tilde_b,log_beta_GA,log_beta_GP,log_theta_SG,log_theta_P,theta_A,
                   epsilon_G,log_gamma,zeta,log_nu,sigma,lambda_A,lambda_F,
                   mu,mu_F,mu_A,tilde_m_H,tilde_b_H,tilde_m_C,tilde_b_C,rho_C,rho_H,log_beta_CA,
                   log_beta_HA,log_beta_CP,log_beta_HP,epsilon_C,epsilon_H,log_theta_SH,
                   log_theta_SC,k,PG0,AG0,FG0,RG0,SH0,PC0,AC0,FC0,RC0):
    '''
    Define a model wrapper based on the parameter space in main().

    Arguments are all the parameters and inital values defined in the SALib 
    problem definition. Note: This does not include S_G_0 or S_C_0.

    Returns the the values of the gen pop and comm classes at the final 
    time.  
    '''

    # When to start and end the model
    t0 = ls.t0
    tf = ls.tf

    # Create a dict for parameters
    params = OrderedDict()

    params['tilde_m']    = tilde_m
    params['tilde_b']    = tilde_b
    params['log_beta_GA']    = log_beta_GA
    params['log_beta_GP']    = log_beta_GP
    params['log_theta_SG']   = log_theta_SG
    params['log_theta_P']    = log_theta_P
    params['theta_A']    = theta_A
    params['epsilon_G']  = epsilon_G
    params['log_gamma']      = log_gamma
    params['zeta']       = zeta
    params['log_nu']    = log_nu
    params['sigma']      = sigma
    params['lambda_A']   = lambda_A
    params['lambda_F']   = lambda_F

    params['mu']         = mu
    params['mu_A']       = mu_A
    params['mu_F']       = mu_F

    params['tilde_m_H']  = tilde_m_H
    params['tilde_b_H']  = tilde_b_H
    params['tilde_m_C']  = tilde_m_C
    params['tilde_b_C']  = tilde_b_C
    params['rho_C']      = rho_C
    params['rho_H']      = rho_H
    params['log_beta_CA']    = log_beta_CA
    params['log_beta_HA']    = log_beta_HA
    params['log_beta_CP']    = log_beta_CP
    params['log_beta_HP']    = log_beta_HP
    params['epsilon_C']  = epsilon_C
    params['epsilon_H']  = epsilon_H
    params['log_theta_SH']   = log_theta_SH
    params['log_theta_SC']   = log_theta_SC
    params['k']          = k
    
    
    # Create a dict for initial values
    initial_vals = OrderedDict()
    initial_vals['PG0'] = PG0
    initial_vals['AG0'] = AG0
    initial_vals['FG0'] = FG0
    initial_vals['RG0'] = RG0
    initial_vals['SG0'] = 1 - initial_vals['PG0'] - initial_vals['AG0'] \
        - initial_vals['FG0'] - initial_vals['RG0']

    initial_vals['SH0'] = SH0
    initial_vals['PC0'] = PC0
    initial_vals['AC0'] = AC0
    initial_vals['FC0'] = FC0
    initial_vals['RC0'] = RC0
    initial_vals['SC0'] = 1 - initial_vals['SH0'] - initial_vals['PC0'] \
        - initial_vals['AC0'] - initial_vals['FC0'] - initial_vals['RC0']
    
    # Run the model
    try:
        sol = ode_model.solve_odes(initial_vals, [t0, tf], params=params)
    except:
        # In case of failure, return info about the failure and make it easy
        #   to find in the data.
        return (sys.exc_info()[1],None,None,None,None,None,None,None,None,
                None,None)
    
    # Return the values of S_G, P_G, A_G, F_G, R_G, S_C, S_H, P_C, A_C, F_C, 
    # and R_C at the final time as a list
    return sol.y[:,-1]

def generate_param_ranges(params_file):
    '''
    Generate a dict of param ranges: +/- 50% for the gen pop params and +/- 75% 
        for the comm params
        
    Arguments:
        - params_file: str of name where the gen pop parameter estimates 
            from the least squares
            
    Output:
        - OrderedDict of the parameter ranges in the correct order for the 
            run_full_model function'''
    # load the sen_params from the least squres output
    params_file = params_file + '.pickle' #'raw_output\\' + params_file + '.pickle'
    global_min, sen_params, obj_fun, obj_fun_vals_list = ls.load_results(params_file)    
    sen_params['mu'] = ls.assumed_params['mu'] # add on mu

    # Determine the bounds to be +/-50% of the gen pop parameters
    sen_param_bounds_dict = OrderedDict()
    for key in sen_params.keys():
        if key[:3] == 'log':
            sen_param_bounds_dict[key] = [round(sen_params[key]-1,1), round(sen_params[key]+1,1)] 
        else:
            sen_param_bounds_dict[key] = [0.5*sen_params[key], 
                                    1.5*sen_params[key]]

    # set the bounds for tilde_m and tilde_b
    sen_param_bounds_dict['tilde_m'] = [-0.0284,-0.0174]
    sen_param_bounds_dict['tilde_b'] = [0.2226,0.396]

    # set the bounds for the subcommunity parameters to be +/- 75% where 
    #   corresponding parameter exists in the community
    sen_param_bounds_dict['tilde_m_H'] = [-0.0393, -0.0160]
    sen_param_bounds_dict['tilde_b_H'] = [0.212, 0.444]
    sen_param_bounds_dict['tilde_m_C'] = [-0.0393, -0.0160]
    sen_param_bounds_dict['tilde_b_C'] = [0.212, 0.444]
    sen_param_bounds_dict['rho_C'] =  [0.001, 2.0]
    sen_param_bounds_dict['rho_H'] = [0.001, 2.0]
    sen_param_bounds_dict['log_beta_CA'] = \
        [round(sen_params['log_beta_GA']-1.5,1), round(sen_params['log_beta_GA']+1.5,1)]
    sen_param_bounds_dict['log_beta_HA'] = \
        [round(sen_params['log_beta_GA']-1.5,1), round(sen_params['log_beta_GA']+1.5,1)]
    sen_param_bounds_dict['log_beta_CP'] = \
        [round(sen_params['log_beta_GP']-1.5,1), round(sen_params['log_beta_GP']+1.5,1)]
    sen_param_bounds_dict['log_beta_HP'] = \
        [round(sen_params['log_beta_GP']-1.5,1), round(sen_params['log_beta_GP']+1.5,1)]
    sen_param_bounds_dict['epsilon_C'] = \
        [0.25*sen_params['epsilon_G'],
        1.75*sen_params['epsilon_G']]
    sen_param_bounds_dict['epsilon_H'] = \
        [0.25*sen_params['epsilon_G'],
        1.75*sen_params['epsilon_G']]
    sen_param_bounds_dict['log_theta_SH'] = \
        [round(sen_params['log_theta_SG']-1.5,1), round(sen_params['log_theta_SG']+1.5,1)]
    sen_param_bounds_dict['log_theta_SC'] = \
        [round(sen_params['log_theta_SG']-1.5,1), round(sen_params['log_theta_SG']+1.5,1)]
    sen_param_bounds_dict['k'] = [1.0e-10, 1.0]

    sen_param_bounds_dict['PC0'] = \
        [0.25*sen_params['PG0'], 
        1.75*sen_params['PG0']]
    sen_param_bounds_dict['AC0'] = \
        [0.25*sen_params['AG0'], 
        1.75*sen_params['AG0']]
    sen_param_bounds_dict['FC0'] = \
        [0.25*sen_params['FG0'], 
        1.75*sen_params['FG0']]
    sen_param_bounds_dict['RC0'] = \
        [0.25*sen_params['RG0'], 
        1.75*sen_params['RG0']]

    sen_param_bounds_dict['SH0'] = [0.0,0.5]

    # make sure the upper and lower bound are in the correct order
    for key in sen_param_bounds_dict.keys():
        if sen_param_bounds_dict[key][0]>sen_param_bounds_dict[key][1]:
            sen_param_bounds_dict[key] = [sen_param_bounds_dict[key][1],
                                        sen_param_bounds_dict[key][0]]

    # check to make sure that lamda_A and lamda_F aren't greater than one
    if sen_param_bounds_dict['lambda_A'][1] > 1.0:
        sen_param_bounds_dict['lambda_A'][1] = 1.0
    if sen_param_bounds_dict['lambda_F'][1] > 1.0:
        sen_param_bounds_dict['lambda_F'][1] = 1.0

    # reorder the sen_param_bounds_dict to match order of inputs for 
    #   run_full_model
    order = ['tilde_m','tilde_b','log_beta_GA','log_beta_GP','log_theta_SG',
             'log_theta_P','theta_A','epsilon_G','log_gamma','zeta','log_nu',
             'sigma','lambda_A','lambda_F','mu','mu_F','mu_A','tilde_m_H',
             'tilde_b_H','tilde_m_C','tilde_b_C','rho_C','rho_H','log_beta_CA','log_beta_HA',
             'log_beta_CP','log_beta_HP','epsilon_C','epsilon_H',
             'log_theta_SH','log_theta_SC','k','PG0','AG0','FG0',
             'RG0','SH0','PC0','AC0','FC0','RC0']
    reorder_sen_param_bounds_dict = OrderedDict()
    for key in order:
        reorder_sen_param_bounds_dict[key] = sen_param_bounds_dict[key]
    sen_param_bounds_dict = reorder_sen_param_bounds_dict.copy()

    return sen_param_bounds_dict


def main(N, filename, params_file, pool=None):
    '''Runs parameter sensivity on the opioid model.
    
    Arguments:
        - N: N*(2D+2) samples from parameter space
        - filename: name of where to store the results for plotting later
        - param_file: name of the file where the param ranges are stored 
            from the least squares output
        - pool: '''

    ###########################################################################
    # Define the parameter space 
    ###########################################################################

    sen_param_bounds_dict = generate_param_ranges(params_file)

    # turn into a numpy array 
    sen_param_bounds = np.array(list(sen_param_bounds_dict.values()))
    
    problem = {
        'num_vars': len(list(sen_param_bounds_dict.keys())),
        'names': list(sen_param_bounds_dict.keys()), 
        'bounds': sen_param_bounds.tolist()
    }

    ###########################################################################
    # Create an N*(2D+2) by num_var matrix of parameter and IC values, where D 
    #   is the number of parameters we are finding sensitivity on
    ###########################################################################
    param_and_IC_values = saltelli.sample(problem, N, calc_second_order=True)

    ###########################################################################
    # Run the model
    ###########################################################################
    print('Examining the parameter space.')
    if args.ncores is None:
        poolsize = os.cpu_count()
    else:
        poolsize = args.ncores
    chunksize = param_and_IC_values.shape[0]//poolsize

    output = pool.starmap(run_full_model, param_and_IC_values, 
                          chunksize=chunksize)
    
    

    ###########################################################################
    # Parse and save the output
    ###########################################################################
    print('Saving and reviewing the results...')
    param_and_IC_values = pd.DataFrame(param_and_IC_values, 
                                       columns=problem['names'])

    # write data to temporary location in case of errors
    with open("raw_result_data.pickle", "wb") as f:
        result = {'output':output, 'param_and_IC_values':param_and_IC_values}
        pickle.dump(result, f)

    # Look for errors
    error_num = 0
    error_places = []
    for n, result in enumerate(output):
        if result[1] is None:
            error_num += 1
            error_places.append(n)
    if error_num > 0:
        print("Errors discovered in output.")
        print("Parameter locations: {}".format(error_places))
        print("Please review pickled output.")
        return
    
    # Save results in HDF5 as dataframe
    print('Parsing the results...')
    output = np.array(output)
    # Resave as dataframe in hdf5
    store = pd.HDFStore(filename+'.h5')
    store['param_and_IC_values'] = param_and_IC_values
    store['raw_output'] = pd.DataFrame(output, columns=['S_G', 'P_G', 'A_G', 
                                                        'F_G', 'R_G', 'S_C', 
                                                        'S_H', 'P_C', 'A_C', 
                                                        'F_C', 'R_C'])
    os.remove('raw_result_data.pickle')

    ###########################################################################
    # Analyze the results and view using Pandas 
    ###########################################################################
    # Conduct the sobol analysis and pop out the S2 results to a dict
    S2 = {}
    S_G_sens = sobol.analyze(problem, output[:,0], calc_second_order=True)
    S2['S_G'] = pd.DataFrame(S_G_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['S_G_conf'] = pd.DataFrame(S_G_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    P_G_sens = sobol.analyze(problem, output[:,1], calc_second_order=True)
    S2['P_G'] = pd.DataFrame(P_G_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['P_G_conf'] = pd.DataFrame(P_G_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    A_G_sens = sobol.analyze(problem, output[:,2], calc_second_order=True)
    S2['A_G'] = pd.DataFrame(A_G_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['A_G_conf'] = pd.DataFrame(A_G_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    F_G_sens = sobol.analyze(problem, output[:,3], calc_second_order=True)
    S2['F_G'] = pd.DataFrame(F_G_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['F_G_conf'] = pd.DataFrame(F_G_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    R_G_sens = sobol.analyze(problem, output[:,4], calc_second_order=True)
    S2['R_G'] = pd.DataFrame(R_G_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['R_G_conf'] = pd.DataFrame(R_G_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    S_C_sens = sobol.analyze(problem, output[:,5], calc_second_order=True)
    S2['S_C'] = pd.DataFrame(S_C_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['S_C_conf'] = pd.DataFrame(S_C_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    S_H_sens = sobol.analyze(problem, output[:,6], calc_second_order=True)
    S2['S_H'] = pd.DataFrame(S_H_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['S_H_conf'] = pd.DataFrame(S_H_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    P_C_sens = sobol.analyze(problem, output[:,7], calc_second_order=True)
    S2['P_C'] = pd.DataFrame(P_C_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['P_C_conf'] = pd.DataFrame(P_C_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    A_C_sens = sobol.analyze(problem, output[:,8], calc_second_order=True)
    S2['A_C'] = pd.DataFrame(A_C_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['A_C_conf'] = pd.DataFrame(A_C_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    F_C_sens = sobol.analyze(problem, output[:,9], calc_second_order=True)
    S2['F_C'] = pd.DataFrame(F_C_sens.pop('S2'), index=problem['names'],
                           columns=problem['names'])
    S2['F_C_conf'] = pd.DataFrame(F_C_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    R_C_sens = sobol.analyze(problem, output[:,10], calc_second_order=True)
    S2['R_C'] = pd.DataFrame(R_C_sens.pop('S2'), index=problem['names'],
                             columns=problem['names'])
    S2['R_C_conf'] = pd.DataFrame(R_C_sens.pop('S2_conf'), 
                                  index=problem['names'],
                                  columns=problem['names'])
    
    # Gonvert the rest to a pandas dataframe
    S_G_sens = pd.DataFrame(S_G_sens,index=problem['names'])
    P_G_sens = pd.DataFrame(P_G_sens,index=problem['names'])
    A_G_sens = pd.DataFrame(A_G_sens,index=problem['names'])
    F_G_sens = pd.DataFrame(F_G_sens,index=problem['names'])
    R_G_sens = pd.DataFrame(R_G_sens,index=problem['names'])

    S_C_sens = pd.DataFrame(S_C_sens,index=problem['names'])
    S_H_sens = pd.DataFrame(S_H_sens,index=problem['names'])
    P_C_sens = pd.DataFrame(P_C_sens,index=problem['names'])
    A_C_sens = pd.DataFrame(A_C_sens,index=problem['names'])
    F_C_sens = pd.DataFrame(F_C_sens,index=problem['names'])
    R_C_sens = pd.DataFrame(R_C_sens,index=problem['names'])


    ###########################################################################
    # Save the analysis
    ###########################################################################
    print('Saving...')
    store['S_G_sens'] = S_G_sens
    store['P_G_sens'] = P_G_sens
    store['A_G_sens'] = A_G_sens
    store['F_G_sens'] = F_G_sens
    store['R_G_sens'] = R_G_sens

    store['S_C_sens'] = S_C_sens
    store['S_H_sens'] = S_H_sens
    store['P_C_sens'] = P_C_sens
    store['A_C_sens'] = A_C_sens
    store['F_C_sens'] = F_C_sens
    store['R_C_sens'] = R_C_sens
    for key in S2.keys():
        store['S2/'+key] = S2[key]
    store.close()

    ###########################################################################
    # Plot 
    ###########################################################################
    plot_S1_ST(S_G_sens, P_G_sens, A_G_sens, F_G_sens, R_G_sens, S_C_sens, 
               S_C_sens, P_C_sens, A_C_sens, F_C_sens, R_C_sens, False)
    

def load_data(filename):
    '''
    Load analysis data from previous run and return for examination
    (e.g. in iPython). This function will return a Pandas store HDF5 object.
    '''
    return pd.HDFStore(filename)
    
def plot_S1_ST_from_store(store, show=True):
    '''
    Extract and plot S1 and ST sensitivity data directly from a store object
    '''

    # To run from the HDF5 file
    # store = load_data('analysis.h5')
    # plot_S1_ST_from_store(store)

    plot_S1_ST(store['S_G_sens'], store['P_G_sens'], store['A_G_sens'], 
               store['F_G_sens'], store['R_G_sens'], store['S_C_sens'], 
               store['S_H_sens'], store['P_C_sens'], store['A_C_sens'], 
               store['F_C_sens'], store['R_C_sens'], show)

def plot_S1_or_ST_from_store(store, S_string, show=True, show_title=True):
    '''
    Extract and plot S1 and ST sensitivity data directly from a store object
    '''

    # To run from the HDF5 file
    # store = load_data('analysis.h5')
    # plot_S1_ST_from_store(store, 'S1')

    plot_S1_or_ST(store['S_G_sens'], store['P_G_sens'], store['A_G_sens'], 
               store['F_G_sens'], store['R_G_sens'], store['S_C_sens'], 
               store['S_H_sens'], store['P_C_sens'], store['A_C_sens'], 
               store['F_C_sens'], store['R_C_sens'], 
               S_string, show, show_title)

def print_max_conf(store):
    '''
    Print off the max confidence interval for each variable in the store,
    for both first-order and total-order indices
    '''

    for var in ['S_G_sens', 'P_G_sens', 'A_G_sens', 'F_G_sens', 'R_G_sens',
                'S_C_sens', 'S_H_sens', 'P_C_sens', 'A_C_sens', 'F_C_sens', 
                'R_C_sens',]:
        print('----------- '+var+' -----------')
        print('S1_conf_max: {}'.format(store[var]['S1_conf'].max()))
        print('ST_conf_max: {}'.format(store[var]['ST_conf'].max()))
        print(' ')

def plot_S1_ST(S_G_sens, P_G_sens, A_G_sens, F_G_sens, R_G_sens, S_C_sens, 
               S_H_sens, P_C_sens, A_C_sens, F_C_sens, R_C_sens, show=True):
    # Gather the S1 and ST results
    S1 = pd.concat([S_G_sens['S1'], P_G_sens['S1'], A_G_sens['S1'], 
                    F_G_sens['S1'], R_G_sens['S1'], S_C_sens['S1'], 
                    S_H_sens['S1'], P_C_sens['S1'], A_C_sens['S1'], 
                    F_C_sens['S1'], R_C_sens['S1']], 
                    keys=['$S_G$','$P_G$','$A_G$','$F_G$','$R_G$','$S_C$',
                          '$S_H$','$P_C$','$A_C$','$F_C$','$R_C$'], axis=1) #produces copy
    ST = pd.concat([S_G_sens['ST'], P_G_sens['ST'], A_G_sens['ST'], 
                    F_G_sens['ST'], R_G_sens['ST'], S_C_sens['ST'],
                    S_H_sens['ST'], P_C_sens['ST'], A_C_sens['ST'], 
                    F_C_sens['ST'], R_C_sens['ST']], 
                    keys=['$S_G$','$P_G$','$A_G$','$F_G$','$R_G$','$S_C$',
                          '$S_H$','$P_C$','$A_C$','$F_C$','$R_C$'], axis=1)
    # Ghange to greek
    for id in S1.index:
        if id in ['PG0', 'AG0', 'FG0', 'RG0', 'SH0', 'PC0', 'AC0', 
                  'FC0', 'RC0']:
            S1.rename(index={id: r'${}_{{ {}_0 }}$'.format(id[0],id[1])}, inplace=True)
        elif id in ['tilde_m','tilde_b','tilde_c','tilde_d','tilde_e']:
            S1.rename(index={id: r'$\~{}$'.format(id[-1])}, inplace=True)
        elif id in ['log_theta_SG', 'log_theta_SH', 'log_theta_SC']:
            S1.rename(index={id: r'$\log(\theta_{{ {} }})$'.format(id[-2:])}, inplace=True)
        elif id in ['log_beta_GA','log_beta_GP','log_beta_CA','log_beta_HA','log_beta_CP','log_beta_HP']:
            S1.rename(index={id: r'$\log(\beta_{{ {} }})$'.format(id[-2:])}, inplace=True)
        elif id in ['tilde_m_H', 'tilde_b_H','tilde_m_C', 'tilde_b_C']:
            S1.rename(index={id: r'$\~{}_{}$'.format(id[-3],id[-1])}, inplace=True)
        elif id[:3] == 'log':
            S1.rename(index={id: r'$\log(\{})$'.format(id[4:])}, inplace=True)
        elif id == 'k':
            S1.rename(index={id: r'${}$'.format(id)}, inplace=True)
        else:
            S1.rename(index={id: r'$\{}$'.format(id)}, inplace=True)
    for id in ST.index:
        if id in ['PG0', 'AG0', 'FG0', 'RG0', 'SH0', 'PC0', 'AC0', 
                  'FC0', 'RC0']:
            ST.rename(index={id: r'${}_{{ {}_0 }}$'.format(id[0],id[1])}, inplace=True)
        elif id in ['tilde_m','tilde_b','tilde_c','tilde_d','tilde_e']:
            ST.rename(index={id: r'$\~{}$'.format(id[-1])}, inplace=True)
        elif id in ['log_theta_SG', 'log_theta_SH', 'log_theta_SC']:
            ST.rename(index={id: r'$\log(\theta_{{ {} }})$'.format(id[-2:])}, inplace=True)
        elif id in ['log_beta_GA','log_beta_GP','log_beta_CA','log_beta_HA','log_beta_CP','log_beta_HP']:
            ST.rename(index={id: r'$\log(\beta_{{ {} }})$'.format(id[-2:])}, inplace=True)
        elif id in ['tilde_m_H', 'tilde_b_H','tilde_m_C', 'tilde_b_C']:
            ST.rename(index={id: r'$\~{}_{}$'.format(id[-3],id[-1])}, inplace=True)
        elif id[:3] == 'log':
            ST.rename(index={id: r'$\log(\{})$'.format(id[4:])}, inplace=True)
        elif id == 'k':
            ST.rename(index={id: r'${}$'.format(id)}, inplace=True)
        else:
            ST.rename(index={id: r'$\{}$'.format(id)}, inplace=True)

    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))

    # Make more colors
    colors=[ '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5']

    S1.plot.bar(stacked=True, ax=axes[0], rot=0, width=0.8, color=colors)
    ST.plot.bar(stacked=True, ax=axes[1], rot=0, width=0.8, color=colors)

    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.tick_params(axis='x', labelrotation=67)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axes[0].set_title('First-order indices', fontsize=20)
    axes[1].set_title('Total-order indices', fontsize=20)

    axes[0].set_ylabel('Sobol Index')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig("param_sens_{}.pdf".format(time.strftime("%m_%d_%H%M")))
    return (fig, axes)

def plot_S1_or_ST(S_G_sens, P_G_sens, A_G_sens, F_G_sens, R_G_sens, S_C_sens, 
               S_H_sens, P_C_sens, A_C_sens, F_C_sens, R_C_sens, S_string, 
               show=True, show_title=True):
    '''Plot S1 or ST given by argument S which is a string'''

    # make sure S_string is 'S1' or 'ST'
    try:
        assert S_string == 'S1' or S_string == 'ST', 'not a proper S_string argument'
    except AssertionError as e:
        e.args += ('S_string = {}'.format(S_string),)
        raise

    # Gather the S1 or ST results
    S = pd.concat([S_G_sens[S_string], P_G_sens[S_string], A_G_sens[S_string], 
                    F_G_sens[S_string], R_G_sens[S_string], S_C_sens[S_string], 
                    S_H_sens[S_string], P_C_sens[S_string], A_C_sens[S_string], 
                    F_C_sens[S_string], R_C_sens[S_string]], 
                    keys=['$S_G$','$P_G$','$A_G$','$F_G$','$R_G$','$S_G$',
                        '$S_H$','$P_G$','$A_G$','$F_G$','$R_G$'], axis=1) #produces copy
    
    # Change to greek
    for id in S.index:
        if id in ['PG0', 'AG0', 'FG0', 'RG0']:
            S.rename(index={id: r'${}_{{ {}_0 }}$'.format(id[0],'G')}, inplace=True)
        elif id in ['PC0', 'AC0', 'FC0', 'RC0']:
            S.rename(index={id: r'${}_{{ {}_0 }}$'.format(id[0],'G')}, inplace=True)
        elif id in ['SH0']:
            S.rename(index={id: r'${}_{{ {}_0 }}$'.format(id[0],id[1])}, inplace=True)
        elif id in ['tilde_m','tilde_b','tilde_c','tilde_d','tilde_e']:
            S.rename(index={id: r'$\~{}_{}$'.format(id[-1],'G')}, inplace=True)
        elif id in ['log_theta_SG']:
            S.rename(index={id: r'$\log(\theta_{{ S{} }})$'.format('G')}, inplace=True)
        elif id in ['log_theta_SH']:
            S.rename(index={id: r'$\log(\theta_{{ S{} }})$'.format('H')}, inplace=True)
        elif id in ['log_theta_SC']:
            S.rename(index={id: r'$\log(\theta_{{ S{} }})$'.format('G')}, inplace=True)
        elif id in ['log_beta_GA','log_beta_GP']:
            S.rename(index={id: r'$\log(\beta_{{ G{} }})$'.format(id[-1])}, inplace=True)
        elif id in ['log_beta_CA','log_beta_CP']:
            S.rename(index={id: r'$\log(\beta_{{ G{} }})$'.format(id[-1])}, inplace=True)
        elif id in ['log_beta_HA','log_beta_HP']:
            S.rename(index={id: r'$\log(\beta_{{ H{} }})$'.format(id[-1])}, inplace=True)
        elif id in ['tilde_m_H', 'tilde_b_H']:
            S.rename(index={id: r'$\~{}_{}$'.format(id[-3],id[-1])}, inplace=True)
        elif id in ['tilde_m_C', 'tilde_b_C']:
            S.rename(index={id: r'$\~{}_G$'.format(id[-3])}, inplace=True)
        elif id[:3] == 'log':
            S.rename(index={id: r'$\log(\{})$'.format(id[4:])}, inplace=True)
        elif id == 'k':
            S.rename(index={id: r'${}$'.format(id)}, inplace=True)
        else:
            if id[-1] == 'G':
                S.rename(index={id: r'$\{}G$'.format(id[:-1])}, inplace=True)
            elif id[-1] == 'C':
                S.rename(index={id: r'$\{}G$'.format(id[:-1])}, inplace=True)
            else:
                S.rename(index={id: r'$\{}$'.format(id)}, inplace=True)


    # Plot
    fig, axes = plt.subplots(figsize=(10, 6))

    # Make more colors to cycle through
    colors=[ '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5']

    S.plot.bar(stacked=True, ax=axes, rot=0, width=0.8, color=colors)
    


    axes.tick_params(labelsize=10)
    axes.tick_params(axis='x', labelrotation=67)
    axes.set_ylim(0.0)
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) #If you need mupltiple columns then ncols = 3
    axes.set_ylabel('Sobol Index')
    if show_title:
        if S_string == 'S1':
            axes.set_title('First-order indices', fontsize=20)
        else:
            axes.set_title('Total-order indices', fontsize=20)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        fig.savefig("param_sens_{}.pdf".format(time.strftime("%m_%d_%H%M")))
    return (fig, axes)

def param_ranges_to_latex(params_file, log_form=True):
    '''
    Print the param ranges in latex format.
        
    Arguments:
        - params_file: str of name where the community parameter estimates 
            from the least squares
        - log_form: if true, write with log (e.g., log(beta) instead of beta)
    '''

    param_range_dict = generate_param_ranges(params_file).copy()

    # round to three signficant figures adn convert to str
    for key in param_range_dict.keys():
        if key[:3] =='log' and not log_form: # convert log parameters back
            lower_bound = '%s' % float('%.3g' % 10**param_range_dict[key][0])
            upper_bound = '%s' % float('%.3g' % 10**param_range_dict[key][1])
        else:
            lower_bound = '%s' % float('%.3g' % param_range_dict[key][0])
            upper_bound = '%s' % float('%.3g' % param_range_dict[key][1])

        # if 'e' in bound, change to '10 *'
        if 'e' in lower_bound:
            e_index = lower_bound.index('e')
            lower_bound = '{}*10^{{{}}}'.format(lower_bound[:e_index],
                                                lower_bound[e_index+1:])
        if 'e' in upper_bound:
            e_index = upper_bound.index('e')
            upper_bound = '{}*10^{{{}}}'.format(upper_bound[:e_index],
                                                upper_bound[e_index+1:])

        # write whole thing as formarted string with bounds
        param_range_dict[key] = '$[{},{}]$'.format(lower_bound,upper_bound)

    # convert to pandas 
    df = pd.DataFrame.from_dict(param_range_dict,orient='index')
    df.index.names = ['Parameter']
    df = df.rename(columns={0:'Range'})

    # make the parameter name latex appropriate
    for id in df.index:
        if id in ['PG0', 'AG0', 'FG0', 'RG0', 'SH0', 'PC0', 'AC0', 
                  'FC0', 'RC0']:
            df = df.rename(index={id:'$'+id[0]+'_{'+id[1]+'_0}$'})
        elif id in ['tilde_m','tilde_b','tilde_c','tilde_d','tilde_e']:
            df = df.rename(index={id:'$\\tilde{'+id[-1]+'}$'})
        elif id in ['log_theta_SG', 'log_theta_SH', 'log_theta_SC']:
            if log_form:
                df = df.rename(index={id:'$\log(\\theta_{'+id[-2:]+'})$'})
            else:
                df = df.rename(index={id:'$\\theta_{'+id[-2:]+'}$'})
        elif id in ['log_beta_GA','log_beta_GP','log_beta_CA','log_beta_HA','log_beta_CP','log_beta_HP']:
            if log_form:
                df = df.rename(index={id:'$\log(\\beta_{'+id[-2:]+'})$'})
            else:
                df = df.rename(index={id:'$\\beta_{'+id[-2:]+'}$'})
        elif id in ['tilde_m_H', 'tilde_b_H','tilde_m_C', 'tilde_b_C']:
            df = df.rename(index={id:'$\\tilde{'+id[-3]+'}_'+id[-1]+'$'})
        elif id[:3] == 'log':
            if log_form:
                df = df.rename(index={id:'$\log(\\'+id[4:]+')$'})
            else:
                df = df.rename(index={id:'$\\'+id[4:]+'$'})
        elif id == 'k':
            df = df.rename(index={id:'$'+id+'$'})
        else:
            df = df.rename(index={id:'$\\'+id+'$'})

    # print as latex 
    final_print = df.to_latex(header = True,
                        label = 'table:sobol_param_ranges',
                        caption = 'enter caption here',
                        escape=False)
    final_print = final_print.replace('\\\\\n', '\\\\ \\hline\n')
    print(final_print)

def sen_index_table_to_latex(store, pop_class):
    '''
    Print the param ranges in latex format.
        
    Arguments:
        - store: store object containing results of sobol run
        - pop_class: Strign chose from SG, AG, ..., SC, SH, ...
    '''

    sens_name = '/{}_{}_sens'.format(pop_class[0],pop_class[1])

    # make sure the chosen pop_class is in the store keys
    try:
        assert sens_name in store.keys(), 'not a proper pop_class argument'
    except AssertionError as e:
        e.args += ('pop_class = {}'.format(pop_class),)
        raise

    # get the dataframe 
    df = store[sens_name]

    # round to 5 significant figures and convert to string
    for col in df.columns:
        for index in df.index:
            df[col].loc[index] = '%s' % float('%.5g' % df[col].loc[index]) 

    # make the parameter name latex appropriate
    for id in df.index:
        if id in ['PG0', 'AG0', 'FG0', 'RG0', 'SH0', 'PC0', 'AC0', 
                  'FC0', 'RC0']:
            df = df.rename(index={id:'$'+id[0]+'_{'+id[1]+'_0}$'})
        elif id in ['tilde_m','tilde_b','tilde_c','tilde_d','tilde_e']:
            df = df.rename(index={id:'$\\tilde{'+id[-1]+'}$'})
        elif id in ['log_theta_SG', 'log_theta_SH', 'log_theta_SC']:
            # df = df.rename(index={id:'$\log(\theta_{'+id[-2:]+'})$'})
            df = df.rename(index={id:'$\\theta_{'+id[-2:]+'}$'})
        elif id in ['log_beta_GA','log_beta_GP','log_beta_CA','log_beta_HA','log_beta_CP','log_beta_HP']:
            # df = df.rename(index={id:'$\log(\\beta_{'+id[-2:]+'})$'})
            df = df.rename(index={id:'$\\beta_{'+id[-2:]+'}$'})
        elif id in ['tilde_m_H', 'tilde_b_H','tilde_m_C', 'tilde_b_C']:
            df = df.rename(index={id:'$\\tilde{'+id[-3]+'}_'+id[-1]+'$'})
        elif id[:3] == 'log':
            # df = df.rename(index={id:'$\log(\\'+id[4:]+')$'})
            df = df.rename(index={id:'$\\'+id[4:]+'$'})
        elif id == 'k':
            df = df.rename(index={id:'$'+id+'$'})
        else:
            df = df.rename(index={id:'$\\'+id+'$'})

    # print as latex 
    final_print = df.to_latex(header = True,
                        label = 'table:sobol_index_table_{}'.format(pop_class),
                        caption = 'enter caption here',
                        escape=False)
    final_print = final_print.replace('\\\\\n', '\\\\ \\hline\n')
    print(final_print)
    
def print_three_most_sens(store, pop_class, order='ST'):
    '''
    Print the three parameters that the pop_class selected is most sensitive to.

    Arguments:
        - store: store object containing results of sobol run
        - pop_class: Strign chose from SG, AG, ..., SC, SH, ...
        - order: string of S1 or ST for first or total order
    '''

    sens_name = '/{}_{}_sens'.format(pop_class[0],pop_class[1])

    # make sure the chosen pop_class is in the store keys
    try:
        assert sens_name in store.keys(), 'not a proper pop_class argument'
    except AssertionError as e:
        e.args += ('pop_class = {}'.format(pop_class),)
        raise

    # get the dataframe 
    df = store[sens_name]

    print('Largest indicies for {} order of {}:{}'.format(order, pop_class, df[order].nlargest(3).index.tolist()))

if __name__ == "__main__":
    __spec__ = None
    args = parser.parse_args()

    if args.ncores is None:
        with Pool() as pool:
            main(args.N, 
            filename=args.filename, 
            params_file=args.params_file, 
            pool=pool)
    else:
        with Pool(args.ncores) as pool:
            main(args.N, 
                filename=args.filename, 
                params_file=args.params_file, 
                pool=pool)

# to run in terminal use: python sobol_sensitivity_analysis_log_scale.py -o sobol_debug -p least_squares_sol_AG2016_False_x0_1000_assumemuAmuF_False -N 2**10

