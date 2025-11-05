
'''
Used to estimate the parameters of the community model 
'''

import pandas as pd
import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy import optimize
import matplotlib.pyplot as plt
from collections import OrderedDict
from multiprocessing import Pool
import pickle

import opioid_model_ODE_solver_log_scale_alphaC as ODE

# seed the random number generator
np.random.seed(5)

# default start year
start_year = 2016

# default inital and final times in terms of years from start_year
t0 = 0
tf = 4
time_range = [t0, tf]

# assumed parameter values 
assumed_params = OrderedDict()
assumed_params['omega'] = 10e-10
assumed_params['mu']    = 0.0143

assumed_params['XG_0'] = 0.0
assumed_params['YG_0'] = 0.0
assumed_params['ZG_0'] = 0.0
assumed_params['JG_0'] = 0.0
assumed_params['KG_0'] = 0.0

# parameter range tuples for estimation
param_ranges = OrderedDict()
param_ranges['tilde_m']        = (-0.1, -0.001) 
param_ranges['tilde_b']        = (0.1, 0.5)
param_ranges['log_beta_GA']    = (math.log10(1.0e-3), math.log10(0.1)) 
param_ranges['log_beta_GP']    = (math.log10(1.0e-4), math.log10(0.01))
param_ranges['log_theta_SG']   = (math.log10(1.0e-3), math.log10(0.5))  
param_ranges['log_theta_P']    = (math.log10(1.0e-3), math.log10(1.0)) 
param_ranges['theta_A']        = (30, 50.0)     
param_ranges['epsilon_G']      = (0.333, 52.0)
param_ranges['log_gamma']      = (math.log10(1.0e-5), math.log10(0.1)) 
param_ranges['zeta']           = (0.0001, 0.9)
param_ranges['log_nu']         = (math.log10(1.0e-3), math.log10(0.9)) 
param_ranges['sigma']          = (0.1, 2.0) 
param_ranges['lambda_A']       = (1.0e-10, 1.0) # have to use 1.0e-10 instead of 0
param_ranges['lambda_F']       = (1.0e-10, 1.0) # have to use 1.0e-10 instead of 0

param_ranges['PG0'] = (0.001, 0.4)
param_ranges['AG0'] = (1.0e-4, 0.01)
param_ranges['FG0'] = (1.0e-5, 0.01)
param_ranges['RG0'] = (1.0e-5, 0.1)

# Values to use for mu_A and mu_F if assume_muA_muF
assumed_mu_A  = 0.0222
assumed_mu_F  = 0.327

# Ranges to use for mu_A and mu_F if not assume_muA_muF
param_range_mu_A = (0.001,1.0) 
param_range_mu_F  = (0.001,1.0) 


###############################################################################
# Get the Data vectors
###############################################################################
def get_Data_vectors(include_AG2016):
    '''
    Returns a list of the Data vectors.
    Arguments:
        - include_AG2016: if True, include the data point for AG 2016, else 
            don't include it in calculations
    '''
    # Read in the data 
    pop_est_df  = pd.read_csv('cleaned_gen_pop_data/population_estimates.csv', 
                            index_col=0)
    overdose_df = pd.read_csv('cleaned_gen_pop_data/overdose_estimates.csv', 
                            index_col=0)
    quart_pres_df = pd.read_csv('cleaned_gen_pop_data/quarterly_prescription_estimates.csv', 
                            index_col=0)

    # convert the columns to ints 
    pop_est_df.columns = pd.to_numeric(pop_est_df.columns)
    overdose_df.columns = pd.to_numeric(overdose_df.columns)
    quart_pres_df.columns = pd.to_numeric(quart_pres_df.columns)

    # Time ranges (in years) for the population data and the overdose data for 
    #   which we have data
    pop_time_range = np.arange(t0,tf)+start_year
    overdose_time_range = np.arange(t0,tf)+start_year

    # Define Data1-5 as described in paper 
    Data1 = pop_est_df.loc['Prescription Opioid Users (excludes RX addicts)'][pop_time_range].values / \
            pop_est_df.loc['Total Population 12+'][pop_time_range].values
    
    Data2 = []
    if include_AG2016:
        Data2 = pop_est_df.loc['Pain Reliever Use Disorder'][pop_time_range].values / \
            pop_est_df.loc['Total Population 12+'][pop_time_range].values
    else:
        Data2 = pop_est_df.loc['Pain Reliever Use Disorder'][pop_time_range[1:]].values / \
            pop_est_df.loc['Total Population 12+'][pop_time_range[1:]].values
    
    Data3 = pop_est_df.loc['Heroin Substance Use Disorder'][pop_time_range].values / \
            pop_est_df.loc['Total Population 12+'][pop_time_range].values
    
    Data4 = overdose_df.loc['Prescription Opioid Use Disorder Fatal Overdoses'][overdose_time_range].values / \
            pop_est_df.loc['Total Population 12+'][overdose_time_range].values
    
    Data5 = overdose_df.loc['Fentanyl/Heroin Use Disorder Fatal Overdoses'][overdose_time_range].values / \
            pop_est_df.loc['Total Population 12+'][overdose_time_range].values

    # Define Data6 as described in paper 
    Data6 = []
    for year_quarter in quart_pres_df.columns:
        year = np.floor(year_quarter)
        quarter = year_quarter - year

        if int(year) in pop_time_range:
            Data6.append(quart_pres_df.loc['knox_quart_pres_without_OUD'][year_quarter] / \
                        pop_est_df.loc['Total Population 12+'][year])
    Data6 = np.array(Data6)

    all_data = [Data1, Data2, Data3, Data4, Data5, Data6]

    # data_points = 0
    # for item in all_data:
    #     data_points += len(item)
    # print('There are {} total data points.'.format(data_points))
    
    return all_data

###############################################################################
# Get the Estim Vectors
###############################################################################
def odes_to_solve(t, X, params, add_ODEs=True):
    '''Define our community model as a system of ODEs.
    t -- time 
    X -- values of SG, PG, AG, FG, RG, XG, YG, ZG, JG, and KG
    params -- dict of parameters
    add_ODEs -- solve with the additiaonl ODEs or not'''

    if add_ODEs:
        SG, PG, AG, FG, RG, XG, YG, ZG, JG, KG = X
    else:
        SG, PG, AG, FG, RG = X

    # Let's first assume that alpha_G is just linear 
    # alpha_G = params['alpha_G'] 
    alpha_G = params['tilde_m']*t + params['tilde_b']

    # throw an error and kill the program (with raise) if alpha_G goes negative
    try:
        assert np.any(alpha_G >= 10e-10), 'alpha_G is negative'
    except AssertionError as e:
        e.args += ('alpha_G = {}'.format(alpha_G),
                   'tilde_m = {}'.format(params['tilde_m']),
                   'tilde_b = {}'.format(params['tilde_b']),
                   't = {}'.format(t))
        raise

    # Let's also first assume that mu_A is constant 
    mu_A = params['mu_A']

    # General Population Model
    dSGdt = - alpha_G*SG - 10**(params['log_beta_GA'])*SG*AG \
        - 10**(params['log_beta_GP'])*SG*PG - 10**(params['log_theta_SG'])*SG*FG \
        + params['epsilon_G']*PG + params['mu']*(PG+AG+FG+RG) + mu_A*AG \
        + params['mu_F']*FG
    dPGdt = - params['epsilon_G']*PG - params['mu']*PG - 10**(params['log_gamma'])*PG \
        - 10**(params['log_theta_P'])*PG*FG  + alpha_G*SG
    dAGdt = - params['zeta']*AG \
        - params['theta_A']*AG*FG \
        - (params['mu']+mu_A)*AG \
        + 10**(params['log_beta_GA'])*SG*AG \
        + 10**(params['log_beta_GP'])*SG*PG \
        + 10**(params['log_gamma'])*PG \
        + params['sigma']*RG*((params['lambda_A']*AG \
        + (1-params['lambda_F'])*FG)/(AG+FG+params['omega']))  
    dFGdt = - 10**(params['log_nu'])*FG - (params['mu']+params['mu_F'])*FG \
        + 10**(params['log_theta_SG'])*SG*FG + 10**(params['log_theta_P'])*PG*FG \
        + params['theta_A']*AG*FG \
        + params['sigma']*RG*(((1-params['lambda_A'])*AG \
        + params['lambda_F']*FG)/(AG+FG+params['omega'])) 
    dRGdt = - params['sigma']*RG*((params['lambda_A']*AG \
            + (1-params['lambda_F'])*FG)/(AG+FG+params['omega'])) \
            - params['sigma']*RG*(((1-params['lambda_A'])*AG \
            + params['lambda_F']*FG)/(AG+FG+params['omega'])) \
            - params['mu']*RG + params['zeta']*AG + 10**(params['log_nu'])*FG
    
    # Additional ODEs needed for objective function 
    if add_ODEs:
        dXGdt = alpha_G*SG
        dYGdt =  10**(params['log_beta_GA'])*SG*AG \
            + 10**(params['log_beta_GP'])*SG*PG \
            + 10**(params['log_gamma'])*PG \
            + params['sigma']*RG*((params['lambda_A']*AG \
                + (1-params['lambda_F'])*FG)/(AG+FG+params['omega']))
        dZGdt = 10**(params['log_theta_SG'])*SG*FG \
            + 10**(params['log_theta_P'])*PG*FG \
            + params['theta_A']*AG*FG \
            + params['sigma']*RG*(((1-params['lambda_A'])*AG \
                + params['lambda_F']*FG)/(AG+FG+params['omega']))
        dJGdt = mu_A*AG 
        dKGdt = params['mu_F']*FG

        Y = [dSGdt, dPGdt, dAGdt, dFGdt, dRGdt, dXGdt, dYGdt, dZGdt, dJGdt, dKGdt]
    else:
        Y = [dSGdt, dPGdt, dAGdt, dFGdt, dRGdt]

    return Y

def get_Estim_vectors(all_params, include_AG2016):
    '''
    Return a list of the Estim vectors. 

    Arguments:
        - all_params: OrderedDict of all the params (assumed and estimated) and 
                        initial values
        - include_AG2016: if True, include the data point for AG 2016, else 
            don't include it in calculations
    '''

    # Determine initial value for SG
    all_params['SG0'] = 1 - all_params['PG0'] - all_params['AG0'] \
        - all_params['FG0'] - all_params['RG0']

    # obtain the initial values 
    init_val_keys = ['SG0', 'PG0', 'AG0', 'FG0', 'RG0', 'XG_0', 'YG_0', 
                     'ZG_0', 'JG_0', 'KG_0']
    initial_vals = [all_params[key] for key in init_val_keys]
    
    # Determine the solutions to our ODEs.
    # Note: that t_eval is the times at which the solution will be stores.  We 
    #   only need the solutions at the integers between (and including) t0 
    #   and tf.
    sol = solve_ivp(odes_to_solve, 
                    time_range, 
                    initial_vals, 
                    t_eval=np.arange(t0,tf+0.25,0.25), 
                    args=[all_params.copy()])
    SG_sol, PG_sol, AG_sol, FG_sol, RG_sol, \
        XG_sol, YG_sol, ZG_sol, JG_sol, KG_sol= sol.y
    
    # throw an error and kill the program (with raise) ode solver didn't get whole solution
    try:
        assert np.any(XG_sol.size == 4*tf+1), 'ODE solver did not work'
    except AssertionError as e:
        e.args += (tuple(all_params.values()))
        raise

    # Determine Estim1, Estim2, ..., Estim6
    Estim1 = [all_params['PG0']+XG_sol[1*4], PG_sol[1*4]+XG_sol[2*4]-XG_sol[1*4],
              PG_sol[2*4]+XG_sol[3*4]-XG_sol[2*4], PG_sol[3*4]+XG_sol[4*4]-XG_sol[3*4]]
    
    Estim2 = []
    if include_AG2016:
        Estim2 = [all_params['AG0']+YG_sol[1*4], AG_sol[1*4]+YG_sol[2*4]-YG_sol[1*4],
              AG_sol[2*4]+YG_sol[3*4]-YG_sol[2*4], AG_sol[3*4]+YG_sol[4*4]-YG_sol[3*4]]
    else:
        Estim2 = [AG_sol[1*4]+YG_sol[2*4]-YG_sol[1*4],
              AG_sol[2*4]+YG_sol[3*4]-YG_sol[2*4], AG_sol[3*4]+YG_sol[4*4]-YG_sol[3*4]]
    
    Estim3 = [all_params['FG0']+ZG_sol[1*4], FG_sol[1*4]+ZG_sol[2*4]-ZG_sol[1*4],
              FG_sol[2*4]+ZG_sol[3*4]-ZG_sol[2*4], FG_sol[3*4]+ZG_sol[4*4]-ZG_sol[3*4]]
    
    Estim4 = [JG_sol[1*4]-JG_sol[0*4], JG_sol[2*4]-JG_sol[1*4], JG_sol[3*4]-JG_sol[2*4], 
              JG_sol[4*4]-JG_sol[3*4]]

    Estim5 = [KG_sol[1*4]-KG_sol[0*4], KG_sol[2*4]-KG_sol[1*4], KG_sol[3*4]-KG_sol[2*4], 
              KG_sol[4*4]-KG_sol[3*4]]

    Estim6 = [all_params['PG0'] +XG_sol[int(0.25*4)], 
              PG_sol[int(0.25*4)]+XG_sol[int(0.50*4)]-XG_sol[int(0.25*4)],
              PG_sol[int(0.50*4)]+XG_sol[int(0.75*4)]-XG_sol[int(0.50*4)],
              PG_sol[int(0.75*4)]+XG_sol[int(1*4)]   -XG_sol[int(0.75*4)],
              PG_sol[int(1*4)]   +XG_sol[int(1.25*4)]-XG_sol[int(1*4)],
              PG_sol[int(1.25*4)]+XG_sol[int(1.50*4)]-XG_sol[int(1.25*4)],
              PG_sol[int(1.50*4)]+XG_sol[int(1.75*4)]-XG_sol[int(1.50*4)],
              PG_sol[int(1.75*4)]+XG_sol[int(2*4)]   -XG_sol[int(1.75*4)],
              PG_sol[int(2*4)]   +XG_sol[int(2.25*4)]-XG_sol[int(2*4)],
              PG_sol[int(2.25*4)]+XG_sol[int(2.50*4)]-XG_sol[int(2.25*4)],
              PG_sol[int(2.50*4)]+XG_sol[int(2.75*4)]-XG_sol[int(2.50*4)],
              PG_sol[int(2.75*4)]+XG_sol[int(3*4)]   -XG_sol[int(2.75*4)],
              PG_sol[int(3*4)]   +XG_sol[int(3.25*4)]-XG_sol[int(3*4)],
              PG_sol[int(3.25*4)]+XG_sol[int(3.50*4)]-XG_sol[int(3.25*4)],
              PG_sol[int(3.50*4)]+XG_sol[int(3.75*4)]-XG_sol[int(3.50*4)],
              PG_sol[int(3.75*4)]+XG_sol[int(4*4)]   -XG_sol[int(3.75*4)]]
    
    return [Estim1, Estim2, Estim3, Estim4, Estim5, Estim6]

###############################################################################
# Run Least Squares 
###############################################################################
def objective_function(params_to_estimate, assumed_params, include_AG2016):
    '''
    Return the objective function. 

    Arguments:
        - params_to_estimate: the parameters and initial conditions we are 
            trying to estimate (tuple)
        - assumed_params: the paramters and intial conditions we are already 
            assuming (OrderedDict)
        - include_AG2016: if True, include the data point for AG 2016, else 
            don't include it in calculations
    '''

    # create a OrderedDict of all the params and initial values (from the 
    #   ranges and the assumed)
    all_params = assumed_params.copy() # the assumed
    all_params.update(OrderedDict(zip(param_ranges.keys(),
                                      list(params_to_estimate)))) # the ranges

    # determine the Data vectors  
    Data = get_Data_vectors(include_AG2016)

    # determine the Estim vectors
    Estim = get_Estim_vectors(all_params.copy(),include_AG2016)

    # determine the Diff vectors
    Diff = []
    for i in range(len(Data)):
        Diff.append(Data[i] - Estim[i])

    # determine the objective function result
    obj_func_result = 0
    for i in range(len(Diff)):
        obj_func_result = obj_func_result + \
            (((Diff[i]**2).sum()) / ((Data[i]**2).sum()))**(1/2)
        
    return obj_func_result


def minimize_obj_func(numb_x0, include_AG2016, assume_muA_muF, pool=None):
    '''
    Determine the global minimum of the objective function by running 
    scipy.optimize.minimize(method='SLSQP') (which is a local multivariate 
    optimizer) with numb_x0 number initial random guesses within our parameter 
    ranges.  Whichever local minimum shows up the most will be our global 
    minimum. 
    
    Arguments:
        - numb_x0: number of initial guess for the local min (int)
        - include_AG2016: if True, include the data point for AG 2016, else 
            don't include it in calculations
        - assume_muA_muF: if True, assume mu_A and mu_A otherwise estimate them
        - pool: Pool object used for multiprocessing; if None then do not run 
            in parallel 
            
    Returns:
        - global_min: returns the determined global min as a OptimizeResult 
            object
        - local_min_list: list of all of the local mins (OptimizeResult 
            objects)
        - obj_fun_vals_list: list of all of the objective function values 
            that the local mins determined'''
        
    # store the resulting OptimizeResult objects
    local_min_list = []

    # store the resulting function values
    obj_fun_vals_list = []

    if assume_muA_muF:
        assumed_params['mu_A']  = assumed_mu_A
        assumed_params['mu_F']  = assumed_mu_F
    else:
        param_ranges['mu_A']  = param_range_mu_A
        param_ranges['mu_F']  = param_range_mu_F

    # value ranges to choose from for initial random guess for local min 
    #   calculations 
    range_array = np.array(list(param_ranges.values()))

    # first, gather all initial guesses
    init_guesses = np.array([np.random.uniform(range_array[:,0],range_array[:,1])
                                for n in range(numb_x0)])

    # Make sure that alpha_G = tilde_m*tf+tilde_b is not negative in 
    #   init_guess. If it is, regenerate tilde_m and tilde_b until it is 
    #   not.
    tilde_m_index = list(param_ranges.keys()).index('tilde_m')
    tilde_b_index = list(param_ranges.keys()).index('tilde_b')
    for i in range(numb_x0):
        while init_guesses[i,tilde_m_index]*tf+init_guesses[i,tilde_b_index] < 0:
            init_guesses[i,tilde_m_index] = np.random.uniform(
                range_array[tilde_m_index,0],range_array[tilde_m_index,1])
            init_guesses[i,tilde_b_index] = np.random.uniform(
                range_array[tilde_b_index,0],range_array[tilde_b_index,1])
    
    if pool is None:
        # run in serial

        # make a constraint such that alpha_G won't be negative at time tf 
        #   (note this is for linear alpha_G)
        cons = {'type':'ineq', 
                'fun': lambda x: x[tilde_m_index]*tf+x[tilde_b_index]-10e-4}
        
        # generate a list of bounds for the parameter ranges 
        bounds_list = list(param_ranges.values())
        
        for i in range(numb_x0):

            print('Running {} of {}'.format(i,numb_x0))

            # if an error occurs (alpha_G being negative), skip over this 
            #   itereation and print a message, but continue running 
            try:
                local_result = optimize.minimize(objective_function, 
                                                x0=init_guesses[i], 
                                                args=(assumed_params.copy(),
                                                    include_AG2016), 
                                                bounds=bounds_list, 
                                                constraints=cons, 
                                                method='SLSQP')
            except AssertionError as e:
                print('issue with iteration i = {}, error: {}'.format(i, e.args))
                continue
            
            local_min_list.append(local_result)
    else:
        # run in parallel

        args = [(init_guesses[n],include_AG2016,assume_muA_muF) for n in range(numb_x0)]

        # returns a list of results from the workers, applying the iterable of 
        #   arguments (tuples) to the function.
        local_min_list = pool.starmap(opt_func, args)

        # remove any iterations that failed (due to a constraint issue)
        local_min_list = [item for item in local_min_list if not isinstance(item, str)]

    # collect the list of function values 
    obj_fun_vals_list = [obj.fun for obj in local_min_list]

    # determine the global min by taking the mins of the resulting mins 
    global_min = local_min_list[obj_fun_vals_list.index(min(obj_fun_vals_list))]

    # # write the final parameter dict to a file
    # if save_params_to_file == True:
    #     final_params = OrderedDict(zip(param_ranges.keys(), list(global_min.x)))
    #     with open('final_params.txt','w') as data:  
    #         data.write(str(final_params))

    print('{} out of {} runs succeeded.'.format(len(local_min_list), numb_x0))

    return global_min, local_min_list, obj_fun_vals_list


def opt_func(guess, include_AG2016, assume_muA_muF):
    '''Pickleable function for parallelization.'''

    if assume_muA_muF:
        assumed_params['mu_A']  = assumed_mu_A
        assumed_params['mu_F']  = assumed_mu_F
    else:
        param_ranges['mu_A']  = param_range_mu_A
        param_ranges['mu_F']  = param_range_mu_F
    
    bounds_list = list(param_ranges.values())

    tilde_m_index = list(param_ranges.keys()).index('tilde_m')
    tilde_b_index = list(param_ranges.keys()).index('tilde_b')
    # make a constraint such that alpha_G won't be negative at time tf 
    #   (note this is for linear alpha_G)
    cons = {'type':'ineq', 
            'fun': lambda x: x[tilde_m_index]*tf+x[tilde_b_index]-10e-4}

    # if an error occurs (alpha_G being negative), skip over this 
    #   itereation and print a message, but continue running 
    try:
        local_result = optimize.minimize(objective_function, 
                            x0=guess,
                            args=(assumed_params.copy(),include_AG2016),
                            bounds=bounds_list, 
                            constraints=cons,
                            method='SLSQP')
    except AssertionError as e:
        print('issue with minimize. error: {}'.format(e.args))
        local_result = 'error occured' # all the 0's will be removed later
    
    return local_result
    

###############################################################################
# Galculate AIG between models 
###############################################################################

def calculate_corrected_AIG(global_min, final_params, include_AG2016):
    '''
    Determine the corrected AIG score. 
    '''
    # determine the Data vectors  
    Data = get_Data_vectors(include_AG2016)

    N = 0
    for item in Data: 
        N = N + len(item)

    K = len(final_params) + 1

    OF = global_min.fun

    AIG_c = N*math.log(OF/N) + 2*K + (2*K*(K-1))/(N-K-1)

    print('N = {}, K = {}, OF = {}, AIG_c = {}'.format(N, K, OF, AIG_c))

    return AIG_c

###############################################################################
# Plot the solutions
###############################################################################

def plot_solutions(final_params):
    '''
    Plot the community model solutions.
    
    Arguments:
    - final_params: the params resulting from the least squares estimation.
    '''

    all_params = final_params.copy()

    # collect the initial values in a OrderedDict in the correct order
    init_val_keys = ['SG0', 'PG0', 'AG0', 'FG0', 'RG0']
    init_vals = OrderedDict()
    for key in init_val_keys[1:]:
        init_vals[key] = all_params.pop(key)
    init_vals['SG0'] = 1 - init_vals['PG0'] - init_vals['AG0'] \
        - init_vals['FG0'] - init_vals['RG0']
    initial_vals_reordered = OrderedDict()
    for key in init_val_keys:
        initial_vals_reordered[key] = init_vals[key]
    init_vals = initial_vals_reordered

    for key in assumed_params.keys():
        all_params[key] = assumed_params[key]

    sol = ODE.solve_odes(initial_vals=init_vals, params=all_params)

    ODE.plot_genpop_or_comm(sol, plot_genpop=True)


def plot_data_with_results(min_result, include_AG2016, assume_muA_muF, show=True):
    '''
    Plot the data with the model results.
    
    Arguments:
    - min_result: the globla min results from minimize_obj_func()
    - include_AG2016: include the datapoint AG 2016
    - show: show it, else save teh figure as a png
    '''

    if assume_muA_muF:
        assumed_params['mu_A']  = assumed_mu_A
        assumed_params['mu_F']  = assumed_mu_F
    else:
        param_ranges['mu_A']  = param_range_mu_A
        param_ranges['mu_F']  = param_range_mu_F

    estimated_param_vals = min_result.x

    # create a OrderedDict of all the params and initial values (from the ranges and the assumed)
    all_params = assumed_params.copy() # from the assumed
    all_params.update(OrderedDict(zip(param_ranges.keys(),
                               list(estimated_param_vals)))) # from the ranges
    all_params['SG0'] = 1 - all_params['PG0'] - all_params['AG0'] \
            - all_params['FG0'] - all_params['RG0']

    # determine the Data vectors  
    Data_list = get_Data_vectors(include_AG2016)
    Estim_list = get_Estim_vectors(all_params.copy(),include_AG2016)

    # obtain the initial values 
    init_val_keys = ['SG0', 'PG0', 'AG0', 'FG0', 'RG0', 'XG_0', 'YG_0', 
                        'ZG_0', 'JG_0', 'KG_0']
    initial_vals = [all_params[key] for key in init_val_keys]

    time_step = 1/20
    sol = solve_ivp(odes_to_solve, time_range, 
        initial_vals, 
        t_eval = np.arange(time_range[0], time_range[1]+time_step, time_step), 
        args=[all_params.copy()])
    SG_sol, PG_sol, AG_sol, FG_sol, RG_sol, \
        XG_sol, YG_sol, ZG_sol, JG_sol, KG_sol = sol.y
    
    yearly_time_span = np.arange(t0,tf) + start_year
    quarterly_time_span = np.arange(t0,tf,0.25) + start_year

    # Plot yearly prescription data and solutions 
    plt.figure(figsize=(9, 6)) 
    plt.subplot(2, 2, 1)
    plt.plot(sol.t[:int(1/time_step)*(tf-1)+1] + start_year, 
        PG_sol[:int(1/time_step)*(tf-1)+1] + XG_sol[int(1/time_step):] \
            - XG_sol[:int(1/time_step)*(tf-1)+1], 
        color = 'r')
    plt.scatter(yearly_time_span, Data_list[0], color = 'r')
    plt.legend(['$P_G$ Model output', '$P_G$ Data'], shadow=True, framealpha=0.25)
    plt.xticks(np.arange(t0+start_year,tf+start_year,step=1))
    plt.xlabel('Year')
    plt.ylabel('Proportion of Population')

    plt.subplot(2, 2, 2)
    plt.plot(sol.t[:int(len(sol.t)//tf*(tf-1)+len(sol.t)//tf*0.75+1)] + start_year, 
            PG_sol[:int(len(sol.t)//tf*(tf-1)+len(sol.t)//tf*0.75+1)] + XG_sol[int(len(sol.t)//tf*0.25):] \
                - XG_sol[:int(len(sol.t)//tf*(tf-1)+len(sol.t)//tf*0.75+1)], color = 'r')
    plt.scatter(quarterly_time_span, Data_list[5], color = 'r')
    # plt.scatter(quarterly_time_span, Estim_list[5], color = 'red')
    plt.legend(['$P_G$ Model output', '$P_G$ Data'], shadow=True, framealpha=0.25)
    plt.xticks(np.arange(t0+start_year,tf+start_year,step=1))
    plt.xlabel('Year')
    plt.ylabel('Proportion of Population')

    plt.subplot(2, 2, 3)

    if include_AG2016:
        plt.plot(sol.t[:int(1/time_step)*(tf-1)+1] + start_year, 
                AG_sol[:int(1/time_step)*(tf-1)+1] + YG_sol[int(1/time_step):] \
                    - YG_sol[:int(1/time_step)*(tf-1)+1], color = 'g')
        plt.scatter(yearly_time_span, Data_list[1], color = 'g')
        plt.plot(sol.t[:int(1/time_step)*(tf-1)+1] + start_year, 
                FG_sol[:int(1/time_step)*(tf-1)+1] + ZG_sol[int(1/time_step):] \
                    - ZG_sol[:int(1/time_step)*(tf-1)+1], color = 'b')
        plt.scatter(yearly_time_span, Data_list[2], color = 'b')
        plt.legend(['$A_G$ Model output', '$A_G$ Data', 
                    '$F_G$ Model output', '$F_G$ Data'], shadow=True, framealpha=0.25)
        plt.ylabel('Proportion of Population')
        plt.xticks(np.arange(t0+start_year,tf+start_year,step=1))
        plt.xlabel('Year')
    else:
        plt.plot(sol.t[int(1/time_step)*1:int(1/time_step)*(tf-1)+1] + start_year, 
        AG_sol[int(1/time_step)*1:int(1/time_step)*(tf-1)+1] + YG_sol[int(1/time_step)*2:] \
            - YG_sol[int(1/time_step)*1:int(1/time_step)*(tf-1)+1], color = 'g')
        plt.scatter(yearly_time_span[1:], Data_list[1], color = 'g')
        plt.plot(sol.t[:int(1/time_step)*(tf-1)+1] + start_year, 
                FG_sol[:int(1/time_step)*(tf-1)+1] + ZG_sol[int(1/time_step):] \
                    - ZG_sol[:int(1/time_step)*(tf-1)+1], color = 'b')
        plt.scatter(yearly_time_span, Data_list[2], color = 'b')
        plt.legend(['$A_G$ Model output', '$A_G$ Data', 
                    '$F_G$ Model output', '$F_G$ Data'], shadow=True, framealpha=0.25)
        plt.ylabel('Proportion of Population')
        plt.xticks(np.arange(t0+start_year,tf+start_year,step=1))
        plt.xlabel('Year')

    plt.subplot(2, 2, 4)
    plt.plot(sol.t[:int(1/time_step)*(tf-1)+1] + start_year, 
            JG_sol[int(1/time_step):] - JG_sol[:int(1/time_step)*(tf-1)+1], color = 'g')
    plt.scatter(yearly_time_span, Data_list[3], color = 'g')
    plt.plot(sol.t[:int(1/time_step)*(tf-1)+1] + start_year, 
            KG_sol[int(1/time_step):] - KG_sol[:int(1/time_step)*(tf-1)+1], color = 'b')
    plt.scatter(yearly_time_span, Data_list[4], color = 'b')
    plt.legend(['$A_G$ Model output', '$A_G$ Data', 
                '$F_G$ Model output', '$F_G$ Data'], shadow=True, framealpha=0.25)
    plt.ylabel('Proportion Fatal ODs of Populations')
    plt.xticks(np.arange(t0+start_year,tf+start_year,step=1))
    plt.xlabel('Year')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('opioid_model_least_squares_with_data.png')


###############################################################################
# Format final results 
###############################################################################

def to_latex_table(final_params, log_form=False):
    '''Print out formatted table to be used pasted in Latex of final values.
        
            - final_params: OrderedDict of final param values
            - log_form: print with log or not'''

    # round to three signficant figures
    for key in final_params.keys():
        # convert log parameters back
        if key[:3] =='log' and log_form == False:
            final_params[key] = 10**final_params[key]
        final_params[key] = '%s' % float('%.3g' % final_params[key])

    
    # convert to pandas 
    df = pd.DataFrame.from_dict(final_params,orient='index')
    df.index.names = ['Input']
    df = df.rename(columns={0:'Estimated Value'})
    df['Units'] = '$\\frac{1}{\\text{year}}$'

    for id in df.index:
        if id in ['PG0', 'AG0', 'FG0', 'RG0', 'SH0', 'PC0', 'AC0', 
                  'FC0', 'RC0']:
            df.loc[id,'Units'] = 'dimensionless'
            df = df.rename(index={id:'$'+id[0]+'_{'+id[1]+'_0}$'})
        elif id in ['tilde_m','tilde_b','tilde_c','tilde_d','tilde_e']:
            df = df.rename(index={id:'$\\tilde{'+id[-1]+'}$'})
        elif id[:9] == 'log_theta':
            if log_form == True:
                df = df.rename(index={id:'$\\log(\\theta_{'+id[-2:]+'})$'})
            else:
                df = df.rename(index={id:'$\\theta_{'+id[-2:]+'}$'})
        elif id[:8] == 'log_beta':
            if log_form == True:
                df = df.rename(index={id:'$\\log(\\beta_{'+id[-2:]+'})$'})
            else:
                df = df.rename(index={id:'$\\beta_{'+id[-2:]+'}$'})
        elif id[:5] == 'tilde':
            df = df.rename(index={id:'$\\tilde{'+id[-3]+'}_'+id[-1]+'$'})
        elif id[:3] == 'log':
            if log_form == True:
                df = df.rename(index={id:'$\\log(\\'+id[4:]+')$'})
            else:
                df = df.rename(index={id:'$\\'+id[4:]+'$'})
        elif id == 'k':
            df = df.rename(index={id:'$'+id+'$'})
        else:
            df = df.rename(index={id:'$\\'+id+'$'})

    # print as latex 
    final_print = df.to_latex(header = True,
                        label = 'table:least_squares_case_1_results',
                        caption = 'enter caption here',
                        escape=False)
    final_print = final_print.replace('\\\\\n', '\\\\ \\hline\n')
    print(final_print)

###############################################################################
# Load final results from file
###############################################################################

def load_results(file_name):
    '''
    Open a pickle file with the final results from a least squares estiamte 
        and return the results. 

    Arguments:
        - file_name: 'string of file name

    Output: 
        - global_min: results from least squares
        - final_params: ordered dict of the final params
        - obj_fun: objective function value
        - obj_fun_vals_list: list off all objective functions from all 
    '''

    # to open the file
    with open(file_name,'rb') as fp:
        [global_min, obj_fun_vals_list] = pickle.load(fp) 

    final_params = OrderedDict(zip(param_ranges.keys(), list(global_min.x)))
    obj_fun = global_min.fun

    # print('The objective function value {}.'.format(obj_fun))

    return global_min, final_params, obj_fun, obj_fun_vals_list 

###############################################################################
# Run the program
###############################################################################

if __name__ == "__main__":
    RUNPARALLEL = True # whether or not to run in parallel
    numb_x0 = 1 # number of initial guesses
    include_AG2016 = False # whether or not to include the AG 2016 data point
    assume_muA_muF = True # whether of not to assume the values of mu_A and mu_F or estimate them
    if RUNPARALLEL:
        # can specify number of workers; default is num of processors
        with Pool() as pool:
            global_min, all_local_mins, obj_fun_vals_list = minimize_obj_func(numb_x0, 
                                                                              include_AG2016, 
                                                                              assume_muA_muF, 
                                                                              pool=pool)
    else:
        global_min, all_local_mins, obj_fun_vals_list = minimize_obj_func(numb_x0, 
                                                                          include_AG2016,
                                                                          assume_muA_muF)

    to_save = [global_min, obj_fun_vals_list]

    # save the resulting globlal_min information to a file 
    file_name = "least_squares_sol_AG2016_{}_x0_{}_assumemuAmuF_{}.pickle".format(include_AG2016,
                                                                                  numb_x0,
                                                                                  assume_muA_muF)
    with open(file_name,'wb') as fp:
        pickle.dump(to_save,fp)



