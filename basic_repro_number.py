
import math 
import numpy as np
import least_squares_parameter_estimation_log_scale as ls
from collections import OrderedDict
import opioid_model_ODE_solver_log_scale_alphaC as ODE
import sobol_sensitivity_analysis_log_scale as sobol
from scipy import optimize
from multiprocessing import Pool
import pandas as pd
import pickle

params_file = 'raw_output/least_squares_sol_AC2016_False_x0_1000_assumemuAmuF_False'

# load the community params as estiamed by the least squares 
global_min, comm_params, obj_fun, obj_fun_vals_list = ls.load_results(params_file + '.pickle')  
comm_params['mu'] = ls.assumed_params['mu']
comm_params['omega'] = ls.assumed_params['omega']

# load the param bounds from sobol (note contains bounds for comm and subcomm)
param_bounds_dict = sobol.generate_param_ranges(params_file)

# get just the subcomm params and not the inital values
subcomm_param_bounds = OrderedDict()
for key in param_bounds_dict.keys():
    if key not in comm_params.keys() and key[-1] != '0':
        subcomm_param_bounds[key] = param_bounds_dict[key]

t = 0 #calculating at year 2016

def objective_function(subcomm_params, comm_params):
    '''
    Return the objective function: the minimum of the absolute values 
        of the eigenvalues of FV^-1 (i.e., R_0) times -1. 

    Arguments:
        - subcomm_params: the parameters and initial conditions from 
            the subcommunity (tuple)
        - comm_params: the paramters and intial conditions from the 
            community, that were estiamed with least sqaures (OrderedDict)
    '''

    # create a OrderedDict of all the subcomm and comm params and initial values 
    all_params = comm_params.copy() 
    all_params.update(OrderedDict(zip(subcomm_param_bounds.keys(),
                                      list(subcomm_params)))) # the ranges

    betaCA = 10**all_params['log_beta_CA']
    thetaSC = 10**all_params['log_theta_SC']
    thetaP = 10**all_params['log_theta_P']
    nu = 10**all_params['log_nu']
    alpha_C = all_params['tilde_m']*t+all_params['tilde_b']
    mu = all_params['mu']
    muF = all_params['mu_F']
    muA = all_params['mu_A']
    varepsilon_C = all_params['epsilon_C']
    zeta = all_params['zeta']

    alpha_V = all_params['tilde_m_V']*t+all_params['tilde_b_V']
    alpha_H = all_params['tilde_m_H']*t+all_params['tilde_b_H']
    k = all_params['k']
    varepsilon_V = all_params['epsilon_V']
    varepsilon_H = all_params['epsilon_H']
    rho_H = all_params['rho_H']
    rho_V = all_params['rho_V']
    thetaSV = 10**all_params['log_theta_SV']
    thetaSH = 10**all_params['log_theta_SH']
    betaVA = 10**all_params['log_beta_VA']
    betaHA = 10**all_params['log_beta_HA']

    q = alpha_V*(varepsilon_H + mu + rho_H) \
        + alpha_H*(alpha_V + varepsilon_V + mu + rho_V) \
        + (varepsilon_H + varepsilon_V + mu)*(mu + rho_H + rho_V)

    SC_star = (varepsilon_C+mu) / (varepsilon_C+mu+alpha_C)
    PC_star = alpha_C / (varepsilon_C+mu+alpha_C)
    SV_star = (alpha_H*(varepsilon_V + mu) + (varepsilon_H + varepsilon_V + mu)*(mu + rho_H))/q
    SH_star = (alpha_V*varepsilon_H + (varepsilon_H + varepsilon_V + mu)*rho_V)/q
    PV_star = (alpha_V*(mu + rho_H) + alpha_H*(alpha_V + rho_V))/q

    e1 = betaCA*SC_star / (zeta + mu + muA)
    e2 = ((1-k)*(betaVA*SV_star+betaHA*SH_star)) / (zeta + mu + muA)
    e3 = (thetaSC*SC_star + thetaP*PC_star) / (nu+mu+muF)
    e4 = ((1-k)*(thetaSV*SV_star + thetaSH*SH_star + thetaP*PV_star)) / (nu + mu + muF)

    return min(-abs(e1),-abs(e2),-abs(e3),-abs(e4))


def minimize_obj_func(numb_x0, pool=None):
    '''
    Determine the global maximize of the objective function by running 
    scipy.optimize.minimize(method='SLSQP') (which is a local multivariate 
    optimizer) with numb_x0 number initial random guesses within our parameter 
    ranges by minimizing its negative.  Whichever local maximize shows up 
    the most will be our global maximize. 
    
    Arguments:
        - numb_x0: number of initial guess for the local min (int)
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

    # value ranges to choose from for initial random guess for local min 
    #   calculations 
    range_array = np.array(list(subcomm_param_bounds.values()))

    # first, gather all initial guesses
    init_guesses = np.array([np.random.uniform(range_array[:,0],range_array[:,1])
                                for n in range(numb_x0)])

    if pool is None:
        # run in serial
        
        # generate a list of bounds for the subcomm parameter ranges 
        bounds_list = list(subcomm_param_bounds.values())
        
        for i in range(numb_x0):

            print('Running {} of {}'.format(i,numb_x0))

            # if an error occurs, skip over this itereation and print a 
            #   message, but continue running 
            try:
                local_result = optimize.minimize(objective_function, 
                                                x0=init_guesses[i], 
                                                args=(comm_params), 
                                                bounds=bounds_list, 
                                                method='SLSQP')
            except AssertionError as e:
                print('issue with iteration i = {}, error: {}'.format(i, e.args))
                continue
            
            local_min_list.append(local_result)
    else:
        # run in parallel

        args = [(init_guesses[n],) for n in range(numb_x0)]

        # returns a list of results from the workers, applying the iterable of 
        #   arguments (tuples) to the function.
        local_min_list = pool.starmap(opt_func, args)

        # remove any iterations that failed (due to a constraint issue)
        local_min_list = [item for item in local_min_list if not isinstance(item, str)]

    # collect the list of function values 
    obj_fun_vals_list = [obj.fun for obj in local_min_list]

    # determine the global min by taking the mins of the resulting mins 
    global_min = local_min_list[obj_fun_vals_list.index(min(obj_fun_vals_list))]

    print('{} out of {} runs succeeded.'.format(len(local_min_list), numb_x0))

    return global_min, local_min_list, obj_fun_vals_list


def opt_func(guess):
    '''Pickleable function for parallelization.'''
    
    bounds_list = list(subcomm_param_bounds.values())

    # if an error occurs (alpha_C being negative), skip over this 
    #   itereation and print a message, but continue running 
    try:
        local_result = optimize.minimize(objective_function, 
                            x0=guess,
                            args=comm_params.copy(),
                            bounds=bounds_list, 
                            method='SLSQP')
    except AssertionError as e:
        print('issue with minimize. error: {}'.format(e.args))
        local_result = 'error occured' # all the 0's will be removed later
    
    return local_result

def param_ranges_to_latex(log_form=True):
    '''
    Print the param ranges in latex format.
        
    Arguments:
        - log_form: if true, write with log (e.g., log(beta) instead of beta)
    '''

    param_range_dict = subcomm_param_bounds.copy()

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
        if id in ['PC0', 'AC0', 'FC0', 'RC0', 'SH0', 'PV0', 'AV0', 
                  'FV0', 'RV0']:
            df = df.rename(index={id:'$'+id[0]+'_{'+id[1]+'_0}$'})
        elif id in ['tilde_m','tilde_b','tilde_c','tilde_d','tilde_e']:
            df = df.rename(index={id:'$\\tilde{'+id[-1]+'}$'})
        elif id in ['log_theta_SC', 'log_theta_SH', 'log_theta_SV']:
            if log_form:
                df = df.rename(index={id:'$\log(\\theta_{'+id[-2:]+'})$'})
            else:
                df = df.rename(index={id:'$\\theta_{'+id[-2:]+'}$'})
        elif id in ['log_beta_CA','log_beta_CP','log_beta_VA','log_beta_HA','log_beta_VP','log_beta_HP']:
            if log_form:
                df = df.rename(index={id:'$\log(\\beta_{'+id[-2:]+'})$'})
            else:
                df = df.rename(index={id:'$\\beta_{'+id[-2:]+'}$'})
        elif id in ['tilde_m_H', 'tilde_b_H','tilde_m_V', 'tilde_b_V']:
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

if __name__ == "__main__":
    RUNPARALLEL = True # whether or not to run in parallel
    numb_x0 = 1000 # number of initial guesses
    if RUNPARALLEL:
        # can specify number of workers; default is num of processors
        with Pool() as pool:
            global_min, all_local_mins, obj_fun_vals_list = minimize_obj_func(numb_x0, pool=pool)
    else:
        global_min, all_local_mins, obj_fun_vals_list = minimize_obj_func(numb_x0)

    final_params = OrderedDict(zip(subcomm_param_bounds.keys(), list(global_min.x)))
    obj_fun = global_min.fun

    ls.to_latex_table(final_params, log_form=True)

    print('R_0 = {}'.format(-1*obj_fun))
