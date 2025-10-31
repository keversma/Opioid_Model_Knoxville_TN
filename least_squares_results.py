
import least_squares_parameter_estimation_log_scale as ls

# file_name = 'least_squares_sol_AG2016_False_x0_1000_assumemuAmuF_False.pickle'
file_name = 'least_squares_sol_AG2016_False_x0_1000_assumemuAmuF_True.pickle'

include_AG2016 = True
if file_name[25] == 'F':
    include_AG2016 = False

assume_muA_muF = True
if file_name[-12] == 'F':
    assume_muA_muF = False

global_min, final_params, obj_fun, obj_fun_vals_list = ls.load_results(file_name)
ls.plot_data_with_results(global_min,include_AG2016, assume_muA_muF)
# to_latex_table(final_params.copy())
# plot_solutions(final_params)
# calculate_corrected_AIG(global_min, final_params, include_AG2016)