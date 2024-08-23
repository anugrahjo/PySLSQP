'''Optimal control of Starship landing with coarsely discretized dynamics'''

import numpy as np
import matplotlib.pyplot as plt
import time

from modopt import PySLSQP, SNOPT, IPOPT, COBYQA, TrustConstr
from modopt.postprocessing import load_variables
from modopt.utils.profiling import profiler

from examples.ex_16_4starship_landing_jax import get_problem as get_jax_prob

num_steps = 20
sol = 16.804
obj_error = 0.168
prob = get_jax_prob(num_steps)

history      = {}
performance  = {}

def print_stats_and_save_performance(prob, solver, results, x, f,
                                     o_time, perf_dict, hist_dict):
    nev       = prob._callback_count
    o_evals   = prob._obj_count
    g_evals   = prob._grad_count
    h_evals   = prob._hess_count

    bd_viol_l  = np.maximum(0, prob.x_lower - x)
    bd_viol_u  = np.maximum(0, x - prob.x_upper)

    con         = prob._compute_constraints(x)
    con_viol_l  = np.maximum(0, prob.c_lower - con)
    con_viol_u  = np.maximum(0, con - prob.c_upper)

    feas = np.sum(bd_viol_l) + np.sum(bd_viol_u) + np.sum(con_viol_l) + np.sum(con_viol_u)

    success = np.isclose(f, sol, atol=obj_error) and feas < 1e-6

    print('\tTime:', o_time, 's')
    print('\tSuccess:', success)
    print('\tEvaluations:', nev)
    print('\tObj evals:', o_evals)
    print('\tGrad evals:', g_evals)
    print('\tHess evals:', h_evals)
    print('\tOptimized vars:', x)
    print('\tOptimized obj:', f)
    
    print('\tFeasibility:', feas)
    perf_dict[prob.problem_name, solver] = {'time': o_time,
                                            'success': success,
                                            'nev': nev,
                                            'nfev': o_evals,
                                            'ngev': g_evals,
                                            'nhev': h_evals,
                                            'objective': f,
                                            'feasibility': feas}

    obj_hist = load_variables(f"{results['out_dir']}/record.hdf5", 'obj')['callback_obj']
    hist_dict[prob.problem_name, solver] = obj_hist

if __name__ == '__main__':

    print('\nProblem:', prob.problem_name)
    print('='*50)

    # PySLSQP
    alg = 'PySLSQP'
    print(f'\t{alg} \n\t------------------------')
    optimizer = PySLSQP(prob, 
                        solver_options={'maxiter': 200, 'acc': 1e-6, 'iprint': 0, 'visualize': True, 'visualize_vars': ['objective', 'feasibility']}, 
                        recording=True)
    start_time = time.time()
    results = optimizer.solve()
    opt_time = time.time() - start_time

    print_stats_and_save_performance(prob, alg, results, results['x'], results['objective'],
                                     opt_time, performance, history)

    # SNOPT
    alg = 'SNOPT'
    print(f'\t{alg} \n\t------------------------')
    optimizer = SNOPT(prob, solver_options={'Major iterations': 200, 'Major optimality': 1e-7, 'Verbose': False},
                      recording=True)

    start_time = time.time()
    results = optimizer.solve()
    opt_time = time.time() - start_time
    
    print_stats_and_save_performance(prob, alg, results, results['x'], results['objective'],
                                     opt_time, performance, history)

    # TrustConstr
    alg = 'TrustConstr'
    print(f'\t{alg} \n\t------------------------')
    optimizer = TrustConstr(prob, solver_options={'maxiter': 200, 'gtol':1e-2, 'xtol':1e-6, 'ignore_exact_hessian':True}, 
                            recording=True)

    start_time = time.time()
    results = optimizer.solve()
    opt_time = time.time() - start_time
    
    print_stats_and_save_performance(prob, alg, results, results['x'], results['obj'],
                                     opt_time, performance, history)

    # IPOPT
    alg = 'IPOPT'
    print(f'\t{alg} \n\t------------------------')
    optimizer = IPOPT(prob, solver_options={'max_iter': 100, 'tol': 1e-3, 'print_level': 0,'accept_after_max_steps': 10},
                      recording=True)

    start_time = time.time()
    results = optimizer.solve()
    opt_time = time.time() - start_time
    
    print_stats_and_save_performance(prob, alg, results, results['x'], results['f'],
                                     opt_time, performance, history)

    algs = ['PySLSQP', 'SNOPT', 'TrustConstr', 'IPOPT']

    plt.rcParams['xtick.labelsize']=8
    plt.rcParams['ytick.labelsize']=8

    plt.figure()
    for alg in algs:
        y_data = history[prob.problem_name, alg]
        plt.plot(y_data, label=f"{alg} ({len(y_data)})")
    plt.xlabel('Evaluations', fontsize=9)
    plt.ylabel('Objective', fontsize=9)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title(f'{prob.problem_name} minimization')
    plt.legend(fontsize=8)
    plt.grid()
    plt.savefig(f"{prob.problem_name}-objective-cb.pdf", bbox_inches='tight')
    plt.savefig(f"{prob.problem_name}-objective-cb.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print performance
    print('\nPerformance')
    print('='*50)
    for key, value in performance.items():
        print(f"{str(key):40}:", value)