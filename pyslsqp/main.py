"""
This module provides a wrapper for the Sequential Least Squares Programming
(SLSQP) optimization algorithm, originally implemented by Dieter Kraft.
The wrapper provides a modified interface to the optimization problem 
and many additional features compared to the SciPy implementation.
Certain portions of the code in this file, comprising only a few lines, 
are adapted from the SciPy implementation and are licensed under the 
BSD 3-Clause "New" or "Revised" License, 
as provided in the file: "./slsqp/LICENSE.txt".

The original algorithm is described in the following papers:
1. Dieter Kraft, "A software package for sequential quadratic programming",
   Technical Report DFVLR-FB 88-28, 1988. 
   DLR German Aerospace Center - Institute for Flight Mechanics, Koln, Germany.
2. Dieter Kraft, "Algorithm 733: TOMP - Fortran modules for optimal control
   calculations", ACM Transactions on Mathematical Software, 20(3):262-281, 1994.
"""

import warnings
import os, sys
import time
import numpy as np
from numpy import array, isfinite, linalg

_epsilon = np.sqrt(np.finfo(float).eps)

from pyslsqp.save_and_load import save_iteration
from pyslsqp._slsqp import slsqp
from pyslsqp.visualize import Visualizer

try:
    import h5py
except ImportError:
    warnings.warn("h5py not found, saving and loading data disabled")
    h5py = None


class Problem:
    '''
    Container class for optimization objective, constraints, gradient, and Jacobian functions.
    Keeps track of the number of function and gradient evaluations and time taken for each.
    Caches the function and derivatives values for the same input x to avoid redundant, consecutive evaluations.
    '''
    def __init__(self, x0, obj, con, grad, jac):
        self.x0 = x0
        self.obj = obj
        self.con = con
        self.grad = grad
        self.jac = jac

        self.nfev = 0
        self.ngev = 0
        self.fev_time = 0.0
        self.gev_time = 0.0
        self.warm_x = np.random.rand(len(x0))
        self.warm_x_derivs = np.random.rand(len(x0))
        self._funcs(x0)
        self._derivs(x0)

    def _funcs(self, x):
        '''
        Compute the objective and constraints at the given x, if x is different from the previous x.
        '''
        if not np.array_equal(x, self.warm_x):
            f_start = time.perf_counter()
            self.f = self.obj(x)
            self.c = self.con(x)
            self.warm_x = x * 1.0
            self.nfev += 1
            self.fev_time += time.perf_counter() - f_start
        return self.f, self.c

    def _derivs(self, x):
        '''
        Compute the gradient and Jacobian at the given x, if x is different from the previous x.
        '''
        if not np.array_equal(x, self.warm_x_derivs):
            g_start = time.perf_counter()
            self.g = self.grad(x)
            self.j = self.jac(x)
            self.warm_x_derivs = x * 1.0
            self.ngev += 1
            self.gev_time += time.perf_counter() - g_start
        return self.g, self.j

    
def check_update_scalar(scalar, name, size, ref_name):
    """
    Check and update the scalar value to match the size of the reference array.
    """
    if isinstance(scalar, (int, float, np.float_, np.int_, np.int32, np.int64, np.float32, np.float64)):
        return np.full(size, scalar, dtype=float)
    elif not isinstance(scalar, np.ndarray):
        raise ValueError(f"{name} must be a scalar or a 1-D array.")
    if len(scalar) != size:
        raise ValueError(f"{name} must have the same length as the {ref_name} ({size},).")
    return np.asfarray(scalar) # Convert to float array if an integer array is provided

def check_load_variables(read_file, iter, x, vars):
    """
    Check if the given x matches the x from the loaded read_file at given iteration.
    If yes, return the values of the vars at the given iteration.
    Otherwise, raise a warning.

    Parameters
    ----------
    read_file : h5py.File
        The file object to load the variables from.
    iter : int
        The iteration number to load the variables from.
    x : np.ndarray
        The optimization variables x.
    vars : list
        List of variables to load from the saved file.
    """
    grp = read_file['iter_' + str(iter)]
    x_hs = grp['x'][()]

    if not np.array_equal(x, x_hs):
        if iter == 0:
            warnings.warn("Given x0 do not match the x0 from the saved file. Resetting given x0 with the saved x0...")
        else:
            warnings.warn(f"The optimization variables x do not match the saved x at iteration {iter}. Falling back to normal function evaluations...")
            return 

    return [grp[var][()] for var in vars]


def get_default_options():
    """
    Returns the default options for the ``optimize()`` function as a dictionary.

    Examples
    --------
    >>> options = get_default_options()
    >>> options # doctest: +NORMALIZE_WHITESPACE
    {'obj': None, 'grad': None, 'con': None, 'jac': None, 'meq': 0, 'callback': None, 'xl': None, 'xu': None, 
    'x_scaler': 1.0, 'obj_scaler': 1.0, 'con_scaler': 1.0, 'maxiter': 100, 'acc': 1e-06, 'iprint': 1, 
    'finite_diff_abs_step': None, 'finite_diff_rel_step': 1.4901161193847656e-08, 'summary_filename': 'slsqp_summary.out', 
    'warm_start': False, 'hot_start': False, 'load_filename': None, 'save_itr': None, 'save_filename': 'slsqp_recorder.hdf5', 
    'save_vars': ['x', 'objective', 'optimality', 'feasibility', 'step', 'iter', 'majiter', 'ismajor', 'mode'], 
    'visualize': False, 'visualize_vars': ['objective', 'optimality', 'feasibility'], 'keep_plot_open': False, 
    'save_figname': 'slsqp_plot.pdf'}
    """
    options = {
        'obj': None,
        'grad': None,
        'con': None,
        'jac': None,
        'meq': 0,
        'callback': None,
        'xl': None,
        'xu': None,
        'x_scaler': 1.0,
        'obj_scaler': 1.0,
        'con_scaler': 1.0,
        'maxiter': 100,
        'acc': 1.0E-6,
        'iprint': 1,
        'finite_diff_abs_step': None,
        'finite_diff_rel_step': _epsilon,
        'summary_filename': 'slsqp_summary.out',
        'warm_start': False,
        'hot_start': False,
        'load_filename': None,
        'save_itr': None,
        'save_filename': 'slsqp_recorder.hdf5',
        'save_vars': ['x', 'objective', 'optimality', 'feasibility', 'step', 'iter', 'majiter', 'ismajor', 'mode'],
        'visualize': False,
        'visualize_vars': ['objective', 'optimality', 'feasibility'],
        'keep_plot_open': False,
        'save_figname': 'slsqp_plot.pdf',
    }

    return options


def optimize(x0, obj=None, grad=None, 
            con=None, jac=None, meq=0, callback=None,
            xl=None, xu=None, x_scaler=1.0, obj_scaler=1.0, con_scaler=1.0,
            maxiter=100, acc=1.0E-6, iprint=1,
            finite_diff_abs_step=None, finite_diff_rel_step=_epsilon, 
            summary_filename='slsqp_summary.out', warm_start=False, hot_start=False, load_filename=None,
            save_itr=None, save_filename='slsqp_recorder.hdf5', save_vars=['x', 'objective', 'optimality', 'feasibility', 'step', 'iter', 'majiter', 'ismajor', 'mode'],
            visualize=False, visualize_vars=['objective', 'optimality', 'feasibility'], keep_plot_open= False, save_figname='slsqp_plot.pdf'):
    """
    Minimize a scalar function of one or more variables using Sequential
    Least Squares Programming (SLSQP).
    This function is a wrapper to the original SLSQP implementation by Dieter
    Kraft. The wrapper provides a slightly modified interface to the
    optimization problem and many additional features, compared to the Scipy wrapper.

    This function solves the general nonlinear programming problem: ::

        minimize            f(x)
        subject to          c_i(x) = 0,                 i = 1,...,meq
                            c_i(x) >= 0,                i = meq+1,...,m
                            xl_i <= x_i <= xu_i ,       i = 1,...,n

    where `x` is a vector of variables with size `n`, `f(x)` is the objective,
    `c(x)` is the constraint function, and `xl` and `xu` are vectors of lower and
    upper bounds, respectively. The first `meq` constraints are equalities
    while the remaining `(m - meq)` constraints are inequalities.

    Parameters
    ----------
    x0 : np.ndarray
        Initial guess for the optimization variables. 
        Array of real elements of size `(n,)`, where 'n' is the
        number of independent variables.
        ``x0`` is a necessary argument (unlike other arguments which are optional) 
        to inform the optimizer about the number of optimization variables.
    obj : callable
        Objective function to be minimized. The function is called as
        ``obj(x)``, where ``x`` is the array of independent variables.
    con : callable
        Vector-valued constraint function of size m. 
        The function is called as ``con(x)``, where ``x`` is 
        the array of optimization variables.
        The first `meq` constraints are treated as equality constraints,
        while the remaining `(m - meq)` constraints are treated as
        inequality constraints.
    xl : np.ndarray or scalar, default=None
        Lower bounds on optimization variables. Defaults to `None`, in which
        case bounds are assumed to be ``-np.inf``.
        xl can be a scalar, in which case all variables have the same lower
        bound, or an array of real elements of size `(n,)`.
    xu : np.ndarray or scalar, default=None
        Upper bounds on optimization variables. Defaults to `None`, in which
        case bounds are assumed to be ``np.inf``.
        xu can be a scalar, in which case all variables have the same upper
        bound, or an array of real elements of size (`n,)`.
    x_scaler : float or np.ndarray, default=1.0
        Factor by which the optimization variables are scaled before sending to SLSQP.
        If float, all variables are scaled by the same factor.
        If np.ndarray of size `(n,)`, each variable is scaled by the corresponding factor.
    obj_scaler : float, default=1.0
        Factor by which the objective function value is scaled before sending to SLSQP.
    con_scaler : float or np.ndarray, default=1.0
        Factor by which the constraint function values are scaled before sending to SLSQP.
        If float, all constraints are scaled by the same factor.
        If np.ndarray of size `(m,)`, each constraint is scaled by the corresponding factor.
    meq : int, default=0
        The number of equality constraints. Defaults to 0.
    grad : callable, default=None
        Gradient of the objective function. If `None`, the gradient will be
        approximated using finite differences.
    jac : callable, default=None
        Jacobian of the constraint function. If `None`, the Jacobian will be
        approximated using finite differences.
    maxiter : int, default=100
        Maximum number of iterations.
    acc : float, default=1.0E-6
        abs(acc) is the stopping criterion and controls the final accuracy.
        If ``acc`` < 0, a maximization problem is solved.
        Otherwise, a minimization problem is solved.
    iprint : int, default=1
        Controls the verbosity of the SLSQP algorithm. 
        Set ``iprint <= 0`` to suppress all console outputs.
        Set ``iprint  = 1`` to print only the final results upon completion.
        Set ``iprint >= 2`` to print the status of each major iteration and the final results.
    finite_diff_abs_step : None or array_like, default=None
        The absolute step size to use for numerical approximation of the derivatives. 
        If None (default), then step is selected using finite_diff_rel_step.
    finite_diff_rel_step : None or array_like, default=None
        The relative step size to use for numerical approximation of the derivatives. 
        The absolute step size is computed as ``h = rel_step * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. Not used if finite_diff_abs_step is given.
        By default, it is selected automatically as 
        ``_epsilon = np.sqrt(np.finfo(float).eps)`` approximately 1e-8.
    callback : callable, default=None
        Function to be called after each major iteration. The function is called as
        ``callback(x)``, where ``x`` is the optimization variable vector from the current major iteration.
    save_itr : {None, 'all', 'major'}, default=None
        If 'all', all iterations are saved. If 'major', only major iterations are saved.
        By default, ``save_itr`` is None, and no iterations are saved.
    summary_filename : str, default='slsqp.out'
        Name of the file to save the summary of the optimization process. 
        By default, the file is saved as ``'slsqp_summary.out'``.
    save_filename : str, default='slsqp_recorder.hdf5'
        Name of the file to save the iterations. 
        By default, the file is saved as ``'slsqp_recorder.hdf5'``.
    save_vars : list, default=['x', 'objective', 'optimality', 'feasibility', 'step', 'mode', 'iter', 'majiter', 'ismajor']
        List of variables to save. The full list of variables available are 
        ``['x', 'objective', 'optimality', 'feasibility', 'step', 'mode', 'iter', 'majiter', 'ismajor', 'constraints', 'gradient', 'multipliers', 'jacobian']``.
    warm_start : bool, default=False
        If True, the optimization algorithm will use the previous solution from the last optimization as the initial guess.
    hot_start : bool, default=None
        If True, the optimization algorithm will use the saved objective, constraints, gradient, and jacobian 
        functions from the previous optimization until the iterations reach the last saved iteration.
        Note that this only works if save_itr for the previous optimization was set to 'all'.
        This is useful when the objective, constraints, gradient, and jacobian functions are expensive to compute
        and the optimization process was interrupted in a prior run.
    load_filename : str, default=None
        Name of the file to load the previous optimization solution or iterates for warm or hot start.
        If None, the ``load_filename`` is assumed to be the same as the save_filename.
        If ``load_filename`` is same as the provided ``save_filename`` will be updated as:
        'save_filename without extension' + '_warm.hdf5' or '_hot.hdf5' depending on the warm_start or hot_start.
    visualize : bool, default=False
        Set to True to visualize the optimization process.
        Only major iterations are visualized.
    visualize_vars : list, default=['objective', 'optimality', 'feasibility']
        List of scalar variables to visualize. Available variables are
        ``['x[i]', 'objective', 'optimality', 'feasibility', 'constraints[i]', 'gradient[i]', 'multipliers[i]', 'jacobian[i,j]']``.
    keep_plot_open : bool, default=False
        Set to True to keep the plot window open after the optimization process is complete.
    save_figname : str, default='slsqp_plot.pdf'
        Name of the file to save the plot.

    Examples
    --------
    >>> import numpy as np
    >>> from pyslsqp import optimize
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> xl = 0.0
    >>> xu = np.array([1, 1])
    >>> x0 = np.array([0.5, 0.5])
    >>> results = optimize(x0, obj=obj, grad=grad, xl=xl, xu=xu)  # doctest: +ELLIPSIS
    No constraints defined. Running an unconstrained optimization problem...
    Optimization terminated successfully    (Exit mode 0)
                Final objective value                : 0.000000e+00
                Final optimality                     : 0.000000e+00
                Final feasibility                    : 0.000000e+00
                Number of major iterations           : 2
                Number of function evaluations       : 2
                Number of derivative evaluations     : 2
                Average Function evaluation time     : ... s per evaluation
                Average Derivative evaluation time   : ... s per evaluation
                Total Function evaluation time       : ... s [ ...%]
                Total Derivative evaluation time     : ... s [ ...%]
                Optimizer time                       : ... s [ ...%]
                Processing time                      : ... s [ ...%]
                Visualization time                   : ... s [  0.00%]
                Total optimization time              : ... s [100.00%]
                Summary saved to                     : slsqp_summary.out
    >>> results['x']
    array([0., 0.])
    >>> con = lambda x: np.array([x[0] - 0.1, x[1] - 0.2])
    >>> jac = lambda x: np.array([[1, 0], [0, 1]])
    >>> meq = 1
    >>> results = optimize(x0, obj=obj, grad=grad, con=con, jac=jac, meq=meq, xl=xl, xu=xu) # doctest: +ELLIPSIS
    Optimization terminated successfully    (Exit mode 0)
                Final objective value                : 5.000000e-02
                Final optimality                     : 1.538763e-16
                Final feasibility                    : 1.942890e-16
                Number of major iterations           : 2
                Number of function evaluations       : 2
                Number of derivative evaluations     : 2
                Average Function evaluation time     : ... s per evaluation
                Average Derivative evaluation time   : ... s per evaluation
                Total Function evaluation time       : ... s [ ...%]
                Total Derivative evaluation time     : ... s [ ...%]
                Optimizer time                       : ... s [ ...%]
                Processing time                      : ... s [ ...%]
                Visualization time                   : ... s [  0.00%]
                Total optimization time              : ... s [100.00%]
                Summary saved to                     : slsqp_summary.out
    >>> results['x']
    array([0.1, 0.2])

    """
    main_start = time.perf_counter()
    import copy

    in_xl = copy.copy(xl)
    in_xu = copy.copy(xu)

    if (obj is None) and (con is None):
        raise ValueError("At least one of the objective or constraint functions must be defined.")
    
    if x0 is None:
        raise ValueError("Some initial guess 'x0' must be provided to inform the optimizer about the number of optimization variables n.")
                         
    if visualize:
        if not isinstance(visualize_vars, (list, str)):
            raise TypeError("visualize_vars must be a list of strings or a string.")
        if isinstance(visualize_vars, str):
            visualize_vars = [visualize_vars]
        
        # Check if the variables in visualize_vars list are valid
        prefixes = ['x[', 'constraints[', 'gradient[', 'multipliers[', 'jacobian[']
        for var in visualize_vars:
            # prefix = var.split('[')[0]
            if var not in ['objective', 'optimality', 'feasibility']:
                if any(var.startswith(prefix) for prefix in prefixes) and var.endswith(']'):
                    indices = var.split('[')[1].split(']')[0].split(',')
                    if not all(idx.isdigit() for idx in indices):
                        raise ValueError(f"Invalid index in {var}. Index must be an integer.")
                    if any(var.startswith(prefix) for prefix in ['x[', 'constraints[', 'gradient[', 'multipliers[']):
                        if len(indices) != 1:
                            raise ValueError(f"Invalid index in {var}. Expected one index.")
                    if var.startswith('jacobian['):
                        if len(indices) != 2:
                            raise ValueError(f"Invalid index in {var}. Expected two indices.")
                else:
                    raise ValueError(f"Invalid variable {var} in visualize_vars. Must be one of ['objective', 'optimality', 'feasibility', 'x[i]', 'constraint[i]', 'gradient[i]', 'multipliers[i]', 'jacobian[i,j]'].")
        
        visualizer = Visualizer(visualize_vars, summary_filename, save_figname)

    # Transform x0 into an array.
    x = np.asfarray(x0).flatten()

    # Warm start from previous optimization solution
    hot_run = False
    if warm_start or hot_start:
        if warm_start and hot_start:
            raise ValueError("Warm start and hot_start are mutually exclusive. Only one of warm_start or hot_start can be True.")
        if load_filename is None:
            load_filename = save_filename
        if save_filename == load_filename:
            if warm_start:
                save_filename = save_filename.split('.')[0] + '_warm.hdf5'
            else:
                save_filename = save_filename.split('.')[0] + '_hot.hdf5'
        if h5py is None:
            raise ImportError("h5py is required for loading previous solution. Install h5py to use warm or hot start.")
        try:
            read_file = h5py.File(load_filename, 'r')
        except FileNotFoundError:
            raise FileNotFoundError(f"File {load_filename} not found or not a valid h5py file. Cannot perform warm or hot start.")
        
    if warm_start:
        print(f"Warm starting from previous optimization solution x from {load_filename}...")
        if 'results' in read_file.keys():
            x = read_file['results']['x'][()]
        else:
            print(f"No results found for warm-start in {load_filename}. Trying to load the last iteration...")
            warm_start_success = False
            num_saves = len(read_file.keys()) # = number of iter/majiter - 1 (since results were not found and counting starts from 0th iteration)
            for k in range(num_saves-1, -1, -1):
                if f'iter_{k}' in read_file.keys():
                    x = read_file[f'iter_{k}']['x'][()]
                    warm_start_success = True
                    print(f"Success loading x from iteration {k} for warm-start.")
                    break
            if not warm_start_success:
                raise ValueError(f"No iterations found in {load_filename}. Cannot perform warm start.")

            x = read_file['iter_0']['x'][()]
        if len(x) != len(x0.flatten()):
            raise ValueError(f"Given x0 and saved x do not have the same length. Expected {len(x0)} but got {len(x)} from {load_filename}.")

    if hot_start:
        saved_itr = read_file.attrs['save_itr']
        if saved_itr != 'all':
            raise ValueError(f"Hot start requires all iterations to be saved. Cannot perform hot start with {load_filename} whose 'save_itr' was set to {saved_itr}.")
        
        saved_vars         = read_file.attrs['save_vars']
        minimum_saved_vars = ['x', 'objective', 'constraints', 'gradient', 'jacobian']
        if not all(ms_var in saved_vars for ms_var in minimum_saved_vars):
            raise ValueError(f"All of [objective, constraints, gradient, and jacobian] are not available in {load_filename}. Cannot perform hot start.")
        
        hot_niter = len(read_file.keys()) - 2 # Number of iterations saved in the file [excludes 0th iteration and results]
        
        print(f"Hot starting using saved x, objective, constraints, gradient, and jacobian from {load_filename}...")

    # n: number of optimization variables
    n = len(x)

    if xl is None:
        xl = np.full(n, -np.inf)
    if xu is None:
        xu = np.full(n, np.inf)

    xl = copy.copy(xl)  # Copied so that the original is not modified, if used later by the user
    xu = copy.copy(xu)  # Copied so that the original is not modified, if used later by the user

    # Check and update xl and xu to match the size of x0
    xl = check_update_scalar(xl, 'xl', n, 'optimization variables x0')
    xu = check_update_scalar(xu, 'xu', n, 'optimization variables x0')
    if any(xl > xu):
        raise ValueError("The lower bounds (xl) must be less than or equal to the upper bounds (xu) for each variable.")
    
    # lb and ub are the actual bounds; xl and xu will be marked with nans for infinite bounds for Fortran
    lb = np.copy(xl)
    ub = np.copy(xu)
    
    # Mark infinite bounds with nans; the Fortran code understands this
    xl[~isfinite(xl)] = np.nan
    xu[~isfinite(xu)] = np.nan

    # clip the initial guess to bounds
    x = np.clip(x, lb, ub)

    exit_modes = {-1: "Gradient evaluation required (g & a)",
                   0: "Optimization terminated successfully",
                   1: "Function evaluation required (f & c)",
                   2: "More equality constraints than independent variables",
                   3: "More than 3*n iterations in LSQ subproblem",
                   4: "Inequality constraints incompatible",
                   5: "Singular matrix E in LSQ subproblem",
                   6: "Singular matrix C in LSQ subproblem",
                   7: "Rank-deficient equality constraint subproblem HFTI",
                   8: "Positive directional derivative for linesearch",
                   9: "Iteration limit reached"}

    # Define a function to clip x to bounds before calling the objective, constraint, gradient, or jacobian functions    
    def _clip_x_for_func(func, lb, ub):
        def _func(x):
            if np.any(x < lb) or np.any(x > ub):
                warnings.warn("At least one entry in x was outside the bounds during a minimize step. Clipping to the bounds....")
                return func(np.clip(x, lb, ub))
            return func(x)
        return _func
    
    r_step = finite_diff_rel_step
    a_step = finite_diff_abs_step
    
    if obj is None:
        _obj  = lambda x: 0.0
        _grad = lambda x: np.zeros(n, dtype=float)
        warnings.warn("Objective function 'obj' is not defined. Running a feasibility problem...")
    else:
        # gh11403 SLSQP sometimes exceeds bounds by 1 or 2 ULP, make sure this
        # doesn't get sent to the func/grad evaluator.
        _obj = _clip_x_for_func(obj, lb, ub)
        if grad is None:
            # Note: FD grad() uses unclipped objective function to avoid errors in the finite difference calculation.
            # Note also that input x for grad() is already clipped and within bounds when it is called through _grad().
            def grad(x):
                if a_step is None:
                    h = r_step * np.maximum(1, np.abs(x))
                else:
                    h = a_step * np.ones(n)

                fd_grad = np.full((n,), -obj(x), dtype=float)
                for i in range(n):
                    # Check if the step will exceed the bounds and reverse the step if necessary
                    if x[i]+h[i] > xu[i]: # h is always positive so no need to check for lower bound
                        h[i] = -h[i]
                    e = np.zeros(n)
                    e[i] = h[i]
                    fd_grad[i] += obj(x + e)

                fd_grad /= h
                return fd_grad

        _grad = _clip_x_for_func(grad, lb, ub)

    if con is None:
        print("No constraints defined. Running an unconstrained optimization problem...")
        _con = lambda x: np.array([0.], dtype=float)
        _jac = lambda x: np.zeros((1, n), dtype=float)
    else:
        _con = _clip_x_for_func(con, lb, ub)
        if jac is None:
            # Note: FD jac() uses unclipped constraint function to avoid errors in the finite difference calculation.
            # Note also that input x for jac() is already clipped and within bounds when it is called through _jac().
            def jac(x):
                if a_step is None:
                    h = r_step * np.maximum(1, np.abs(x))
                else:
                    h = a_step * np.ones(n)

                fd_jac = np.outer(con(x), -np.ones(n))
                for i in range(n):
                    # Check if the step will exceed the bounds and reverse the step if necessary
                    if x[i]+h[i] > xu[i]: # h is always positive so no need to check for lower bound
                        h[i] = -h[i]
                    e = np.zeros(n)
                    e[i] = h[i]
                    fd_jac[:, i] += con(x + e)
                fd_jac /= h # Note: fd_jac has shape (m, n) and h has shape (n,) so broadcasting is done correctly
                return fd_jac
            
        _jac = _clip_x_for_func(jac, lb, ub)

    prob = Problem(x, _obj, _con, _grad, _jac)
    fx, c = prob._funcs(x)
    
    # Compute the constants that Fortran SLSQP module needs
    # m: total number of constraints
    m = len(c) if con is not None else 0
    # la: The number of constraints, or 1 if there are no constraints
    la = max(1, m)

    # Allocate the array workspaces needed by the Fortran SLSQP module
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    len_jw = mineq
    w = np.zeros(len_w)
    jw = np.zeros(len_jw)

    # Set the accuracy as acc, the mode as 0, and the major iteration counter as maxiter
    mode = array(0, int)
    acc = array(acc, float)
    majiter = array(maxiter-1, int)
    majiter_prev = 0

    # Initialize internal SLSQP state variables
    alpha = array(0, float)
    f0 = array(0, float)
    gs = array(0, float)
    h1 = array(0, float) # h1 is the optimality (~complementarity) measure,  h1 = mu*max(0, -c[:meq])
    # h2 is the feasibility measure, h2 = sum(constraint violations) ; con_viol > 0 for infeasible constraints
    h2 = array(0, float) # h2 is the feasibility measure, h2 = sum(abs(c[:meq]) + sum(max(0, -c[meq:]))
    h3 = array(0, float)
    h4 = array(0, float)
    t = array(0, float)
    t0 = array(0, float)
    tol = array(0, float)
    iexact = array(0, int)
    incons = array(0, int)
    ireset = array(0, int)
    itermx = array(0, int)
    line = array(0, int)
    n1 = array(0, int)
    n2 = array(0, int)
    n3 = array(0, int)
    # Although Lagrange multipliers `mu `are supposed to be in w[:m], it is only used in the L1 test function and gets updated there with estimates
    # so we use the lsq multipliers r() which are the actual Lagrange multipliers
    # wref indicates where lsq multipliers start in w 
    wref = int(la+ (n+1)*n/2 + 1 + n)

    if save_itr is not None:
        if save_itr not in ['all', 'major']:
            raise ValueError("'save_itr' must be 'all' or 'major'")
        
        if h5py is None:
            raise ImportError("h5py is required for saving iterations. Install h5py to use this feature.")
        # If file exists, delete it
        try:
            os.remove(save_filename)
        except FileNotFoundError:
            pass

        # 'ismajor', 'iter', `majiter` are appended to the save_vars list by default to indicate if the iteration is a major iteration
        in_save_vars = copy.copy(save_vars)
        if 'ismajor' not in save_vars:
            save_vars.append('ismajor')
        if 'iter' not in save_vars:
            save_vars.append('iter')
        if 'majiter' not in save_vars:
            save_vars.append('majiter')

        if not set(save_vars).issubset(['x', 'objective', 'optimality', 'feasibility', 'step', 'mode', 'iter', 'majiter', 'ismajor', 'constraints', 'gradient', 'multipliers', 'jacobian']):
            raise ValueError("Invalid variable in save_vars. Must be one of " \
                             "'x', 'objective', 'optimality', 'feasibility', 'step', 'mode', 'iter', 'majiter', 'ismajor', 'constraints', 'gradient', 'multipliers', or 'jacobian'.")
        
        file = h5py.File(save_filename, 'a')
        file.attrs['n'] = n
        file.attrs['m'] = m
        file.attrs['meq'] = meq
        
        file.attrs['x0'] = x0
        file.attrs['xl'] = in_xl if in_xl is not None else 'None (undefined)'
        file.attrs['xu'] = in_xu if in_xu is not None else 'None (undefined)'

        file.attrs['x_scaler'] = x_scaler
        file.attrs['obj_scaler'] = obj_scaler
        file.attrs['con_scaler'] = con_scaler

        file.attrs['maxiter'] = maxiter
        file.attrs['acc'] = acc
        file.attrs['iprint'] = iprint

        if finite_diff_abs_step is not None:
            file.attrs['finite_diff_abs_step'] = finite_diff_abs_step
        else:
            file.attrs['finite_diff_abs_step'] = 'None (undefined)'
        
        file.attrs['finite_diff_rel_step'] = finite_diff_rel_step
        file.attrs['summary_filename'] = summary_filename
        file.attrs['save_itr'] = save_itr
        file.attrs['save_filename'] = save_filename
        file.attrs['save_vars'] = in_save_vars
        file.attrs['warm_start'] = warm_start
        file.attrs['hot_start'] = hot_start
        if load_filename is not None:
            file.attrs['load_filename'] = load_filename
        else:
            file.attrs['load_filename'] = 'None (undefined)'
        file.attrs['visualize'] = visualize
        file.attrs['visualize_vars'] = visualize_vars
        file.attrs['keep_plot_open'] = keep_plot_open
        file.attrs['save_figname'] = save_figname

    # mode is zero on entry, so call objective, constraints and gradients
    # there should be no func evaluations here because it's cached from prob
    if hot_start:
        hot_run = True # Turn on hot_run which indicates that the optimization is using the saved variables until hot_run is turned off
        x, fx, c, g, a = check_load_variables(read_file, 0, x, vars=['x', 'objective', 'constraints', 'gradient', 'jacobian'])
        prob.nfev = 1 # Counter for number of function evaluations in the hot start
        prob.ngev = 1 # Counter for number of gradient evaluations in the hot start
    else:
        fx, c = prob._funcs(x)
        g,  a = prob._derivs(x)
    g = np.append(g, 0.0)
    a = np.concatenate((a, np.zeros([la, 1])), 1)

    iter = 0
    opt_time = 0.0

    out_dict = {}
    out_dict['iter'] = iter
    out_dict['majiter'] = 0
    out_dict['ismajor'] = True
    out_dict['mode'] = mode
    out_dict['x'] = x
    out_dict['objective'] = fx
    out_dict['constraints'] = c[:m]
    out_dict['gradient'] = g[:-1]
    out_dict['multipliers'] = w[wref:wref+m]
    out_dict['jacobian'] = a[:, :-1]
    out_dict['optimality'] = 99.0    # Optimality is not available in the 0th iteration
    out_dict['feasibility'] = 99.0   # Feasibility is not available in the 0th iteration
    out_dict['step'] = 99.0          # Step is undefined in the 0th iteration

    if save_itr is not None: # Note majiter and iter are the same for the first iteration
        save_iteration(file, iter, save_vars, out_dict)

    # Print the header if iprint >= 2
    if iprint >= 2:
        print("%5s %5s %5s %16s %16s %16s %16s %16s %16s" % ("MAJOR", "NFEV", "NGEV", "OBJFUN", "GNORM", "CNORM", "FEAS", "OPT", "STEP"))
        print("%5i %5i %5i %16.6E %16.6E %16.6E %16.6E %16.6E %16.6E" % (0, 1, 1, fx, linalg.norm(g), linalg.norm(c), 99.0, 99.0, 99.0))

        # with open('opt_slsqp.out', 'w') as f:
        #     np.savetxt(f, [h2])
        # with open('duals_slsqp_maj.out', 'w') as f:
        #     np.savetxt(f, w[wref:wref+m].reshape(1,m))

    # Write the header to the summary file regardless of the iprint value
    with open(summary_filename, 'w') as f:
        f.write("%5s %5s %5s %16s %16s %16s %16s %16s %16s \n" % ("MAJOR", "NFEV", "NGEV", "OBJFUN", "GNORM", "CNORM", "FEAS", "OPT", "STEP"))
        f.write("%5i %5i %5i %16.6E %16.6E %16.6E %16.6E %16.6E %16.6E \n" % (0, 1, 1, fx, linalg.norm(g), linalg.norm(c), 99.0, 99.0, 99.0))

    if visualize:
        visualizer.update_plot(out_dict)

    # Scaler check and initialization
    x_scaler   = copy.copy(x_scaler)     # Copied so that the original is not modified, if used later by the user
    con_scaler = copy.copy(con_scaler)   # Copied so that the original is not modified, if used later by the user
    obj_scaler = copy.copy(obj_scaler)   # Copied so that the original is not modified, if used later by the user
    x_scaler   = check_update_scalar(x_scaler, 'x_scaler', n, 'optimization variables x0')
    con_scaler = check_update_scalar(con_scaler, 'con_scaler', la, 'constraints con(x)') # size of (la,)
    obj_scaler = check_update_scalar(obj_scaler, 'obj_scaler', 1, 'objective function f(x)')[0]
    
    x_inv_scaler = np.append(1.0 / x_scaler, 0.0) # size of (n+1,)
    g_scaler = obj_scaler * x_inv_scaler
    
    # Apply scaling to optimization variables and its bounds
    xl_scaled = xl * x_scaler
    xu_scaled = xu * x_scaler
    x_scaled  = x  * x_scaler

    while 1:
        # Scale the objective, constraints, gradients and jacobian
        fx_scaled = fx * obj_scaler                         # scalar
        c_scaled  = c  * con_scaler                         # size of (la,)
        g_scaled  = g  * g_scaler                           # size of (n+1,)
        a_scaled  = a  * np.outer(con_scaler, x_inv_scaler) # size of (la, n+1)

        iter += 1
        # Call SLSQP
        opt_start = time.perf_counter()
        slsqp(m, meq, x_scaled, xl_scaled, xu_scaled, fx_scaled, c_scaled, g_scaled, a_scaled, acc, majiter, mode, w, jw,
              alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
              iexact, incons, ireset, itermx, line,
              n1, n2, n3)
        opt_time += time.perf_counter() - opt_start

        if majiter > majiter_prev and majiter != majiter_prev + 1:
            warnings.warn(f"SLSQP Bug: Major iteration counter jumped from {majiter_prev} to {majiter}. Resetting to {majiter_prev + 1}.")
            majiter = majiter_prev + 1

        x = x_scaled / x_scaler

        if hot_run:
            if iter == hot_niter:
                # Hot start is complete. Turn off hot_run and begin normal function evaluations
                print(f"Hot start is complete at iteration {iter}. Starting normal function evaluations...")
                hot_run = False
                hot_nfev = prob.nfev * 1    # Number of function evaluations that used saved variables
                hot_ngev = prob.ngev * 1    # Number of derivative evaluations that used saved variables
            else:
                try:
                    if mode == 1:   # objective and constraint evaluation required
                        fx, c = check_load_variables(read_file, iter, x, vars=['objective', 'constraints'])
                        prob.nfev += 1      # update problem nfev counter along with hot fevals
                    if mode == -1:   # derivative evaluation required
                        g, a = check_load_variables(read_file, iter, x, vars=['gradient', 'jacobian'])
                        g = np.append(g, 0.0)
                        a = np.concatenate((a, np.zeros([la, 1])), 1)
                        prob.ngev += 1      # update problem ngev counter along with hot gevals
                    
                except:
                    # Hot start failed. Turn off hot_run and begin normal function evaluations
                    print(f"Hot start failed at iteration {iter}. Starting normal function evaluations...")
                    hot_run = False
                    hot_nfev = prob.nfev * 1    # Number of function evaluations that used saved variables
                    hot_ngev = prob.ngev * 1    # Number of derivative evaluations that used saved variables
        
        # Below should not be else, as hot_run can be turned off in the above block 
        if not hot_run:
            if mode == 1:  # objective and constraint evaluation required
                fx, c = prob._funcs(x)
        
            if mode == -1:  # derivative evaluation required
                g, a = prob._derivs(x)
                g = np.append(g, 0.0)
                a = np.concatenate((a, np.zeros([la, 1])), 1)

        # Check if slsqp exits with a mode other than +/- 1 meaning it has completed
        # SLSQP sometimes forgets to update the majiter when it exits with abs(mode) != 1 (Possible bug?)
        if abs(mode) != 1:
            if majiter == majiter_prev: # If majiter has not incremented when exiting, increment it
                majiter = int(majiter) + 1
            
        out_dict['iter'] = iter
        out_dict['majiter'] = majiter * 1 # Copy the value of majiter to out_dict
        out_dict['ismajor'] = True if majiter > majiter_prev else False
        out_dict['mode'] = mode
        out_dict['x'] = x
        out_dict['objective'] = fx
        out_dict['constraints'] = c[:m]
        out_dict['gradient'] = g[:-1]
        out_dict['multipliers'] = w[wref:wref+m]
        out_dict['jacobian'] = a[:, :-1]
        out_dict['optimality'] = h1    # Optimality from Fortran SLSQP
        # out_dict['feasibility'] = h2 # Feasibility from Fortran SLSQP
        out_dict['feasibility'] = feas_calc = np.sum(np.abs(c[:meq])) + np.sum(np.maximum(0, -c[meq:]))
        out_dict['step'] = alpha

        ########### BETTER OPTIMALITY AND FEASIBILITY CALCULATION ###########
        # # The following optimality calculation ignores Lagrange multipliers for variable bounds
        # # since SLSQP does not calculate them. They are just set to NaN in the Fortran code.
        # opt_lmults = 0.0
        # opt_compl  = 0.0
        # max_lmult  = 0.0
        # if m != 0:
        #     if meq != m:
        #         opt_lmults = np.max(np.maximum(0, -w[wref+meq:wref+m])) # Nonnegativity of multipliers for inequality constraints
        #     opt_compl  = np.max(np.abs(w[wref:wref+m]*c[:m]))  # Complementarity condition for all constraints
        #     # opt_lgrad  = np.max(np.abs(g[:-1] - a[:, :-1].T @ w[wref:wref+m])) # Lagrangian gradient optimality condition (missing bound multipliers)
        #     max_lmult  = np.max(np.abs(w[wref:wref+m])) # Maximum Lagrange multiplier for constraints
        # out_dict['optimality'] = opt_calc = max(opt_lmults, opt_compl) / (1 + max_lmult) # Optimality measure

        # A different feasibility measure that normalizes the max. constraint violation by the maximum of the absolute values of x
        # max_constraint_violation = np.max(np.concatenate((np.abs(c[:meq]), np.maximum(0, -c[meq:]))))
        # out_dict['feasibility'] = feas_calc = max_constraint_violation / (1 + np.max(np.abs(x))) # Normalized feasibility
        ########################################################################

        if save_itr == 'all':
            save_iteration(file, iter, save_vars, out_dict)

        if majiter > majiter_prev:
            if save_itr == 'major':
                save_iteration(file, majiter*1, save_vars, out_dict)
            # call callback if major iteration has incremented
            if callback is not None:
                callback(np.copy(x))

            # Print the status of the current major iterate if iprint >= 2
            if iprint >= 2:
                print("%5i %5i %5i %16.6E %16.6E %16.6E %16.6E %16.6E %16.6E" % (majiter, prob.nfev, prob.ngev,
                                                   fx, linalg.norm(g), linalg.norm(c), feas_calc, h1, alpha))
                # with open('opt_slsqp.out', 'a') as f:
                #     np.savetxt(f, [h2])
                # with open('duals_slsqp_maj.out', 'a') as f:
                #     np.savetxt(f, w[wref:wref+m].reshape(1,m))

            # Write the status of the current iteration to the summary file regardless of the iprint value
            with open(summary_filename, 'a') as f:
                f.write("%5i %5i %5i %16.6E %16.6E %16.6E %16.6E %16.6E %16.6E \n" % (majiter, prob.nfev, prob.ngev,
                                                fx, linalg.norm(g), linalg.norm(c), feas_calc, h1, alpha))
            if visualize:
                visualizer.update_plot(out_dict)

        # If exit mode is not -1 or 1, slsqp has completed
        if abs(mode) != 1:            
            break

        majiter_prev = int(majiter)

    vis_time = 0.0
    vis_wait = 0.0
    if visualize:
        if keep_plot_open:
            visualizer.keep_plot()
            vis_wait = visualizer.wait_time
        else:
            visualizer.close_plot()
        vis_time = visualizer.vis_time   

    total_time = time.perf_counter() - main_start - vis_wait
    processing_time = total_time - prob.fev_time - prob.gev_time - vis_time - opt_time

    # Optimization loop complete. Print summary if iprint >= 1
    if iprint >= 1:
        print(exit_modes[int(mode)] + "    (Exit mode " + str(mode) + ')')
        print("            Final objective value                : {:.6e}".format(fx))
        print("            Final optimality                     : {:.6e}".format(out_dict['optimality']))
        print("            Final feasibility                    : {:.6e}".format(out_dict['feasibility']))
        print("            Number of major iterations           : {:d}".format(majiter))
        if hot_start:
            print(f"            Num fun evals (reused in hotstart)   : {prob.nfev:d} ({hot_nfev:.0f})")
            print(f"            Num deriv evals (reused in hotstart) : {prob.ngev:d} ({hot_ngev:.0f})")
        else:
            print("            Number of function evaluations       : {:d}".format(prob.nfev))
            print("            Number of derivative evaluations     : {:d}".format(prob.ngev))
        print("            Average Function evaluation time     : {:.6f} s per evaluation".format(prob.fev_time/prob.nfev))
        print("            Average Derivative evaluation time   : {:.6f} s per evaluation".format(prob.gev_time/prob.ngev))
        print("            Total Function evaluation time       : {:.6f} s [{:6.2f}%]".format(prob.fev_time, prob.fev_time/total_time*100))
        print("            Total Derivative evaluation time     : {:.6f} s [{:6.2f}%]".format(prob.gev_time, prob.gev_time/total_time*100))
        print("            Optimizer time                       : {:.6f} s [{:6.2f}%]".format(opt_time, opt_time/total_time*100))
        print("            Processing time                      : {:.6f} s [{:6.2f}%]".format(processing_time, processing_time/total_time*100))
        print("            Visualization time                   : {:.6f} s [{:6.2f}%]".format(vis_time, vis_time/total_time*100))
        print("            Total optimization time              : {:.6f} s [{:6.2f}%]".format(total_time, 100.00))
        print("            Summary saved to                     : " + summary_filename)
        if save_itr is not None:
            print("            Iteration data saved to              : " + save_filename)
        if visualize:
            print("            Plot saved to                        : " + save_figname)

    results = {}
    results['x'] = x
    results['objective'] = fx
    results['optimality'] = h1
    results['feasibility'] = feas_calc
    results['constraints'] = c[:m]
    results['multipliers'] = w[wref:wref+m]
    results['gradient'] = g[:-1]
    results['jacobian'] = a[:m, :-1]
    results['num_majiter'] = int(majiter)
    results['nfev'] = prob.nfev
    results['ngev'] = prob.ngev
    if hot_start:
        results['nfev_reused_in_hotstart'] = hot_nfev if hot_start else 0
        results['ngev_reused_in_hotstart'] = hot_ngev if hot_start else 0
    results['fev_time'] = prob.fev_time
    results['gev_time'] = prob.gev_time
    results['optimizer_time'] = opt_time
    results['processing_time'] = processing_time
    results['visualization_time'] = vis_time
    results['total_time'] = total_time
    results['status'] = int(mode)
    results['message'] = exit_modes[int(mode)]
    results['success'] = (mode == 0)
    results['summary_filename'] = summary_filename
    if save_itr is not None:
        results['save_filename'] = save_filename
    if visualize:
        results['plot_filename'] = save_figname

    if save_itr is not None:
        file.create_group('results')
        for key in results.keys():
            file['results'][key] = results[key]
        file.close()
    
    return results

if __name__ == "__main__":
    import doctest
    doctest.testmod()