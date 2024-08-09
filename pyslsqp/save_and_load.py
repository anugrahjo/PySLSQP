import warnings
import numpy as np
try:
    import h5py
except ImportError:
    warnings.warn("h5py not found, saving and loading data disabled")
    h5py = None

def import_h5py_file(filepath):
    if h5py is None:
        raise ImportError("h5py not found, saving and loading data disabled")
    try:
        return h5py.File(filepath, 'r')
    except:
        raise FileNotFoundError(f"File {filepath} not found or not a valid h5py file.")
    
def save_iteration(file, iter, vars, out_dict):
    '''
    Save the data from one iteration to an active file.

    Parameters
    ----------
    file : str
        Loaded file.
    iter : int
        Iteration number.
    vars : list
        List of variable names to save.
    out_dict : dict
        Dictionary with variable names as keys and variable values as values.
    '''
    file.create_group('iter_' + str(iter))
    for var in vars:
        file[f'iter_{iter}'][var] = out_dict[var]
    
def print_dict_as_table(data):
    """
    Print any input dictionary as a table.

    Parameters
    ----------
    data : dict
        Dictionary to print as a table.

    Examples
    --------
    >>> data = {'a': 0, 'b': "string", 'c': ['a', 'b', 'c']}
    >>> print_dict_as_table(data)
    --------------------------------------------------
            a                        : 0
            b                        : string
            c                        : ['a', 'b', 'c']
    --------------------------------------------------
    """
    print("--------------------------------------------------")
    for key, value in data.items():
        print(f"        {key:24} : {value}")
    print("--------------------------------------------------")
    
def print_file_contents(filepath):
    '''
    Print the contents of the saved file.
    
    Parameters
    ----------
    filepath : str
        Path to the saved file.        

    Examples
    --------
    >>> import numpy as np
    >>> from pyslsqp import optimize
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> xl = 0.0
    >>> xu = np.array([1, 1])
    >>> x0 = np.array([0.5, 0.5])
    >>> results = optimize(x0, obj=obj, grad=grad, xl=xl, xu=xu, save_itr='major', save_vars=['objective', 'optimality', 'x'])  # doctest: +ELLIPSIS
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
                Iteration data saved to              : slsqp_recorder.hdf5
    >>> from pyslsqp.postprocessing import print_file_contents
    >>> print_file_contents('slsqp_recorder.hdf5')  # doctest: +NORMALIZE_WHITESPACE
    Available data in the file:
    ---------------------------
         Attributes of optimization  : ['acc', 'con_scaler', 'finite_diff_abs_step', 'finite_diff_rel_step', 
         'hot_start', 'iprint', 'keep_plot_open', 'load_filename', 'm', 'maxiter', 'meq', 'n', 'obj_scaler', 
         'save_figname', 'save_filename', 'save_itr', 'save_vars', 'summary_filename', 'visualize', 'visualize_vars', 
         'warm_start', 'x0', 'x_scaler', 'xl', 'xu']
         Saved variable iterates     : ['ismajor', 'iter', 'majiter', 'objective', 'optimality', 'x']
         Results of Optimization     : ['constraints', 'feasibility', 'fev_time', 'gev_time', 'gradient', 'jacobian', 
         'message', 'multipliers', 'nfev', 'ngev', 'num_majiter', 'objective', 'optimality', 'optimizer_time', 'processing_time', 
         'save_filename', 'status', 'success', 'summary_filename', 'total_time', 'visualization_time', 'x']

    '''
    file = import_h5py_file(filepath)
    print("Available data in the file:")
    print("---------------------------")
    try:
        print("     Attributes of optimization  :", list(file.attrs.keys()))
    except:
        warnings.warn("No attributes found in the file.")
    try:
        print("     Saved variable iterates     :", list(file['iter_0'].keys()))
    except:
        warnings.warn("No variable iterates found in the file.")
    try:
        print("     Results of Optimization     :", list(file['results'].keys()))
    except:
        warnings.warn("No results found in the file.")
    file.close()

def load_variables(filepath, vars, itr_start=0, itr_end=-1, major_only=False):
    '''
    Load specified variable iterates between ``itr_start`` and ``itr_end`` from the saved file.
    Returns a dictionary with the variable names as keys and list of variable iterates as values.
    Note the variables at ``itr_start`` and ``itr_end`` are included in the output.

    Parameters
    ----------
    filepath : str
        Path to the saved file.
    vars : str or list
        Variable names to load from the saved file.
        If only specific scalar variables are needed from an array, 
        use the format 'var_name[idx]'.
        For example, 'x[0]' will load the iterates for the first element of the array 'x', and
        'jacobian[i,j]' will load the iterates for the (i,j)-th element of the array 'jacobian'.
    itr_start : int, default=0
        Starting iteration to load the variables from.
        Negative indices are allowed with -1 representing the last iteration
        and -2 representing the second last iteration and so on.
    itr_end : int, default=-1
        Ending iteration to load the variables from. 
        Negative indices are allowed with -1 representing the last iteration
        and -2 representing the second last iteration and so on.
    major_only : bool, default=False
        If True, only major iterations are loaded.
        If False, all iterations are loaded irrespective of major or line search iterations.

    Returns
    -------
    out_data : dict
        Dictionary with variable names as keys and list of variable iterates as values.

    Examples
    --------
    >>> import numpy as np
    >>> from pyslsqp import optimize
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> xl = 0.0
    >>> xu = np.array([1, 1])
    >>> x0 = np.array([0.5, 0.5])
    >>> results = optimize(x0, obj=obj, grad=grad, xl=xl, xu=xu, save_itr='major', save_vars=['objective', 'optimality', 'x'])  # doctest: +ELLIPSIS
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
                Iteration data saved to              : slsqp_recorder.hdf5
    >>> from pyslsqp.postprocessing import load_variables
    >>> load_variables('slsqp_recorder.hdf5', ['objective', 'optimality', 'x[0]'], itr_start=0, itr_end=-1, major_only=True)
    {'objective': [0.5, 0.0, 0.0], 'optimality': [99.0, 0.0, 0.0], 'x[0]': [0.5, 0.0, 0.0]}

    '''
    if not isinstance(filepath, str):
        raise ValueError("filepath must be a string.")
    
    if not isinstance(vars, (str, list)):
        raise ValueError("vars must be a string or a list of strings")
    if isinstance(vars, str):
        vars = [vars]
    if not all(isinstance(var, str) for var in vars):
        raise ValueError("vars must be a string or a list of strings")
    
    if not isinstance(itr_start, int):
        raise ValueError("itr_start must be an integer.")
    if not isinstance(itr_end, int):
        raise ValueError("itr_end must be an integer.")
    
    file  = import_h5py_file(filepath)
    niter = len(file.keys()) - 2 # Number of iterations saved in the file [excludes 0th iteration and results]
    num_saves = len(file.keys()) - 1 # Number of iterations saved [includes 0th iteration but excludes results]
    if major_only:
        niter = file['results']['num_majiter'][()]
    if (-(niter+1) <= itr_start <= niter) and (-(niter+1) <= itr_end <= niter):
        start = itr_start * 1
        if itr_start < 0:
            start = niter + itr_start + 1
        end = itr_end * 1
        if itr_end < 0:
            end = niter + itr_end + 1
        if start > end:
            raise ValueError(f"itr_start index ({start}) must be less than itr_end index ({end}).")
    else:
        raise ValueError(f"itr_start {itr_start} and itr_end {itr_end} must be within bounds (>={-(niter+1)} and <={niter}).")
    

    out_data = {}
    for var in vars:
        if var.split('[')[0] not in file['iter_0'].keys():
            raise ValueError(f"Variable {var} not found in the file.")
        
        out_data[var] = []

    if major_only:
        for i in range(num_saves):
            if file[f'iter_{i}']['ismajor'][()] and file[f'iter_{i}']['majiter'][()] >= start:
                for var in vars:
                    if '[' in var:
                        name = var.split('[')[0]
                        if name == 'jacobian':
                            idx1, idx2 = map(int, var.split('[')[1].split(']')[0].split(','))
                            out_data[var].append(file[f'iter_{i}'][name][idx1, idx2])
                        else:
                            idx = int(var.split('[')[1].split(']')[0])
                            out_data[var].append(file[f'iter_{i}'][name][idx])
                    else:
                        out_data[var].append(file[f'iter_{i}'][var][()])
                if file[f'iter_{i}']['majiter'][()] == end:
                    break
    else:
        for i in range(start, end+1):
            for var in vars:
                if '[' in var:
                    name = var.split('[')[0]
                    if name == 'jacobian':
                        idx1, idx2 = map(int, var.split('[')[1].split(']')[0].split(','))
                        out_data[var].append(file[f'iter_{i}'][name][idx1, idx2])
                    else:
                        idx = int(var.split('[')[1].split(']')[0])
                        out_data[var].append(file[f'iter_{i}'][name][idx])
                else:
                    out_data[var].append(file[f'iter_{i}'][var][()])
    
    file.close()

    return out_data

def load_results(filepath):
    '''
    Load the results of optimization from the saved file as a dictionary.

    Parameters
    ----------
    filepath : str
        Path to the saved file.

    Returns
    -------
    out_data : dict
        Dictionary with optimization results.

    Examples
    --------
    >>> import numpy as np
    >>> from pyslsqp import optimize
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> xl = 0.0
    >>> xu = np.array([1, 1])
    >>> x0 = np.array([0.5, 0.5])
    >>> results = optimize(x0, obj=obj, grad=grad, xl=xl, xu=xu, save_itr='major', save_vars=['objective', 'optimality', 'x'])  # doctest: +ELLIPSIS
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
                Iteration data saved to              : slsqp_recorder.hdf5
    >>> from pyslsqp.postprocessing import load_results
    >>> load_results('slsqp_recorder.hdf5')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'constraints': array([], dtype=float64), 'feasibility': 0.0, 'fev_time': ..., 'gev_time': ..., 'gradient': array([0., 0.]), 
    'jacobian': array([], shape=(0, 2), dtype=float64), 'message': 'Optimization terminated successfully', 
    'multipliers': array([], dtype=float64), 'nfev': 2, 'ngev': 2, 'num_majiter': 2, 'objective': 0.0, 'optimality': 0.0, 
    'optimizer_time': ..., 'processing_time': ..., 'save_filename': 'slsqp_recorder.hdf5', 'status': 0, 'success': True, 
    'summary_filename': 'slsqp_summary.out', 'total_time': ..., 'visualization_time': 0.0, 'x': array([0., 0.])}

    '''
    file = import_h5py_file(filepath)
    result_dict = {}
    for key in file['results'].keys():
        result_dict[key] = file['results'][key][()]
        if key in ['message', 'save_filename', 'summary_filename']:
            result_dict[key] = result_dict[key].decode('utf-8')
    file.close()
    return result_dict

def load_attributes(filepath):
    '''
    Load the attributes of optimization from the saved file as a dictionary.

    Parameters
    ----------
    filepath : str
        Path to the saved file.

    Returns
    -------
    out_data : dict
        Dictionary with optimization attributes.

    Examples
    --------
    >>> import numpy as np
    >>> from pyslsqp import optimize
    >>> obj = lambda x: np.sum(x**2)
    >>> grad = lambda x: 2*x
    >>> xl = 0.0
    >>> xu = np.array([1, 1])
    >>> x0 = np.array([0.5, 0.5])
    >>> results = optimize(x0, obj=obj, grad=grad, xl=xl, xu=xu, save_itr='major', save_vars=['objective', 'optimality', 'x'])  # doctest: +ELLIPSIS
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
                Iteration data saved to              : slsqp_recorder.hdf5
    >>> from pyslsqp.postprocessing import load_attributes
    >>> load_attributes('slsqp_recorder.hdf5')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    {'acc': 1e-06, 'con_scaler': 1.0, 'finite_diff_abs_step': 'None (undefined)', 'finite_diff_rel_step': 1.4901161193847656e-08, 
    'hot_start': False, 'iprint': 1, 'keep_plot_open': False, 'load_filename': 'None (undefined)', 'm': 0, 'maxiter': 100, 'meq': 0, 'n': 2, 'obj_scaler': 1.0, 
    'save_figname': 'slsqp_plot.pdf', 'save_filename': 'slsqp_recorder.hdf5', 'save_itr': 'major', 'save_vars': ['objective', 'optimality', 'x'], 
    'summary_filename': 'slsqp_summary.out', 'visualize': False, 'visualize_vars': ['objective', 'optimality', 'feasibility'], 'warm_start': False, 
    'x0': array([0.5, 0.5]), 'x_scaler': 1.0, 'xl': 0.0, 'xu': array([1, 1])}

    '''
    file = import_h5py_file(filepath)
    attr_dict = {}
    for key in file.attrs.keys():
        attr_dict[key] = file.attrs[key]
        if key in ['save_vars', 'visualize_vars']:
            attr_dict[key] = list(file.attrs[key])
    file.close()
    return attr_dict

if __name__ == "__main__":
    import doctest
    doctest.testmod()
