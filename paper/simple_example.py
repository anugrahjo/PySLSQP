import numpy as np

def objective(x):
    return x[0]**2 + x[1]**2

def constraints(x):
    return  np.array([x[0] + x[1] - 1, 3*x[0] + 2*x[1] - 1])

def jacobian(x):
    return np.array([[1, 1], [3, 2]])

# Variable bounds
x_lower = np.array([0.4, -np.inf])
x_upper = np.array([np.inf, 0.6])

# Number of equality constraints
m_eq = 1

# Initial guess
x0 = np.array([2,3])

# Scaling factors
x_s = 10.0
o_s =  2.0
c_s = np.array([1., 0.5])

from pyslsqp import optimize

results = optimize(x0, obj=objective, con=constraints, jac=jacobian, 
                   meq=m_eq, xl=x_lower, xu=x_upper, finite_diff_abs_step=1e-6,
                   x_scaler=x_s, obj_scaler=o_s, con_scaler=c_s,
                   save_itr='major', save_vars=['majiter', 'x', 'objective'],
                   save_filename="save_file.hdf5",
                   visualize=True, visualize_vars=['objective', 'x[0]'])

# Print the returned results dictionary
print(results)