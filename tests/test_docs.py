'''
This file contains tests for the code snippets in the .md and .ipynb files in the docs.
'''
import os
from testing_utils import extract_python_code_from_md, execute_python_code_snippet, check_timing, extract_python_code_from_nb
here = os.path.abspath(os.path.dirname(__file__))

def test_readme():
    from numpy.testing import assert_array_almost_equal
    
    filepath = os.path.join(here, '../README.md')
    python_code = extract_python_code_from_md(filepath)
    local_vars = execute_python_code_snippet(python_code[0])
    x0 = local_vars['x0']
    results = local_vars['results']

    assert_array_almost_equal(x0, [1., 1.], decimal=11)
    assert results['success'] == True
    assert results['status'] == 0
    assert results['message'] == 'Optimization terminated successfully'
    assert results['summary_filename'] == 'slsqp_summary.out'
    assert_array_almost_equal(results['objective'], 0.0)
    assert_array_almost_equal(results['x'], [0., 0.], decimal=11)
    assert_array_almost_equal(results['optimality'], [2.22044607e-16], decimal=11)
    assert_array_almost_equal(results['feasibility'], [0.], decimal=11)
    assert_array_almost_equal(results['constraints'], [], decimal=11)
    assert_array_almost_equal(results['multipliers'], [], decimal=11)
    assert_array_almost_equal(results['gradient'], [1.49011612e-08, 1.49011612e-08], decimal=11)
    assert results['num_majiter'] == 2
    assert results['nfev'] == 3
    assert results['ngev'] == 2

    check_timing(results)

def test_getting_started(): # same example as in README
    from numpy.testing import assert_array_almost_equal

    filepath = os.path.join(here, '../docs/src/getting_started.md')
    python_code = extract_python_code_from_md(filepath)
    local_vars = execute_python_code_snippet(python_code[0])
    x0 = local_vars['x0']
    results = local_vars['results']

    assert_array_almost_equal(x0, [1., 1.], decimal=11)
    assert results['success'] == True
    assert results['status'] == 0
    assert results['message'] == 'Optimization terminated successfully'
    assert results['summary_filename'] == 'slsqp_summary.out'
    assert_array_almost_equal(results['objective'], 0.0)
    assert_array_almost_equal(results['x'], [0., 0.], decimal=11)
    assert_array_almost_equal(results['optimality'], [2.22044607e-16], decimal=11)
    assert_array_almost_equal(results['feasibility'], [0.], decimal=11)
    assert_array_almost_equal(results['constraints'], [], decimal=11)
    assert_array_almost_equal(results['multipliers'], [], decimal=11)
    assert_array_almost_equal(results['gradient'], [1.49011612e-08, 1.49011612e-08], decimal=11)
    assert results['num_majiter'] == 2
    assert results['nfev'] == 3
    assert results['ngev'] == 2

    check_timing(results)


def test_basic_notebook():
    import numpy as np
    from numpy.testing import assert_array_almost_equal
    
    filepath = os.path.join(here, '../docs/src/basic.ipynb')
    python_code = extract_python_code_from_nb(filepath)
    
    # Use global variables since we execute multiple code snippets in the notebook
    # each of which depends on the previous ones
    global_vars = globals()

    # Defining your optimization problem
    global_vars.update(execute_python_code_snippet(python_code[0], global_vars=global_vars))

    assert_array_almost_equal(global_vars['x0'], [2, 3], decimal=11)
    assert_array_almost_equal(global_vars['x_lower'], [0.4, -np.inf], decimal=11)
    assert_array_almost_equal(global_vars['x_upper'], [np.inf, 0.6], decimal=11)
    assert global_vars['num_eqcon'] == 1

    # Solving the optimization problem
    global_vars.update(execute_python_code_snippet(python_code[1], global_vars=global_vars))
    results = global_vars['results']

    assert results['success'] == True
    check_timing(results)

    assert_array_almost_equal(results['objective'], 0.5)
    assert_array_almost_equal(results['x'], [0.5, 0.5], decimal=11)
    assert_array_almost_equal(results['optimality'], [1.232595164407831e-31], decimal=11)
    assert_array_almost_equal(results['feasibility'], [0.], decimal=11)
    assert_array_almost_equal(results['constraints'], [0., 1.5], decimal=11)
    assert_array_almost_equal(results['multipliers'], [1., 0.], decimal=11)
    assert_array_almost_equal(results['gradient'], [1., 1.], decimal=11)
    assert results['num_majiter'] == 4
    assert results['nfev'] == 4
    assert results['ngev'] == 4

    # Viewing the results
    global_vars.update(execute_python_code_snippet(python_code[2], global_vars=global_vars))

    # Scaling the problem
    global_vars.update(execute_python_code_snippet(python_code[3], global_vars=global_vars))
    assert_array_almost_equal(global_vars['x_sc'], [10., 0.1], decimal=11)
    assert_array_almost_equal(global_vars['f_sc'], 0.01, decimal=11)
    assert_array_almost_equal(global_vars['c_sc'], 2000, decimal=11)

    results = global_vars['results']
    assert results['success'] == True
    check_timing(results)

    assert_array_almost_equal(results['objective'], 0.5)
    assert_array_almost_equal(results['x'], [0.5, 0.5], decimal=11)
    assert_array_almost_equal(results['optimality'], [7.682743e-16], decimal=11)
    assert_array_almost_equal(results['feasibility'], [3.84137e-14], decimal=11)
    assert_array_almost_equal(results['constraints'], [-3.84137167e-14, 1.50000000e+00], decimal=11)
    assert_array_almost_equal(results['multipliers'], [5.e-06, 0.e+00], decimal=11)
    assert_array_almost_equal(results['gradient'], [1., 1.], decimal=11)

    assert results['num_majiter'] == 9
    assert results['nfev'] == 10
    assert results['ngev'] == 9
    assert results['summary_filename'] == 'scaled_summary.out'

    if 0: # Skip this test everytime, otherwise it will open a plot window even for `pytest -m "not visualize"`
        # Live visualization
        if os.getenv("GITHUB_ACTIONS") is None: # Skip this test on GitHub Actions since it requires a display
            global_vars.update(execute_python_code_snippet(python_code[4], global_vars=global_vars))
            results = global_vars['results']
            assert results['success'] == True
            check_timing(results, visualize=True)

            assert_array_almost_equal(results['objective'], 0.5)
            assert_array_almost_equal(results['x'], [0.5, 0.5], decimal=11)
            assert_array_almost_equal(results['optimality'], [1.232595e-31], decimal=11)
            assert_array_almost_equal(results['feasibility'], [0.], decimal=11)
            assert_array_almost_equal(results['constraints'], [0., 1.5], decimal=11)
            assert_array_almost_equal(results['multipliers'], [1., 0.], decimal=11)
            assert_array_almost_equal(results['gradient'], [1., 1.], decimal=11)

            assert results['num_majiter'] == 4
            assert results['nfev'] == 4
            assert results['ngev'] == 4
            assert results['summary_filename'] == 'visualized_summary.out'

    # Writing optimization data to a file
    global_vars.update(execute_python_code_snippet(python_code[5], global_vars=global_vars))
    results = global_vars['results']
    assert results['success'] == True
    check_timing(results)

    assert_array_almost_equal(results['objective'], 0.5)
    assert_array_almost_equal(results['x'], [0.5, 0.5], decimal=11)
    assert_array_almost_equal(results['optimality'], [1.232595e-31], decimal=11)
    assert_array_almost_equal(results['feasibility'], [0.], decimal=11)
    assert_array_almost_equal(results['constraints'], [0., 1.5], decimal=11)
    assert_array_almost_equal(results['multipliers'], [1., 0.], decimal=11)
    assert_array_almost_equal(results['gradient'], [1., 1.], decimal=11)

    assert results['num_majiter'] == 4
    assert results['nfev'] == 4
    assert results['ngev'] == 4
    assert results['summary_filename'] == 'slsqp_summary.out'
    assert results['save_filename'] == 'save_file.hdf5'

    # Other optimizer options
    global_vars.update(execute_python_code_snippet(python_code[6], global_vars=global_vars))

def test_postprocessing_notebook():
    import numpy as np
    from numpy.testing import assert_array_almost_equal
    
    filepath = os.path.join(here, '../docs/src/postprocessing.ipynb')
    python_code = extract_python_code_from_nb(filepath)
    
    # Use global variables since we execute multiple code snippets in the notebook
    # each of which depends on the previous ones
    global_vars = globals()

    # Saving a file
    global_vars.update(execute_python_code_snippet(python_code[0], global_vars=global_vars))

    assert_array_almost_equal(global_vars['x0'], [2, 3], decimal=11)
    assert_array_almost_equal(global_vars['x_lower'], [0.4, -np.inf], decimal=11)
    assert_array_almost_equal(global_vars['x_upper'], [np.inf, 0.6], decimal=11)
    assert global_vars['num_eqcon'] == 1

    results = global_vars['results']
    assert results['success'] == True
    check_timing(results)

    assert_array_almost_equal(results['objective'], 0.5)
    assert_array_almost_equal(results['x'], [0.5, 0.5], decimal=11)
    assert_array_almost_equal(results['optimality'], [1.232595164407831e-31], decimal=11)
    assert_array_almost_equal(results['feasibility'], [0.], decimal=11)
    assert_array_almost_equal(results['constraints'], [0., 1.5], decimal=11)
    assert_array_almost_equal(results['multipliers'], [1., 0.], decimal=11)
    assert_array_almost_equal(results['gradient'], [1., 1.], decimal=11)
    assert results['num_majiter'] == 4
    assert results['nfev'] == 4
    assert results['ngev'] == 4
    assert results['summary_filename'] == 'slsqp_summary.out'
    assert results['save_filename'] == 'postprocessing.hdf5'

    # Viewing saved file contents
    global_vars.update(execute_python_code_snippet(python_code[1], global_vars=global_vars))

    # Loading results and attributes
    global_vars.update(execute_python_code_snippet(python_code[2], global_vars=global_vars))
    
    # Loading variable iterates
    global_vars.update(execute_python_code_snippet(python_code[3], global_vars=global_vars))

    # Loading variable iterates with major iterations only
    global_vars.update(execute_python_code_snippet(python_code[4], global_vars=global_vars))

    if 0: # Skip this test everytime, otherwise it will open a plot window even for `pytest -m "not visualize"`
        # Visualizing saved optimization
        if os.getenv("GITHUB_ACTIONS") is None: # Skip this test on GitHub Actions since it requires a display
            global_vars.update(execute_python_code_snippet(python_code[5], global_vars=global_vars))

    # Warm start
    global_vars.update(execute_python_code_snippet(python_code[6], global_vars=global_vars))
    results = global_vars['results']
    assert results['success'] == True
    check_timing(results)

    assert_array_almost_equal(results['objective'], 0.5)
    assert_array_almost_equal(results['x'], [0.5, 0.5], decimal=11)
    assert_array_almost_equal(results['optimality'], [2.465190e-31], decimal=11)
    assert_array_almost_equal(results['feasibility'], [0.], decimal=11)
    assert_array_almost_equal(results['constraints'], [0., 1.5], decimal=11)
    assert_array_almost_equal(results['multipliers'], [1., 0.], decimal=11)
    assert_array_almost_equal(results['gradient'], [1., 1.], decimal=11)

    assert results['num_majiter'] == 1
    assert results['nfev'] == 1
    assert results['ngev'] == 1
    assert results['summary_filename'] == 'slsqp_summary.out'

    # Hot start
    global_vars.update(execute_python_code_snippet(python_code[7], global_vars=global_vars))
    results = global_vars['results']
    assert results['success'] == True
    check_timing(results)

    assert_array_almost_equal(results['objective'], 0.5)
    assert_array_almost_equal(results['x'], [0.5, 0.5], decimal=11)
    assert_array_almost_equal(results['optimality'], [1.232595164407831e-31], decimal=11)
    assert_array_almost_equal(results['feasibility'], [0.], decimal=11)
    assert_array_almost_equal(results['constraints'], [0., 1.5], decimal=11)
    assert_array_almost_equal(results['multipliers'], [1., 0.], decimal=11)
    assert_array_almost_equal(results['gradient'], [1., 1.], decimal=11)

    assert results['num_majiter'] == 4
    assert results['nfev'] == 4
    assert results['ngev'] == 4
    assert results['nfev_reused_in_hotstart'] == 4
    assert results['ngev_reused_in_hotstart'] == 4
    assert results['summary_filename'] == 'slsqp_summary.out'

if __name__ == '__main__':
    test_readme()
    test_getting_started()
    test_basic_notebook()
    test_postprocessing_notebook()