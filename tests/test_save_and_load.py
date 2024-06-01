'''
This script tests the save_and_load module.
'''
def test_save_and_load():
    import numpy as np
    from numpy.testing import assert_almost_equal
    from pyslsqp import optimize
    
    # This is the same example as res11 and res12 in test_optimize()

    def obj(x):
        return np.sum(x**2)
    def grad(x):
        return 2*x
      
    def coneqineq(x):
        return np.array([x[0] - 1., x[1] - 3., x[2] - 5.])
    def jaceqineq(x):
        return np.eye(3, 10)
    
    x0 = np.ones(10)

    # Equality and inequality constrained problem with saving, vector/scalar scaling and bounds
    res1 = optimize(x0, obj, grad=grad, con=coneqineq, jac=jaceqineq, meq=2, xl=0.2, xu=10., acc=1.0E-6,
                    summary_filename='save_load_slsqp.out', save_itr='all', save_filename='save_load_slsqp.hdf5',
                    save_vars=['iter', 'majiter', 'mode', 'x', 'objective', 'constraints', 'gradient', 'jacobian'],
                    x_scaler=0.02, obj_scaler=100., con_scaler=0.02)
    

    from pyslsqp.save_and_load import save_iteration, load_variables, \
                                      load_attributes, load_results, \
                                      print_dict_as_table, print_file_contents
    # Print file contents
    print_file_contents('save_load_slsqp.hdf5')
    
    # Load all iterations of all variables in the input list
    vars = load_variables('save_load_slsqp.hdf5', ['iter', 'majiter', 'mode', 'x', 'objective'])
    assert_almost_equal(vars['iter'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert_almost_equal(vars['majiter'], [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5])
    assert_almost_equal(vars['mode'], [0, 1, 1, -1, 1, -1, 1, -1, 1, -1, 0])
    assert_almost_equal(vars['objective'][0], 10, decimal=11)
    assert_almost_equal(vars['objective'][-1], 35.28, decimal=2)
    assert_almost_equal(vars['x'][0], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    assert_almost_equal(vars['x'][-1], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

    # Print dict as table
    print_dict_as_table(vars)

    # Load only the last 2 iterations of all variables in the input list
    vars = load_variables('save_load_slsqp.hdf5', ['iter', 'majiter', 'mode', 'x', 'objective'], itr_start=-2, itr_end=-1)
    assert_almost_equal(vars['iter'], [9, 10])
    assert_almost_equal(vars['majiter'], [4, 5])
    assert_almost_equal(vars['mode'], [-1, 0])
    assert_almost_equal(vars['objective'][-1], 35.28, decimal=2)
    assert_almost_equal(vars['x'][-1], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

    # Load only the major iterations of all variables in the input list
    vars = load_variables('save_load_slsqp.hdf5', ['iter', 'majiter', 'mode', 'x', 'objective'], major_only=True)
    assert_almost_equal(vars['iter'], [0, 1, 4, 6, 8, 10])
    assert_almost_equal(vars['majiter'], [0, 1, 2, 3, 4, 5])
    assert_almost_equal(vars['mode'], [0, 1, 1, 1, 1, 0])
    assert_almost_equal(vars['objective'][0], 10, decimal=11)
    assert_almost_equal(vars['objective'][-1], 35.28, decimal=2)
    assert_almost_equal(vars['x'][0], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    assert_almost_equal(vars['x'][-1], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

    # Load only the last 2 major iterations of all variables in the input list
    vars = load_variables('save_load_slsqp.hdf5', ['iter', 'majiter', 'mode', 'x', 'objective'], itr_start=-2, itr_end=-1, major_only=True)
    assert_almost_equal(vars['iter'], [8, 10])
    assert_almost_equal(vars['majiter'], [4, 5])
    assert_almost_equal(vars['mode'], [1, 0])
    assert_almost_equal(vars['objective'][-1], 35.28, decimal=2)
    assert_almost_equal(vars['x'][-1], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

    # Load all attributes
    attrs = load_attributes('save_load_slsqp.hdf5')
    assert attrs['n'] == 10
    assert attrs['m'] == 3
    assert attrs['meq'] == 2

    assert attrs['maxiter'] == 100
    assert attrs['acc'] == 1.0E-6
    assert attrs['iprint'] == 1

    assert attrs['finite_diff_abs_step'] == 'None (undefined)'
    assert attrs['finite_diff_rel_step'] == np.sqrt(np.finfo(float).eps)

    assert attrs['summary_filename'] == 'save_load_slsqp.out'
    assert attrs['save_itr'] == 'all'
    assert attrs['save_filename'] == 'save_load_slsqp.hdf5'
    assert set(attrs['save_vars']) == set(['iter', 'majiter', 'mode', 'x', 'objective', 'constraints', 'gradient', 'jacobian'])
    assert attrs['warm_start'] == False
    assert attrs['hot_start'] == False
    assert attrs['load_filename'] == 'None (undefined)'
    assert attrs['visualize'] == False
    assert set(attrs['visualize_vars']) == set(['objective', 'optimality', 'feasibility'])
    assert attrs['keep_plot_open'] == False
    assert attrs['save_figname'] == 'slsqp_plot.pdf'
    
    assert_almost_equal(attrs['x0'], np.ones(10))
    assert_almost_equal(attrs['xl'], 0.2*np.ones(10))
    assert_almost_equal(attrs['xu'], 10*np.ones(10))

    assert_almost_equal(attrs['x_scaler'], 0.02)
    assert_almost_equal(attrs['obj_scaler'], 100.)
    assert_almost_equal(attrs['con_scaler'], 0.02)

    # Load all results
    results = load_results('save_load_slsqp.hdf5')
    assert_almost_equal(results['x'], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)
    assert_almost_equal(results['objective'], 35.28, decimal=2)
    assert_almost_equal(results['optimality'], 0., decimal=3)
    assert_almost_equal(results['feasibility'], 0., decimal=3)
    assert_almost_equal(results['constraints'], [0., 0., 0.], decimal=3)
    assert_almost_equal(results['gradient'], [2., 6., 10., 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], decimal=3)
    assert_almost_equal(results['jacobian'], np.eye(3, 10), decimal=11)

    assert results['num_majiter'] == 5
    assert results['nfev'] == 6
    assert results['ngev'] == 5

    assert results['status'] == 0
    assert results['message'] == 'Optimization terminated successfully'
    assert results['success'] == True

    assert results['summary_filename'] == 'save_load_slsqp.out'
    assert results['save_filename'] == 'save_load_slsqp.hdf5'

if __name__ == '__main__':
    test_save_and_load()
