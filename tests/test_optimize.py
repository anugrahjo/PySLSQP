'''
This script tests the optimize function in the pyslsqp module.
'''

def test_get_default_options():
    import numpy as np
    from pyslsqp import get_default_options
    options = get_default_options()
    
    assert options == {
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
                        'finite_diff_rel_step': np.sqrt(np.finfo(float).eps),
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
    
def test_optimize():
    import numpy as np
    from pyslsqp import optimize
    from numpy.testing import assert_almost_equal
    from testing_utils import check_timing

    def obj(x):
        return np.sum(x**2)
    def grad(x):
        return 2*x

    x0 = np.ones(10)

    # Unconstrained problem and summary file
    res1 = optimize(x0, obj, grad=grad, summary_filename='uncon_slsqp.out')
    res2 = optimize(x0, obj, summary_filename='uncon_fd_slsqp.out')

    assert res1['success'] == True
    assert res2['success'] == True
    assert res1['summary_filename'] == 'uncon_slsqp.out'
    assert res2['summary_filename'] == 'uncon_fd_slsqp.out'
    assert res1['x'].shape == (10,)
    assert res2['x'].shape == (10,)
    assert res1['gradient'].shape == (10,)
    assert res2['gradient'].shape == (10,)
    assert res1['num_majiter'] == 2
    assert res2['num_majiter'] == 2
    assert res1['nfev'] == 3
    assert res2['nfev'] == 3
    assert res1['ngev'] == 2
    assert res2['ngev'] == 2

    check_timing(res1)
    check_timing(res2)

    assert_almost_equal(res1['objective'], 0., decimal=11)
    assert_almost_equal(res1['optimality'], 0., decimal=11)
    assert_almost_equal(res1['feasibility'], 0., decimal=11)
    assert_almost_equal(res1['x'], np.zeros(10), decimal=11)
    assert_almost_equal(res1['gradient'], np.zeros(10), decimal=11)

    assert_almost_equal(res2['objective'], 0., decimal=11)
    assert_almost_equal(res2['optimality'], 0., decimal=11)
    assert_almost_equal(res2['feasibility'], 0., decimal=11)
    assert_almost_equal(res2['x'], np.zeros(10), decimal=11)
    assert_almost_equal(res2['gradient'], np.ones(10)*1.5e-8, decimal=7)

    # Feasibility problem with scalar scaling, and scalar bounds
    def con(x):
        return x[:4] - 0.4
    def jac(x):
        return np.eye(4, 10)
    
    x0 = -np.ones(10)*100
    res3 = optimize(x0, con=con, jac=jac, xl=0.2, xu =0.9, 
                    summary_filename='feas_slsqp.out', x_scaler=0.02, obj_scaler=100.)
    res4 = optimize(x0, con=con, xl=0.2, xu =0.9, 
                    summary_filename='feas_fd_slsqp.out', x_scaler=0.02, obj_scaler=100.)

    assert res3['success'] == True
    assert res4['success'] == True
    assert res3['summary_filename'] == 'feas_slsqp.out'
    assert res4['summary_filename'] == 'feas_fd_slsqp.out'
    
    assert_almost_equal(res3['objective'], 0.0, decimal=11)
    assert_almost_equal(res3['optimality'], 0.0, decimal=11)
    assert_almost_equal(res3['feasibility'], 0.0, decimal=11)
    assert_almost_equal(res3['x'], [0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=11)

    assert_almost_equal(res4['objective'], 0.0, decimal=11)
    assert_almost_equal(res4['optimality'], 0.0, decimal=11)
    assert_almost_equal(res4['feasibility'], 0.0, decimal=11)
    assert_almost_equal(res4['x'], [0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=11)


    x0 = np.ones(10)

    # Bound-constrained problem with vector bounds and vector scaling
    res5 = optimize(x0, obj, grad=grad, xl=0.2*np.ones(10), xu =0.9*np.ones(10), 
                    summary_filename='bdcon_slsqp.out', x_scaler=0.02*np.ones(10), obj_scaler=100.)
    res6 = optimize(x0, obj, xl=0.2*np.ones(10), xu =0.9*np.ones(10), 
                    summary_filename='bdcon_fd_slsqp.out', x_scaler=0.02*np.ones(10), obj_scaler=100.)

    assert res5['success'] == True
    assert res6['success'] == True
    
    assert_almost_equal(res5['objective'], 0.4, decimal=7)
    assert_almost_equal(res5['optimality'], 4e-9, decimal=7)
    assert_almost_equal(res5['feasibility'], 0.0, decimal=11)
    assert_almost_equal(res5['x'], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=7)

    assert_almost_equal(res6['objective'], 0.4, decimal=7)
    assert_almost_equal(res6['optimality'], 0.0, decimal=7)
    assert_almost_equal(res6['feasibility'], 0.0, decimal=11)
    assert_almost_equal(res6['x'], [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=7)

    # Inequality constrained problem with higher accuracy, saving variables and vector/scalar scaling    
    res7 = optimize(x0, obj, grad=grad, con=con, jac=jac, acc=1.0E-10,
                    summary_filename='ineq_slsqp.out', 
                    save_filename='ineq_slsqp.hdf5', save_itr='major', save_vars=['x', 'objective', 'majiter'], 
                    x_scaler=0.02, obj_scaler=100., con_scaler=0.02)
    res8 = optimize(x0, obj, con=con, acc=1.0E-10,
                    summary_filename='ineq_fd_slsqp.out', 
                    save_filename='ineq_fd_slsqp.hdf5', save_itr='major', save_vars=['x', 'objective', 'majiter'], 
                    x_scaler=0.02*np.ones(10), obj_scaler=100., con_scaler=0.02*np.ones(4))
    
    assert res7['success'] == True
    assert res8['success'] == True
    assert res7['save_filename'] == 'ineq_slsqp.hdf5'
    assert res8['save_filename'] == 'ineq_fd_slsqp.hdf5'

    assert_almost_equal(res7['objective'], 0.64, decimal=10)
    assert_almost_equal(res7['optimality'], 0.0, decimal=10)
    assert_almost_equal(res7['feasibility'], 0.0, decimal=10)
    assert_almost_equal(res7['x'], [0.4, 0.4, 0.4, 0.4, 0., 0., 0., 0., 0., 0.], decimal=7)

    assert_almost_equal(res8['objective'], 0.64, decimal=10)
    assert_almost_equal(res8['optimality'], 0.0, decimal=10)
    assert_almost_equal(res8['feasibility'], 0.0, decimal=10)
    assert_almost_equal(res8['x'], [0.4, 0.4, 0.4, 0.4, 0., 0., 0., 0., 0., 0.], decimal=7)


    # Equality constrained problem
    def coneq(x):
        return x[:5] - 0.5
    def jaceq(x):
        return np.eye(5, 10)
    
    res9 = optimize(x0, obj, grad=grad, con=coneq, jac=jaceq, meq=5, 
                    summary_filename='eq_slsqp.out', save_itr='all', save_filename='eq_slsqp.hdf5', 
                    x_scaler=0.02, obj_scaler=100., con_scaler=0.02)
    res10 = optimize(x0, obj, con=coneq, meq=5, 
                     summary_filename='eq_fd_slsqp.out', save_itr='all', save_filename='eq_fd_slsqp.hdf5', 
                     x_scaler=0.02*np.ones(10,), obj_scaler=100., con_scaler=0.02*np.ones(5))
    
    assert res9['success'] == True
    assert res10['success'] == True

    assert_almost_equal(res9['objective'], 1.25, decimal=10)
    assert_almost_equal(res9['optimality'], 1e-8, decimal=7)
    assert_almost_equal(res9['feasibility'], 0.0, decimal=10)
    assert_almost_equal(res9['x'], [0.5, 0.5, 0.5, 0.5, 0.5, 0., 0., 0., 0., 0.], decimal=6)

    assert_almost_equal(res10['objective'], 1.25, decimal=10)
    assert_almost_equal(res10['optimality'], 0.0, decimal=7)
    assert_almost_equal(res10['feasibility'], 0.0, decimal=10)
    assert_almost_equal(res10['x'], [0.5, 0.5, 0.5, 0.5, 0.5, 0., 0., 0., 0., 0.], decimal=6)

    # Equality and inequality constrained problem with saving, vector/scalar scaling and bounds
    def coneqineq(x):
        return np.array([x[0] - 1., x[1] - 3., x[2] - 5.])
    
    def jaceqineq(x):
        return np.eye(3, 10)
    
    res11 = optimize(x0, obj, grad=grad, con=coneqineq, jac=jaceqineq, meq=2, xl=0.2, xu=10., acc=1.0E-6,
                    summary_filename='eqineq_slsqp.out', save_itr='all', save_filename='eqineq_slsqp.hdf5', 
                    x_scaler=0.02, obj_scaler=100., con_scaler=0.02)
    res12 = optimize(x0, obj, con=coneqineq, meq=2, xl=0.2*np.ones(10), xu=10*np.ones(10), acc=1.0E-6,
                    summary_filename='eqineq_fd_slsqp.out', save_itr='all', save_filename='eqineq_fd_slsqp.hdf5', 
                    x_scaler=0.02*np.ones(10), obj_scaler=100., con_scaler=0.02*np.ones(3))
    
    assert res11['success'] == True
    assert res12['success'] == True

    assert_almost_equal(res11['objective'], 35.28, decimal=2)
    assert_almost_equal(res11['optimality'], 0.0, decimal=10)
    assert_almost_equal(res11['feasibility'], 0.0, decimal=3)
    assert_almost_equal(res11['x'], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

    assert_almost_equal(res12['objective'], 35.28, decimal=2)
    assert_almost_equal(res12['optimality'], 0.0, decimal=10)
    assert_almost_equal(res12['feasibility'], 0.0, decimal=7)
    assert_almost_equal(res12['x'], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

import pytest

@pytest.mark.visualize
def test_visualize():
    import os
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

    if os.getenv("GITHUB_ACTIONS") is None: # Skip this test on GitHub Actions since it requires a display
        # This example discovered a bug in SLSQP: majiter jumps from 4 to 9 after 4th major iteration
        # NOTE: <IMPORTANT> This test should not be modified since it exposes a bug in SLSQP, explicit bug fix in pySLSQP.
        # Equality and inequality constrained problem with saving, vector/scalar scaling and bounds
        res1 = optimize(x0, obj, grad=grad, con=coneqineq, jac=jaceqineq, meq=2, xl=0.2, xu=10., acc=1.0E-6,
                        summary_filename='vis_eqineq_slsqp.out', save_itr='all', save_filename='vis_eqineq_slsqp.hdf5', 
                        x_scaler=0.02, obj_scaler=100., con_scaler=0.02, keep_plot_open=True,
                        visualize=True, visualize_vars=['objective', 'optimality', 'feasibility', 'x[0]', 'constraints[0]', 'gradient[0]', 'multipliers[0]', 'multipliers[2]', 'jacobian[0,0]'])
        res2 = optimize(x0, obj, con=coneqineq, meq=2, xl=0.2*np.ones(10), xu=10*np.ones(10), acc=1.0E-6,
                        summary_filename='vis_eqineq_fd_slsqp.out', save_itr='all', save_filename='vis_eqineq_fd_slsqp.hdf5', 
                        x_scaler=0.02*np.ones(10), obj_scaler=100., con_scaler=0.02*np.ones(3), keep_plot_open=True,
                        visualize=True, visualize_vars=['objective', 'optimality', 'feasibility', 'x[0]', 'constraints[0]', 'gradient[0]', 'multipliers[0]', 'multipliers[2]', 'jacobian[0,0]'])
        
        assert res1['success'] == True
        assert res2['success'] == True

        assert_almost_equal(res1['objective'], 35.28, decimal=2)
        assert_almost_equal(res1['optimality'], 0.0, decimal=10)
        assert_almost_equal(res1['feasibility'], 0.0, decimal=3)
        assert_almost_equal(res1['x'], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

        assert_almost_equal(res2['objective'], 35.28, decimal=2)
        assert_almost_equal(res2['optimality'], 0.0, decimal=10)
        assert_almost_equal(res2['feasibility'], 0.0, decimal=7)
        assert_almost_equal(res2['x'], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

def test_warm_and_hot_start():
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
                    summary_filename='save_eqineq_slsqp.out', save_itr='all', save_filename='save_eqineq_slsqp.hdf5',
                    save_vars=['x', 'objective', 'constraints', 'gradient', 'jacobian'],
                    x_scaler=0.02, obj_scaler=100., con_scaler=0.02)
    
    assert res1['success'] == True
    assert_almost_equal(res1['objective'], 35.28, decimal=2)
    assert_almost_equal(res1['optimality'], 0.0, decimal=10)
    assert_almost_equal(res1['feasibility'], 0.0, decimal=3)
    assert_almost_equal(res1['x'], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

    assert res1['num_majiter'] == 5
    assert res1['nfev'] == 6
    assert res1['ngev'] == 5
    
    # Warm start
    res2 = optimize(x0, obj, grad=grad, con=coneqineq, jac=jaceqineq, meq=2, xl=0.2, xu=10., acc=1.0E-6,
                    summary_filename='warm_eqineq_slsqp.out', warm_start=True, load_filename='save_eqineq_slsqp.hdf5', 
                    save_itr='all', save_filename='save_eqineq_slsqp.hdf5', x_scaler=0.02, obj_scaler=100., con_scaler=0.02)
    
    assert res2['success'] == True
    assert res2['save_filename'] == 'save_eqineq_slsqp_warm.hdf5'
    assert_almost_equal(res2['objective'], 35.28, decimal=2)
    assert_almost_equal(res2['optimality'], 0.0, decimal=10)
    assert_almost_equal(res2['feasibility'], 0.0, decimal=3)
    assert_almost_equal(res2['x'], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

    assert res2['num_majiter'] == 1
    assert res2['nfev'] == 1
    assert res2['ngev'] == 1

    # Hot start
    res3 = optimize(x0, obj, grad=grad, con=coneqineq, jac=jaceqineq, meq=2, xl=0.2, xu=10., acc=1.0E-6,
                    summary_filename='hot_eqineq_slsqp.out', hot_start=True, load_filename='save_eqineq_slsqp.hdf5',
                    save_itr='all', save_filename='hot_eqineq_slsqp.hdf5', x_scaler=0.02, obj_scaler=100., con_scaler=0.02)
    
    assert res3['success'] == True
    assert res3['save_filename'] == 'hot_eqineq_slsqp.hdf5'
    assert_almost_equal(res3['objective'], 35.28, decimal=2)
    assert_almost_equal(res3['optimality'], 0.0, decimal=10)
    assert_almost_equal(res3['feasibility'], 0.0, decimal=3)
    assert_almost_equal(res3['x'], [1., 3., 5., 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], decimal=3)

    assert res3['num_majiter'] == 5
    assert res3['nfev'] == 6
    assert res3['ngev'] == 5
    assert res3['nfev_reused_in_hotstart'] == 6
    assert res3['ngev_reused_in_hotstart'] == 5

    
if __name__ == "__main__":
    test_optimize()
    test_visualize()
    test_get_default_options()
    test_warm_and_hot_start()