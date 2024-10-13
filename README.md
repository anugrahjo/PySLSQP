# PySLSQP

[![GitHub Actions Test Badge](https://github.com/anugrahjo/PySLSQP/actions/workflows/build_install_test.yml/badge.svg)](https://github.com/anugrahjo/PySLSQP/actions)
[![Coverage Status](https://coveralls.io/repos/github/anugrahjo/PySLSQP/badge.svg?branch=main)](https://coveralls.io/github/anugrahjo/PySLSQP?branch=main)
[![Documentation Status](https://readthedocs.org/projects/pyslsqp/badge/?version=latest)](https://pyslsqp.readthedocs.io/en/latest/?badge=main)
[![Pypi version](https://img.shields.io/pypi/v/pyslsqp)](https://pypi.org/project/pyslsqp/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/anugrahjo/PySLSQP/blob/main/LICENSE.txt)
<!-- [![PyPI Monthly Downloads][https://img.shields.io/pypi/dm/pyslsqp]][https://pypi.org/project/pyslsqp/] -->
<!-- [![Forks](https://img.shields.io/github/forks/anugrahjo/PySLSQP.svg)](https://github.com/anugrahjo/PySLSQP/network) -->
<!-- [![Issues](https://img.shields.io/github/issues/anugrahjo/PySLSQP.svg)](https://github.com/anugrahjo/PySLSQP/issues) -->
<!-- [![Python](https://img.shields.io/pypi/pyversions/PySLSQP)](https://img.shields.io/pypi/pyversions/PySLSQP) -->
<!-- [![Pypi](https://img.shields.io/pypi/v/PySLSQP)](https://pypi.org/project/PySLSQP/) -->

The SLSQP algorithm is designed to solve nonlinear programming (NLP) problems.
PySLSQP is a Python package that wraps the original SLSQP algorithm 
implemented in Fortran by Dieter Kraft.
While the Fortran code is sourced from 
[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html), 
PySLSQP extends its functionality
by offering new features for further analysis of optimization problems, 
thereby significantly improving the utility of the original algorithm.
The prebuilt packages for various system architectures are available on PyPI for download, 
thus avoiding the need for the user to compile the Fortran sources.


Some of the additional features provided by PySLSQP include:

- **Data Saving**: PySLSQP allows you to save optimization data during the optimization process. 
  This can be useful for analyzing the progress of the optimization, for post-processing purposes, 
  or for warm/hot restart of the optimization.

- **Warm/Hot Starting**: PySLSQP supports warm/hot starting, enabling users to initialize the optimization 
  algorithm with a previously saved solution. This can be particularly useful when you want to restart an 
  optimization from a previous solution or continue an optimization that was terminated
  for various reasons.

- **Live Visualization**: PySLSQP provides the capability to visualize the optimization progress in real-time,
  enabling users to monitor the convergence via the optimality and feasibility measures.
  This also helps the users understand how the objective function, constraints, and design variables 
  are changing during the optimization.

- **Scaling**: PySLSQP allows you to independently scale the objective function, constraints, and design variables, 
  separate from their original definitions.
  Scaling can help improve the convergence behavior of the optimization algorithm and make it more robust.

- **More Internal Optimization Variables**: PySLSQP provides access to additional internal optimization variables
  such as optimality, feasibility, Lagrange multipliers, etc. which can be useful for advanced analysis 
  of the optimization problem. 
  In `scipy.optimize.minimize`, the original callback function 
  returns only the optimization variables, and only for the major iterations.

These additional features make PySLSQP a powerful tool for solving constrained optimization problems in Python.
In addition, PySLSQP also supports the estimation of gradients using first-order finite differencing, 
as in the Scipy version. 

<!-- ## Dependencies
Before installing PySLSQP, make sure you have the dependencies installed.
Numpy is the minimum requirement for using PySLSQP. 
[numpy](https://numpy.org/install/) can be installed from PyPI with
```sh
pip install numpy
```
Additionally, if you need to save optimization data and visualize different variables during the optimization,
install `h5py` and `matplotlib` respectively.
All the dependencies can be installed at once with 
```sh
pip install numpy h5py matplotlib
``` -->

## Installation

To install the latest release of PySLSQP on PyPI, run on the terminal or command line
```sh
pip install pyslsqp
```

> **Warning**
> 
> Precompiled wheels for common Ubuntu, macOS, and Windows architectures are available on PyPI.
  However, if a wheel for your system's architecture is not available,
  the above installation will compile the source distribution directly on your machine.
  In such scenarios, if your Fortran compilers aren't compatible, you may encounter compilation errors.
  Additional troubleshooting may be required to resolve these errors depending on their specifics.


To install the latest commit from the main branch, run
```sh
pip install git+https://github.com/anugrahjo/PySLSQP.git@main
```
Note that this installation method will compile the Fortran sources locally on your machine.
Therefore, we only recommend this method if you are a developer looking to modify the package for your own use case.

To upgrade PySLSQP from an older version to the latest released version on PyPI, run
```sh
pip install --upgrade pyslsqp
```

To uninstall PySLSQP, run
```sh
pip uninstall pyslsqp
```

## Testing
To test if the package works correctly and as intended, install `pytest` using
```sh
pip install pytest
```
and run the following line on the terminal from the project's root directory:
```sh
pytest -m "not visualize"
```

## Usage
Most features of the PySLSQP package can be accessed through the `optimize` function.
However, there are some additional utility functions that are available for post-processing.
Here is a small optimization example that minimizes `x^2 + y^2`.

```python
import numpy as np
from pyslsqp import optimize

# `v` represents the vector of optimization variables
def objective(v):
    # the objective function
    return v[0]**2 + v[1]**2

x0 = np.array([1., 1.])
# optimize() returns a dictionary that contains the results from optimization
results = optimize(x0, obj=objective)
print(results)
```
Note that we did not provide the gradient for the objective function above.
In the absence of user-provided gradients, `optimize` estimates the gradients
using first-order finite differencing.
However, it is always more efficient for the user to provide the exact gradients.
Note also that we did not have any constraints or variable bounds in this problem.
Examples with user-defined gradients, constraints, and bounds
can be found in the [Basic User Guide](https://pyslsqp.readthedocs.io/en/latest/src/basic.html).

## Documentation
For API reference and more details on installation and usage, visit the [documentation](https://pyslsqp.readthedocs.io/).

## Citation
If you use PySLSQP in your work, please use the following reference for citation:

```
@article{joshy2024pyslsqp,
  title={PySLSQP: A transparent Python package for the SLSQP optimization algorithm modernized with utilities for visualization and post-processing},
  author={Joshy, Anugrah Jo and Hwang, John T},
  journal={arXiv preprint},
  year={2024},
  doi= {10.48550/arXiv.2408.13420},
}
```

## Bugs, feature requests, questions
Please use the [GitHub issue tracker](https://github.com/anugrahjo/PySLSQP/issues) for reporting bugs, requesting new features, or any other questions.

## Contributing
We always welcome contributions to PySLSQP. 
Please refer the [`CONTRIBUTING.md`](https://github.com/anugrahjo/PySLSQP/blob/main/CONTRIBUTING.md) 
file for guidelines on how to contribute.

## License
This project is licensed under the terms of the [BSD 3-Clause "New" or "Revised" License](https://github.com/anugrahjo/PySLSQP/blob/main/LICENSE.txt).
