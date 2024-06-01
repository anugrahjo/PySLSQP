# Getting Started
This page provides instructions for installing PySLSQP
and running a minimal example.

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

To install the latest commit from the main branch, run
```sh
pip install git+https://github.com/anugrahjo/PySLSQP.git@main
```
<!-- Note that this will compile the package locally on your computer.
Therefore, this is  not recommended unless you are a developer and wants to edit the package for your use. -->

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

and run the following line on the terminal from the project root directory:
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
# optimize returns a dictionary that contains the results from optimization
results = optimize(x0, obj=objective)
print(results)
```
Note that we did not provide the gradient for the objective function above.
In the absence of user-provided gradients, `optimize` estimates the gradients
using first-order finite differencing.
However, it is always more efficient for the user to provide the exact gradients.
Note also that we did not have any constraints or variable bounds in this problem.
Examples with user-defined gradients, constraints, and bounds
can be found in the upcoming sections.
