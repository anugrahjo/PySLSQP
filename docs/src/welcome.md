# Welcome to PySLSQP

The SLSQP algorithm is designed to solve nonlinear programming (NLP) problems.
PySLSQP is a Python package that wraps the original SLSQP algorithm 
implemented in Fortran by Dieter Kraft {cite:p}`kraft1988software, kraft1994algorithm`.
While the Fortran code in PySLSQP is sourced from 
[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html), 
{cite:p}`virtanen2020scipy`, 
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
  such as optimality, feasibility, Lagrange multipliers, etc. at every iteration, 
  which can be useful for advanced analysis of the optimization problem. 
  In `scipy.optimize.minimize`, the original callback function 
  returns only the optimization variables, and only for the major iterations.

These additional features make PySLSQP a powerful tool for solving constrained optimization problems in Python.
In addition, PySLSQP also supports the estimation of gradients using first-order finite differencing, 
as in the Scipy version. 

## Getting Started
To install and start using PySLSQP, please read the [Getting Started](src/getting_started.md) page.

## Citation
If you use PySLSQP in your work, please use the following reference for citation:

```
@Misc{pyslsqp,
author = "Anugrah Jo Joshy",
title = "PySLSQP",
howpublished = "\url{https://github.com/anugrahjo/PySLSQP}",
year = "2024",
}
```

<!-- ## Bugs, feature requests, questions
Please use the [GitHub issue tracker](https://github.com/anugrahjo/PySLSQP_alpha/issues) for reporting bugs, requesting new features, or any other questions.

## Contributing
We always welcome contributions to PySLSQP. 
Please refer the [`CONTRIBUTING.md`](https://github.com/anugrahjo/PySLSQP_alpha/blob/main/CONTRIBUTING.md) 
file for guidelines on how to contribute.

## License
This project is licensed under the terms of the **BSD license**. -->


## Contents

```{toctree}
:maxdepth: 2

src/getting_started
src/basic
src/postprocessing
src/api
src/contributing
src/changelog
src/license
```

## References

```{bibliography} src/references.bib
```