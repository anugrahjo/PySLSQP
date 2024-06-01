# Contributing to PySLSQP

Contributions are always welcome and appreciated!
This document is intended for developers who would like to contribute to PySLSQP.

## Reporting Bugs / Feature suggestions
Please use the [GitHub issue tracker](https://github.com/anugrahjo/PySLSQP_alpha/issues) 
for reporting any bugs that you come across, and suggesting new features.
Before doing so, please review the list of open issues to see if the same issue has been reported before. 
If it's already been reported, please comment on the existing issue instead of creating a new one.

If you already know the fix for a reported bug or know how to implement a suggested new feature, 
please feel free to make the necessary code changes on your fork and then submit a pull request. 
See below for more details on submitting pull requests.

## Setting up the package for local development
To contribute to PySLSQP:
1. First fork PySLSQP to create your own separate copy of the `PySLSQP` repository.
2. Clone your fork locally by running `git clone https://github.com/your_username/PySLSQP.git`.
3. Install PySLSQP in development mode by running `pip install -e .` at the project root directory.
4. Make a new branch for local development by running `git checkout -b name-of-your-bugfix-or-feature`.
5. Make the necessary changes to your code locally. Add tests, comments, docstrings, and documentation, if needed.
6. After ensuring all tests pass and the documentation builds as intended, 
   commit your changes and push your branch to GitHub:
    ```sh
    git add .
    git commit -m "A detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```
7. Submit a pull request to the main branch of the `PySLSQP` repository on the GitHub website for review and
   incorporation of your changes into the main repository.

## Pull Requests
Before contributing to the `main` branch, ensure the following requirements are met:
- [ ] All existing tests have passed
- [ ] All documentation is building correctly (if new documentation was added) 
- [ ] No local merge conflicts
- [ ] Code is commented
- [ ] Added tests for new features

Once you finish making changes on your fork, create a pull request on GitHub with the descriptions of changes.
Make sure to fill out the pull request template and assign reviewers (repo admin(s)).
<!-- To create the pull request, follow the steps:

1. Pull from `main` branch
```sh
git pull # Get most up-to-update version
```
1. Merge with main and check for conflicts
```sh
git merge main # merge with main locally on your branch to check for conflicts
```
1. Run tests
```sh
pytest # standard testing
pytest --disable-warnings # tests without displaying warnings
pytest -rP # tests while displaying print statements
```

1. Push changes

```sh
git push
```

5. Create pull request on GitHub with descriptions for changes.
 - Fill out pull request template
 - Assign reviewers (repo admin(s)) -->

### Pull Request Review
A submitted pull request will be approved after review if:
 - Requirements above are met
 - GitHub actions tests pass
 - The code runs locally on the reviewer's computer (ideally after running new cases to try and break the new code)

When rejecting a pull request, a precise description of what needs to 
improved/fixed will be provided in the comment section.

## Testing
For test-driven development, create tests before implementing code.
Create test files with names starting with `test_` in the `tests` directory and 
write test functions with `test_` prefix in the test files.
To run the tests, install `pytest` with 
```sh
pip install pytest
``` 
and run any one of the following lines
```sh
pytest                              # standard testing
pytest -m "not visualize"           # skip testing visualization since it opens multiple windows
pytest --disable-warnings           # tests without displaying warnings
pytest -rP                          # tests while displaying print statements
pytest tests/ --cov=pyslsqp         # testing while also computing line coverage
pytest --cov=pyslsqp --cov-branch   # testing while also computing branch coverage
pytest --cov=pyslsqp --cov-report html    # detailed coverage report generated at htmlcov/index.html
                                          # clicking specific files in the report shows 
                                          # which lines were missed in testing
```
on the terminal or command line at the project root directory.

## Documentation
After you have made your changes to the code and tested them successfully, make sure to add any
documentation necessary to guide users to start using the new features.
This could be done by including docstrings and comments in your new code.
Additionally, you can include new documentation pages and/or text/code examples to existing documentation
written in the `./docs/src` folder, if you have made new additions to the API.
New pages can be written using markdown or Jupyter notebooks, as per the
developer's preferences.

To build the documentation locally, you need to first install the following dependencies:
```sh
pip install sphinx myst-nb sphinx_rtd_theme sphinx-copybutton sphinx-autoapi numpydoc sphinxcontrib-bibtex
```
Once all the source code is written for your documentation, 
build the new documentation on your system by going into the `./docs` folder and running `make html`.
This will build all the html pages locally and you can verify if the documentation was built as intended by
opening the `docs/_build/html/index.html` on your web browser.

## License
By contributing to PySLSQP, you agree to make anything contributed to the repository available 
under the [BSD 3-Clause "New" or "Revised" License](https://github.com/anugrahjo/PySLSQP_alpha/blob/main/LICENSE.txt).
