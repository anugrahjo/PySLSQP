# Contributing to PySLSQP

Contributions are always welcome and appreciated!
This document is intended for developers who would like to contribute to PySLSQP.

## Reporting Bugs / Feature suggestions
Please use the [GitHub issue tracker](https://github.com/anugrahjo/PySLSQP/issues) 
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
3. Ideally, create a virtual Python environment (`python3 -m venv pyslsqp_env`), 
   activate it (`source pyslsqp_env/bin/activate` or `.\pyslsqp_env\Scripts\activate` for Windows),
   and perform the remaining steps within the created virtual environment.
4. Install development dependencies by running `pip install -r requirements.txt`.
5. Install PySLSQP in development mode by running `pip install -e .` in the project's root directory.
   For a successful installation, ensure your Fortran compilers are setup correctly.
6. Make a new branch for local development by running `git checkout -b name-of-your-bugfix-or-feature`.
7. Make the necessary changes to your code locally. Add tests, comments, docstrings, and documentation, if needed.
8.  After ensuring all tests pass and the documentation builds as intended, 
   commit your changes and push your branch to GitHub:
    ```sh
    git add .
    git commit -m "A detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```
9.  Submit a pull request to the main branch of the `PySLSQP` repository on the GitHub website for review and
   incorporation of your changes into the main repository.

## Pull Requests
Before contributing to the `main` branch, ensure the following requirements are met:
- [ ] All existing tests have passed
- [ ] All documentation is building correctly (if new documentation was added) 
- [ ] No local merge conflicts
- [ ] Code is commented
- [ ] Added tests for new features

Once you finish making changes on your fork, create a pull request on GitHub with 
a description of the changes.
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

When rejecting a pull request, a precise description of what needs to be
improved/fixed will be provided in the comment section.

## Testing
For test-driven development, create tests before implementing code.
Create test files with names starting with `test_` in the `tests` directory and 
write test functions with the `test_` prefix in the test files.
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
After making changes to the code and testing them successfully, 
ensure you add any necessary documentation to guide users in using the new features.
This can be done by including docstrings and comments in your new code.
Additionally, if you have made new additions to the API, you can create new documentation pages 
and/or add text or code examples to existing documentation in the `./docs/src` folder.
New pages can be written using Markdown or Jupyter notebooks, according to the
developer's preference.

Before building the documentation locally, ensure that you have installed all the development dependencies
as stated in step 4 by using `pip install -r requirements.txt`.
Once you have written all the source code for your documentation, 
build the new documentation on your system by navigating to the `./docs` folder and running `make html`.
This will build all the HTML pages locally, and you can verify if the documentation was built as intended by
opening `docs/_build/html/index.html` on your web browser.

## Versioning
PySLSQP follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html), summarized below:

```{code-block} none
Given a version number MAJOR.MINOR.PATCH, increment the:

 1. MAJOR version when you make incompatible API changes.
 2. MINOR version when you add functionality in a backward compatible manner.
 3. PATCH version when you make backward compatible bug fixes.

 Additional labels for pre-release and build metadata can be included as extensions 
 to the MAJOR.MINOR.PATCH format.

 1. A pre-release version MAY be denoted by appending a hyphen and a series of 
    dot separated identifiers immediately following the patch version, 
    e.g., 1.0.0-alpha.1 .
 2. Build metadata MAY be denoted by appending a plus sign and a series of 
    dot separated identifiers immediately following the patch or pre-release version, 
    e.g., 1.0.0-alpha.1+001 .
```

## License
By contributing to PySLSQP, you agree to make anything contributed to the repository available 
under the [BSD 3-Clause "New" or "Revised" License](https://github.com/anugrahjo/PySLSQP/blob/main/LICENSE.txt).
