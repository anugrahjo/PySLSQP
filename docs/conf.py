# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))     # for autodoc

def get_version_string(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        for line in fp:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

__version__ = get_version_string('../pyslsqp/__init__.py')

# -- Project information -----------------------------------------------------

project = 'PySLSQP'
copyright = '2024, Anugrah Jo Joshy'
author = 'Anugrah'
version = __version__
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    # "sphinx.ext.autodoc",           # for auto generating API reference
    "numpydoc",                 
    "sphinx_copybutton",            # allows copying code embedded in the docs rendered from .md or .ipynb files
    "myst_nb",                      # renders .md, .myst, .ipynb files
    "sphinx.ext.viewcode",          # adds the source code for classes and functions in auto generated api ref
    "sphinxcontrib.bibtex",         # for references and citations
]

autodoc_typehints = 'none'
autosummary_generate = True
# add_module_names = False

# sphinxcontrib.bibtex options
bibtex_bibfiles = ['src/references.bib']

# myst_options
myst_heading_anchors = 3 # To use heading anchors for all headings of level 3 and above

# myst_nb options
myst_title_to_header = True
myst_enable_extensions = ["dollarmath", "amsmath", "tasklist"]
nb_execution_mode = 'off'

# napoleon options
# napoleon_use_ivar = False
# napoleon_use_param = False

root_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['README.md', '_build', 'Thumbs.db', '.DS_Store', 'src/welcome.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme' # other theme options: 'sphinx_book_theme', 'sphinx_rtd_theme', 
                                # 'alabaster', 'classic', 'sphinxdoc', 'nature', 'bizstyle', ...

# html_theme_options for sphinx_rtd_theme
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',   # other valid colors: 'white', ...
    # toc options
    'collapse_navigation': False,   # default: True
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': True     # default: False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']