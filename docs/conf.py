# docs/conf.py
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# Insert the project root directory (one level up from 'docs') into the path
# This allows Sphinx to find the 'dcs' package for autodocumentation
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Dynamic Causal Strength'
# Update copyright year and holder as needed
copyright = '2025, Salar Nouri / CMC-Lab' # Example update
author = 'Salar Nouri'

# Attempt to get the version number dynamically from your package's __init__.py
# This avoids having to update the version in multiple places.
try:
    # Assuming your main package __init__ file defines __version__
    from dcs import __version__ as version_str # [cite: uploaded:dynamic-causal-strength/dcs/__init__.py]
except ImportError:
    # Fallback if the import fails (e.g., during early setup)
    version_str = '0.1.0' # [cite: uploaded:dynamic-causal-strength/dcs/__init__.py] referenced for fallback value basis

# The short X.Y version (e.g., '0.1')
version = '.'.join(version_str.split('.')[:2])
# The full version, including alpha/beta/rc tags (e.g., '0.1.0')
release = version_str


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',      # Core library for autodoc functionality
    'sphinx.ext.napoleon',     # Parses NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other projects' documentation (Python, NumPy, etc.)
    'sphinx.ext.viewcode',     # Add links to source code in the documentation
    'sphinx.ext.githubpages', # Creates .nojekyll file for GitHub Pages deployment
    # 'sphinx.ext.mathjax',    # Uncomment if you use math notations in docstrings
    # Add other extensions here if needed, e.g., 'sphinx_copybutton'
]

# Configuration for intersphinx: specify mapping to other documentations
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    # Add mappings for other libraries you depend on and want to link to
    # 'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Configuration for Napoleon (docstring parsing)
napoleon_google_docstring = True   # Enable Google style docstrings
napoleon_numpy_docstring = True    # Enable NumPy style docstrings
napoleon_include_init_with_doc = True # Include __init__ docstring after class docstring
napoleon_include_private_with_doc = False # Do not include private members (_ membros)
napoleon_include_special_with_doc = True  # Include special members (__members__)
napoleon_use_admonition_for_examples = False # Use .. code-block:: python directive for examples
napoleon_use_ivar = False          # Use :ivar: role for instance variables
napoleon_use_param = True          # Use :param: role for parameters
napoleon_use_rtype = True          # Use :rtype: role for return types

# Configuration for Autodoc
autodoc_member_order = 'bysource'  # Order members by source code order ('alphabetical', 'groupwise')
autoclass_content = 'both'       # Include docstrings from both the class and its __init__ method
autodoc_preserve_defaults = True # Show default values in function signatures
autodoc_typehints = 'description' # Show typehints in the description, not the signature (cleaner signature)
# autodoc_mock_imports = ['heavy_dependency'] # List dependencies to mock if they cause issues during build

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index' # Corresponds to index.rst

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme' # Popular theme resembling Read the Docs

# Theme options are theme-specific and customize the look and feel of a theme
# further.
# html_theme_options = {
#     'collapse_navigation': False,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static'] # Uncomment if you add custom CSS or JS

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'DCSdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',

    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'DCS.tex', 'Dynamic Causal Strength Documentation',
     'Salar Nouri', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'dcs', 'Dynamic Causal Strength Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'DCS', 'Dynamic Causal Strength Documentation',
     author, 'DCS', 'Causal inference in time series.',
     'Miscellaneous'),
]
