# Configuration file for the Sphinx documentation builder.

import os
import sys
import warnings

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "Dynamic Causal Strength"
author = "Salar Nouri"
copyright = "2025, Salar Nouri / CMC-Lab"

# Try dynamic versioning
try:
    from dcs import __version__ as version_str
except ImportError:
    warnings.warn(
        "Could not import version from dcs.__init__.py. Falling back to '0.1.0'."
    )
    version_str = "0.1.0"

version = ".".join(version_str.split(".")[:2])
release = version_str

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_sitemap",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "myst_parser",  # Markdown support
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__init__",
    "show-inheritance": True,
}

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
html_baseurl = "https://dynamic-causal-strength.readthedocs.io"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "sticky_navigation": True,
}
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "DCSdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}
latex_documents = [
    (
        master_doc,
        "DCS.tex",
        "Dynamic Causal Strength Documentation",
        "Salar Nouri",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "dcs", "Dynamic Causal Strength Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "DCS",
        "Dynamic Causal Strength Documentation",
        author,
        "DCS",
        "Causal inference in time series.",
        "Miscellaneous",
    ),
]
