# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../muffin'))
sys.path.insert(0, os.path.abspath('../examples'))
print(sys.path)
project = 'MUFFIN'
copyright = '2023, Pierre de Langen'
author = 'Pierre de Langen'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [ "sphinx.ext.autodoc",
                "sphinx.ext.intersphinx",
                "sphinx.ext.mathjax",
                "sphinx.ext.viewcode",
                'sphinx.ext.autosectionlabel',
                'sphinx.ext.napoleon',
                'nbsphinx'
                ]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = [".rst", ".md"]

html_css_files = [
    'css/custom.css',  # replace with the actual path to your .css file relative to the _static directory
]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
