# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# sys.path.insert(0, os.path.abspath('../src/pd_explain'))
autodoc_mock_imports = ['pandas', 'numpy~=1.20.3']

project = 'pd_Explain'
copyright = '2023, Eden Isakov, DR Amit Somech'
author = 'Eden Isakov, DR Amit Somech'
release = '0.0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.napoleon',
    # 'sphinx.ext.duration',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    # 'sphinx.ext.intersphinx',
    'nbsphinx',
    'sphinx_rtd_theme',
    'autoapi.extension'
]

numpydoc_show_class_members = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

intersphinx_disabled_domains = ['std']


templates_path = ['_templates']
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for EPUB output
# epub_show_urls = 'footnote'

# Autoapi conf
autoapi_dirs = [
    '../src/pd_explain/'
]

autoapi_options = [
    'members', 'undoc-members'
]

autoapi_ignore = ['*/consts.py']
