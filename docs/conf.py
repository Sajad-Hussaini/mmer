project = 'MMER'
copyright = '2025, Sajad Hussaini'
author = 'Sajad Hussaini'
from mmer import __version__ as release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx']

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

autodoc_typehints = "none"
