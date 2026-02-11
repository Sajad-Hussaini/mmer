project = 'MMER'
copyright = '2026, Sajad Hussaini'
author = 'Sajad Hussaini'
from mmer import __version__ as release

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx_copybutton',
    'nbsphinx',
]

# Notebook settings
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

# General configuration
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output options
html_theme = 'pydata_sphinx_theme'

# PyData Theme Config - OPTIMIZED FOR READABILITY
html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "show_toc_level": 2,
    # This aligns the logo/text properly
    "navbar_align": "content", 
    # This prevents the secondary sidebar from taking up too much space on small screens
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    # Footer configuration (Updated for newer pydata-sphinx-theme versions)
    "footer_start": ["copyright", "sphinx-version"],
    "footer_end": ["theme-version"],
}

# Syntax highlighting
pygments_style = 'sphinx' 

# Autodoc settings (CLEAN LOOK)
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
    'exclude-members': '__weakref__, __dict__, __module__',
}

# TYPE HINT SETTINGS: 
# "description" moves type hints to the parameter description instead of the signature.
# This makes the main function definition line SHORTER and easier to read.
autodoc_typehints = 'description' 
autodoc_typehints_format = 'short'

# Napoleon settings (NumPy style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_include_init_with_doc = False  # Don't duplicate __init__ docs
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Math settings
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
