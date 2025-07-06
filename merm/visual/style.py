import matplotlib.pyplot as plt
from contextlib import contextmanager

"""
Matplotlib style configuration module.

This module provides consistent styling for matplotlib plots used throughout the project.
It defines default styling parameters optimized for publication-quality figures
and provides a context manager for easy application of these styles.

Example:
    >>> from merm.style import style
    >>> with style():
    ...     plt.plot([1, 2, 3])
    ...     plt.show()
    
    # Override specific parameters
    >>> with style({"font.size": 12}):
    ...     plt.plot([1, 2, 3])
    ...     plt.show()
"""

_rc_config = {
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 10,

    'lines.linewidth': 0.5,
    'lines.markersize': 1,

    'axes.titlesize': 'medium',
    'axes.linewidth': 0.2,

    'xtick.major.width': 0.2,
    'ytick.major.width': 0.2,
    'xtick.minor.width': 0.15,
    'ytick.minor.width': 0.15,

    'legend.framealpha': 1.0,
    'legend.frameon': False,

    'figure.dpi': 1000,
    'figure.figsize': (9/2.54, 9/2.54),
    'figure.constrained_layout.use': True,

    'patch.linewidth': 0.5,
    }

@contextmanager
def style(overrides=None):
    """
    Context manager to temporarily apply custom matplotlib style settings.
    Args:
        overrides: Dict of matplotlib rc parameters to override the defaults
    Returns:
        Context manager for use in with statements
    """
    rc_settings = _rc_config.copy()
    if overrides:
        rc_settings.update(overrides)
    
    with plt.rc_context(rc=rc_settings):
        yield