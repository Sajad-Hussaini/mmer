<div align="center">
  <img src="https://raw.githubusercontent.com/Sajad-Hussaini/mmer/main/MMER_Icon.png" alt="MMER Logo" width="300"/>
  <br>
  <h1>MMER: Multivariate Mixed Effects Regression</h1>

  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python"></a>
    <a href="https://opensource.org/licenses/GPL-3.0"><img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License"></a>
    <a href="https://pypi.org/project/mmer"><img src="https://img.shields.io/pypi/v/mmer.svg" alt="PyPI"></a>
    <a href="https://mmer.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mmer/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://doi.org/10.5281/zenodo.18068839"><img src="https://zenodo.org/badge/942170020.svg" alt="DOI"></a>
  </p>
</div>  

**MMER** is a flexible Python framework for multivariate mixed-effects regression. Its defining feature is a plug-and-play architecture that allows you to seamlessly integrate any generic regressor to model the fixed effects, from standard parametric algorithms to advanced machine learning models like Neural Networks, Random Forests, and etc. It natively handles multiple correlated outcomes across various grouping structures, providing direct access to the full random effect and residual covariance matrices [[1]](#references).

## Table of Contents
- [Installation](#installation)
- [Documentation & License](#documentation--license)
- [Contact & Support](#contact--support)
- [References](#references)

## Installation

**Stable release (recommended):** Install the latest stable version from [PyPI](https://pypi.org/project/mmer):

```bash
pip install mmer
```

**Development version:**  To use the latest development version (may include experimental or untested changes), install directly from the [GitHub repository](https://github.com/Sajad-Hussaini/mmer):

```bash
pip install git+https://github.com/Sajad-Hussaini/mmer.git
```

## Documentation & License

📖 **[Explore the Full Documentation, Tutorials, and API Reference](https://mmer.readthedocs.io/)** available at [mmer.readthedocs.io](https://mmer.readthedocs.io/en/latest/?badge=latest).


**MMER** is distributed under the [**GNU General Public License v3 (GPLv3)**](https://opensource.org/licenses/GPL-3.0). See the [LICENSE](LICENSE) file for the full text.


> You are free to use, modify, and distribute this software for academic and research purposes. Any commercial use or distribution of modified versions requires the entire project to be open-sourced under the same GPLv3 license. For proprietary commercial exemptions, please refer to the Contact section.

## Contact & Support

For any questions, assistance, suggestions, or requests to modify API, please feel free to contact:

**S. M. Sajad Hussaini**  
📧 [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

> Please include "MMER" in the subject line for a quicker response.

## References

Please cite the following references for any formal study:  

**[1] Primary Reference**  
*A Multivariate Mixed-Effects Regression Framework for Ground Motion Modeling: Integrating Parametric and Machine Learning Approaches*  
*DOI: https://doi.org/10.1002/eqe.70168*  
(Journal of Earthquake Engineering and Structural Dynamics)

**[2] MMER Package**  
*Multivariate Mixed Effects Regression*  
*DOI: https://doi.org/10.5281/zenodo.18068839*  
