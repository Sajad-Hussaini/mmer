<div align="center">
  <img src="https://raw.githubusercontent.com/Sajad-Hussaini/mmer/main/MMER_Icon.png" alt="MMER Logo" width="300"/>
  <br>
  <h1>MMER: Multivariate Mixed Effects Regression</h1>

  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <a href="https://pypi.org/project/mmer"><img src="https://img.shields.io/pypi/v/mmer.svg" alt="PyPI"></a>
    <a href="https://mmer.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mmer/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://doi.org/10.5281/zenodo.18068839"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18068839.svg" alt="DOI"></a>
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


MMER is released under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for the full text.

## Contact & Support

For any questions, assistance, suggestions, or requests to modify API, please feel free to contact:

**S. M. Sajad Hussaini**  
📧 [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

> Please include "MMER" in the subject line for a quicker response.

> If you find this package useful, contributions to help maintain and improve it, are always appreciated. [![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg)](https://www.paypal.com/paypalme/sajadhussaini)

## References

Please cite the following references for any formal study:  

**[1] Primary Reference**  
*A Multivariate Mixed-Effects Regression Framework for Ground Motion Modeling: Integrating Parametric and Machine Learning Approaches*  
*DOI: https://doi.org/10.1002/eqe.70168*  
(Journal of Earthquake Engineering and Structural Dynamics)

**[2] MMER Package**  
*Multivariate Mixed Effects Regression*  
*DOI: https://doi.org/10.5281/zenodo.18068839*  
