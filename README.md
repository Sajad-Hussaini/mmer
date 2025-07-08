# MERM - Multivariate Mixed Effects Regression Model

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/merm.svg)](https://pypi.org/project/merm/)

**MERM** is a Python package for performing multivariate mixed-effects regression. Its key innovation is allowing fixed effects to be modeled by a variety of machine learning regressors (e.g., Artificial Neural Networks, Random Forest, XGBoost) in addition to standard parametric models. It supports multiple responses (multivariate), multiple grouping variables, and multiple random effects (i.e., intercept, slopes) with flexible random effects structures.

> 💡 **Tip**: To ensure compatibility with the user guide, it's recommended to use the latest **release** available on **GitHub**, **PyPI**, or **Zenodo**.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [User Guide](#user-guide)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Features

✅ **Multivariate responses**: Handle multiple dependent variables simultaneously  
✅ **Flexible random effects**: Support for random intercepts and slopes across multiple grouping factors  
✅ **Scalable computation**: Uses Stochastic Lanczos Quadrature (SLQ) for efficient log-determinant estimation  
✅ **Scikit-learn integration**: Use ML models compatible with any scikit-learn regressor, or your own custom models for fixed effects  
✅ **Parallel processing**: Memory-efficient computation leveraging multi-core processing for large datasets  

## Installation

### Install from PyPI (Recommended)
```bash
pip install merm
```

### Install from Source
```bash
git clone https://github.com/Sajad-Hussaini/merm.git
cd merm
pip install .
```

## User Guide

For step-by-step examples and tutorials on using **MERM**, explore [Examples](merm/examples/).

> 📚 **Note**: The User Guide will be updated with more detailed instructions.

## License

MERM is released under the [MIT License](https://opensource.org/licenses/MIT).  
See the [LICENSE](LICENSE) file for the full text.

## Contact

For questions or assistance, please contact:

**S.M. Sajad Hussaini**  
📧 [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

> Please include "MERM" in the subject line for faster response.

### Support the Project

If you find this package useful, contributions to help maintain and improve it are always appreciated.

[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg)](https://www.paypal.com/paypalme/sajadhussaini)

## References

Please cite the following references for any formal study:

**[1] Primary Reference**  
*Title of the paper*  
DOI: [To be added] (Journal of Earthquake Engineering and Structural Dynamics)

**[2] MERM Package**  
*MERM: Multivariate Mixed Effects Regression Model*  
DOI: [To be added]
