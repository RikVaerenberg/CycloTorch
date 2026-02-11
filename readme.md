# CycloTorch: A library for GPU accelerated cyclostationary analysis

CycloTorch is a library developed for cyclostationary analysis of real signals using PyTorch in order to enable GPU accelerated cyclostationary analysis.

Currently, there are five Cyclic Spectral Correlation (CSC) estimators implemented (each with their respective asymmetric and symmetric estimators): the averaged cyclic periodogram (ACP), the frequency smoothing method (FSM), the Fast SC, the Faster SC and the Strip Spectral Correlation Algorithm (SSCA).

In addition, the optimum and adaptive FRESH filter are implemented.

The code can be found on gitlab [here](https://gitlab.kuleuven.be/lmsd-cm/public/cyclotorch).
The documentation can be found [here](https://lmsd-cm.pages.gitlab.kuleuven.be/public/cyclotorch).

## Running the code 
Create the required anaconda environment:
```bash
conda env create -f environment.yaml
```
And then use it to run_demos.py!
Note that the library works best on a CUDA enabled device (and is required for the TimingDemo).

## Local version of the documentation

To create a locally hosted version of the documentation, just install mkdocs using pip
```bash
pip install -r mkdocs_requirements.txt
```
And then just run the command:
```bash
mkdocs build
```

## Read more about cyclostationarity
- [[1]](https://doi.org/10.1016/j.ymssp.2017.01.011) Antoni, J., Xin, G., & Hamzaoui, N. (2017). Fast computation of the spectral correlation. Mechanical Systems and Signal Processing, 92, 248-277.

- [2] Gardner, William A. Statistical Spectral Analysis: A Nonprobabilistic Theory. Prentice-Hall, Inc., 1986.


- [[3]](https://doi.org/10.1109/78.340776)  Spooner, Chad M., and William A. Gardner. "The cumulant theory of cyclostationary time-series. II. Development and applications." IEEE Transactions on Signal Processing 42.12 (2002): 3409-3429.

