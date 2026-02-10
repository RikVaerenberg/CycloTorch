# Welcome to CycloTorch

CycloTorch is a python library based on [Pytorch](https://pytorch.org/) intended for cyclostationary analysis.
It currently includes implementation different cyclic spectral correlation (CSC) (or coherence) estimators and the implementation of FRESH filters.


## Packages

### [CSC_estimators](CSC_estimators.md)

### [FRESHfilt](FRESHfilt.md)


## References
- [[1](https://doi.org/10.1016/j.ymssp.2017.01.011)] Antoni, J., Xin, G., & Hamzaoui, N. (2017). Fast computation of the spectral correlation. Mechanical Systems and Signal Processing, 92, 248-277.

- [2] Gardner, William A. Statistical Spectral Analysis: A Nonprobabilistic Theory. Prentice-Hall, Inc., 1986.

- [[3](https://doi.org/10.1109/78.340776)] Spooner, Chad M., and William A. Gardner. "The cumulant theory of cyclostationary time-series. II. Development and applications." IEEE Transactions on Signal Processing 42.12 (2002): 3409-3429.

- [[4](https://doi.org/10.1016/j.ymssp.2018.03.059)] Borghesani, Pietro, and Jérôme Antoni. "A faster algorithm for the calculation of the fast spectral correlation." Mechanical Systems and Signal Processing 111 (2018): 113-118.

- [[5](https://doi.org/10.1109/26.212375)] Gardner, William A. "Cyclic Wiener filtering: theory and method." IEEE Transactions on communications 41.1 (2002): 151-163.



<!-- ## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files. -->
