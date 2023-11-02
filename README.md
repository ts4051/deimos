# DEnsItyMatrixOscillationSolver - DEIMOS

A simple python-based neutrino oscillation solver using the density matrix formalism.

Author: Tom Stuttard (stuttard@icecube.wisc.edu)

## Installation

1) Clone this repository locally in the usual manner, e.g.

```
cd <path/to/install/location>
git clone https://github.com/ts4051/deimos.git
```

2) Install anaconda3 (see https://www.anaconda.com/download), if not already installed.

3) Install DEIMOS and its dependencies using the installation script in this repository, as follows. Note that this will create a dedicated conda env for this software.

```
cd <path/to/install/location>/deimos
python install.py -ad <path/to/anaconda/top/directory> [-ow]
```

where:
* `-ab` is the path to your anaconda installation, e.g. the directory containing the `bin` directory
* `-ow` can optionally be used to force overwritr on any existing DEMIOS installation and associated conda env

This process will generate a script `setup_deimos.sh` that can be used to active the DEIMOS environment. Do this as foloows (must be done for each new shell session):

```
source setup_deimos.sh
```

Note that this installation process only supports unix-based systems (linux, OSX).
