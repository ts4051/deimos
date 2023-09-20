# DEnsItyMatrixOscillationSolver - DEIMOS

A simple python-based neutrino oscillation solver using the density matrix formalism.

Author: Tom Stuttard (stuttard@icecube.wisc.edu)

## Installation

1) Clone this repository locally in the usual manner, e.g.

```
cd <path/to/install/location>
git clone https://github.com/ts4051/deimos.git
```

2) Install dependencies as follows. I recommend using a conda env, and if you do this make sure you have activated your conda env before running this command.

```
pip install numpy matplotlib scipy odeintw
```

3) Add `deimos` to your PYTHONPATH

```
export PYTHONPATH=<path/to/install/location>/deimos:$PYTHONPATH
```
You need to do this for each new terminal session, or add it to an env scetup script such as `~/.bashrc`.
