# from setuptools import setup
from setuptools import setup, find_packages

setup(
    name='deimos',
    version='00.01',
    description='DEnsItyMatrixOscillationSolver',
    author='Tom Stuttard',
    author_email='thomas.stuttard@nbk.ku.dk',
    packages=find_packages(),
    # packages=['deimos'],
    install_requires=[
        'odeintw', # Part of the core solver
        'numpy', # Part of the core solver
        'matplotlib', # For plotting/user-scripts only, can be removed for a minimal installation
        "python-ternary", # For plotting/user-scripts only, can be removed for a minimal installation
        "astropy", # For plotting/user-scripts only, can be removed for a minimal installation
    ],
)
