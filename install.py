'''
Install DEIMOS, and optionally some dependencies

Tom Stuttard
'''


import os, sys, subprocess

#
# Helper functions
#

def check_env() :
    '''
    Check for suitable python env
    '''

    # Check python version. Not compatible with py3.
    py_version = sys.version_info[0]
    assert py_version > 2, "Not compatible with python2"

    # Check if a conda/virtualenv is active (recommended)
    conda_env_active = "CONDA_PREFIX" in os.environ
    virtual_env_active = "VIRTUAL_ENV" in os.environ
    if not (conda_env_active or virtual_env_active) :
        response = input("No conda/virtualenv is active, are you sure you want to proceed? [y/n]").lower().replace(" ", "")
        if response == "y" :
            pass
        elif response == "n" :
            print("Exiting...")
            sys.exit()
        else :
            raise Exception("Unknown response : %s" % response)


def execute_commands(commands) :
    '''
    Run a list of shell commands in a single envionrment/shell
    Check output of each command before continuing
    '''

    assert isinstance(commands,list)

    overall_command = ""

    for command in commands :

        assert isinstance(command,str)

        if len(overall_command) > 0 : 
            overall_command += " ; "

        overall_command += command

        overall_command += " || exit 1" # Error code handling

    print(overall_command)

    return_code = subprocess.call( overall_command, shell=True, executable="/bin/bash" )
    assert return_code == 0



def clone_git_repo(
    repo_path, 
    target_dir, 
    branch=None, 
    git_protocol=None,
) :
    '''
    Clone a git repo
    '''

    # Defaults
    if git_protocol is None :
        git_protocol = "https"

    # Handle git prototcol (e.g. ssh vs https), if one is specified
    if git_protocol is not None :
        if git_protocol.lower() == "https" :
            if repo_path.startswith("git") :
                repo_path = repo_path.replace("git@","https://").replace(".com:",".com/")
            assert repo_path.startswith("https")
        elif git_protocol.lower() == "ssh" :
            if repo_path.startswith("https") :
                repo_path = repo_path.replace("https://","git@").replace(".com/",".com:")
            assert repo_path.startswith("git")
        else :
            raise Exception( "Unknown git protocol '%s', choose between 'ssh' or 'https' (or None)" % git_protocol )

    print("Cloning git repo '%s" % repo_path)

    # Check if it already exists
    if os.path.exists(target_dir) :
        print("Target directory already exist, not cloning")    #TODO check branch
        return

    # Clone
    command = "git clone"
    if branch is not None :
        command += " -b %s" % branch
    command += " %s %s" % (repo_path,target_dir)
    print( "Clone git repo : %s" % command )
    return_code = subprocess.call( command , shell=True )
    assert return_code == 0, "Failed to clone git repo"



#
# Tool installation functions
#

def install_mceq() :
    '''
    Install MCEq (https://github.com/mceq-project/MCEq)
    Used for calculating atmospheric neutrino flux
    '''

    # Simply install using PyPi (e.g. not from source)
    execute_commands(["pip install MCEq"])


def install_nusquids(
    target_dir=None,
    squids_repo_path=None, nusquids_repo_path=None, 
    squids_branch=None, nusquids_branch=None, 
    git_protocol=None,
) :
    '''
    Install nuSQuIDS (https://github.com/arguelles/nuSQuIDS)
    Used for calculating oscillation probabilities (including BSM effects)

    This installation assumes that you are using conda, other methods are not supported here
    '''

    print("\nInstalling (nu)SQuIDS...")


    #
    # Check inputs
    #

    # Check for conda env
    assert "CONDA_PREFIX" in os.environ, "This script only supports nuSQuIDS installation via conda"
    prefix = os.environ["CONDA_PREFIX"]

    # Defaults
    if target_dir is None :
        target_dir = os.path.join( os.path.dirname(__file__), ".." ) # Same location as DEIMOS

    if squids_repo_path is None :
        squids_repo_path = "git@github.com:jsalvado/SQuIDS.git"
    if squids_branch is None :
        squids_branch = "master"

    if nusquids_repo_path is None :
        nusquids_repo_path = "git@github.com:arguelles/nuSQuIDS.git"
    if nusquids_branch is None :
        nusquids_branch = "master"

    # Absoute paths
    target_dir = os.path.abspath(target_dir)

    # Define paths to (nu)SQuIDS
    squids_target_dir = os.path.join(target_dir, "SQuIDS")
    nusquids_target_dir = os.path.join(target_dir, "nuSQuIDS")


    #
    # Install dependencies
    #

    # Define (nu)SQuIDS dependencies
    dependencies = [
        "gsl",
        "hdf5",
        "numpy",
        "boost",
    ]

    # Add to the commands
    commands = []
    commands.append( f"conda install -y " + " ".join(dependencies) )


    #
    # Install SQuIDS
    #

    # Git clone the repo
    clone_git_repo(
        repo_path=squids_repo_path,
        target_dir=squids_target_dir,
        branch=squids_branch,
        git_protocol=git_protocol,
    ) 

    # Enter the source directory
    commands.append( "cd " + squids_target_dir )

    # Configure (point all dependency paths to the conda env)
    commands.append(f"./configure --prefix={prefix} --with-gsl-incdir={prefix}/include --with-gsl-libdir={prefix}/lib")

    # Compile
    commands.append( "make clean" )
    commands.append( "make" )
    commands.append( "make install" )


    #
    # Install nuSQuIDS
    #

    # Git clone the repo
    clone_git_repo(
        repo_path=nusquids_repo_path,
        target_dir=nusquids_target_dir,
        branch=nusquids_branch,
        git_protocol=git_protocol,
    )

    # Enter the source directory
    commands.append( "cd " + nusquids_target_dir )

    # Configure (point all dependency paths to the conda env)
    commands.append( f"./configure --with-python-bindings --prefix={prefix} --with-gsl={prefix} --with-hdf5={prefix} --with-boost={prefix} --with-squids={squids_target_dir}" )

    # Compile
    commands.append( "make clean" )
    commands.append( "make" )
    commands.append( "make python" ) # This is the pybindings
    commands.append( "make install" )
    commands.append( "make python-install" )

    # Check can import the pybindings
    # commands.append( "python -c 'import nuSQuIDS'" )


    #
    # Done
    #

    # Finally, run the commands
    execute_commands(commands)

    print(">>> (nu)SQuIDS installation complete!\n")


def install_deimos() :
    '''
    Install DEIMOS (e.g. this project)

    Assumes you;ve already clone the repository (since you need this script)
    '''

    # Get DEIMOS clone directory
    deimos_dir = os.path.join( os.path.dirname(__file__), ".." )

    # Install using pip
    execute_commands([
        f"cd {deimos_dir}", # Go to top directory
        "pip install .", # Install from source, using pip
    ])



#
# Main
#

if __name__ == "__main__" :

    check_env()
    install_nusquids()
    install_mceq()
    install_deimos()
