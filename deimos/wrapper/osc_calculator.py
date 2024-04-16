'''
Wrapper class providing a common interface to a range of oscillation solvers,
both from this project and external.

Tom Stuttard
'''

# Import plotting tools
# Do this before anything else, as want the matplotlib backend handling dealt with before any other packages called
from deimos.utils.plotting import *

import sys, os, collections, numbers, copy, re, datetime
import numpy as np
import warnings
import healpy as hp

try:
    import cPickle as pickle
except:
    import pickle

# Import nuSQuIDS
NUSQUIDS_AVAIL = False
try:
    import nuSQuIDS as nsq # Modern
    NUSQUIDS_AVAIL = True
except ImportError as e:
    try:
        import nuSQUIDSpy as nsq # Old (backwards compatibility)
        NUSQUIDS_AVAIL = True
    except ImportError as e:
        pass

# Import nuSQuIDS decoherence implementation - now deprecated (import from nsq)
# NUSQUIDS_DECOH_AVAIL = False
# try:
#     from nuSQUIDSDecohPy import nuSQUIDSDecoh, nuSQUIDSDecohAtm
#     NUSQUIDS_DECOH_AVAIL = True
# except ImportError as e:
#     pass

# Import prob3
PROB3_AVAIL = False
try:
    from BargerPropagator import * 
    PROB3_AVAIL = True
except ImportError as e:
    pass

# General DEIMOS imports
from deimos.utils.constants import *
from deimos.density_matrix_osc_solver.density_matrix_osc_solver import DensityMatrixOscSolver, get_pmns_matrix, get_matter_potential_flav
from deimos.utils.oscillations import calc_path_length_from_coszen
from deimos.utils.coordinates import *


#
# Globals
#

DEFAULT_CALC_BASIS = "nxn"
DEFAULT_DECOHERENCE_GAMMA_BASIS = "sun"


#
# Calculator
#

class OscCalculator(object) :
    '''
    A unified interface to a range of oscillation + decoherence calculation tools.
    Allows easy comparison of methods.
    ''' 

    def __init__(self,
        tool, # Name of the underlying calculion tool
        atmospheric, # Bool indicating calculating in atmospheric parameter space (e.g. zenith instead of baseline)
        flavors=None,
        # Osc params
        mixing_angles_rad=None,
        mass_splittings_eV2=None,
        deltacp_rad=None,
        cache_dir=None,
        **kw
    ) :

        # Store args
        self.tool = tool
        self.atmospheric = atmospheric
        self.flavors = flavors
        self.cache_dir = cache_dir

        # User must specify flavors, or take default
        if self.flavors is None :
            self.flavors = FLAVORS
        assert isinstance(self.flavors, list)
        assert len(self.flavors) == len(set(self.flavors)), "Duplicate flavors provided"
        assert all([ (f in FLAVORS) for f in self.flavors ]), "Unknown flavors provided"
        self.num_neutrinos = len(self.flavors)

        # Checks
        assert self.num_neutrinos in [2,3]

        # Useful derived values
        self.num_sun_basis_vectors = self.num_neutrinos ** 2

        # Init
        if self.tool == "nusquids" :
            self._init_nusquids(**kw)
        elif self.tool == "deimos" :
            self._init_deimos(**kw)
        elif self.tool == "prob3" :
            self._init_prob3(**kw)
        else :
            raise Exception("Unrecognised tool : %s" % self.tool)

        # Set some default values for parameters
        self.set_matter("vacuum")

        if mass_splittings_eV2 is None :
            if self.num_neutrinos == 3 :
                mass_splittings_eV2 = MASS_SPLITTINGS_eV2
            else :
                raise Exception("Must specify 'mass_splittings_eV2' when not in 3 flavor mode")

        if mixing_angles_rad is None :
            if self.num_neutrinos == 3 :
                mixing_angles_rad = MIXING_ANGLES_rad
            else :
                raise Exception("Must specify 'mixing_angles_rad' when not in 3 flavor mode")

        if deltacp_rad is None :
            if self.num_neutrinos == 2 :
                assert deltacp_rad is None, "deltacp not relevent for 2 flavor oscillations"
                deltacp_rad = 0.
            elif self.num_neutrinos == 3 :
                deltacp_rad = DELTACP_rad
            else :
                raise Exception("Must specify 'deltacp_rad' when using >3 flavors")

        # Update osc params
        self.set_mixing_angles(*mixing_angles_rad, deltacp=deltacp_rad)
        self.set_mass_splittings(*mass_splittings_eV2)
        self.set_calc_basis(DEFAULT_CALC_BASIS)
        # self.set_decoherence_D_matrix_basis(DEFAULT_DECOHERENCE_GAMMA_BASIS)

        # Init some variables related to astrohysical coordinates   #TODO are thes DEIMOS-specific? If so, init in _init_deimos()
        self.detector_coords = None
        self._neutrino_source_kw = None

        # Caching
        if self.cache_dir is None :
            self.cache_dir = os.path.realpath( os.path.join( os.path.dirname(__file__), "..", "..", ".cache" ) )
            if not os.path.exists(self.cache_dir) :
                os.mkdir(self.cache_dir)
        assert os.path.isdir(self.cache_dir) 




    def parse_pisa_config(self,config) :
        '''
        Parse settings from a PISA config file and apply them
        '''

        pass #TODO


    def _init_nusquids(self,
        energy_nodes_GeV=None,
        coszen_nodes=None,
        interactions=False,
        nusquids_variant=None, # Specify nuSQuIDS variants (nuSQuIDSDecoh, nuSQUIDSLIV, etc)
        error=1.e-6,
    ) :

        assert NUSQUIDS_AVAIL, "Cannot use nuSQuIDS, not installed"


        #
        # Handle nuSQuIDS variants
        #

        # Store arg
        self._nusquids_variant = nusquids_variant

        # Aliases
        if self._nusquids_variant in ["decoh", "decoherence" ] :
            self._nusquids_variant = "nuSQUIDSDecoh"
        if self._nusquids_variant in ["liv", "LIV", "sme", "SME" ] :
            self._nusquids_variant = "nuSQUIDSLIV"


        #
        # Calculation nodes
        #

        # Energy node definition
        self.energy_nodes_GeV = energy_nodes_GeV
        if self.energy_nodes_GeV is False :
            pass # Single-energy mode
        elif self.energy_nodes_GeV is None :
            # Provide default nodes if none provided
            self.energy_nodes_GeV = np.logspace(0.,3.,num=100)

        # cos(zenith) node definition
        # Only relevant in atmospheric mode
        if self.atmospheric :
            self.coszen_nodes = coszen_nodes
            if self.coszen_nodes is None :
                # Provide default nodes if none provided
                self.coszen_nodes = np.linspace(-1.,1.,num=100)
        else :
            assert coszen_nodes is None, "`coszen_nodes` argument only valid in `atmospheric` mode"

        # In theory should be able to provide a single energy node for single energy mode, but finding some issues with this (constructor definitions)
        # Instead, provide two energy nodes in this case to by pass this minimal extra computation time
        if np.isscalar(self.energy_nodes_GeV) :
            self.energy_nodes_GeV = np.array([ self.energy_nodes_GeV, self.energy_nodes_GeV*2. ])

        # Check formats
        assert isinstance(self.energy_nodes_GeV, np.ndarray) and self.energy_nodes_GeV.ndim, "nuSQuIDs energy_nodes_GeV must be a 1D numpy array"
        if self.atmospheric :
            assert isinstance(self.coszen_nodes, np.ndarray) and self.coszen_nodes.ndim, "nuSQuIDs coszen_nodes must be a 1D numpy array"


        #
        # Instantiate nuSQuIDS
        #

        # Get nuSQuiDS units
        self.units = nsq.Const()

        # Get neutrino type
        # Alwys do both, not the most efficient but simplifies things
        nu_type = nsq.NeutrinoType.both 

        # Toggle between atmo. vs regular modes
        if self.atmospheric :

            # Instantiate nuSQuIDS atmospheric calculator
            args = [
                self.coszen_nodes,
                self.energy_nodes_GeV * self.units.GeV,
                self.num_neutrinos,
                nu_type,
                interactions,
            ]

            if self._nusquids_variant is None :
                self.nusquids = nsq.nuSQUIDSAtm(*args)

            elif self._nusquids_variant == "nuSQUIDSDecoh" :
                # assert NUSQUIDS_DECOH_AVAIL, "Could not find nuSQuIDS decoherence implementation"
                # self.nusquids = nuSQUIDSDecohAtm(*args) #TODO Needs updating to modern nuSQuIDS pybindings format
                assert hasattr(nsq, "nuSQUIDSDecohAtm"), "Could not find nuSQuIDS Decoh implementation"
                self.nusquids = nsq.nuSQUIDSDecohAtm(*args)

            elif self._nusquids_variant == "nuSQUIDSLIV" :
                assert hasattr(nsq, "nuSQUIDSLIVAtm"), "Could not find nuSQuIDS LIV implementation"
                self.nusquids = nsq.nuSQUIDSLIVAtm(*args)

            else :
                raise Exception("Unknown nusquids varint : %s" % self._nusquids_variant)
            
            # Enable tau regeneration if using interactions
            if interactions :
                self.nusquids.Set_TauRegeneration(True)

        else :

            # Instantiate nuSQuIDS regular calculator
            if self.energy_nodes_GeV is False :
                # Single-energy mode
                assert not interactions, "`interactions` cannot be set in single energy mode"
                assert nu_type in [nsq.NeutrinoType.neutrino, nsq.NeutrinoType.antineutrino], "Single-energy mode does not support neutrino and anitneutrino calculation simultaneously" 
                args = [
                    self.num_neutrinos,
                    nu_type,
                ]
            else :
                args = [
                    self.energy_nodes_GeV * self.units.GeV,
                    self.num_neutrinos,
                    nu_type,
                    interactions,
                ]

            if self._nusquids_variant is None :
                self.nusquids = nsq.nuSQUIDS(*args)

            elif self._nusquids_variant == "nuSQUIDSDecoh" :
                # assert NUSQUIDS_DECOH_AVAIL, "Could not find nuSQuIDS decoherence implementation"
                # self.nusquids = nuSQUIDSDecoh(*args) #TODO Needs updating to modern nuSQuIDS pybindings format
                assert hasattr(nsq, "nuSQUIDSDecoh"), "Could not find nuSQuIDS Decoh implementation"
                self.nusquids = nsq.nuSQUIDSDecoh(*args)

            elif self._nusquids_variant == "nuSQUIDSLIV" :
                assert hasattr(nsq, "nuSQUIDSLIV"), "Could not find nuSQuIDS LIV implementation"
                self.nusquids = nsq.nuSQUIDSLIV(*args)

            else :
                raise Exception("Unknown nusquids varint : %s" % self._nusquids_variant)

            # Enable tau regeneration if using interactions
            if interactions :
                self.nusquids.Set_TauRegeneration(True)

                
        #
        # Various settings
        #

        self.nusquids.Set_rel_error(error)
        self.nusquids.Set_abs_error(error)

        self.nusquids.Set_ProgressBar(False)


    def _init_deimos(self,
        **kw
    ) :

        # Init persistent state variables required
        # These are things that are passed to `calc_osc_probs` basically
        # self._decoh_D_matrix_eV = None
        self._decoh_n = None
        self._decoh_E0_eV = None
        self._calc_basis = None
        # self._decoherence_D_matrix_basis = None
        self._decoh_model_kw = None
        self._lightcone_model_kw = None
        self._sme_model_kw = None
        self.skymap_use = None
        
        # Instantiate solver
        self.solver = DensityMatrixOscSolver(
            num_states=self.num_neutrinos,
            **kw
        )


    def _init_prob3(self,
        **kw
    ) :

        assert PROB3_AVAIL, "Prob3 not installed"

        # Create a dict to hold settings
        self._prob3_settings = {}

        # Create propagator
        self._propagator = BargerPropagator()




    def set_matter(self, matter, **kw) :

        # Re-initalise any persistent matter-related setting
        # Mostly don't use this, only for "layers" mode currently 
        self._matter_settings = { "matter":matter }

        #
        # Vacuum
        #

        if (matter == "vacuum") or (matter is None) :

            if self.tool == "nusquids" :
                self.nusquids.Set_Body(nsq.Vacuum())

            elif self.tool == "deimos" :
                self.solver.set_matter_potential(None)

            elif self.tool == "prob3" :
                self._prob3_settings["matter"] = None

        #
        # Earth
        #

        elif matter == "earth" :

            if self.tool == "nusquids" :
                if self.atmospheric :
                    self.nusquids.Set_EarthModel(nsq.EarthAtm())
                else :
                    raise Exception("`earth` is only an option in atmospheric mode")

            elif self.tool == "deimos" :
                raise Exception("`%s` does have an Earth model" % self.tool)

            elif self.tool == "prob3" :
                self._prob3_settings["matter"] = "earth"


        #
        # Uniform matter density
        #

        elif matter == "constant" :

            # Check required kwargs present
            assert "matter_density_g_per_cm3" in kw
            assert "electron_fraction" in kw

            if self.tool == "nusquids" :
                self.nusquids.Set_Body(nsq.ConstantDensity(kw["matter_density_g_per_cm3"], kw["electron_fraction"]))

            elif self.tool == "deimos" :
                V = get_matter_potential_flav(flavors=self.flavors, matter_density_g_per_cm3=kw["matter_density_g_per_cm3"], electron_fraction=kw["electron_fraction"], nsi_matrix=None)
                self.solver.set_matter_potential(V)


            elif self.tool == "prob3" :
                self._prob3_settings["matter"] = "constant"
                self._prob3_settings["matter_density_g_per_cm3"] = kw["matter_density_g_per_cm3"]



        #
        # Matter layers (of constant density)
        #

        elif matter == "layers" :

            # Check required kwargs present
            assert "layer_endpoint_km" in kw # The endpoint of each layer. The startpoint is either L=0 (first layer) or the end of the previous layer
            assert "matter_density_g_per_cm3" in kw # Density in each layer
            assert "electron_fraction" in kw # Electron fraction in each layer

            # Check their format (e.g. one value per layer)
            assert isinstance(kw["matter_density_g_per_cm3"], np.ndarray) and (kw["matter_density_g_per_cm3"].ndim == 1), "'matter_density_g_per_cm3' should be an array of float values in 'layers' mode"
            assert isinstance(kw["electron_fraction"], np.ndarray) and (kw["electron_fraction"].ndim == 1), "'electron_fraction' should be an array of float values in 'layers' mode"
            assert kw["layer_endpoint_km"].size == kw["electron_fraction"].size, "'layer_endpoint_km', 'matter_density_g_per_cm3' and 'electron_fraction' do not have the same length (should be one per layer)"
            assert kw["layer_endpoint_km"].size == kw["matter_density_g_per_cm3"].size, "'layer_endpoint_km', 'matter_density_g_per_cm3' and 'electron_fraction' do not have the same length (should be one per layer)"

            # Check layer endpoints as ascending
            assert np.all(kw["layer_endpoint_km"][:-1] <= kw["layer_endpoint_km"][1:]), "'layer_endpoint_km' must be ascending"

            if self.tool == "nusquids" :
                # Store the laters for use during state evolution
                self._matter_settings["layer_endpoint_km"] = kw["layer_endpoint_km"]
                self._matter_settings["matter_density_g_per_cm3"] = kw["matter_density_g_per_cm3"]
                self._matter_settings["electron_fraction"] = kw["electron_fraction"]

            elif self.tool == "deimos" :
                raise NotImplemented("'layers' mode for matter effects not implemented for deimos")

            elif self.tool == "prob3" :
                raise NotImplemented("'layers' mode for matter effects not implemented for prob3")


        #
        # Error handling
        #

        else :
            raise Exception("Unrecognised `matter` : %s" % matter)



    def set_mixing_angles(self,theta12, theta13=None, theta23=None, deltacp=0.) :
        '''
        Units: radians
        '''

        if self.num_neutrinos == 2 :
            assert theta13 is None
            assert theta23 is None
        else :
            assert theta13 is not None
            assert theta23 is not None

        if self.tool == "nusquids" :
            self.nusquids.Set_CPPhase( 0, 2, deltacp ) #TODO check indices
            self.nusquids.Set_MixingAngle( 0, 1, theta12 )
            if self.num_neutrinos > 2 :
                self.nusquids.Set_MixingAngle( 0, 2, theta13 )
                self.nusquids.Set_MixingAngle( 1, 2, theta23 )

        elif self.tool == "deimos" :
            self.solver.set_mixing_angles( np.array([ t for t in [theta12,theta13,theta23] if t is not None ]), deltacp=deltacp )
            # self.solver.set_mixing_angles( -1. * np.array([ t for t in [theta12,theta13,theta23] if t is not None ]) ) #TODO

        elif self.tool == "prob3" :
            # Just store for passing an propagation time to solver
            self._prob3_settings["theta12"] = theta12
            self._prob3_settings["theta13"] = theta13
            self._prob3_settings["theta23"] = theta23
            self._prob3_settings["deltacp"] = deltacp


    def get_mixing_angles(self) :

        if self.tool == "deimos" :
            return self.solver.theta_rad

        elif self.tool == "prob3" :
            assert self.num_neutrinos == 3
            return (self._prob3_settings["theta12"],  self._prob3_settings["theta13"], self._prob3_settings["theta23"])

        else :
            raise Exception("TODO")


    def get_deltacp(self) :

        if self.tool == "deimos" :
            return self.solver.deltacp

        elif self.tool == "prob3" :
            return self._prob3_settings["deltacp"]

        else :
            raise Exception("TODO")


    def set_deltacp(self, deltacp) :

        if self.tool == "nusquids" :
            self.nusquids.Set_CPPhase( 0, 2, deltacp )

        elif self.tool == "deimos" :
            raise Exception("Cannot set delta CP on its own for `deimos`, use `set_mixing_angles`")

        elif self.tool == "prob3" :
            self._prob3_settings["deltacp"] = deltacp


    def set_mass_splittings(self, deltam21, deltam31=None) :
        '''
        # Note: deltam31 is +ve for normal ordering and -ve for inverted ordering

        Units: eV**2
        '''

        if self.num_neutrinos == 2 :
            assert deltam31 is None
        else :
            assert deltam31 is not None

        if self.tool == "nusquids" :
            self.nusquids.Set_SquareMassDifference( 1, deltam21*self.units.eV*self.units.eV )
            if deltam31 is not None :
                self.nusquids.Set_SquareMassDifference( 2, deltam31*self.units.eV*self.units.eV )

        elif self.tool == "deimos" :
            self.solver.set_mass_splittings( np.array([ dm2 for dm2 in [deltam21, deltam31] if dm2 is not None ]) )

        elif self.tool == "prob3" :
            self._prob3_settings["deltam21"] = deltam21
            self._prob3_settings["deltam31"] = deltam31


    def get_mass_splittings(self) :
        '''
        Units: eV**2
        '''

        if self.tool == "nusquids" :
            mass_splittings_eV2 = [ self.nusquids.Get_SquareMassDifference(1)/(self.units.eV*self.units.eV) ]
            if self.num_neutrinos > 2 :
                mass_splittings_eV2.append( self.nusquids.Get_SquareMassDifference(2)/(self.units.eV*self.units.eV) )
            return tuple(mass_splittings_eV2)

        elif self.tool == "deimos" :
            return self.solver.get_mass_splittings()

        elif self.tool == "prob3" :
            return ( self._prob3_settings["deltam21"], self._prob3_settings["deltam31"] )


    def set_std_osc(self) :
        '''
        Use standard oscillations (e.g. disable any BSM effects)
        '''

        if self.tool == "nusquids" :
            self.set_calc_basis(DEFAULT_CALC_BASIS)

            if self._nusquids_variant == "nuSQUIDSDecoh" :
                # self.set_decoherence_D_matrix_basis(DEFAULT_CALC_BASIS)
                self.set_decoherence_D_matrix(D_matrix_eV=np.zeros((self.num_sun_basis_vectors,self.num_sun_basis_vectors)), n=0, E0_eV=1.)

            elif self._nusquids_variant == "nuSQUIDSLIV" :
                null_matrix = np.zeros((3,self.num_neutrinos,self.num_neutrinos))
                self.set_sme(directional=True, basis="mass", a_eV=null_matrix, c=null_matrix, e=null_matrix, ra_rad=0., dec_rad=0.)
        
        else :
            self._decoh_model_kw = None
            self._lightcone_model_kw = None
            self._sme_model_kw = None
            self._neutrino_source_kw = None


    def set_calc_basis(self, basis) :

        if self.tool == "nusquids" :
            assert basis == "nxn" #TODO is this correct?

        elif self.tool == "deimos" :
            self._calc_basis = basis # Store for use later

        elif self.tool == "prob3" :
            pass # Basis not relevent here, not solving Linblad master equation

        else :
            raise Exception("`%s` does not support setting calculation basis" % self.tool)



    #
    # Nu flux
    #

    def get_neutrino_flux(self, energy_GeV, coszen, source, tool=None, grid=False, overwrite_cache=False) :
        '''
        Function to return the atmospheric neutrino flux, for a given model or calculation method

        Output flux shape is: [E, cz, flavor, nu/nubar]
        '''

        #TODO azimuth

        # Check inputs
        assert energy_GeV.ndim == 1 #TODO support 2D, scalar, etc
        assert coszen.ndim == 1

        # Toggle neutrino source
        if source.lower() in [ "atmospheric", "atmo" ] :


            #
            # Atmospheric flux
            #

            # Checks
            assert self.atmospheric, "Must be in atmospheric mode"
            assert self.num_neutrinos > 2, "Atmospheric flux not defined for 2nu system"

            # Default tool
            if tool is None :
                tool = "mceq"

            # Toggle tool used to generate flux
            if tool.lower() == "honda" :
                raise NotImplemented("TODO: Honda flux")

            elif tool.lower() == "daemon" :
                raise NotImplemented("TODO: Daemon flux")

            elif tool.lower() == "mceq" :
                return self._get_atmo_neutrino_flux_mceq(energy_GeV=energy_GeV, coszen=coszen, grid=grid, overwrite_cache=overwrite_cache)

            else :
                raise NotImplemented("Unknown tool for atmospheric flux")


        else :

            #
            # High energy astrophysical flux (as detected by IceCube)
            #

            # Checks
            assert self.atmospheric, "Must be in atmospheric mode"
            assert self.num_neutrinos > 2, "Atmospheric flux not defined for 2nu system"

            # Default tool
            if tool is None :
                tool = "spl"

            # Toggle tool used to generate flux
            if tool in [ "spl", "single_power_law" ] :

                # Generate a basic   using roughly the spectral index from IceCube observations
                # Not rigorous, good for quick checks though
                # Assumes 1:1:1 flavor, and 1:1 nu:nubar, isotropic (which means uniform in coszen)

                # Using flux from IceCube HESE 2020 (arXiv:2011.03545)
                norm_100_TeV = 6.5e-18 / 6. # GeV^{-1} sr^{-1} s^{-1} cm^{-2}. Dividing by 6 since this is the total flux for all flavors and nu/nubar       #TODO get more precise value
                spectral_index = -2.87

                phi_E = norm_100_TeV * np.power( energy_GeV / 1e5, spectral_index ) 

                output_flux = np.full( (energy_GeV.size, coszen.size, self.num_neutrinos, 2), np.NaN ) # shape =  (same as used by e.g. calc_osc_probs)
                for cz in range(coszen.size) :
                    for f in range(self.num_neutrinos) :
                        for r in range(2) :
                            output_flux[:, cz, f, r] = phi_E
                assert np.all( np.isfinite(output_flux) )

                return output_flux



    def _get_atmo_neutrino_flux_mceq(self, energy_GeV, coszen, grid=False, overwrite_cache=False) :
        '''
        Run MCEq to compute flux for some zenith/energy nodes, then spline them to get continuous description

        Cache the spline for later so don't have to regenerate it every time 
        '''

        #TODO option to load splines from PISA?

        from scipy.interpolate import RectBivariateSpline

        #
        # Prepare
        #

        # Define path to cache file, and check if it exists already
        cache_file = os.path.join(self.cache_dir, "atmo_flux_mceq.pckl") #TODO include model names

        # Check if cache already exists
        cache_exists = os.path.exists(cache_file)

        # Mapping of MCEq and DEIMOS flavor labels
        # Note that this labels with return the combined conventional and prompt atmospheric flux from MCEq
        flavor_mapping = collections.OrderedDict()
        flavor_mapping["nue"] = (0, False) # flavor, nubar
        flavor_mapping["antinue"] = (0, True)
        flavor_mapping["numu"] = (1, False)
        flavor_mapping["antinumu"] = (1, True)
        flavor_mapping["nutau"] = (2, False)
        flavor_mapping["antinutau"] = (2, True)


        #
        # Generate or load splines
        #

        # Generate splines (a) if no cache exists, or (b) is user wants to overwrite
        generate_splines = overwrite_cache or (not cache_exists)
        if generate_splines :  

            #
            # Use MCEq to compute flux and generate splints
            #

            start_time = datetime.datetime.now()

            # Import MCEq
            try :
                from MCEq.core import config, MCEqRun
                import crflux.models as crf
            except Exception as e :
                raise Exception("MCEq not installed")

            # Initalize MCEq by creating the user interface object MCEqRun   #TODO Make model steerable
            mceq = MCEqRun(
                interaction_model='SIBYLL23C', # High-energy hadronic interaction model
                primary_model=(crf.GlobalSplineFitBeta, None), # cosmic ray flux at the top of the atmosphere
                theta_deg=0., # This will be overwritten later
            )

            # Define zenith values to solve for
            coszen_nodes = np.linspace(-1., 1., num=21)
            # coszen_nodes = np.array([0.]) # MCEq only considers down-going, need to mirror later for up-going

            # Init flux container
            mceq_flux = collections.OrderedDict()
            for f in flavor_mapping.keys() :
                mceq_flux[f] = []

            # Loop over zenith angles
            for coszen_node in coszen_nodes :

                # Solve the equation system, for this zenith angle
                # MCEq only considers down-going, need to mirror later for up-going   #TODO more efficient just to copy array, rathe than re-comoute
                mceq.set_theta_deg( np.rad2deg(np.arccos(np.abs(coszen_node))) ) # The abs doe the zenith mirroring
                mceq.solve()

                # Extract the flux
                for f in mceq_flux.keys() :
                    mceq_flux[f].append( mceq.get_solution(f).tolist() )

            # numpy-ify
            for f in mceq_flux.keys() :
                mceq_flux[f] = np.array(mceq_flux[f]).T # -> [E, coszen]

            # Spline the flux so can interpolate to get values at arbitrary coszen/E values
            splines = collections.OrderedDict()
            for flavor, flavor_flux in mceq_flux.items() :
                splines[flavor] = RectBivariateSpline(mceq.e_grid, coszen_nodes, flavor_flux)

            # Save the splines
            with open(cache_file,"wb") as f :
                pickle.dump(splines, f, protocol=-1) # protocol=-1 -> use fastest mode available

            end_time = datetime.datetime.now()
            print("MCEq flux spline generation complete : Took %s" % (end_time-start_time))


        else :

            #
            # Load the cached splines, if didn't just generate them
            #

            with open(cache_file, "rb") as f:
                splines = pickle.load(f)


        # 
        # Evaluate splines for specified [E, coszen] grid
        #

        # Get flux for each flavor at the specified nodes
        # Output as 4D array in same format used elsewhere in code
        output_flux = np.full( (energy_GeV.size, coszen.size, self.num_neutrinos, 2), np.NaN ) # shape = [E, cz, flavor, nu/nubar] (same as used by e.g. calc_osc_probs)
        for mceq_flavor, spline in splines.items() :
            flavor, nubar = flavor_mapping[mceq_flavor]
            rho = 1 if nubar else 0
            output_flux[:, :, flavor, rho] = spline(energy_GeV, coszen, grid=grid) # Include mapping from MCEq to DEIMOS flavor

        # Checks
        assert np.all( np.isfinite(output_flux) )

        return output_flux




    #
    # Decoherence member functions
    #

    # def set_decoherence_D_matrix_basis(self, basis) :

    #     if self.tool == "nusquids" :
    #         assert basis == "sun"

    #     elif self.tool == "deimos" :
    #         self._decoherence_D_matrix_basis = basis # Store for use later

    #     else :
    #         raise Exception("`%s` does not support setting decoherence gamma matrix basis" % self.tool)


    def set_decoherence_D_matrix(self,
        D_matrix_eV,
        n, # energy-dependence
        E0_eV,
    ) :
        '''
        Set the decoherence D matrix, plus energy dependence

        Definitions in arXiv:2007.00068, e.g.:
          - D matrix -> eqn 10
          - energy-dependence (steered by n and E9) -> eqn 18
        '''

        from deimos.models.decoherence.decoherence_operators import check_decoherence_D_matrix


        #
        # Check inputs
        #

        # If user specified the full matrix, check dimensions
        assert isinstance(D_matrix_eV, np.ndarray)

        # Check all relevent matrix conditions
        check_decoherence_D_matrix(num_neutrinos=self.num_neutrinos, D=D_matrix_eV)


        #
        # Set values
        #

        if self.tool == "nusquids" :
            assert self._nusquids_variant == "nuSQUIDSDecoh"
            assert np.allclose(D_matrix_eV.imag, 0.), "nuSQuIDS decoherence implementation currently does not support imaginary gamma matrix"
            self.nusquids.Set_DecoherenceGammaMatrix(D_matrix_eV.real * self.units.eV)
            self.nusquids.Set_DecoherenceGammaEnergyDependence(n)
            self.nusquids.Set_DecoherenceGammaEnergyScale(E0_eV)

        elif self.tool == "deimos" :
            self._decoh_model_kw = {
                "D_matrix0_eV" : D_matrix_eV, # Put 0 in this line!
                "n" : n,
                "E0_eV" : E0_eV,
                "D_matrix_basis" : "sun" # Added this line!
            }

    def set_decoherence_model(self, model_name, **kw) :
        '''
        Set the decoherence model to be one of the pre-defined models
        '''

        from deimos.models.decoherence.nuVBH_model import get_randomize_phase_decoherence_D_matrix, get_randomize_state_decoherence_D_matrix, get_neutrino_loss_decoherence_D_matrix
        from deimos.models.decoherence.decoherence_operators import get_generic_model_decoherence_D_matrix

        #
        # Unpack args
        #

        kw = copy.deepcopy(kw)

        assert "gamma0_eV" in kw
        gamma0_eV = kw.pop("gamma0_eV")

        assert "n" in kw
        n = kw.pop("n")

        assert "E0_eV" in kw
        E0_eV = kw.pop("E0_eV")

        assert len(kw) == 0


        #
        # nu-VBH interaction models
        #

        get_D_matrix_func = None

        # Check if model is one of the nuVBH models, and get the D matrix definition function if so
        if model_name == "randomize_phase" :
            get_D_matrix_func = get_randomize_phase_decoherence_D_matrix

        elif model_name == "randomize_state" :
            get_D_matrix_func = get_randomize_state_decoherence_D_matrix

        elif model_name == "neutrino_loss" :
            get_D_matrix_func = get_neutrino_loss_decoherence_D_matrix

        # Check if found a match
        if get_D_matrix_func is not None :
            D_matrix_basis, D_matrix0_eV = get_D_matrix_func(num_states=self.num_neutrinos, gamma=gamma0_eV)


        #
        # Generic models
        #

        # Otherwise, try the generic models
        else :
            D_matrix0_eV = get_generic_model_decoherence_D_matrix(name=model_name, gamma=gamma0_eV)
            # D_matrix_basis = "mass" #TODO


        #
        # Pass to the solver
        #

        self.set_decoherence_D_matrix( D_matrix_eV=D_matrix0_eV, n=n, E0_eV=E0_eV ) #TODO what about the basis?


    #
    # Lightcone flucutation member functions
    #

    def set_lightcone_fluctuations(
        self,
        dL0_km,
        L0_km,
        E0_eV,
        n,
        m,
    ) :
        '''
        Set lightcone fluctuation model parameters
        '''

        from deimos.utils.model.lightcone_fluctuations.lightcone_fluctuation_model import get_lightcone_decoherence_D_matrix

        if self.tool == "nusquids" :
            print("NotImplemented. This is placeholder code:")
            damping_power = 2
            D_matrix = np.diag([0,1,1,0,1,1,1,1,0])
            self.nusquids.Set_DecoherenceGammaMatrix(D_matrix.real)
            self.nusquids.Set_DecoherenceGammaEnergyDependence(n)
            self.nusquids.Set_DecoherenceGammaEnergyScale(E0_eV * self.units.eV)
            self.nusquids.Set_UseLightconeFluctuations(True)
            self.nusquids.Set_mLengthDependenceIndex(m)
            self.nusquids.Set_dL0(dL0_km * self.units.km)
            self.nusquids.Set_L0LengthScale(L0_km * self.units.km)
            self.nusquids.Set_DampingPower(damping_power)


        elif self.tool == "deimos" :
            self._lightcone_model_kw = {
                "dL0_km" : dL0_km,
                "L0_km" : L0_km,
                "E0_eV" : E0_eV,
                "n" : n,
                "m" : m,
            }

    #
    # SME member functions
    #

    def set_sme(self,
        directional, # bool
        basis=None,       # string: "mass" or "flavor"
        a_eV=None,        # 3 x Num_Nu x Num_nu
        c=None,           # 3 x Num_Nu x Num_nu
        e=None,           # 3 x Num_Nu x Num_nu
        ra_rad=None,
        dec_rad=None,
    ) :
        '''
        TODO
        '''

        #
        # Check inputs
        #

        if basis is None :
            basis = "mass"
        assert basis in ["flavor", "mass"]

        if directional :   #TODO Maybe not relevant anymore? (Non-directional does currently not work in nuSQuIDS)
            operator_shape = (3, self.num_neutrinos, self.num_neutrinos) # shape is (num spatial dims, N, N), where N is num neutrino states
            if a_eV is None: 
                a_eV = np.zeros(operator_shape)
            if c is None:
                c = np.zeros(operator_shape)
            if e is None :
                e = np.zeros(operator_shape)

            assert isinstance(a_eV, np.ndarray) and (a_eV.shape == operator_shape)
            assert isinstance(c, np.ndarray) and (c.shape == operator_shape) 
            assert isinstance(e, np.ndarray) and (e.shape == operator_shape) 

            assert (ra_rad is not None) and (dec_rad is not None), "Must provide ra and dec when using directional SME"

        else :
            operator_shape = (self.num_neutrinos, self.num_neutrinos) # shape is (N, N), where N is num neutrino states

            if a_eV is None: 
                a_eV = np.zeros(operator_shape)
            if c is None:
                c = np.zeros(operator_shape)

            assert isinstance(a_eV, np.ndarray) and (a_eV.shape == operator_shape)
            assert isinstance(c, np.ndarray) and (c.shape == operator_shape) 
            assert e is None, "e not implemented yet for isotropic SME"

            assert (ra_rad is None) and (dec_rad is None), "ra and dec not relevent for isotropic SME"


        #
        # Set values
        #

        if self.tool == "nusquids" :
            assert directional, "Istropic SME not implemented in nuSQuIDS yet"
            assert basis == "mass", "Only mass basis SME implemented in nuSQuIDS currently"
            self.nusquids.Set_LIVCoefficient(a_eV, c, e, ra_rad, dec_rad)

        elif self.tool == "deimos" :
            if directional :
                self._sme_model_kw = {
                    "directional" : True,
                    "basis" : basis,
                    "a_eV" : a_eV,
                    "c" : c,
                    "e": e,
                    "ra_rad" : ra_rad,
                    "dec_rad" : dec_rad,
                }
            else :
                self._sme_model_kw = {
                    "directional" : False,
                    "basis" : basis,
                    "a_eV" : a_eV,
                    "c" : c,
                    # "e": e,
                }

        else :
            raise NotImplemented("SME not yet wrapped for %s" % self.tool) #TODO this is already supported by prob3, just need to wrap it


    def set_detector_location(
        self,
        lat_deg,
        long_deg, 
        height_m,
    ) :
        '''
        Define detector position
        '''

        self.detector_coords = DetectorCoords(
            detector_lat=lat_deg, 
            detector_long=long_deg, 
            detector_height_m=height_m,  #TODO consistency with detector depth in the L<->coszen calculation
        )
        

    def set_detector(
        self,
        name,
    ) :
        '''
        Set detector (position, etc), choosing from known detectors
        '''

        #
        # Real detectors
        #

        if name.lower() == "icecube" :
            self.set_detector_location(
                lat_deg="89°59′24″S",
                long_deg="63°27′11″W",
                height_m=-1400.,
            )

        elif name.lower() == "dune" :
            self.set_detector_location(
                lat_deg=44.3517,
                long_deg=-103.7513,
                height_m=-1.5e3,
            )

        elif name.lower() == "arca" :
            self.set_detector_location(
                lat_deg="36°15′36″N",
                long_deg="16°06′00″E",
                height_m=-1500.,
            )


        #
        # Toy detectors
        #

        elif name.lower() == "toy_equator" : # Toy detector at sea level on the equator, at 0 deg longitude
            self.set_detector_location(
                lat_deg="0°S",
                long_deg="0°W",
                height_m=0.,
            )

        elif name.lower() == "toy_north_pole" : # Same as 'toy_equator' but at the North Pole (only latitude changes)
            self.set_detector_location(
                lat_deg="90°N",
                long_deg="0°W",
                height_m=0.,
            )

        elif name.lower() == "toy_south_pole" : # Same as 'toy_equator' but at the South Pole (only latitude changes)
            self.set_detector_location(
                lat_deg="90°S",
                long_deg="0°W",
                height_m=0.,
            )


        # Error handling
        else :
            raise NotImplemented("Unknown detector : %s" % name)




    
    # def set_neutrino_source(self,
    #                         # Date, Time and Timezone
    #                         date_str,
    #                         # Location on the sky
    #                         ra_deg=None, 
    #                         dec_deg=None,
    #                         ):

    #     if self.tool == "nusquids" :
    #         raise NotImplementedError()

    #     elif self.tool == "deimos" :
    #         #Set date, time and location of neutrino source
    #         coszen_neutrino_source, altitude_neutrino_source, azimuth_neutrino_source = self.detector_coords.get_coszen_altitude_and_azimuth(
    #             ra_deg = ra_deg, 
    #             dec_deg = dec_deg,
    #             date_str = date_str
    #             )
            
    #         self._neutrino_source_kw = {
    #             # Horizontal Coordinate System
    #             "coszen" : coszen_neutrino_source,
    #             "altitude" : altitude_neutrino_source,
    #             "azimuth" : azimuth_neutrino_source,
    #             # Equatorial Coordinate System
    #             "ra" : ra_deg,
    #             "dec" : dec_deg,
    #             # Store date_str for skymap
    #             "date_str" : date_str,
    #             "sidereal_time" : self.detector_coords.get_local_sidereal_time(date_str)
    #         }
         
    #     else :
    #         raise NotImplemented()


    def calc_osc_prob(self,
        energy_GeV,
        initial_flavor=None,
        initial_state=None,
        distance_km=None,
        coszen=None,
        nubar=False,
        **kw
    ) :
        '''
        For the given model state, calcukate oscillation probabilities for neutrinos as specified in the inputs
        '''

        #TODO caching
        #TODO Option for different final rho to allow nu->nubar transitions

        #
        # Check inputs
        # 
 
         # Handle arrays vs single values for energy
        if isinstance(energy_GeV, (list, np.ndarray)) :
            single_energy, energy_size = False, len(energy_GeV)
        else :
            assert isinstance(energy_GeV, numbers.Number)
            single_energy, energy_size = True, 1

        # Indexing
        if initial_flavor is not None :
            initial_flavor = self._get_flavor_index(initial_flavor)

        #
        # Handle atmospheric mode
        #
        
        if self.atmospheric :

            # Want coszen, not distance
            assert ( (coszen is not None) and (distance_km is None) ), "Must provide `coszen` (and not `distance_km`) in atmospheric mode"  #TODO option to provide distance still in atmo mode

            # Handle single vs array of distances
            if isinstance(coszen, (list, np.ndarray)) :
                coszen = np.array(coszen)
                assert coszen.ndim == 1
                single_dist, dist_size = False, len(coszen)
            else :
                assert isinstance(coszen, numbers.Number)
                coszen = [coszen]
                single_dist, dist_size = True, 1

        else :

            # Want distance, not coszen
            assert ( (distance_km is not None) and (coszen is None) ), "Must provide `distance_km` (and not `coszen`) in non-atmospheric mode" 

            # Handle single vs array of distances
            if isinstance(distance_km, (list, np.ndarray)) :
                single_dist, dist_size = False, len(distance_km)
            else :
                assert isinstance(distance_km, numbers.Number)
                single_dist, dist_size = True, 1



        #
        # Calculate
        #

        # Call sub-function for relevent solver
        if self.tool == "nusquids" :
            osc_probs = self._calc_osc_prob_nusquids( initial_flavor=initial_flavor, initial_state=initial_state, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, nubar=nubar, **kw ) #TODO use single E value for single E mode

        elif self.tool == "deimos" :
            assert initial_flavor is not None, "must provide `initial_flavor` (`initial_state` not currently supported for %s" % self.tool
            osc_probs = self._calc_osc_prob_deimos( initial_flavor=initial_flavor, nubar=nubar, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, **kw )
       
        elif self.tool == "prob3" :
            osc_probs = self._calc_osc_prob_prob3( initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, nubar=nubar, **kw )



        #
        # Done
        #

        # Check shape of output array
        expected_shape = ( energy_size, dist_size, self.num_neutrinos )
        assert osc_probs.shape == expected_shape

        # Remove single-valued dimensions, and check shape again
        # osc_probs = np.squeeze(osc_probs)
        # expected_shape = []
        # if not single_energy :
        #     expected_shape.append(energy_size)
        # if not single_dist :
        #     expected_shape.append(dist_size)
        # expected_shape.append(self.num_neutrinos)
        # expected_shape = tuple(expected_shape)
        # assert osc_probs.shape = expected_shape
        if single_energy and single_dist :
            osc_probs = osc_probs[0,0,:]
        elif single_energy :
            osc_probs = osc_probs[0,:,:]
        elif single_dist :
            osc_probs = osc_probs[:,0,:]

        # Checks
        assert np.all( np.isfinite(osc_probs) ), "Found non-finite osc probs"

        return osc_probs



    def calc_osc_prob_sme(self,
        # Neutrino properties
        energy_GeV,
        ra_rad,
        dec_rad,
        time,
        initial_flavor,
        nubar=False,
        # SME properties
        std_osc=False, # Can toggle standard oscillations (rather than SME)
        basis=None,
        a_eV=None,
        c=None,
        e=None,
        # Args to pass down to the standard osc prob calc
        **kw
    ) :
        '''
        Similar to calc_osc_prob, but for the specific case of the SME where there is also a RA/declination/time dependence 

        Aswell as osc probs, also return the computed direction information
        '''

        #TODO option to provide detector coord info (coszen, azimuth) instead of ra/dec
        #TODO anything required to support skymaps?


        #
        # Check inputs
        #

        # Handle arrays vs single values for RA/dec     #TODO option to pass one of RA/dec as single valued and one as array
        if isinstance(ra_rad, (list, np.ndarray)) :
            assert isinstance(dec_rad, (list, np.ndarray)), "ra_rad and dec_rad must either both be array-like or both scalars"
            ra_rad_values = np.array(ra_rad)
            dec_rad_values = np.array(dec_rad)
            assert ra_rad_values.ndim == 1
            assert dec_rad_values.ndim == 1
            assert ra_rad_values.size == dec_rad_values.size
            single_dir = False
        else :
            assert isinstance(ra_rad, numbers.Number)
            assert isinstance(dec_rad, numbers.Number)
            ra_rad_values = [ra_rad]
            dec_rad_values = [dec_rad]
            single_dir = True

        # Handle arrays vs single values for time
        if isinstance(time, (list, np.ndarray)) :
            time_values = time
            assert np.ndim(time_values) == 1
            single_time = False
        else :
            time_values = [time]
            single_time = True

        # Handle SME vs standard osc
        if std_osc :
            assert basis is None
            assert a_eV is None
            assert c is None
            assert e is None


        #
        # Loop over directions
        #

        osc_probs = []
        coszen_values, azimuth_values = [], []

        # Loop over directions
        for ra_rad, dec_rad in zip(ra_rad_values, dec_rad_values) :

            osc_probs_vs_time = []
            coszen_values_vs_time, azimuth_values_vs_time = [], []


            #
            # Set SME model params 
            #

            # Cannot do this before calling this function as for most oscillation models, due to the RA/declination/time dependence of the Hamiltonian
            # Also might use standard oscillations here, depending on what user requestes

            if std_osc :
                self.set_std_osc()

            else :
                self.set_sme(
                    directional=True,
                    basis=basis,
                    a_eV=a_eV,
                    c=c,
                    e=e,
                    ra_rad=ra_rad,
                    dec_rad=dec_rad,
                )


            # 
            # Loop over times
            #

            for time in time_values :


                #
                # Handle atmospheric vs regular case
                #

                if self.atmospheric :

                    #
                    # Atmospheric case
                    #

                    # Need to know the detector location to get coszen/azimuth from RA/dec
                    assert self.detector_coords is not None, "Must set detector position"

                    # Get local direction coords
                    coszen, altitude, azimuth = self.detector_coords.get_coszen_altitude_and_azimuth(ra_deg=np.rad2deg(ra_rad), dec_deg=np.rad2deg(dec_rad), time=time)

                    # Standard osc prob calc, so this particular direction/time
                    _osc_probs = self.calc_osc_prob(
                        initial_flavor=initial_flavor,
                        nubar=nubar,
                        energy_GeV=energy_GeV,
                        coszen=coszen,
                        **kw # Pass down kwargs
                    )



                else :

                    #
                    # Regular (1D) case
                    #

                    raise NotImplemented("Non-atmospheric case not yet implemented for celestial coords")


                # Merge into the overall output array
                if single_time :
                    osc_probs_vs_time = _osc_probs
                    coszen_values_vs_time = coszen
                    azimuth_values_vs_time = azimuth
                else :
                    osc_probs_vs_time.append( _osc_probs )
                    coszen_values_vs_time.append( coszen )
                    azimuth_values_vs_time.append( azimuth )

            # Merge into the overall output array
            if single_dir :
                osc_probs = osc_probs_vs_time
                coszen_values = coszen_values_vs_time
                azimuth_values = azimuth_values_vs_time
            else :
                osc_probs.append( osc_probs_vs_time )
                coszen_values.append( coszen_values_vs_time )
                azimuth_values.append( azimuth_values_vs_time )

        #
        # Done
        #

        # Array-ify
        osc_probs = np.array(osc_probs)
        coszen_values = np.array(coszen_values)
        azimuth_values = np.array(azimuth_values)

        # Check size
        #TODO

        # Checks
        assert np.all( np.isfinite(osc_probs) ), "Found non-finite osc probs"

        # Return
        return_values = [osc_probs]
        if self.atmospheric :
            return_values.extend([ coszen_values, azimuth_values ])
        return tuple(return_values)



    def _calc_osc_prob_nusquids(self,
        energy_GeV,
        initial_flavor=None,
        initial_state=None,
        nubar=False,
        distance_km=None,
        coszen=None,
    ) :
        '''
        Calculate oscillation probability for the model

        Returned result has following structure: [ energy, coszen, final flavor ]
        '''


        #
        # Prepare
        #

        assert not ( (initial_flavor is None) and (initial_state is None) ), "Must provide `initial_flavor` or `initial_state`"
        assert not ( (initial_flavor is not None) and (initial_state is not None) ), "Must provide `initial_flavor` or `initial_state`, not both"

        # Calculate all final state flavors
        final_flavors = self.states

        # Handle scalars vs arrays
        energy_GeV = np.asarray( [energy_GeV] if np.isscalar(energy_GeV) else energy_GeV )
        if distance_km is not None :
            distance_km = np.asarray( [distance_km] if np.isscalar(distance_km) else distance_km )
        if coszen is not None :
            coszen = np.asarray( [coszen] if np.isscalar(coszen) else coszen )

        # Arrays must be 1D
        assert energy_GeV.ndim == 1
        if distance_km is not None :
            assert distance_km.ndim == 1
        if coszen is not None :
            assert coszen.ndim == 1

        # Handle nubar
        rho = 1 if nubar else 0

        # # Handle nubar
        # if initial_flavor is not None :
        #     if initial_flavor < 0 :
        #         assert include_nubar
        #         initial_flavor = -1* initial_flavor
        #         rho = 1
        #     else :
        #         rho = 0


        #
        # Atmospheric case
        #

        if self.atmospheric :

            randomize_atmo_prod_height = False #TODO support

            # Init results container
            # results = np.full( (energy_GeV.size, coszen.size, final_flavors.size, 2 ), np.NaN )
            results = np.full( (energy_GeV.size, coszen.size, final_flavors.size ), np.NaN )

            # Determine shape of initial state vector in nuSQuIDS
            state_shape = [ self.nusquids.GetNumCos(), self.nusquids.GetNumE() ]
            state_shape.append( 2 ) # nu/nubar
            state_shape.append( final_flavors.size )
            state_shape = tuple(state_shape)

            # Define initial state if not provided, otherwise verify the one provided and re-order to match DEIMOS format
            if initial_state is None :
                initial_state = np.full( state_shape, 0. )
                initial_state[ :, :, rho, initial_flavor ] = 1. # dims = [ cz node, E node, nu(bar), flavor ]
            else :
                # DEIMOS expects [E, cz, flavor] shape (for either nu or nubar), but nuSQuIDS expects [cz, E, nu/nubar, flavor]. Reformat...
                input_initial_state = initial_state
                assert input_initial_state.shape == (state_shape[1], state_shape[0], state_shape[3]), "User provided an incompatible shape for initial state"
                input_initial_state = np.swapaxes(initial_state, 1, 0) # swap E and cz dimensions
                initial_state = np.full( state_shape, 0. )
                for f in range(final_flavors.size) :
                    for r in range(2) : 
                        if r == rho :
                            np.copyto(src=input_initial_state[:,:,f], dst=initial_state[:,:,r,f])
            assert initial_state.shape == state_shape, "Wrong shape for initial state"

            # Set the intial state
            self.nusquids.Set_initial_state(initial_state, nsq.Basis.flavor)

            # Evolve the state
            self.nusquids.EvolveState()

            # Evaluate the flavor at each grid point to get oscillation probabilities
            for i_E,E in enumerate(energy_GeV) :
                for i_cz,cz in enumerate(coszen) :
                    for i_f,final_flavor in enumerate(final_flavors) :
                        # results[i_E,i_cz,i_f] = self.nusquids.EvalFlavor( final_flavor, cz, E*self.units.GeV )#, rho ) #TODO Add randomize prod height arg
                        results[i_E,i_cz,i_f] = self.nusquids.EvalFlavor( int(final_flavor), cz, E*self.units.GeV, int(rho), randomize_atmo_prod_height) #TODO add nubar

            return results


        #
        # Distance case
        #

        else :

            # Init results container
            results = np.full( (energy_GeV.size, distance_km.size, final_flavors.size), np.NaN )
            # results = np.full( (energy_GeV.size, distance_km.size, final_flavors.size, 2), np.NaN )

            # Determine shape of initial state vector in nuSQuIDS
            state_shape = [ self.nusquids.GetNumE() ]
            state_shape.append(2) # nu/nubar
            state_shape.append( final_flavors.size )
            state_shape = tuple(state_shape)

            # Define initial state if not provided, otherwise verify the one provided
            if initial_state is None :
                initial_state = np.full( state_shape, 0. )
                initial_state[ :, rho, initial_flavor ] = 1. # dims = [ E node, nu(bar), flavor ]
            else :
                assert initial_state.shape == state_shape, "Incompatible shape for initial state : Expected %s, found %s" % (state_shape, initial_state.shape)

            # Loop over distance nodes
            for i_L, L in enumerate(distance_km) :


                #
                # Propagate the neutrino in 1D
                #

                #TODO keep evolving from previous (shorter) distance node rather than re-calculating from 0 every time (for efficiency)?

                # Set the track (e.g. neutrino travel path), taking medium into account. Then propagate
                if self._matter_settings["matter"] == "vacuum" :

                    # Vacuum is easy: Just propagate in vacuum
                    self.nusquids.Set_Track(nsq.Vacuum.Track(L*self.units.km))
                    self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
                    self.nusquids.EvolveState()

                elif self._matter_settings["matter"] == "constant" :

                    # Constant density is easy: Just propagate in constant density medium
                    self.nusquids.Set_Track(nsq.ConstantDensity.Track(L*self.units.km))
                    self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor )
                    self.nusquids.EvolveState()

                elif self._matter_settings["matter"] == "layers" :

                    # Layers on constant density are a bit more tricky. Step through them, evolving the state though each later, then changing density and continuing the state evolution (wuthout resetting it)
                    # Take care to cut off when reach the requested propagation distance

                    # Check the layers cover the full path length
                    assert self._matter_settings["layer_endpoint_km"][-1] >= L, "Matter layers do not cover the full baseline"

                    # Loop through layers
                    L_so_far = 0.
                    for endpoint, density, efrac in zip(self._matter_settings["layer_endpoint_km"], self._matter_settings["matter_density_g_per_cm3"], self._matter_settings["electron_fraction"]) :

                        # Bail out if have reached travel distance
                        if L_so_far > endpoint :
                            break

                        # Figure out how far we will travel in this layer
                        if L < endpoint :
                            endpoint = L # Do not step past endpoint
                        L_layer = endpoint - L_so_far

                        # Set the body and track, and propagate
                        self.nusquids.Set_Body(nsq.ConstantDensity(density, efrac))
                        self.nusquids.Set_Track(nsq.ConstantDensity.Track(L_layer*self.units.km))
                        if L_so_far == 0 :
                            self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor ) # Only first step
                        self.nusquids.EvolveState()

                        # Update distance counter
                        L_so_far += L_layer

                else :
                    raise Exception("Unknown matter : %s" % self._matter_settings["matter"]) 


                #
                # Evaluate final state
                #

                # Loop over energies
                for i_e, E in enumerate(energy_GeV) :

                    # Evaluate final state flavor composition
                    for i_f, final_flavor in enumerate(final_flavors) :
                        # for rho in [0, 1] :
                        #     results[i_e,i_L,i_f,rho] = self.nusquids.EvalFlavor( int(final_flavor), float(E*self.units.GeV), int(rho) )
                        results[i_e,i_L,i_f] = self.nusquids.EvalFlavor( int(final_flavor), float(E*self.units.GeV), int(rho) )

            return results


    def _calc_osc_prob_prob3(self,
        initial_flavor,
        energy_GeV,
        distance_km=None,
        coszen=None,
        nubar=False,
    ) :


        #
        # Check inputs
        #

        # Check num flavors
        assert self.num_neutrinos == 3, "prob3 wrapper only supporting 3-flavor oscillations currently" #TODO probably can add supoort for N != 3

        # Note that coszen vs distance handling already done in level above

        #TODO coszen->distance conversion for vacuum atmo case



        #
        # Define system
        #

        # Get osc propeties
        theta12, theta13, theta23 = self.get_mixing_angles()
        sin2_theta12, sin2_theta13, sin2_theta23 = np.square(np.sin(theta12)), np.square(np.sin(theta13)), np.square(np.sin(theta23))
        deltacp = self.get_deltacp()
        dm21, dm31 = self.get_mass_splittings()
        dm32 = dm31 - dm21 #TODO careful with mass ordering
        KNuType = -1 if nubar else +1

        # Dertemine all final states
        final_flavors = self.states

        # Determine matter
        earth, matter_density_g_per_cm3 = False, None
        vacuum = self._prob3_settings["matter"] is None
        if not vacuum :
            if self._prob3_settings["matter"] == "earth" :
                earth = True
            else :
                assert self._prob3_settings["matter"] == "constant"
                matter_density_g_per_cm3 = self._prob3_settings["matter_density_g_per_cm3"]


        #
        # Loop over energy/coszen
        #

        # Array-ify
        energy_GeV = np.asarray( [energy_GeV] if np.isscalar(energy_GeV) else energy_GeV )
        distance_km = np.asarray( [distance_km] if np.isscalar(distance_km) else distance_km )
 
        # Init outputs container
        energy_dim = np.size(energy_GeV)
        distance_dim = np.size(distance_km)
        results = np.full( (energy_dim, distance_dim, self.num_neutrinos), np.NaN )

        # Loop over energy
        for i_E in range(energy_dim) :

            # Update propagator settings
            # Must do this each time energy changes, but don't need to for distance
            self._propagator.SetMNS(
                sin2_theta12, # sin2_theta12,
                sin2_theta13, # sin2_theta13,
                sin2_theta23, # sin2_theta23,
                dm21, # dm12,
                dm32, # dm23,
                deltacp, # delta_cp [rad]   #TODO get diagreeement between prob3 and other solvers (DEIMOS, nuSQuIDS) when this is >0, not sure why?
                energy_GeV[i_E], # Energy
                True, # True means expect sin^2(theta), False means expect sin^2(2*theta)
                KNuType,
            )

            # Loop over distance f
            for i_L in range(distance_dim) :

                # Loop over flavor
                for i_f, final_flavor in enumerate(final_flavors) :


                    #
                    # Calc osc probs
                    #

                    # Handle prob3 flavor index format: uses [1,2,3], not [0,1,3]
                    initial_flavor_prob3 = initial_flavor + 1 # prob3 
                    final_flavor_prob3 = final_flavor + 1

                    # Calculation depends of matter type
                    if vacuum :

                        # Run propagation and calc osc probs
                        P = self._propagator.GetVacuumProb( initial_flavor_prob3, final_flavor_prob3 , energy_GeV[i_E],  distance_km[i_L])

                    else :

                        # Toggle between Earth vs constant density
                        if earth :

                            # Propagate in Earth
                            raise Exception("Not yet implemented") #TODO need to handle coszen, etc
                            self._propagator.DefinePath( cosineZ, prod_height )
                            self._propagator.propagate( KNuType )

                        else :

                            # Propagate in constant density matter
                            self._propagator.propagateLinear( KNuType, distance_km[i_L] , matter_density_g_per_cm3 )

                        # Calc osc probs for mater case (after propagation done abopve already)
                        P = self._propagator.GetProb( initial_flavor_prob3, final_flavor_prob3 )


                    # Set to output array
                    assert np.isscalar(P)
                    results[i_E, i_L, i_f] = P


        return results


    def _calc_osc_prob_deimos(self,

        # Neutrino definition
        initial_flavor,
        energy_GeV,
        distance_km=None,
        coszen=None,
        nubar=False,

        # Neutrino direction in celestial coords - only required for certain models (such as the SME)
        ra_rad=None,
        dec_rad=None,

        **kw

    ) :

        #
        # Prepare
        #

        # Calculate all final state flavors
        final_flavors = self.states

        # Handle scalars vs arrays
        energy_GeV = np.asarray( [energy_GeV] if np.isscalar(energy_GeV) else energy_GeV )
        if distance_km is not None :
            distance_km = np.asarray( [distance_km] if np.isscalar(distance_km) else distance_km )
        if coszen is not None :
            coszen = np.asarray( [coszen] if np.isscalar(coszen) else coszen )

        # Arrays must be 1D
        assert energy_GeV.ndim == 1
        if distance_km is not None :
            assert distance_km.ndim == 1
        if coszen is not None :
            assert coszen.ndim == 1


        #
        # Calculate
        #

        # coszen -> L conversion (for atmospheric case)
        if self.atmospheric :
            production_height_km = DEFAULT_ATMO_PROD_HEIGHT_km #TODO steerable, randomizable
            detector_depth_km = DEFAULT_ATMO_DETECTOR_DEPTH_km if self.detector_coords is None else self.detector_coords.detector_depth_m*1e-3 # Use detector position, if available    #TODO should we really be defining this as height?
            distance_km = calc_path_length_from_coszen(cz=coszen, h=production_height_km, d=detector_depth_km)

        # DensityMatrixOscSolver doesn't like decending distance values in the input arrays,
        # and this is what you get from coszen arrays often
        flip = False
        if distance_km[-1] < distance_km[0] : 
            flip = True
            distance_km = np.flip(distance_km)

        # Run solver
        # 'results' has shape [N energy, N distance, N flavor]
        results = self.solver.calc_osc_probs(
            E_GeV=energy_GeV,
            L_km=distance_km,
            initial_state=initial_flavor,
            initial_basis="flavor",
            nubar=nubar,
            calc_basis=self._calc_basis,
            # D_matrix_basis=self._decoherence_D_matrix_basis,
            decoh_opts=self._decoh_model_kw,
            lightcone_opts=self._lightcone_model_kw,
            sme_opts=self._sme_model_kw,
            verbose=False,
            **kw
        )

        # Handle flip in results (L dimension)
        if flip :
            results = np.flip(results, axis=1)

        return results



    def calc_final_flux(self,
        source,
        energy_GeV,
        coszen,
        nubar=False,
        tool=None,
    ) :
        '''
        Propagate an initial flux to a final flux, accounting for oscillations, matter, etc

        This largely re-uses calc_osc_probs() but with a different format for the initial state
        '''

        #
        # Check inputs
        # 

        # Some limitations of current implementation
        assert self.atmospheric, "Currently only supporting flux propagation in atmospheric mode"


        #
        # Get initial flux
        #

        # Get the initital flux before any propagation  #TODO ideally let user provide any flux of their choosing, but run into issues with nusquids since it needs the flux values at its nodes
        get_neutrino_flux_kw = dict(grid=True, source=source, tool=tool, overwrite_cache=False)
        initial_flux = self.get_neutrino_flux(energy_GeV=energy_GeV, coszen=coszen, **get_neutrino_flux_kw)

        # Remove nubar dim
        rho = 1 if nubar else 0
        initial_flux = initial_flux[:, :, :, rho]*1e18 #TODO remove this factor, which is just to make the calculation work for now


        #
        # Handle differently for different solvers
        #

        # Check tool
        if self.tool == "nusquids" :


            #
            # nuSQuIDS
            #

            # For nuSQuIDS, must define the initital state vector as the flux at the E and coszen nodes of the nuSQuIDSAtm instance
            initial_state = self.get_neutrino_flux(energy_GeV=self.energy_nodes_GeV, coszen=self.coszen_nodes, **get_neutrino_flux_kw)
            initial_state = initial_state[:, :, :, rho]*1e18
            print("Initial state at an energy of {} GeV:".format(energy_GeV[0]) ,initial_state[0,0,:])

            # Propagate, (ab)using the calc_osc_prob function
            final_flux = self.calc_osc_prob(
                energy_GeV=energy_GeV,
                initial_state=initial_state,
                coszen=coszen,
                nubar=nubar,
            )
            #print("Final state at an energy of {} GeV:".format(energy_GeV[0]) ,final_flux[0,0,:] )

            #TODO potential issues due to differing nodes for the generation of the initial and final state here (MCEq interpolation and nuSQuIDS interpolation). Make it safer though by setting the plotting grid as the nuSQuIDS nodes...

        else :

            raise NotImplemented("TODO: calc_final_flux not implemented for %s" % self.tool)


        #
        # Done
        #

        # Check shapes match
        assert initial_flux.shape == final_flux.shape

        # Checks
        assert np.all( np.isfinite(initial_flux) ), "Found non-finite initial flux"
        assert np.all( np.isfinite(final_flux) ), "Found non-finite final flux"

        return initial_flux, final_flux



    def _get_flavor_index(self,flavor) :

        index = None

        if isinstance(flavor, str) :
            if flavor in ["e","nue"] :
                index = 0
            elif flavor in ["mu","numu"] :
                index = 1
            elif flavor in ["tau","nutau"] :
                index = 2

        else :
            assert flavor in [0,1,2]
            index = flavor

        assert flavor < self.num_neutrinos

        return index 

    def set_colors(self, nu_colors) :
        self.nu_colors = nu_colors


    @property
    def states(self) :
        return np.array(range(self.num_neutrinos))


    def get_flavor_tex(self, i) :
        '''
        Get tex representation of flavor i (e.g. e, mu, tau)
        '''

        assert i < self.num_neutrinos
        flavor = self.flavors[i]

        if flavor == "e" :
            return r"e"
        elif flavor == "mu" :
            return r"\mu"
        elif flavor == "tau" :
            return r"\tau"
        else :
            raise Exception("Unknown flavor : %s" % flavor)


    def get_nu_flavor_tex(self, i=None, nubar=False) :
        '''
        Get tex representation of neutrino flavor i (e.g. nue, numu, nutau)
        '''

        nu_tex = r"\nu"

        if nubar :
            nu_tex = r"\bar{" + nu_tex + r"}"

        if i is None :
            nu_tex += r"_{\rm{all}}"
        else :
            flavor_tex = self.get_flavor_tex(i)
            nu_tex += r"_{" + flavor_tex + r"}"

        return nu_tex


    def get_nu_mass_tex(self, i=None, nubar=False) :
        '''
        Get tex representation of neutrino mass state i (e.g. nu_1, numu_2, nutau_3)
        '''

        nu_tex = r"\nu"

        if nubar :
            nu_tex = r"\bar{" + nu_tex + r"}"

        if i is None :
            nu_tex += r"_{\rm{all}}"
        else :
            nu_tex += r"_{" + (i+1) + r"}"

        return nu_tex


    # @property
    # def flavors_tex(self) :
    #     return [ self.get_flavor_tex(i) for i in self.states ]


    # @property
    # def masses_tex(self) :
    #     return [ self.get_mass_tex(i) for i in self.states ]


    # @property
    # def flavors_color(self) :
    #     return [ self.get_flavor_color(i) for i in self.states ]

    # def get_flavor_color(self, flavor) :
    #     return self.nu_colors[flavor]


    def get_transition_prob_tex(self, initial_flavor, final_flavor, nubar=False) :
        return r"P(%s \rightarrow %s)" % ( self.get_nu_flavor_tex(initial_flavor, nubar), self.get_nu_flavor_tex(final_flavor, nubar) )


    @property
    def PMNS(self) :
        '''
        Return the PMNS matrix
        '''

        if self.tool == "nusquids" :
            if self.num_neutrinos == 2 :
                theta = [ self.nusquids.Get_MixingAngle(0,1) ]
            elif self.num_neutrinos == 3 :
                theta = [ self.nusquids.Get_MixingAngle(0,1), self.nusquids.Get_MixingAngle(0,2), self.nusquids.Get_MixingAngle(1,2) ]
            else :
                raise Exception("`PMNS` function only supports 2/3 flavors")
            deltacp = self.nusquids.Get_CPPhase(0,2) #TODO check indices
            return get_pmns_matrix( theta=theta, dcp=deltacp )


        elif self.tool == "deimos" :
            return self.solver.PMNS


    def plot_osc_prob_vs_distance(self, 
        # Steer physics
        initial_flavor, 
        energy_GeV, 
        distance_km=None, coszen=None, 
        nubar=False, 
        final_flavor=None,
        # Plotting
        fig=None, ax=None, 
        label=None, 
        title=None,
        xscale="linear",
        ylim=None,
        **plot_kw
    ) :
        '''
        Compute and plot the oscillation probability, vs propagation distance
        '''

        # Handle distance vs coszen
        if self.atmospheric :
            assert coszen is not None
            dist_kw = {"coszen" : coszen}
            x = coszen
            xlabel = COSZEN_LABEL
        else :
            assert distance_km is not None
            dist_kw = {"distance_km" : distance_km}
            x = distance_km
            xlabel = DISTANCE_LABEL

        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(x, np.ndarray)
        assert np.isscalar(energy_GeV)
        assert isinstance(nubar, bool)
        if final_flavor is not None :
            assert isinstance(final_flavor, int)

        # User may provide a figure, otherwise make one
        ny = ( self.num_neutrinos + 1 ) if final_flavor is None else 1
        if fig is None : 
            fig, ax = plt.subplots( nrows=ny, sharex=True, figsize=( 6, 4 if ny == 1 else 2*ny) )
            if ny == 1 :
                ax = [ax]

        else :
            assert ax is not None
            assert len(ax) == ny

        # Handle title
        if title is not None :
            fig.suptitle(title) 

        # Calc osc probs
        osc_probs = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Plot oscillations to all possible final states
        final_flavor_values = self.states if final_flavor is None else [final_flavor]
        for i, final_flavor in enumerate(final_flavor_values) :
            ax[i].plot( x, osc_probs[:,final_flavor], label=label, **plot_kw )
            ax[i].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )

        # Plot total oscillations to any final state
        if len(final_flavor_values) > 1 :
            osc_probs_flavor_sum = np.sum(osc_probs,axis=1)
            ax[-1].plot( x, osc_probs_flavor_sum, label=label, **plot_kw ) # Dimension 2 is flavor
            ax[-1].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, None, nubar) )

        # Formatting
        if ylim is None :
            ylim = (-0.05, 1.05)
        ax[-1].set_xlabel(xlabel)
        if label is not None :
            ax[0].legend(fontsize=12) # loc='center left', bbox_to_anchor=(1, 0.5), 
        for this_ax in ax :
            this_ax.set_ylim(ylim)
            this_ax.set_xlim(x[0], x[-1])
            this_ax.set_xscale(xscale)
            this_ax.grid(True)
        fig.tight_layout()

        return fig, ax, osc_probs


    def plot_osc_prob_vs_cozen(self, coszen, *args, **kwargs) : # Alias
        return self.plot_osc_prob_vs_distance(coszen=coszen, *args, **kwargs)



    def plot_osc_prob_vs_energy(self, 
        # Steer physics
        initial_flavor, 
        energy_GeV, 
        distance_km=None, coszen=None, 
        nubar=False, 
        final_flavor=None,
        # Plotting
        fig=None, ax=None, 
        label=None, 
        title=None,
        xscale="linear",
        ylim=None,
        plot_LoE=False,
        **plot_kw
    ) :
        '''
        Compute and plot the oscillation probability, vs neutrino energy
        '''

        # Handle distance vs coszen
        if self.atmospheric :
            assert coszen is not None
            dist_kw = {"coszen" : coszen}
            x = coszen
        else :
            assert distance_km is not None
            dist_kw = {"distance_km" : distance_km}
            x = distance_km

        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert np.isscalar(x)
        assert isinstance(nubar, bool)
        if final_flavor is not None :
            assert isinstance(final_flavor, int)

        # User may provide a figure, otherwise make one
        ny = ( self.num_neutrinos + 1 ) if final_flavor is None else 1
        if fig is None : 
            fig, ax = plt.subplots( nrows=ny, sharex=True, figsize=( 6, 4 if ny == 1 else 2*ny) )
            if ny == 1 :
                ax = [ax]
        else :
            assert ax is not None
            assert len(ax) == ny

        # Handle title
        if title is not None :
            fig.suptitle(title) 

        # Calc osc probs
        osc_probs = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Convert to L/E, if requested
        xplot = energy_GeV
        if plot_LoE :
            assert not self.atmospheric, "Need to handle coszen conversion for L/E plot"
            LoE = dist_kw["distance_km"] / energy_GeV
            xplot = LoE

        # Plot oscillations to all possible final states
        final_flavor_values = self.states if final_flavor is None else [final_flavor]
        for i, final_flavor in enumerate(final_flavor_values) :
            ax[i].plot( xplot, osc_probs[:,final_flavor], label=label, **plot_kw )
            ax[i].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, final_flavor, nubar) )

        # Plot total oscillations to any final state
        if len(final_flavor_values) > 1 :
            osc_probs_flavor_sum = np.sum(osc_probs,axis=1)
            ax[-1].plot( xplot, osc_probs_flavor_sum, label=label, **plot_kw ) # Dimension 2 is flavor
            ax[-1].set_ylabel( r"$%s$" % self.get_transition_prob_tex(initial_flavor, None, nubar) )

        # Formatting
        if ylim is None :
            ylim = (-0.05, 1.05)
        if plot_LoE :
            ax[-1].set_xlabel("%s / %s" % (DISTANCE_LABEL, ENERGY_LABEL))
        else :
            ax[-1].set_xlabel(ENERGY_LABEL)
        if label is not None :
            ax[0].legend(fontsize=10) # loc='center left', bbox_to_anchor=(1, 0.5), 
        for this_ax in ax :
            if plot_LoE :
                this_ax.set_xlim(xplot[-1], xplot[0]) # Reverse
            else :
                this_ax.set_xlim(xplot[0], xplot[-1])
            this_ax.set_ylim(ylim)
            this_ax.grid(True)
            this_ax.set_xscale(xscale)
        fig.tight_layout()

        return fig, ax, osc_probs


    def plot_cp_asymmetry() :
        '''
        Plot the CP(T) asymmetry
        '''

        raise NotImplemented("TODO")


    def plot_oscillogram(
        self,
        initial_flavor,
        final_flavor,
        energy_GeV,
        coszen,
        nubar=False,
        title=None,
        ax=None,
        vmax=1.,
    ) :
        '''
        Helper function for plotting an atmospheric neutrino oscillogram
        '''
        from deimos.utils.plotting import plot_colormap, value_spacing_is_linear

        assert self.atmospheric, "`plot_oscillogram` can only be called in atmospheric mode"

        #
        # Steering
        #

        # Plot steering
        transition_prob_tex = self.get_transition_prob_tex(initial_flavor, final_flavor, nubar)
        continuous_map = "jet" # plasma jet
        # diverging_cmap = "seismic" # PuOr_r RdYlGn Spectral


        #
        # Compute osc probs
        #

        # Define osc prob calc settings
        calc_osc_prob_kw = dict(
            initial_flavor=initial_flavor,
            nubar=nubar,
            energy_GeV=energy_GeV,
            coszen=coszen, 
        )

        # Calc osc probs 
        osc_probs = self.calc_osc_prob( **calc_osc_prob_kw )

        # Get chose flavor/rho
        osc_probs = osc_probs[:, :, final_flavor]

        #
        # Plot
        #

        # Create fig
        fig = None
        if ax is None :
            fig, ax = plt.subplots( figsize=(7, 6) )

        # Title
        if title is not None :
            ax.set_title(title) 

        # Plot oscillogram
        plot_colormap( ax=ax, x=energy_GeV, y=coszen, z=osc_probs, vmin=0., vmax=vmax, cmap=continuous_map, zlabel=r"$%s$"%transition_prob_tex )

        # Format
        xscale = "linear" if value_spacing_is_linear(energy_GeV) else "log"
        yscale = "linear" if value_spacing_is_linear(coszen) else "log"
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(energy_GeV[0], energy_GeV[-1])
        ax.set_ylim(coszen[0], coszen[-1])
        ax.set_xlabel(ENERGY_LABEL)
        ax.set_ylabel(COSZEN_LABEL)
        if fig is not None :
            fig.tight_layout()

        return fig, ax, osc_probs


    def plot_right_ascension_vs_energy_2D(
            self,
            # Steer physics
            initial_flavor,
            energy_GeV,
            distance_km=None, coszen=None,
            nubar=False,
            final_flavor=None,
            # Plotting
            fig=None, ax=None,
            label=None,
            title=None,
            xscale="linear",
            ylim=None,
            **plot_kw
    ):
        '''
        Make a 2D plot of oscillation probabilities vs neutrino energy (x-axis) and right ascension (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
            
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            dec_0 = np.deg2rad(self._neutrino_source_kw["dec"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\delta \sim {:.2f}$".format(dec_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
    
        # Calculate probabilities
        probabilities2d = self.calc_osc_prob(
            initial_flavor = initial_flavor,
            energy_GeV = energy_GeV,
            **dist_kw
        )
        
        # Transpose array to plot right ascension vs. energy
        probabilities2d = np. transpose(probabilities2d, (1,0,2) )
    
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(probabilities2d[:, :, i] < -0.1) or np.any(probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of oscillation probabilities outside the range [0, 1].", UserWarning)
                
            # Plot the results
            if self._sme_model_kw: 
                im = ax[i].pcolormesh(energy_GeV,
                                      np.deg2rad(self._neutrino_source_kw["ra"]),
                                      probabilities2d[:, :, i],
                                      vmin=0, vmax=1.0, 
                                      cmap='RdPu')
                ax[i].set_ylabel("Right Ascension (rad)")
            
            else:
                im = ax[i].pcolormesh(energy_GeV,
                                      coszen,
                                      probabilities2d[:, :, i],
                                      vmin=0, vmax=1.0, 
                                      cmap='RdPu')
                ax[i].set_ylabel("Coszen")
            ax[i].set_xscale(xscale)
    
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < 0.9) or np.any(osc_probs_flavor_sum > 1.1):
                warnings.warn("Values outside the range [0.9, 1.1] in osc_probs_flavor_sum.", UserWarning)
            
            if self._sme_model_kw: 
                ax[-1].pcolormesh(energy_GeV,
                    np.deg2rad(self._neutrino_source_kw["ra"]),
                    osc_probs_flavor_sum,
                    vmin=0.9, vmax=1.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Right Asceionsion (rad)")
            
            else:
                im = ax[-1].pcolormesh(energy_GeV,
                    coszen,
                    osc_probs_flavor_sum,
                    vmin=0.9, vmax=1.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Coszen")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[-1], label=r"$P(\nu_{\mu}\rightarrow \nu_{all})$")
    
        plt.tight_layout()
        plt.show()

        return fig, ax, probabilities2d
    
    
    def plot_declination_vs_energy_2D(
        self,
        # Steer physics
        initial_flavor,
        energy_GeV,
        distance_km=None, coszen=None,
        nubar=False,
        final_flavor=None,
        # Plotting
        fig=None, ax=None,
        label=None,
        title=None,
        xscale="linear",
        ylim=None,
        **plot_kw
    ):
        '''
        Make a 2D plot of oscillation probabilities vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Set title of figure     
        # TODO adjust title of plots
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\alpha \sim {:.2f}$".format(ra_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
    
        # Calculate probabilities
        probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
    
        # Transpose array to plot declination vs. energy
        probabilities2d = np.transpose(probabilities2d, (1, 0, 2))
    
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
            
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(probabilities2d[:, :, i] < -0.1) or np.any(probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of oscillation probabilities outside the range [0, 1].", UserWarning)
              
            # Plot the results
            if self._sme_model_kw: 
                im = ax[i].pcolormesh(energy_GeV,
                                      np.deg2rad(self._neutrino_source_kw["dec"]),
                                      probabilities2d[:, :, i],
                                      vmin=0, vmax=1.0, 
                                      cmap='RdPu')
                ax[i].set_ylabel("Declination (rad)")
            
            else:
                im = ax[i].pcolormesh(energy_GeV,
                                      coszen,
                                      probabilities2d[:, :, i],
                                      vmin=0, vmax=1.0, 
                                      cmap='RdPu')
                ax[i].set_ylabel("Coszen")
                
            ax[i].set_xscale(xscale)
    
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < 0.9) or np.any(osc_probs_flavor_sum > 1.1):
                warnings.warn("Values outside the range [0.9, 1.1] in osc_probs_flavor_sum.", UserWarning)
            
            if self._sme_model_kw: 
                ax[-1].pcolormesh(energy_GeV,
                    np.deg2rad(self._neutrino_source_kw["dec"]),
                    osc_probs_flavor_sum,
                    vmin=0.9, vmax=1.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Declination (rad)")
            
            else:
                im = ax[-1].pcolormesh(energy_GeV,
                    coszen,
                    osc_probs_flavor_sum,
                    vmin=0.9, vmax=1.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Coszen")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(im, ax=ax[-1], label=r"$P(\nu_{\mu}\rightarrow \nu_{all})$")
    
        plt.tight_layout()
        plt.show()
    
        return fig, ax, probabilities2d
    
    
    def plot_declination_vs_energy_2D_diff(
    self,
    # Steer physics
    initial_flavor,
    energy_GeV,
    distance_km=None, coszen=None,
    nubar=False,
    final_flavor=None,
    # Plotting
    fig=None, ax=None,
    label=None,
    title=None,
    xscale="linear",
    ylim=None,
    **plot_kw
    ):
        '''
        Make a 2D plot of the difference of oscillation probabilities between standard osc and with SME
        vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            ra_0 = np.deg2rad(self._neutrino_source_kw["ra"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\alpha \sim {:.2f}$".format(ra_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
            
        # Calculate probabilities for the non-standard case
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities2d = sme_probabilities2d - standard_probabilities2d

        # Select a colormap
        cmap_diff = 'bwr'

        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(diff_probabilities2d[:, :, i] < -1.1) or np.any(diff_probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)
            
            # Use the custom colormap to plot the difference of probabilities
            if self._sme_model_kw: 
                im = ax[i].pcolormesh(energy_GeV,
                                      np.deg2rad(self._neutrino_source_kw["dec"]),
                                      diff_probabilities2d[:, :, i],
                                      vmin=-1.0, vmax=1.0, 
                                      cmap=cmap_diff)
                ax[i].set_ylabel("Declination (rad)")
            
            else:
                im = ax[i].pcolormesh(energy_GeV,
                                      coszen,
                                      diff_probabilities2d[:, :, i],
                                      vmin=-1.0, vmax=1.0, 
                                      cmap=cmap_diff)
                ax[i].set_ylabel("Coszen")
            
            ax[i].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
            
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = diff_probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < -0.1) or np.any(osc_probs_flavor_sum > 0.1):
                warnings.warn("Values outside the range [-0.1, 0.1] in osc_probs_flavor_sum.", UserWarning)

            if self._sme_model_kw: 
                ax[-1].pcolormesh(energy_GeV,
                    np.deg2rad(self._neutrino_source_kw["dec"]),
                    osc_probs_flavor_sum,
                    vmin=-0.1, vmax=0.1,
                    cmap=cmap_diff)
                
                ax[-1].set_ylabel("Declination (rad)")
            
            else:
                im = ax[-1].pcolormesh(energy_GeV,
                    coszen,
                    osc_probs_flavor_sum,
                    vmin=-0.1, vmax=0.1,
                    cmap=cmap_diff)
                
                ax[-1].set_ylabel("Coszen")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(im, ax=ax[-1], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$")
    

        return fig, ax, diff_probabilities2d
    
    
    def plot_right_ascension_vs_energy_2D_diff(
    self,
    # Steer physics
    initial_flavor,
    energy_GeV,
    distance_km=None, coszen=None,
    nubar=False,
    final_flavor=None,
    # Plotting
    fig=None, ax=None,
    label=None,
    title=None,
    xscale="linear",
    ylim=None,
    **plot_kw
    ):
        '''
        Make a 2D plot of the difference of oscillation probabilities between standard osc and with SME
        vs neutrino energy (x-axis) and declination (y-axis).
        '''
    
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
    
        # User may provide a figure, otherwise make one
        ny = self.num_neutrinos + 1 if final_flavor is None else 1
        if fig is None:
            fig, ax = plt.subplots(nrows=ny, sharex=True, sharey=False, figsize=(6, 4 * ny))
            if ny == 1:
                ax = [ax]
            if title is not None:
                for this_ax in ax:
                    this_ax.set_title(title)  # Set the same title for all subplots
        else:
            assert ax is not None
            assert len(ax) == ny
            assert title is None
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
            dec_0 = np.deg2rad(self._neutrino_source_kw["dec"][0])
    
        # Set title of figure     
        if self._sme_model_kw:
            fig.suptitle("SME",
                # r"$\delta \sim {:.2f}$".format(dec_0)
                # + r", $a^X = {:.2e} \, \rm GeV$".format(a_eV[0])
                # + r", $a^Y = {:.2e} \, \rm GeV$".format(a_eV[1])
                # + r", $c^X = {:.2e}$".format(c[0])
                # + r", $c^Y = {:.2e}$".format(c[1]),
                fontsize=14,
            )
        else:
            fig.suptitle("Standard osc",
                fontsize=14,
            )
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
            
        # Calculate probabilities for the non-standard case
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )

        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities2d = sme_probabilities2d - standard_probabilities2d

        # Create a custom colormap
        cmap_diff = 'bwr'
        
        # Define the possible final states
        final_states = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        
        # Loop over each final state and create the corresponding plot
        for i, final_flavor in enumerate(final_states):
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(diff_probabilities2d[:, :, i] < -1.1) or np.any(diff_probabilities2d[:, :, i] > 1.1):
                warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)
               
            # Use the custom colormap to plot the difference of probabilities
            if self._sme_model_kw: 
                im = ax[i].pcolormesh(energy_GeV,
                                      np.deg2rad(self._neutrino_source_kw["ra"]),
                                      diff_probabilities2d[:, :, i],
                                      vmin=-1.0, vmax=1.0, 
                                      cmap=cmap_diff)
                ax[i].set_ylabel("Declination (rad)")
            
            else:
                im = ax[i].pcolormesh(energy_GeV,
                                      coszen,
                                      diff_probabilities2d[:, :, i],
                                      vmin=-1.0, vmax=1.0, 
                                      cmap=cmap_diff)
                ax[i].set_ylabel("Coszen")
            ax[i].set_xscale(xscale)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax[i], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor + r"})$")
    
            
        # Plot total oscillations to any final state
        if final_flavor is not None:
            osc_probs_flavor_sum = diff_probabilities2d.sum(axis=-1)
            
            # Check for values outside the range [0.9, 1.1]
            if np.any(osc_probs_flavor_sum < -0.1) or np.any(osc_probs_flavor_sum > 0.1):
                warnings.warn("Values outside the range [-0.1, 0.1] in osc_probs_flavor_sum.", UserWarning)
            
            if self._sme_model_kw: 
                ax[-1].pcolormesh(energy_GeV,
                    np.deg2rad(self._neutrino_source_kw["ra"]),
                    osc_probs_flavor_sum,
                    vmin=-0.1, vmax=0.1,
                    cmap=cmap_diff)
                
                ax[-1].set_ylabel("Right Ascension (rad)")
            
            else:
                im = ax[-1].pcolormesh(energy_GeV,
                    coszen,
                    osc_probs_flavor_sum,
                    vmin=-0.1, vmax=0.1,
                    cmap="RdPu")
                
                ax[-1].set_ylabel("Coszen")
            ax[-1].set_xlabel(ENERGY_LABEL)
            ax[-1].set_xscale(xscale)
        
            #Add colorbar
            cbar = fig.colorbar(im, ax=ax[-1], label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$")
    

        return fig, ax, diff_probabilities2d
    

    def plot_healpix_map(
            self,
            healpix_map,
            visible_sky_map,
            nside,
            title,
            cbar_label,
            cmap='viridis',
            min_val = -1,
            max_val = 1,
            ):
        
        #Plot in the mollview projection
        projected_map = hp.mollview(
                            map = healpix_map, 
                            title=title + "\n", 
                            cmap=cmap,
                            xsize=2000,
                            # rotation in the form (lon, lat, psi) (unit: degrees) : the point at longitude lon and latitude lat will be at the center.
                            # An additional rotation of angle psi around this direction is applied.
                            rot=(180, 0, 0),
                            # equatorial (celestial) coordinate system
                            coord='C',
                            # east towards left, west towards right
                            flip = 'astro',
                            min=min_val,
                            max=max_val,
                            cbar=False,
                            return_projected_map=True,
                            # Allow overlaying
                            hold = True
                            )
        # Overlay the visible_sky map
        hp.mollview(
            map=visible_sky_map,  # Add the visible_sky map as an overlay
            cmap='Greys',  # Set the colormap to 'Greys'
            xsize=2000,
            # An additional rotation of angle psi around this direction is applied.
            rot=(180, 0, 0),
            # equatorial (celestial) coordinate system
            coord='C',
            # east towards left, west towards right
            flip = 'astro',
            min = 0,
            max =1,
            # Set opacity to 0.2
            alpha=visible_sky_map,  
            # Allow overlaying
            reuse_axes=True,
            cbar=False,
            )
        
        # Add meridians and parallels
        hp.graticule()
        
        # Add declination labels
        for dec in np.arange(-75, 0, 15):
            #lonlat If True, theta and phi are interpreted as longitude and latitude in degree, otherwise, as colatitude and longitude in radian
            hp.projtext(359.9, dec, "\n" + f"{dec}°   ", lonlat=True, color="black", ha="right", va="center")  
        for dec in np.arange(0, 76, 15):
            if dec == 0:
                hp.projtext(359.9, dec, r"Declination $\delta$" + "\n\n", lonlat=True, color="black", ha="right", va="center", rotation ='vertical')
                continue
            hp.projtext(359.9, dec, f"{dec}°   " + "\n", lonlat=True, color="black", ha="right", va="center")
            
        # Add the right ascension labels
        hp.projtext(359.9, 0, "24h ", lonlat=True, color="black", ha="right", va="center")
        hp.projtext(0, 0, " 0h", lonlat=True, color="black", ha="left", va="center")
        hp.projtext(180, -90, "\n\n\n12h" + "\nRight Ascension " +  r"$\alpha$", lonlat=True, color="black", ha="center", va="center")
    
        # Create an empty image plot as a mappable for the colorbar
        img = plt.imshow(projected_map, cmap=cmap, vmin=min_val, vmax=max_val)
        cb = plt.colorbar(img, shrink=0.7)  # You can adjust the size of the colorbar using 'shrink' parameter
        cb.set_label(label=cbar_label)  
        
        # Save the plot to a file
        # Replace Greek symbols with English letters
        cbar_label = cbar_label.replace(r"$\Delta", "Delta")
        cbar_label = cbar_label.replace(r"\nu", "nu")
        cbar_label = cbar_label.replace(r"\mu", "mu")
        cbar_label = cbar_label.replace(r"\rightarrow", "to")
        cbar_label = cbar_label.replace(r"_", " ")

        # Remove any remaining LaTeX commands (e.g., "{", "}", "$")
        cbar_label = re.sub(r"{|}|\$", "", cbar_label)

        # Remove spaces and add underscores between words
        cbar_label = cbar_label.replace(" ", "_")
        title = title.replace(" ", "_")
        plt.savefig(title + cbar_label + ".png", bbox_inches='tight')
        plt.show()
        
        # Close the current plot to free memory
        plt.close()  
        
        
    def plot_osc_prob_skymap_2D(
        self,
        # Steer physics
        initial_flavor,
        energy_GeV,
        distance_km=None, coszen=None,
        nubar=False,
        final_flavor=None,
        #Plotting
        resolution= 8,
        cmap='RdPu',
        ) :
        
        '''
        Make a 2D plot of neutrino oscillation probabilities vs right ascension and declination
        for a fixed energy.
        '''
        
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
        assert resolution > 0 and (resolution & (resolution - 1)) == 0, "resolution needs to be a power of 2."
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
        
        # Generate minimal ra and dec to cover all pixels of healpy map
        # Number of pixels of healpy map
        npix = hp.nside2npix(nside=resolution)
        
        # Convert pixel to polar coordinates (in deg)
        right_ascension_flat, declination_flat = hp.pix2ang(nside=resolution, ipix=np.arange(npix), lonlat=True)
        date_str = self._neutrino_source_kw["date_str"]
        self._neutrino_source_kw = None
        self.set_neutrino_source(# Date, Time and Timezone
                                date_str = date_str,
                                # Location on the sky
                                ra_deg = right_ascension_flat, 
                                dec_deg = declination_flat,
                                )
        
        #Store dictionaries for later use
        neutrinos_dict = self._neutrino_source_kw
        sme_dict = self._sme_model_kw
        
        # Evaluate which pixels are above the horizon 
        _, alt, _ = self.detector_coords.get_coszen_altitude_and_azimuth(date_str = date_str, ra_deg = right_ascension_flat, dec_deg = declination_flat)
        # Create a mask for altitudes between 0 and 90 degrees
        mask = (alt >= 0) # & (alt <= 90)
        # Create an array of zeros with the same shape as alt
        visible_sky = np.zeros_like(alt)
        
        # Set the elements where the condition is met to .4
        visible_sky[mask] = .4

        
        
        # Calculate probabilities with SME model
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
       
        #
        # Plot each healpix map
        #
        
        #number of plots
        ny = self.num_neutrinos if final_flavor is None else 1
        
        # Define the possible final states
        if final_flavor is None:
            final_flavor = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        else:
            final_flavor = [final_flavor] 
            
        for i in range(len(energy_GeV)):
            healpix_maps_flavours = sme_probabilities2d[i,:,:]
            
            # Round to two significant digits
            rounded_energy = round(energy_GeV[i], -int(np.floor(np.log10(abs(energy_GeV[i]))) - 1))
            
            # Display in scientific notation
            formatted_energy = f"{rounded_energy:.2e} GeV"
        
            for j in range(ny):
                single_healpix_map = healpix_maps_flavours[:,j]
                # Check if any values are outside the range [-1, 1]
                if np.any(single_healpix_map < -.1) or np.any(single_healpix_map > 1.1):
                    warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)

                #Plot the difference in oscillation probabilities for all flavours
                self.plot_healpix_map(
                    healpix_map=single_healpix_map, 
                    visible_sky_map=visible_sky,
                    nside=resolution, 
                    title=formatted_energy,
                    cbar_label=r"$P(\nu_{\mu}\rightarrow \nu_{" + final_flavor[j] + r"})$", 
                    cmap=cmap,
                    min_val=0
                )
               
            healpix_maps_sum_flavours = np.sum(healpix_maps_flavours, axis=1)
            healpix_maps_sum_flavours = np.squeeze(healpix_maps_sum_flavours)
            
            # Check if any values are outside the range [-0.1, 0.1]
            if np.any(single_healpix_map < -0.1) or np.any(single_healpix_map > 0.1):
                warnings.warn("Values of the sum of the difference of the oscillation probabilities outside the range [-0.1, 0.1].", UserWarning)
            
            # Plot sum of flavours 
            self.plot_healpix_map(
                healpix_map=healpix_maps_sum_flavours,
                visible_sky_map=visible_sky, 
                nside=resolution, 
                title=formatted_energy,
                cbar_label=r"$P(\nu_{\mu}\rightarrow \nu_{all})$", 
                cmap=cmap,
                max_val=1.1,
                min_val=0.9,
            )
            
        return sme_probabilities2d
    
    def plot_osc_prob_skymap_2D_diff(
        self,
        # Steer physics
        initial_flavor,
        energy_GeV,
        distance_km=None, coszen=None,
        nubar=False,
        final_flavor=None,
        #Plotting
        resolution= 8,
        cmap='bwr',
        ) :
        
        '''
        Make a 2D plot of neutrino oscillation probabilities vs right ascension and declination
        for a fixed energy.
        '''
        
        # Check inputs
        assert isinstance(initial_flavor, int)
        assert isinstance(energy_GeV, np.ndarray)
        assert isinstance(nubar, bool)
        if final_flavor is not None:
            assert isinstance(final_flavor, int)
        assert resolution > 0 and (resolution & (resolution - 1)) == 0, "resolution needs to be a power of 2."
    
        # Get a_eV, c and ra for naming the plot
        if self._sme_model_kw:    
            a_eV = self._sme_model_kw.get("a_eV")
            c = self._sme_model_kw.get("c")
        
        # Handle distance vs coszen
        if self.atmospheric:
            assert coszen is not None
            dist_kw = {"coszen": coszen}
        else:
            assert distance_km is not None
            dist_kw = {"distance_km": distance_km}
        
        # Generate minimal ra and dec to cover all pixels of healpy map
        # Number of pixels of healpy map
        npix = hp.nside2npix(nside=resolution)
        
        # Convert pixel to polar coordinates (in deg)
        right_ascension_flat, declination_flat = hp.pix2ang(nside=resolution, ipix=np.arange(npix), lonlat=True)
        date_str = self._neutrino_source_kw["date_str"]
        self._neutrino_source_kw = None
        self.set_neutrino_source(# Date, Time and Timezone
                                date_str = date_str,
                                # Location on the sky
                                ra_deg = right_ascension_flat, 
                                dec_deg = declination_flat,
                                )
        
        #Store dictionaries for later use
        neutrinos_dict = self._neutrino_source_kw
        sme_dict = self._sme_model_kw
        
        # Set coszen values to the values corresponding to the different pixels of the healpix map
        self.skymap_use = True
        
        # Calculate probabilities with SME model
        sme_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        # Call method to set parameters for the standard oscillation case
        self.set_std_osc()
        
        # Set _neutrino_source_kw values again and _sme_model_kw to zero to ensure 
        # that standard_probabilities2d has the same shape as sme_probabilities2d
        self._neutrino_source_kw = neutrinos_dict
        
        # Calculate probabilities for the standard case
        standard_probabilities2d = self.calc_osc_prob(
            initial_flavor=initial_flavor,
            energy_GeV=energy_GeV,
            **dist_kw
        )
        
        # Calculate the difference of probabilities between non-standard and standard cases
        diff_probabilities = sme_probabilities2d - standard_probabilities2d
        
        #
        # Plot each healpix map
        #
        
        #number of plots
        ny = self.num_neutrinos if final_flavor is None else 1
        
        # Define the possible final states
        if final_flavor is None:
            final_flavor = ["e", "\u03BC", "\u03C4"]  # Use unicode characters for mu and tau
        else:
            final_flavor = [final_flavor] 
            
        for i in range(len(energy_GeV)):
            healpix_maps_flavours = diff_probabilities[i,:,:]
            
            # Round to two significant digits
            rounded_energy = round(energy_GeV[i], -int(np.floor(np.log10(abs(energy_GeV[i]))) - 1))
            
            # Display in scientific notation
            formatted_energy = f"{rounded_energy:.2e} GeV"
        
            for j in range(ny):
                single_healpix_map = healpix_maps_flavours[:,j]
                # Check if any values are outside the range [-1, 1]
                if np.any(single_healpix_map < -1.1) or np.any(single_healpix_map > 1.1):
                    warnings.warn("Values of the difference of the oscillation probabilities outside the range [-1, 1].", UserWarning)

                #Plot the difference in oscillation probabilities for all flavours
                self.plot_healpix_map(
                    healpix_map=single_healpix_map, 
                    nside=resolution, 
                    title=formatted_energy,
                    cbar_label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{" + final_flavor[j] + r"})$", 
                    cmap=cmap
                )
               
            healpix_maps_sum_flavours = np.sum(healpix_maps_flavours, axis=1)
            healpix_maps_sum_flavours = np.squeeze(healpix_maps_sum_flavours)
            
            # Check if any values are outside the range [-0.1, 0.1]
            if np.any(single_healpix_map < -0.1) or np.any(single_healpix_map > 0.1):
                warnings.warn("Values of the sum of the difference of the oscillation probabilities outside the range [-0.1, 0.1].", UserWarning)
            
            # Plot sum of flavours 
            self.plot_healpix_map(
                healpix_map=healpix_maps_sum_flavours, 
                nside=resolution, 
                title=formatted_energy,
                cbar_label=r"$\Delta P(\nu_{\mu}\rightarrow \nu_{all})$", 
                cmap=cmap
            )
            
        return sme_probabilities2d, standard_probabilities2d 
            
    def compare_models(
        self,
        model_defs,
        initial_flavors,
        energy_GeV,
        distance_km, 
        include_std_osc=True,
    ) :
        '''
        Compare the different models/cases specified by `model_defs`

        model_defs : list of dicts
            Each dict must/can have the following entries:
                "calc_basis" (required)
                "D_matrix_basis" (required)
                "D_matrix" (required)
                "n" (required)
                "label" (required)
                "color" (optional)
                "linestyle" (optional)
        '''

        #TODO add comparison w.r.t. energy

        # Check inputs
        #TODO

        # Plot steering
        color_scale = ColorScale("hsv", len(model_defs))

        # Output containers
        figures = []

        # Loop over initial flavors
        for initial_flavor in initial_flavors :

            fig, ax = plt.subplots( nrows=self.num_neutrinos+1, sharex=True, figsize=(6,7) )
            figures.append(fig)

            # Plot std osc
            if include_std_osc :
                self.set_std_osc()
                self.plot_osc_prob_vs_distance(fig=fig, initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, color="lightgrey", label="Std osc")

            # Loop over models/cases
            for i_model, model_dict in enumerate(model_defs) :

                # Plot steering
                label = model_dict["label"]
                color = model_dict["color"] if "color" in model_dict else color_scale.get(i_model)
                linestyle = model_dict["linestyle"] if "linestyle" in model_dict else "-"

                # Set physics params
                self.set_calc_basis(model_dict["calc_basis"])
                self.set_decoherence_D_matrix_basis(model_dict["D_matrix_basis"])
                self.set_decoherence_D_matrix( D_matrix_eV=model_dict["D_matrix"], n=model_dict["n"] )

                # Plot
                self.plot_osc_prob_vs_distance(fig=fig, initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, color=color, linestyle=linestyle, label=label ) 

            # Add long range behaviour lines
            #TODO

            # Format
            ax[-1].set_xlabel(DISTANCE_LABEL)
            ax[0].legend( loc="upper right", fontsize=10 ) #TODO put to the right of the ax
            fig.quick_format( ylim=(-0.01,1.01), legend=False )

        return figures




def define_matching_perturbation_and_lindblad_calculators(num_neutrinos=3) :
    '''
    Create DecoherenceToyModel and OscCalculator instances
    with common osc parmas, etc.

    Allows for easy comparison between the models.
    '''

    #TODO Make this more flexible by making this a member function of OscCalculator
    # which returns a compatible DecoherenceToyModel instance.


    from deimos.utils.toy_model.decoherence_toy_model import DecoherenceToyModel, get_neutrino_masses

    #
    # Define system
    #

    # Get the system definition
    flavors = FLAVORS
    mass_splittings_eV2 = MASS_SPLITTINGS_eV2
    mixing_angles_rad = MIXING_ANGLES_rad
    deltacp = DELTACP_rad

    #TODO store flavor labels in class, or integrate with OscCalculator

    # Assuming some (arbitrary, at least w.r.t. oscillations) lightest neutrino mass, get masses from mass splittings
    lowest_neutrino_mass_eV = 0. #1.e-3
    masses_eV = get_neutrino_masses(lowest_neutrino_mass_eV, mass_splittings_eV2)

    # Get PMNS
    PMNS = get_pmns_matrix(mixing_angles_rad, dcp=deltacp)


    #
    # Create toy model
    #

    perturbation_toy_model = DecoherenceToyModel(
        num_states=num_neutrinos,
        mass_state_masses_eV=masses_eV,
        PMNS=PMNS,
        seed=12345,
    )


    #
    # Create Lindblad lindblad_calculator
    #

    lindblad_calculator = OscCalculator(
        tool="dmos", #TODO nusquids
        atmospheric=False,
        num_neutrinos=num_neutrinos,
    )

    lindblad_calculator.set_matter("vacuum")

    lindblad_calculator.set_mass_splittings(*mass_splittings_eV2)
    lindblad_calculator.set_mixing_angles(*mixing_angles_rad, deltacp=deltacp)

    # Handle basis choice
    lindblad_calculator.set_calc_basis("nxn")


    #
    # Checks
    #

    assert np.array_equal(perturbation_toy_model.PMNS, lindblad_calculator.PMNS)
    assert np.isclose(perturbation_toy_model.get_mass_splittings()[0], mass_splittings_eV2[0]) # 21
    if num_neutrinos > 2 :
        assert np.isclose(perturbation_toy_model.get_mass_splittings()[1], mass_splittings_eV2[1]) # 31

    return perturbation_toy_model, lindblad_calculator

