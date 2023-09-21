'''
Wrapper class providing a common interface to a range of oscillation solvers,
both from this project and external.

Tom Stuttard
'''

import sys, os, collections, numbers, copy

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

# Import nuSQuIDS decoherence implementation
NUSQUIDS_DECOH_AVAIL = False
# try:
#     print("+++ OW 5")
#     from nuSQUIDSDecohPy import nuSQUIDSDecoh, nuSQUIDSDecohAtm  #TODO this is hanging, why? for no have commented this out
#     print("+++ OW 6")
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
from deimos.models.decoherence.decoherence_operators import get_model_D_matrix
from deimos.density_matrix_osc_solver.density_matrix_osc_solver import DensityMatrixOscSolver, get_pmns_matrix, get_matter_potential_flav
from deimos.utils.oscillations import calc_path_length_from_coszen


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
        num_neutrinos=3,
        **kw
    ) :

        # Store args
        self.tool = tool
        self.atmospheric = atmospheric
        self.num_neutrinos = num_neutrinos

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
        mass_splitting_eV2, mixing_angles_rad, deltacp, mass_tex, _, flavors_tex, _, nu_colors = get_default_neutrino_definitions(self.num_neutrinos)
        self.set_mixing_angles(*mixing_angles_rad, deltacp=deltacp)
        self.set_mass_splittings(*mass_splitting_eV2)
        self.set_tex_labels(flavor_tex=flavors_tex, mass_tex=mass_tex)
        self.set_colors(nu_colors)
        self.set_calc_basis(DEFAULT_CALC_BASIS)
        # self.set_decoherence_D_matrix_basis(DEFAULT_DECOHERENCE_GAMMA_BASIS)


    def parse_pisa_config(self,config) :
        '''
        Parse settings from a PISA config file and apply them
        '''

        pass #TODO


    def _init_nusquids(self,
        energy_nodes_GeV=None,
        coszen_nodes=None,
        interactions=False,
        error=1.e-6,
    ) :

        assert NUSQUIDS_AVAIL, "Cannot use nuSQuIDS, not installed"

        #
        # Calculation nodes
        #

        # Set some default nodes...

        if energy_nodes_GeV is None :
            energy_nodes_GeV = np.logspace(0.,3.,num=100)
        self.energy_nodes_GeV = energy_nodes_GeV

        if self.atmospheric :
            if coszen_nodes is None :
                coszen_nodes = np.linspace(-1.,1.,num=100)
            self.coszen_nodes = coszen_nodes
        else :
            assert coszen_nodes is None, "`coszen_nodes` argument only valid in `atmospheric` mode"


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
            if NUSQUIDS_DECOH_AVAIL :
                self.nusquids = nuSQUIDSDecohAtm(*args)
            else :
                self.nusquids = nsq.nuSQUIDSAtm(*args)

            # Add tau regeneration
            # if interactions :
            #     self.nusquids.Set_TauRegeneration(True) #TODO results look wrong, disable for now and investigate

        else :

            print(self.energy_nodes_GeV)

            # Instantiate nuSQuIDS regular calculator
            args = [
                self.energy_nodes_GeV * self.units.GeV,
                self.num_neutrinos,
                nu_type,
                interactions,
            ]
            if NUSQUIDS_DECOH_AVAIL :
                self.nusquids = nuSQUIDSDecoh(*args)
            else :
                self.nusquids = nsq.nuSQUIDS(*args)
            

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
                V = get_matter_potential_flav(num_states=self.num_neutrinos, matter_density_g_per_cm3=kw["matter_density_g_per_cm3"], electron_fraction=kw["electron_fraction"], nsi_matrix=None)
                self.solver.set_matter_potential(V)


            elif self.tool == "prob3" :
                print("WARNING : electron fraction not currently handled by prob3 wrapper")
                self._prob3_settings["matter"] = "constant"
                self._prob3_settings["matter_density_g_per_cm3"] = kw["matter_density_g_per_cm3"]


        #
        # Error handling
        #

        else :
            raise Exception("Unrecognised `matter` : %s" % matter)


        self._matter = matter


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
        Use standard oscillations (e.g. disable decoherence)
        '''

        if self.tool == "nusquids" :
            self.set_calc_basis(DEFAULT_CALC_BASIS)
            # self.set_decoherence_D_matrix_basis(DEFAULT_CALC_BASIS)
            self.set_decoherence_D_matrix(D_matrix_eV=np.zeros((self.num_sun_basis_vectors,self.num_sun_basis_vectors)), n=0, E0_eV=1.)
        else :
            self._decoh_model_kw = None
            self._lightcone_model_kw = None
            self._sme_model_kw = None


    def set_calc_basis(self, basis) :

        if self.tool == "nusquids" :
            assert basis == "nxn" #TOO is this correct?

        elif self.tool == "deimos" :
            self._calc_basis = basis # Store for use later

        elif self.tool == "prob3" :
            pass # Basis not relevent here, not solving Linblad master equation

        else :
            raise Exception("`%s` does not support setting calculation basis" % self.tool)


    # def set_decoherence_D_matrix_basis(self, basis) :

    #     if self.tool == "nusquids" :
    #         assert basis == "sun"

    #     elif self.tool == "deimos" :
    #         self._decoherence_D_matrix_basis = basis # Store for use later

    #     else :
    #         raise Exception("`%s` does not support setting decoherence gamma matrix basis" % self.tool)




    #
    # Decoherence member functions
    #

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

        #
        # Check inputs
        #

        # If user specified the full matrix, check dimensions
        assert isinstance(D_matrix_eV, np.ndarray)

        # Check all relevent matrix conditions
        self.check_decoherence_D_matrix(D_matrix_eV)


        #
        # Set values
        #

        if self.tool == "nusquids" :
            assert NUSQUIDS_DECOH_AVAIL
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


    def check_decoherence_D_matrix(self, D) :
        '''
        There exist inequalities between the elements of the D matrix, meaning that the elements are not fully independent

        Enforcing these inequalities here:

         - 2 flavor: https://arxiv.org/pdf/hep-ph/0105303.pdf
         - 3 flavor: https://arxiv.org/pdf/1811.04982.pdf Appendix B
        '''

        #TODO Move this function out of this class, into models dir

        if self.num_neutrinos == 3 :

            #
            # SU(3) case
            #

            #TODO What enforces g1=g1, g4=g5, g6=g7 ?

            assert D.shape == (9, 9)

            #TODO what about 0th row/col?

            # Check everything is real
            assert np.all( D.imag == 0. )

            # Check everything is positive or zero
            assert np.all( D >= 0. )

            # Extract diagonal elements (gamma)
            g1 = D[1,1]
            g2 = D[2,2]
            g3 = D[3,3]
            g4 = D[4,4]
            g5 = D[5,5]
            g6 = D[6,6]
            g7 = D[7,7]
            g8 = D[8,8]

            # Extract off-diagonal elements (beta)
            # Enforce pairs either side of the diagonal match in the process
            b12 = D[1,2]
            assert D[2,1] == b12
            b13 = D[1,3]
            assert D[3,1] == b13
            b14 = D[1,4]
            assert D[4,1] == b14
            b15 = D[1,5]
            assert D[5,1] == b15
            b16 = D[1,6]
            assert D[6,1] == b16
            b17 = D[1,7]
            assert D[7,1] == b17
            b18 = D[1,8]
            assert D[8,1] == b18
            b23 = D[2,3]
            assert D[3,2] == b23
            b24 = D[2,4]
            assert D[4,2] == b24
            b25 = D[2,5]
            assert D[5,2] == b25
            b26 = D[2,6]
            assert D[6,2] == b26
            b27 = D[2,7]
            assert D[7,2] == b27
            b28 = D[2,8]
            assert D[8,2] == b28
            b34 = D[3,4]
            assert D[4,3] == b34
            b35 = D[3,5]
            assert D[5,3] == b35
            b36 = D[3,6]
            assert D[6,3] == b36
            b37 = D[3,7]
            assert D[7,3] == b37
            b38 = D[3,8]
            assert D[8,3] == b38
            b45 = D[4,5]
            assert D[5,4] == b45
            b46 = D[4,6]
            assert D[6,4] == b46
            b47 = D[4,7]
            assert D[7,4] == b47
            b48 = D[4,8]
            assert D[8,4] == b48
            b56 = D[5,6]
            assert D[6,5] == b56
            b57 = D[5,7]
            assert D[7,5] == b57
            b58 = D[5,8]
            assert D[8,5] == b58
            b67 = D[6,7]
            assert D[7,6] == b67
            b68 = D[6,8]
            assert D[8,6] == b68
            b78 = D[7,8]
            assert D[8,7] == b78

            # Now implement all inequalities
            a1 = -g1 + g2 + g3 - (g8/3.) 
            a2 =  g1 - g2 + g3 - (g8/3.)
            a3 =  g1 + g2  -g3 - (g8/3.)

            a4 = -g4 + g5 + g3 + (2.*g8/3.) - (2.*b38/np.sqrt(3.)) # See here that beta38 is somehwat special (since it relates to the special gamma3/8 params)
            a5 =  g4 - g5 + g3 + (2.*g8/3.) - (2.*b38/np.sqrt(3.))
            a6 = -g6 + g7 + g3 + (2.*g8/3.) + (2.*b38/np.sqrt(3.))
            a7 =  g6 - g7 + g3 + (2.*g8/3.) + (2.*b38/np.sqrt(3.))

            a8 = -(g1/3.) - (g2/3.) - (g3/3.) + (2.*g4/3.) + (2.*g5/3.) + (2.*g6/3.) + (2.*g7/3.) - g8

            assert a1 >= 0., "Inequality failure (a1)"
            assert a2 >= 0., "Inequality failure (a2)"
            assert a3 >= 0., "Inequality failure (a3)"
            assert a4 >= 0., "Inequality failure (a4)"
            assert a5 >= 0., "Inequality failure (a5)"
            assert a6 >= 0., "Inequality failure (a1)"
            assert a7 >= 0., "Inequality failure (a7)"
            assert a8 >= 0., "Inequality failure (a8)"

            assert (4.*np.square(b12)) <= ( np.square(g3 - (g8/3.)) - np.square(g1 - g2) )
            assert (4.*np.square(b13)) <= ( np.square(g2 - (g8/3.)) - np.square(g1 - g3) )
            assert (4.*np.square(b23)) <= ( np.square(g1 - (g8/3.)) - np.square(g2 - g3) )

            assert np.square( 4.*np.square(b38) + (g4/np.sqrt(3.)) + (g5/np.sqrt(3.)) - (g6/np.sqrt(3.)) - (g7/np.sqrt(3.)) ) <= (a3*a8)

            #TODO there are still quite a few more involving beta....

        else :
            print("Checks on decoherence D matrix inequalities not yet implemented for a %i neutrino system" % self.num_neutrinos)
            pass


    def set_decoherence_model(self, model_name, **kw) :
        '''
        Set the decoherence model to be one of the pre-defined models
        '''

        from deimos.models.decoherence.nuVBH_model import get_randomize_phase_decoherence_D_matrix, get_randomize_state_decoherence_D_matrix, get_neutrino_loss_decoherence_D_matrix

        get_D_matrix_func = None

        # Check if model is one of the nuVBH models, and get the D matrix definition function if so
        if model_name == "randomize_phase" :
            get_D_matrix_func = get_randomize_phase_decoherence_D_matrix

        elif model_name == "randomize_state" :
            get_D_matrix_func = get_randomize_state_decoherence_D_matrix

        elif model_name == "neutrino_loss" :
            get_D_matrix_func = get_neutrino_loss_decoherence_D_matrix

        else :
            raise Exception("Unknown decoherence model : %s" % model_name) 

        # Check kwarks
        kw = copy.deepcopy(kw)
        assert "gamma0_eV" in kw
        gamma0_eV = kw.pop("gamma0_eV")
        assert "n" in kw
        n = kw.pop("n")
        assert "E0_eV" in kw
        E0_eV = kw.pop("E0_eV")
        assert len(kw) == 0

        # Get the correct D matrix and pass to the solver
        D_matrix_basis, D_matrix0_eV =  get_D_matrix_func(num_states=self.num_neutrinos, gamma=gamma0_eV)
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
        a_eV,
        c,
    ) :
        '''
        TODO
        '''

        #
        # Check inputs
        #

        assert isinstance(a_eV, numbers.Number)
        assert isinstance(c, numbers.Number)


        #
        # Set values
        #

        if self.tool == "nusquids" :
            raise NotImplemented()

        elif self.tool == "deimos" :
            self._sme_model_kw = {
                "a_eV" : a_eV,
                "c" : c,
            }



    #
    # Plotting member functions
    #

    def calc_osc_prob(self,
        energy_GeV,
        initial_flavor=None,
        initial_state=None,
        distance_km=None,
        coszen=None,
        nubar=False,
    ) :

        #TODO caching
        #TODO Option for different final rho to allow nu->nubar transitions

        #
        # Check inputs
        # 

        if self.atmospheric :
            assert ( (coszen is not None) and (distance_km is None) ), "Must provide `coszen` (and not `distance_km`) in atmospheric mode"
        else :
            assert ( (distance_km is not None) and (coszen is None) ), "Must provide `distance_km` (and not `coszen`) in non-atmospheric mode" 

        if initial_flavor is not None :
            initial_flavor = self._get_flavor_index(initial_flavor)


        #
        # Calculate
        #

        # Call sub-function for relevent solver
        if self.tool == "nusquids" :
            osc_probs = self._calc_osc_prob_nusquids( initial_flavor=initial_flavor, initial_state=initial_state, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, nubar=nubar )

        elif self.tool == "deimos" :
            assert initial_flavor is not None, "must provide `initial_flavor` (`initial_state` not currently supported for %s" % self.tool
            osc_probs = self._calc_osc_prob_deimos( initial_flavor=initial_flavor, nubar=nubar, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen)
       
        elif self.tool == "prob3" :
            osc_probs = self._calc_osc_prob_prob3( initial_flavor=initial_flavor, energy_GeV=energy_GeV, distance_km=distance_km, coszen=coszen, nubar=nubar )

        # Checks
        assert np.all( np.isfinite(osc_probs) ), "Found non-finite osc probs"

        return osc_probs


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
            results = np.full( (energy_GeV.size, coszen.size, final_flavors.size, 2 ), np.NaN )

            # Determine shape of initial state vector
            state_shape = [ self.nusquids.GetNumCos(), self.nusquids.GetNumE() ]
            state_shape.append( 2 )
            state_shape.append( final_flavors.size )
            state_shape = tuple(state_shape)

            # Define initial state if not provided, otherwise verify the one provided
            if initial_state is None :
                initial_state = np.full( state_shape, 0. )
                initial_state[ :, :, rho, initial_flavor ] = 1. # dims = [ cz node, E node, nu(bar), flavor ]
            else :
                assert initial_state.shape == state_shape, "Incompatible shape for initial state : Expected %s, found %s" % (state_shape, initial_state.shape)

            # Set the intial state
            self.nusquids.Set_initial_state(initial_state, nsq.Basis.flavor)

            # Evolve the state
            self.nusquids.EvolveState()

            # Evaluate the flavor at each grid point to get oscillation probabilities
            for i_E,E in enumerate(energy_GeV) :
                for i_cz,cz in enumerate(coszen) :
                    for i_f,final_flavor in enumerate(final_flavors) :
                        # results[i_E,i_cz,i_f] = self.nusquids.EvalFlavor( final_flavor, cz, E*self.units.GeV )#, rho ) #TODO Add randomize prod height arg
                        results[i_E,i_cz,i_f] = self.nusquids.EvalFlavor( int(final_flavor), cz, E*self.units.GeV, rho, randomize_atmo_prod_height) #TODO add nubar

            return results


        #
        # Distance case
        #

        else :

            # Init results container
            results = np.full( (energy_GeV.size, distance_km.size, final_flavors.size), np.NaN )
            # results = np.full( (energy_GeV.size, distance_km.size, final_flavors.size, 2), np.NaN )

            # Determine shape of initial state vector
            state_shape = [ self.nusquids.GetNumE() ]
            state_shape.append(2)
            state_shape.append( final_flavors.size )
            state_shape = tuple(state_shape)

            # Define initial state if not provided, otherwise verify the one provided
            if initial_state is None :
                initial_state = np.full( state_shape, 0. )
                initial_state[ :, rho, initial_flavor ] = 1. # dims = [ E node, nu(bar), flavor ]
            else :
                assert initial_state.shape == state_shape, "Incompatible shape for initial state : Expected %s, found %s" % (state_shape, initial_state.shape)

            # Loop over distance nodes
            for i_L,L in enumerate(distance_km) :

                # Set then track, taking medium into account
                if self._matter == "vacuum" :
                    self.nusquids.Set_Track(nsq.Vacuum.Track(L*self.units.km))
                elif self._matter == "constant" :
                    self.nusquids.Set_Track(nsq.ConstantDensity.Track(L*self.units.km))
                else :
                    raise Exception("Unknown body : %s" % body) 

                # Set initial flavor
                self.nusquids.Set_initial_state( initial_state, nsq.Basis.flavor ) 

                # Evolve for the track distance
                self.nusquids.EvolveState()

                # Loop over energies
                for i_e,E in enumerate(energy_GeV) :

                    # Evaluate final state flavor composition
                    for i_f, final_flavor in enumerate(final_flavors) :
                        # for rho in [0, 1] :
                        #     results[i_e,i_L,i_f,rho] = self.nusquids.EvalFlavor( int(final_flavor), float(E*self.units.GeV), int(rho) )
                        results[i_e,i_L,i_f] = self.nusquids.EvalFlavor( int(final_flavor), float(E*self.units.GeV), int(rho) )

            #TODO squeeze unused dimensions?

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
        initial_flavor,
        energy_GeV,
        distance_km=None,
        coszen=None,
        nubar=False,
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
            production_height_km = 22. # common with nuSQuIDS (Although gives differing results???). TODO steerable, and defined in constants.py
            detector_depth_km = 0. # common with nuSQuIDS (Although gives differing results???). TODO steerable, and defined in constants.py
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
            verbose=False
        )

        # Handle flip in results (L dimension)
        if flip :
            results = np.flip(results, axis=1)

        return results


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

    
    def set_tex_labels(self, flavor_tex, mass_tex) :
        self.flavor_tex = flavor_tex
        self.mass_tex = mass_tex


    def set_colors(self, nu_colors) :
        self.nu_colors = nu_colors


    @property
    def states(self) :
        return np.array(range(self.num_neutrinos))

    @property
    def flavors_tex(self) :
        return [ self.get_flavor_tex(i) for i in self.states ]


    @property
    def masses_tex(self) :
        return [ self.get_mass_tex(i) for i in self.states ]


    @property
    def flavors_color(self) :
        return [ self.get_flavor_color(i) for i in self.states ]


    def get_mass_tex(self, state) :
        return self.mass_tex[state]


    def get_flavor_tex(self, flavor, nubar=False) :
        nu_tex = r"\nu"
        if nubar :
            nu_tex = r"\bar{" + nu_tex + r"}"
        if flavor is None :
            nu_tex += r"_{\rm{all}}"
        else :
            nu_tex += r"_{" + self.flavor_tex[flavor] + r"}"
        return nu_tex


    def get_flavor_color(self, flavor) :
        return self.nu_colors[flavor]


    def get_transition_prob_tex(self,initial_flavor, final_flavor, nubar=False) :
        return r"P(%s \rightarrow %s)" % ( self.get_flavor_tex(initial_flavor, nubar), self.get_flavor_tex(final_flavor, nubar) )


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

        import matplotlib.pyplot as plt

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

        # Remove energy dimension, since this is single energy
        osc_probs = osc_probs[0,...]

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

        import matplotlib.pyplot as plt

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

        # Remove distance dimension, since this is single distance
        osc_probs = osc_probs[:,0,...]

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

        import matplotlib.pyplot as plt

        raise NotImplemented("TODO")


    def plot_oscillogram(
        self,
        initial_flavor,
        final_flavor,
        energy_GeV,
        coszen,
        nubar=False,
        title=None,
    ) :
        '''
        Helper function for plotting an atmospheric neutrino oscillogram
        '''

        import matplotlib.pyplot as plt
        from deimos.utils.plotting import plot_colormap, value_spacing_is_linear

        assert self.atmospheric, "`plot_oscillogram` can only be called in atmospheric mode"

        #
        # Steerig
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
        fig, ax = plt.subplots( figsize=(7, 6) )
        if title is not None :
            fig.suptitle(title) 

        # Plot oscillogram
        plot_colormap( ax=ax, x=energy_GeV, y=coszen, z=osc_probs, vmin=0., vmax=1., cmap=continuous_map, zlabel=r"$%s$"%transition_prob_tex )

        # Format
        xscale = "linear" if value_spacing_is_linear(energy_GeV) else "log"
        yscale = "linear" if value_spacing_is_linear(coszen) else "log"
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(energy_GeV[0], energy_GeV[-1])
        ax.set_ylim(coszen[0], coszen[-1])
        ax.set_xlabel(ENERGY_LABEL)
        ax.set_ylabel(COSZEN_LABEL)
        fig.tight_layout()

        return fig, ax, osc_probs


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
    mass_splittings_eV2, mixing_angles_rad, deltacp, _, _, _, _, _ = get_default_neutrino_definitions(num_neutrinos)

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

