'''
Propagate neutrinos as 3 independent mass states.
Can inject perturbations.

Tom Stuttard
'''

#TODO integrate into OscillationCalculator

import sys, copy, collections, os

from deimos.density_matrix_osc_solver.density_matrix_osc_solver import get_pmns_matrix, psi_flav_to_mass_basis, psi_mass_to_flav_basis, psi_flavor_prob
from deimos.utils.oscillations import OSC_FREQUENCY_UNIT_CONVERSION
from deimos.utils.constants import *
from deimos.utils.cache import Cachable


class MassStatePropagator(Cachable) :

    def __init__(
        self,
        num_states,
        mass_state_masses_eV,
        PMNS,
        seed=None,
        cache_dir=None,
    ) :

        # Caching
        if cache_dir is None :
            cache_dir = os.path.join( os.path.dirname(__file__), ".cache" )
        Cachable.__init__(self, cache_dir=cache_dir)

        # Store args
        self.num_states = num_states
        self.mass_state_masses_eV = mass_state_masses_eV
        self.PMNS = PMNS
        self.seed = seed

        # Checks
        assert self.num_states > 1
        assert mass_state_masses_eV.shape == tuple([self.num_states,])
        assert PMNS.shape == tuple([self.num_states,self.num_states])
        #TODO check unitarity


    def init_random_state(self) :
        return np.random.RandomState(self.seed)


    def get_state_dict(self) :
        return copy.deepcopy(self.__dict__)

    
    def get_mass_splittings(self) :
        if self.num_states == 2 :
            return np.array([ np.square(self.mass_state_masses_eV[1]) - np.square(self.mass_state_masses_eV[0]) ])

        elif self.num_states == 3 :
            return np.array([ 
                np.square(self.mass_state_masses_eV[1]) - np.square(self.mass_state_masses_eV[0]),
                np.square(self.mass_state_masses_eV[2]) - np.square(self.mass_state_masses_eV[0]),
                np.square(self.mass_state_masses_eV[2]) - np.square(self.mass_state_masses_eV[1]),
            ])

        else :
            raise Exception("Not implemented for %i states" % self.num_states)


    def propagate_mass_states(self,
        initial_psi_mass, # Initial value of mass state wavefunctions
        E_GeV,
        distance_km,
        perturbation=None,
        perturbation_options=None,
        num_steps=None,
        random_state=None,
    ) :
        '''
        This function takes an initial set of mass states (of the specified masses 
        and energy) and propagates them as plane waves over the requested distance.

        Optionally, perturbations can be applied to the states as they propagate. 
        Various different forms of perturbation are supported, see description of 
        arguments below. 

        These perturbations may result in decoherence (if stochastic and not equally 
        affecting the phase of each mass state), or could produce coherent effects 
        in other circumstances.
        '''

        #
        # Check inputs
        #

        assert len(initial_psi_mass) == self.num_states

        # Set defaults
        if num_steps is None :
            num_steps = 1000


        #
        # Prepare calculation
        #

        if random_state is None :
            random_state = self.init_random_state()

        # Initialise the state
        psi_mass = initial_psi_mass

        # Start phases at initial point
        # Initial phase is always zero (e.g. at some interaction) #TODO check
        initial_phase = np.zeros_like(initial_psi_mass)
        phase = initial_phase.copy()

        # Get the steps, from the observer's perspective (e..g who doesn't know about any warping space-time, varying production height, etc)
        distance_step_observer_km = distance_km / float(num_steps)
        distance_step_true_km = distance_step_observer_km
        distance_so_far_observer_km = 0.
        distance_so_far_neutrino_km = 0.


        #
        # Init perturbation
        #

        if perturbation is not None :

            # Check args
            assert isinstance(perturbation,str)
            if perturbation_options is None :
                perturbation_options = {}
            assert isinstance(perturbation_options, collections.Mapping)


            #
            # Baseline variations
            #

            if perturbation == "baseline" :

                # Get the peturbation params
                assert "perturbation_size_sigma_km" in perturbation_options
                perturbation_function = perturbation_options["perturbation_function"]
                perturbation_size_sigma_km = perturbation_options["perturbation_size_sigma_km"]
                perturbation_size_positive_only = perturbation_options["perturbation_size_positive_only"] if "perturbation_size_positive_only" in perturbation_options else False

                # Generate perturbation
                perturbation_size_km = self._get_perturbation(random_state=random_state, perturbation_function=perturbation_function, perturbation_size=perturbation_size_sigma_km, positive_only=perturbation_size_positive_only)
                
                # Modify the true step size accordingly
                distance_step_true_km = ( distance_km + perturbation_size_km ) / float(num_steps)


            #
            # Production distance variations
            #

            elif perturbation == "production_distance" :

                # Get the peturbation params
                assert "perturbation_size_sigma_km" in perturbation_options
                perturbation_function = perturbation_options["perturbation_function"]
                perturbation_size_sigma_km = perturbation_options["perturbation_size_sigma_km"]
                perturbation_size_positive_only = perturbation_options["perturbation_size_positive_only"] if "perturbation_size_positive_only" in perturbation_options else False

                # Generate perturbation
                perturbation_size_km = self._get_perturbation(random_state=random_state, perturbation_function=perturbation_function, perturbation_size=perturbation_size_sigma_km, positive_only=perturbation_size_positive_only)
                
                # Modify the true step size accordingly
                distance_so_far_neutrino_km += perturbation_size_km


            #
            # Correlated step size variations
            #

            elif perturbation == "fully_correlated_step_size" :

                # Here I perturb the step size much like "step size" flucutations, but in this case the flucutation of each step is maximally correlated
                # This means that for a given neutrino, even step fluctuates by the same amount. This fluctuation differs from neutrino to neutrino though
                # This isn't very realistic (would be like riding a "fluctuation wave"!), but is a bounding case

                # Get the peturbation params
                assert "perturbation_size_sigma_km" in perturbation_options
                perturbation_function = perturbation_options["perturbation_function"]
                perturbation_size_sigma_km = perturbation_options["perturbation_size_sigma_km"]
                perturbation_size_positive_only = perturbation_options["perturbation_size_positive_only"] if "perturbation_size_positive_only" in perturbation_options else False

                # Generate perturbation
                perturbation_size_km = self._get_perturbation(random_state=random_state, perturbation_function=perturbation_function, perturbation_size=perturbation_size_sigma_km, positive_only=perturbation_size_positive_only)


        #
        # Propagate the states
        #

        # Containers to fill
        psi_mass_values = []
        distance_values_observer_km = []
        distance_values_neutrino_km = []

        # Start stepping
        distance_so_far_observer_km = 0.
        while distance_so_far_observer_km < distance_km :

            # For this step, start from the default "true" step size
            distance_step_this_step_km = distance_step_true_km


            #
            # Apply perturbation
            #

            # Check if we are applying epturbations
            if perturbation is not None :


                # Here we apply the various different types of perturbation...

                #TODO document the callable options


                #
                # Phase perturbation
                #

                # Apply a perturbation to the phase of the mass states
                if perturbation == "phase" :

                    # Get the mean free path, and convert to a probability of a perturbation per step
                    assert "mean_free_path_km" in perturbation_options
                    mean_free_path_km = perturbation_options["mean_free_path_km"]
                    perturbation_prob = self._get_pertubation_prob_per_step(step_size_km=distance_step_this_step_km, mean_free_path_km=mean_free_path_km)

                    # Check if applying a perturbation at this step
                    if random_state.uniform(0.,1.) < perturbation_prob :

                        # Get the peturbation strength
                        assert "perturbation_size_rad" in perturbation_options
                        perturbation_size_rad = perturbation_options["perturbation_size_rad"]

                        # Get the state to perturb
                        # Default to randomly picking a state
                        if "state_to_perturb" in perturbation_options :
                            state_to_perturb = perturbation_options["state_to_perturb"]
                        else :
                            state_to_perturb = random_state.randint(0,self.num_states)

                        # Apply the perturbation
                        phase[state_to_perturb] += perturbation_size_rad


                #
                # Step size perturbation
                #

                # Perturb the step size
                # This represents e.g. rapidly fluctuating space-time curvature (metric) at small distance scales
                # Note that am varying EVERY step here, unlike most of the other cases which use a mean free path to choose stochastic cases
                # This is an example of "uncorrelated" step size fluctuations, e.g. each step fluctuation is independent of the others
                elif perturbation == "step_size" :

                    # Get the peturbation params
                    assert "perturbation_size_sigma_km" in perturbation_options
                    perturbation_function = perturbation_options["perturbation_function"]
                    perturbation_size_sigma_km = perturbation_options["perturbation_size_sigma_km"]
                    perturbation_size_positive_only = perturbation_options["perturbation_size_positive_only"] if "perturbation_size_positive_only" in perturbation_options else False

                    # Generate perturbation
                    perturbation_size_km = self._get_perturbation(random_state=random_state, perturbation_function=perturbation_function, perturbation_size=perturbation_size_sigma_km, positive_only=perturbation_size_positive_only)
                    
                    # Apply the perturbation to the step size
                    distance_step_this_step_km += perturbation_size_km

                    # Floor at 0
                    # distance_step_this_step_km = max( distance_step_this_step_km, 0. ) #TODO think about this w.r.t. metric fluctuations


                #
                # Fully correlated step size perturbation
                #

                # Did most of the work here already before the step loop
                # Here just need to modify the true step size by the pre-computed fluctuation
                elif perturbation == "fully_correlated_step_size" :
                    distance_so_far_neutrino_km += perturbation_size_km


                #
                # Metric perturbation
                #

                #TODO docs
                #TODO Only cosidering a4 for now (distance-like)

                elif perturbation == "metric" :

                    from analysis.oscNext_decoherence.theory.decoherence_maths import FLAT_SPACETIME_METRIC_TENSOR, get_ds_from_metric_tensor, get_fluctuated_metric_tensor

                    # Get the peturbation params
                    assert "sigma4" in perturbation_options
                    perturbation_function = perturbation_options["perturbation_function"]
                    sigma4 = perturbation_options["sigma4"]

                    # Generate perturbation
                    a4 = self._get_perturbation(random_state=random_state, perturbation_function=perturbation_function, perturbation_size=sigma4, positive_only=False)

                    # Calculate distance fluctuation resulting from metric fluctuation
                    g = get_fluctuated_metric_tensor(a1=0., a2=0., a3=0., a4=a4)
                    distance_step_this_step_km = get_ds_from_metric_tensor(dx=np.array([0., distance_step_true_km]), g=g) #TODO Can I just consider x, and not time?


                #
                # Interaction
                #

                # The neutrino experiences a weak interaction
                elif perturbation == "weak_interaction" :

                    #TODO this doesn't account for a flavored target, e.g. prob only depends on the neutrino state
                    #TODO Add option for energy loss in the interaction

                    # Get the mean free path, and convert to a probability of a perturbation per step
                    assert "mean_free_path_km" in perturbation_options
                    mean_free_path_km = perturbation_options["mean_free_path_km"]
                    perturbation_prob = self._get_pertubation_prob_per_step(step_size_km=distance_step_this_step_km, mean_free_path_km=mean_free_path_km)

                    # Check if applying a perturbation at this step
                    if random_state.uniform(0.,1.) < perturbation_prob :

                        # Get the current flavor state and choose a particular interaction flavor based on the state amplitudes
                        psi_flav_before = psi_mass_to_flav_basis(psi_mass,self.PMNS) 
                        flav_probs = [ psi_flavor_prob(psi_flav_before,flav) for flav in range(self.num_states) ]
                        flav_probs_rng = [ random_state.uniform(0.,p) for p in flav_probs ]
                        interaction_flavor = np.argmax(flav_probs_rng)
                        psi_flav_after = self._get_initial_psi_flavor(interaction_flavor)

                        # Rotate back to mass basis
                        initial_psi_mass = psi_flav_to_mass_basis(psi_flav_after,self.PMNS)

                        # Reset propagation
                        phase = initial_phase.copy()
                        distance_so_far_neutrino_km = 0. #TODO?


                #
                # Wavefunction collapse
                #

                #TODO Think about this more now that I've made a dedicated "weak_ibteraction" case...

                # The neutrino experiences wave fucntion collapse
                elif perturbation == "wavefunction_collapse" :

                    # Get the mean free path, and convert to a probability of a perturbation per step
                    assert "mean_free_path_km" in perturbation_options
                    mean_free_path_km = perturbation_options["mean_free_path_km"]
                    perturbation_prob = self._get_pertubation_prob_per_step(step_size_km=distance_step_this_step_km, mean_free_path_km=mean_free_path_km)

                    # Check if applying a perturbation at this step
                    if random_state.uniform(0.,1.) < perturbation_prob :

                        # Get the interaction flavor
                        # Default to randomly picking a flavor
                        if "flavor" in perturbation_options :
                            collapsed_flavor = perturbation_options["flavor"]
                            if callable(collapsed_flavor) :
                                collapsed_flavor = collapsed_flavor(phase,distance_step_this_step_km)
                        else :
                            collapsed_flavor = random_state.randint(0,self.num_states)

                        # Snap to flavor (make flavor wavefunctions 100% the interacting flavor, and rotate to mass basis)
                        psi_flav = self._get_initial_psi_flavor(collapsed_flavor)
                        initial_psi_mass = psi_flav_to_mass_basis(psi_flav,self.PMNS)

                        # Reset propagation
                        phase = initial_phase.copy()
                        distance_so_far_neutrino_km = 0. #TODO?


                #
                # Randomize flavor
                #

                # Randomize the neutrino flavor
                # This is a prediction of black hole - neutrino interactions
                elif perturbation == "randomize_flavor" :

                    # Get the mean free path, and convert to a probability of a perturbation per step
                    assert "mean_free_path_km" in perturbation_options
                    mean_free_path_km = perturbation_options["mean_free_path_km"]
                    perturbation_prob = self._get_pertubation_prob_per_step(step_size_km=distance_step_this_step_km, mean_free_path_km=mean_free_path_km)

                    # Check if applying a perturbation at this step
                    if random_state.uniform(0.,1.) < perturbation_prob :

                        # Snap to a random flavor
                        psi_flav = self._get_initial_psi_flavor( random_state.randint(0,self.num_states) )
                        initial_psi_mass = psi_flav_to_mass_basis(psi_flav,self.PMNS)

                        # Reset propagation
                        phase = initial_phase.copy()
                        distance_so_far_neutrino_km = 0. #TODO?


                #
                # Randomize mass state
                #

                # Force to a single mass state (e.g. rotate to 100% of a particular mass state, 0% of the others)
                # This is a prediction of black hole - neutrino interactions
                elif perturbation == "randomize_mass_state" :

                    # Get the mean free path, and convert to a probability of a perturbation per step
                    assert "mean_free_path_km" in perturbation_options
                    mean_free_path_km = perturbation_options["mean_free_path_km"]
                    perturbation_prob = self._get_pertubation_prob_per_step(step_size_km=distance_step_this_step_km, mean_free_path_km=mean_free_path_km)

                    # Check if applying a perturbation at this step
                    if random_state.uniform(0.,1.) < perturbation_prob :

                        # Snap to a random mass state
                        initial_psi_mass = np.zeros( self.num_states, dtype=np.complex128 )
                        initial_psi_mass[random_state.randint(0,self.num_states)] = 1.

                        # Reset propagation
                        phase = initial_phase.copy()
                        distance_so_far_neutrino_km = 0. #TODO?


                #
                # Randomize phase
                #

                # Randomize the neutrino phase
                # This is another way of thinking about black hole - neutrino interactions
                elif perturbation == "randomize_phase" :

                    #TODO what about complex part of phase? should fluctuate that too...

                    # Get the mean free path, and convert to a probability of a perturbation per step
                    assert "mean_free_path_km" in perturbation_options
                    mean_free_path_km = perturbation_options["mean_free_path_km"]
                    perturbation_prob = self._get_pertubation_prob_per_step(step_size_km=distance_step_this_step_km, mean_free_path_km=mean_free_path_km)

                    # Check if applying a perturbation at this step
                    if random_state.uniform(0.,1.) < perturbation_prob :

                        # Randomise the phase of ALL the mass states
                        # Note that only difference between the phase shifts actually impacts oscillations
                        phase = np.random.uniform( 0., 2.*np.pi, size=phase.shape )


                #
                # Neutrino loss
                #

                # Neutrino is lost in the stochastic process
                # e.g. it is sucked into a black hole, or decays to something invisible
                #TODO also allow individual mass states to be lost, either into black hole or via decay
                elif perturbation == "neutrino_loss" :

                    # Get the mean free path, and convert to a probability of a perturbation per step
                    assert "mean_free_path_km" in perturbation_options
                    mean_free_path_km = perturbation_options["mean_free_path_km"]
                    perturbation_prob = self._get_pertubation_prob_per_step(step_size_km=distance_step_this_step_km, mean_free_path_km=mean_free_path_km)

                    # Check if applying a perturbation at this step
                    if random_state.uniform(0.,1.) < perturbation_prob :

                        # The neutrino disappears
                        # Set its wavefunction to 0 for all subsequent steps
                        initial_psi_mass.fill(0.)
                        #TODO could make this more efficient by terminating loop, but would need to take care of other variables being populated


                # Catch case where user provides unknown input
                elif perturbation in [ "baseline", "production_distance", ] : # These cases not handled in this "per step" loop
                    pass
                else :
                    raise Exception("Unknown pertubation type : %s" % perturbation)


            #
            # Update the mass states (plane waves)
            #

            '''
            Mass eigenstate plane wave evolution (ultrarelativistic limit) is defined by:
                |nu_i(L)> = exp{-i (m2_i * L / 2E) + phi } |nu_i(0)>   (where phi is a phase)

            This is what we will use to evolve the mass state over distance.

            There is a standard pre-calculated factor applied to the oscillation term (e.g. the X in sin^2(X) ) 
            in neutrino oscillation transition probability calculations to convert from the commonly used 
            km/eV^2/GeV units to natural units 
            
            e.g.     (dm2 * L) / (4 E) [natural units]
                  --> 1.27 * (dm2 [eV^2] * L [km] / E [GeV] )
            
            Can re-use this for the plane wave evolution, but need to take care of the factor 1/2  (e.g. 1/2E vs 1/4E)

            See https://en.wikipedia.org/wiki/Neutrino_oscillation for more details
            https://indico.cern.ch/event/305391/contributions/701285/attachments/580248/798873/ZZ_NeutrinoPhysics_L2.pdf slide 19 has the conversion
            '''

            plane_wave_unit_conversion = 2. * OSC_FREQUENCY_UNIT_CONVERSION

            psi_mass = np.exp( -1.j * ( ( plane_wave_unit_conversion * np.square(self.mass_state_masses_eV) * distance_so_far_neutrino_km / E_GeV) + phase )  ) * initial_psi_mass

            # Add to containers
            psi_mass_values.append(psi_mass)

            # Add step to distance travelled
            distance_so_far_observer_km += distance_step_observer_km
            distance_so_far_neutrino_km += distance_step_this_step_km

            # Store distance s
            distance_values_observer_km.append(distance_so_far_observer_km)
            distance_values_neutrino_km.append(distance_so_far_neutrino_km)


        #
        # Done
        #

        # Numpy-ify
        distance_values_observer_km = np.asarray(distance_values_observer_km)
        distance_values_neutrino_km = np.asarray(distance_values_neutrino_km)
        psi_mass_values = np.asarray(psi_mass_values)

        #TODO return perturbation positions

        return distance_values_observer_km, distance_values_neutrino_km, psi_mass_values


    def get_osc_probs(self,
        initial_flavor,
        E_GeV,
        distance_km,
        perturbation=None,
        perturbation_options=None,
        num_steps=None,
        use_cache=True,
        random_state=None,
    ) :
        '''
        Calculate oscillation probability for a propagating neutrino.
        Uses `propagate_mass_states` to propagate the neutrino mass states,
        then rotates the results to the flavor basis to calcuate 
        oscillation probability.

        Args :
            initial_psi_flav : (N) array (N is number of flavors)
                Inital flavor state of neutrino
            For all other args, see `propagate_mass_states` function documentation
        '''

        # Load cached results if available
        if use_cache :
            results, func_call_hash = self.load_cached_results("get_osc_probs", locals())

        # Compute if required
        if (not use_cache) or (results is None) :

            # No cached results, run the function...

            # Init random state
            if random_state is None :
                random_state = self.init_random_state()

            # Define initial flavor stat
            initial_psi_flav = self._get_initial_psi_flavor(initial_flavor)

            # Rotate from flavor to mass basis
            initial_psi_mass = psi_flav_to_mass_basis(initial_psi_flav,self.PMNS) 

            # Propagate mass eigenstates
            x_obs_values, x_nu_values, psi_mass_values = self.propagate_mass_states(
                initial_psi_mass=initial_psi_mass,
                E_GeV=E_GeV,
                distance_km=distance_km,
                perturbation=perturbation,
                perturbation_options=perturbation_options,
                num_steps=num_steps,
                random_state=random_state,
            )

            # Rotate mass wavefunctions to flavor basis
            psi_flav_values = np.array([ psi_mass_to_flav_basis(psi_mass,self.PMNS) for psi_mass in psi_mass_values ]) #TODO Array-ify

            # Get osc prob
            osc_prob_values = np.array([ [ psi_flavor_prob(psi_flav,flav) for flav in range(self.num_states) ] for psi_flav in psi_flav_values ] ) #TODO Array-ify

            # Save to cache
            results = {
                "x_obs_values" : x_obs_values,
                "x_nu_values" : x_nu_values,
                "psi_mass_values" : psi_mass_values,
                "psi_flav_values" : psi_flav_values,
                "osc_prob_values" : osc_prob_values,
            }
            if use_cache :
                self.save_results_to_cache("get_osc_probs", func_call_hash, results)

        # Return results
        return results["x_obs_values"], results["x_nu_values"], results["psi_mass_values"], results["psi_flav_values"], results["osc_prob_values"]


    def _get_initial_psi_flavor(self,initial_flavor) :
        '''
        Return a flavor psi state corresponding to a neutrino in 100% of a particular flavor state
        e.g. a neutrino produced in a weak interaction/decay
        '''
        initial_psi_flav = np.zeros( self.num_states, dtype=np.complex128 )
        initial_psi_flav[initial_flavor] = 1.
        return initial_psi_flav


    def _get_pertubation_prob_per_step(self, step_size_km, mean_free_path_km) :
        '''
        Helper function to get the probability for a perturbation in a single 
        integration step, based on a mean free path.
        '''
        assert mean_free_path_km > step_size_km, "Mean free path (%0.3g km) must be larger than solver step size (%0.3g km)" % ( perturbation_mean_free_path_km, step_size_km )
        perturbation_prob = step_size_km / mean_free_path_km
        return perturbation_prob


    def get_osc_probs_loop(self,
        num_events, 
        num_individual_events_to_return=None, # Choose how many of the indiivudal neutrinos to return. Too many creates memory issues. Note that full number is always used for the averging calc, just not stored.
        *args, # args for the underlying `get_osc_probs` call
        **kwargs # kwargs for the underlying `get_osc_probs` call
    ) :
        '''
        Loop to calculate osc probs many times.
        When applying stochastic perturbations, can consider each repeat to represent one neutrino.
        '''

        # Load cached results if available
        results, func_call_hash = self.load_cached_results("get_osc_probs_loop", locals())
        if results is None :

            #
            # Calculate with perturbations
            #

            # Init random state
            random_state = self.init_random_state()

            # Prepare containers
            x_nu_per_event = []
            psi_mass_per_event = []
            psi_flav_per_event = []
            osc_prob_per_event = []

            # Loop to calculate decoherence case many times (random stochasticity each time)
            for i_event in range(num_events) :

                # Report progress
                if i_event % (num_events/100) == 0 :
                    print("%0.3g%% done" % ( 100. * float(i_event) / float(num_events) ) )

                # Calc os probs
                x_obs_values, x_nu_values, psi_mass_values, psi_flav_values, osc_prob_values = self.get_osc_probs(*args, use_cache=False, random_state=random_state, **kwargs)

                # Add to containers
                # For everything apart from `osc_prob_values` (which is needed for the final average) only return subset if `num_individual_events_to_return` specified
                if (num_individual_events_to_return is not None) and (i_event < num_individual_events_to_return) :
                    x_nu_per_event.append(x_nu_values)
                    psi_mass_per_event.append(psi_mass_values)
                    psi_flav_per_event.append(psi_flav_values)
                osc_prob_per_event.append(osc_prob_values)

            # numpy-ify
            x_nu_per_event = np.asarray(x_nu_per_event)
            psi_mass_per_event = np.asarray(psi_mass_per_event)
            psi_flav_per_event = np.asarray(psi_flav_per_event)
            osc_prob_per_event = np.asarray(osc_prob_per_event)

            # Average the osc probs for all events
            osc_prob_average = np.array([ np.mean(osc_prob_per_event[:,:,i],axis=0) for i in range(osc_prob_per_event.shape[2]) ]).T

            # Save to cache
            results = collections.OrderedDict()
            results["x_obs_values"] = x_obs_values
            results["x_nu_per_event"] = x_nu_per_event
            results["psi_mass_per_event"] = psi_mass_per_event
            results["psi_flav_per_event"] = psi_flav_per_event
            results["osc_prob_per_event"] = osc_prob_per_event[:num_individual_events_to_return,:]# Subset if `num_individual_events_to_return` specified
            results["osc_prob_average"] = osc_prob_average
            self.save_results_to_cache("get_osc_probs_loop", func_call_hash, results)

        # Return results
        return tuple(results.values())


    def _get_perturbation(self, random_state, perturbation_function, perturbation_size, positive_only=False) :
        '''
        Helper function to return perturbations 
        '''

        # Gaussian perturbation
        if perturbation_function in ["normal", "gaussian", "gauss"] :
            perturbation_size = random_state.normal(0., perturbation_size)
            if positive_only :
                perturbation_size = np.abs(perturbation_size) 

        # Exponential perturbation
        elif perturbation_function in ["exp", "exponential"] :
            perturbation_size = random_state.exponential(perturbation_size)
            if not positive_only :
                perturbation_size *= random_state.choice([-1., 1.])

        # Uniform perturbation
        elif perturbation_function in ["uniform"] :
            perturbation_size = random_state.uniform( (0. if positive_only else -perturbation_size), perturbation_size )

        else :
            raise Exception("Unknown `perturbation_function` : %s" % perturbation_function)

        return perturbation_size



#
# Helper functions
#

def get_neutrino_masses(lowest_neutrino_mass_eV, mass_splittings_eV2) :
    '''
    Get underlying mass values from mass splitting + lowest mass 
    '''

    masses_eV = np.array( [lowest_neutrino_mass_eV] + np.sqrt( np.square(lowest_neutrino_mass_eV) + mass_splittings_eV2 ).tolist() )
    return masses_eV


#
# Plotting functions
#

def plot_state_evolution(
    initial_flavor,
    x_values,
    psi_mass_values,
    psi_flav_values,
    osc_prob_values,
    fig=None,
    show_imaginary=True,
    imag_alpha=0.3,
    mass_state_labels=[r"\nu_{1}", r"\nu_{2}"],
    flav_state_labels=[r"\nu_{\alpha}", r"\nu_{\beta}"],
) :
    '''
    Plot the evolution of the mass and falvor states, and the resulting oscillation probability.

    Basically takes the outputs of `MassStatePropagator.get_osc_probs` as inputs.
    '''

    if fig is None :
        fig = Figure( ny=3, sharex=True, figsize=(FIG_WIDTH,6) )

    # Hnalde num states
    num_states = psi_mass_values.shape[1]
    assert len(mass_state_labels) == num_states
    assert len(flav_state_labels) == num_states

    # Plot flavor basis
    fig.get_ax(y=0).plot( x_values, psi_flav_values[:,0].real, color="red", linestyle="-", label=r"$%s$ real"%flav_state_labels[0] )
    if show_imaginary :
        fig.get_ax(y=0).plot( x_values, psi_flav_values[:,0].imag, color="red", linestyle="--", label=r"$%s$ imaginary"%flav_state_labels[0], alpha=imag_alpha )
    fig.get_ax(y=0).plot( x_values, psi_flav_values[:,1].real, color="blue", linestyle="-", label=r"$%s$ real"%flav_state_labels[1] )
    if show_imaginary :
        fig.get_ax(y=0).plot( x_values, psi_flav_values[:,1].imag, color="blue", linestyle="--", label=r"$%s$ imaginary"%flav_state_labels[1], alpha=imag_alpha )
    if num_states == 3 :
        fig.get_ax(y=0).plot( x_values, psi_flav_values[:,2].real, color="green", linestyle="-", label=r"$%s$ real"%flav_state_labels[2] )
        if show_imaginary :
            fig.get_ax(y=0).plot( x_values, psi_flav_values[:,2].imag, color="green", linestyle="--", label=r"$%s$ imaginary"%flav_state_labels[2], alpha=imag_alpha )
    fig.get_ax(y=0).set_ylim(-1.1,1.1)
    fig.get_ax(y=0).set_ylabel(r"$\left|\nu_{\rm{flavor}}\right\rangle$")

    # Plot mass basis
    fig.get_ax(y=1).plot( x_values, psi_mass_values[:,0].real, color="orange", linestyle="-", label=r"$%s$ real"%mass_state_labels[0] )
    if show_imaginary : 
        fig.get_ax(y=1).plot( x_values, psi_mass_values[:,0].imag, color="orange", linestyle="--", label=r"$%s$ imaginary"%mass_state_labels[0], alpha=imag_alpha )
    fig.get_ax(y=1).plot( x_values, psi_mass_values[:,1].real, color="purple",linestyle="-", label=r"$%s$ real"%mass_state_labels[1] )
    if show_imaginary : 
        fig.get_ax(y=1).plot( x_values, psi_mass_values[:,1].imag, color="purple",linestyle="--", label=r"$%s$ imaginary"%mass_state_labels[1], alpha=imag_alpha )
    if num_states == 3 :
        fig.get_ax(y=1).plot( x_values, psi_mass_values[:,2].real, color="darkcyan",linestyle="-", label=r"$%s$ real"%mass_state_labels[2] )
        if show_imaginary : 
            fig.get_ax(y=1).plot( x_values, psi_mass_values[:,2].imag, color="darkcyan",linestyle="--", label=r"$%s$ imaginary"%mass_state_labels[2], alpha=imag_alpha )
    fig.get_ax(y=1).set_ylim(-1.1,1.1)
    fig.get_ax(y=1).set_ylabel(r"$\left|\nu_{\rm{mass}}\right\rangle$")

    # Plot osc prob
    fig.get_ax(y=2).plot( x_values, osc_prob_values[:,0], color="red", label=r"P($%s \rightarrow %s$)"%(flav_state_labels[initial_flavor],flav_state_labels[0]) )
    fig.get_ax(y=2).plot( x_values, osc_prob_values[:,1], color="blue", label=r"P($%s \rightarrow %s$)"%(flav_state_labels[initial_flavor],flav_state_labels[1]) )
    if num_states == 3 :
        fig.get_ax(y=2).plot( x_values, osc_prob_values[:,2], color="green", label=r"P($%s \rightarrow %s$)"%(flav_state_labels[initial_flavor],flav_state_labels[2]) )
    fig.get_ax(y=2).set_ylim(-0.05,1.05)
    fig.get_ax(y=2).set_ylabel(r"$P(X \rightarrow Y)$")

    # Format
    fig.get_ax(y=2).set_xlabel(r"Propagation distance [km]")
    # fig.get_ax(y=2).set_xticklabels([])
    for ax in fig.get_all_ax() :
        width = x_values[-1] - x_values[0]
        padding = width / 100.
        ax.set_xlim(x_values[0]-padding, x_values[-1]+padding)
        ax.grid(True,zorder=-1)
        # ax.legend()
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    fig.tight_layout()

    return fig


def plot_perturbed_states_osc_prob(
    initial_flavor,
    final_flavor,
    x_values,
    osc_prob_values_per_event_perturb,
    osc_prob_values_mean_perturb,
    osc_prob_values_no_perturb=None,
    fig=None,
    flav_state_labels=[r"\nu_{\alpha}", r"\nu_{\beta}"],
    max_events_to_plot=200,
) :

    if fig is None :
        fig = Figure(figsize=(FIG_WIDTH,5))

    initial_tex = flav_state_labels[initial_flavor]
    final_tex = flav_state_labels[final_flavor]

    # Plot the standard osc picture
    if osc_prob_values_no_perturb is not None :
        fig.get_ax().plot( x_values, osc_prob_values_no_perturb[:,final_flavor], color="blue", linestyle="-", zorder=2, label=r"$\nu_{\rm{unperturbed}}$" )

    # Plot the individual perturbed curves
    for i_event in range( min( max_events_to_plot, osc_prob_values_per_event_perturb.shape[0] ) ) :
        fig.get_ax().plot( x_values, osc_prob_values_per_event_perturb[i_event,:,final_flavor], color="red", alpha=0.01, zorder=1, label=None )

    # Make a clearer legend element for the individual nu (alpha is too low otherwise)
    xlim = fig.get_ax().get_xlim()
    fig.get_ax().plot( [-101,-100], [0.,0.], color="red", alpha=0.4, label=r"$\nu_{\rm{perturbed}}$" )
    fig.get_ax().set_xlim(xlim) 

    # Plot the average of all perturbed curves
    fig.get_ax().plot( x_values, osc_prob_values_mean_perturb[:,final_flavor], color="red", linestyle="--", zorder=3, label=r"$\nu_{\rm{perturbed,average}}$" )

    # Format
    fig.get_ax().set_ylim(-0.05, 1.05)
    fig.get_ax().set_xlim(x_values[0], x_values[-1])
    fig.get_ax().set_xlabel(r"Propagation distance [km]")
    # fig.get_ax().set_xticklabels([])
    fig.get_ax().set_ylabel( r"P($%s \rightarrow %s$)" % (initial_tex,final_tex) )
    fig.get_ax().grid(True)
    fig.get_ax().legend( loc="%s right"%("lower" if final_flavor == initial_flavor else "upper"))
    fig.tight_layout()

    return fig

