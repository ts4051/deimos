'''
Functions for working with space=time metric fluctuations

For neutrino - virtual black hole interactions, see instead deimos/models/decoherence/nuVBH_model.py

Tom Stuttard
'''

#
# Lightcone/distance/time fluctuations
#

#TODO integrate model here


#
# Space-time metric
#

# Define relevent metrics
FLAT_SPACETIME_METRIC_TENSOR = np.array([ # Minkowski
    [ -1., 0., 0., 0., ], #TODO c ?
    [  0., 1., 0., 0., ],
    [  0., 0., 1., 0., ],
    [  0., 0., 0., 1., ],
])

def get_ds_from_metric_tensor(dx, g=FLAT_SPACETIME_METRIC_TENSOR) :
    '''
    Compute the distance/displacement step, ds, corresponding to some time/spatial dimension steps, dx, 
    for a given space-time metric tensor, g.

    For 4-D space-time, dx = [dt, dx, dy, dz], and the metric tensor is 4x4

    See https://www.mv.helsinki.fi/home/syrasane/cosmo/lect2018_02.pdf

    Note that off-diagonal metric tensor components represent non-orthogonal coordinate systems,
    which are not common so rarely use these.
    '''

    #TODO also do integral to get overall distance here?

    dim = dx.size
    assert g.shape == (dim, dim)

    ds2 = 0.
    for mu in range(dim) :
        for nu in range(dim) :
            ds2 += ( g[mu, nu] * dx[mu] * dx[nu] )

    return np.sqrt(ds2)


def get_fluctuated_metric_tensor(a1, a2, a3, a4) :
    '''
    Using the definition of metric fluctuations From hep-ph/0606048

    Specificially eqn 2.4 (derived from 2.2, 2.3)

    This is 1+1D ([t, x]) metric tensor with fluctuations characterised by static coefficients ai that are Gaussian random variables with <ai> = 0
    Can choose x to lie along particle direction
    
    ai are Gaussian random variables with <ai> = 0 and sigmai

    a4 characterises the distance-only flucutation

    Note that the analytic g expression below is a fluctuation of flat space-time (e.g. Minkowski metric tensor). Could in princip
    '''

    # Check that fluctuations are small or at least comaprable to the overall metric structure
    # This is an assumption used in this model (and a very reasonable one)
    # If have e.g. a4 < -1, ds starts to rise again  with decreasing a4 due to the sqrt( (a4+1)^2 ) term, which is
    # clearly nonsense (and thus a limitations of the parameterisation)
    assert np.all( np.abs(a1) <= 1. ), "Metric perturbation cannot be larger in scale than the unfluctuated metric (parameterisation assumes that the pertubations are small)"
    assert np.all( np.abs(a2) <= 1. ), "Metric perturbation cannot be larger in scale than the unfluctuated metric (parameterisation assumes that the pertubations are small)"
    assert np.all( np.abs(a3) <= 1. ), "Metric perturbation cannot be larger in scale than the unfluctuated metric (parameterisation assumes that the pertubations are small)"
    assert np.all( np.abs(a4) <= 1. ), "Metric perturbation cannot be larger in scale than the unfluctuated metric (parameterisation assumes that the pertubations are small)"

    # Define metric
    # This is specifically for flat space-time + perturbation
    g = np.array([
        [    ( -1. * np.square(a1 + 1) ) + np.square(a2)     ,     ( -1. * a3 * (a1 + 1) ) +  ( a2 * (a4 + 1) )    ],
        [    ( -1. * a3 * (a1 + 1) ) +  ( a2 * (a4 + 1) )    ,     ( -1. * np.square(a3) ) +  np.square(a4 + 1)    ],
    ])


    # Optionally, verify the derivation (see eqns 2.2 and 2.3)
    if False :

        O = np.array([ # This is the fundamental perurbation
            [ a1+1., a2, ],
            [ a3, a4+1, ],
        ])

        eta = np.array([ # Note: Could use a different metric here if desired (this is flat space-time)
            [ -1., 0., ],
            [ 0., 1., ],
        ])

        g_v2 = np.matmul( O, np.matmul(eta, O.T) )

        assert np.array_equal(g, g_v2)

    # Done
    return g
