import os
import random
import numpy  as np
import pandas as pd
import pytest

from pytest                       import mark
from pytest                       import raises

from hypothesis                   import given
from hypothesis.strategies        import floats
from hypothesis.extra.pandas      import data_frames
from hypothesis.extra.pandas      import column
from hypothesis.extra.pandas      import range_indexes

from .. reco    .deconv_functions import cut_and_redistribute_df
from .. reco    .deconv_functions import drop_isolated_sensors
from .. reco    .deconv_functions import interpolate_signal
from .. reco    .deconv_functions import deconvolution_input
from .. reco    .deconv_functions import deconvolve
from .. reco    .deconv_functions import richardson_lucy
from .. reco    .deconv_functions import generate_satellite_mask
from .. reco    .deconv_functions import collect_component_sizes

from .. core    .core_functions   import in_range
from .. core    .core_functions   import shift_to_bin_centers
from .. core    .testing_utils    import assert_dataframes_close

from .. io      .dst_io           import load_dst

from .. types   .symbols          import InterpolationMethod
from .. types   .symbols          import CutType

from .. database.load_db          import DataSiPM

from scipy.stats                  import multivariate_normal


@pytest.fixture(scope='function')
def sat_arr(ICDATADIR):
    '''
    An array made to imitate a z-slice that would be passed through deconvolution
    '''
    arr = np.array([[1.   , 1.   , 0.3  , 0.1  , 1.   ],
                    [1.   , 1.   , 0.1  , 0.1  , 0.1  ],
                    [0.1  , 0.1  , 0.1  , 0.1  , 0.1  ],
                    [0.2  , 0.1  , 1.   , 1.   , 1.   ],
                    [0.1  , 0.1  , 1.   , 1.   , 1.   ]])
    return arr


@pytest.fixture(scope='session')
def compsize_array(ICDATADIR):
    '''
    An array made to imitate a z-slice that would be passed through deconvolution
    This array has specific values such that differing e_cut reveals boolean arrays
    with differing components across the array.

    Described in more detail in test_component_sizes()
    '''
    arr = np.array([[0.5, 0.3, 0.1, 0.3],
                    [0.3, 0.5, 0.1, 0.1],
                    [0.1, 0.3, 0.5, 0.1],
                    [0.1, 0.1, 0.1, 0.5]])
    return arr


@pytest.fixture(scope='session')
def no_satellite_killer(ICDATADIR):
    '''
    The default satellite_killer parameters provided if you don't want it to be run on
    your sample. Used to avoid its usage in certain tests.
    '''
    satellite_params = dict(satellite_start_iter = np.inf, 
                            satellite_max_size   = 10,
                            e_cut                = 0.5, 
                            cut_type             = CutType.abs)
    return satellite_params


@given(data_frames(columns=[column('A', dtype=float, elements=floats(1, 1e3)),
                            column('B', dtype=float, elements=floats(1, 1e3)),
                            column('C', dtype=float, elements=floats(1, 1e3))],
                     index=range_indexes(min_size=2, max_size=10)))


def test_cut_and_redistribute_df(df):
    cut_var       = 'A'
    redist_var    = ['B', 'C']
    cut_val       = round(df[cut_var].mean(), 3)
    cut_condition = f'{cut_var} > {cut_val:.3f}'
    cut_function  = cut_and_redistribute_df(cut_condition, redist_var)
    df_cut        = cut_function(df)
    df_cut_manual = df.loc[df[cut_var].values > cut_val, :].copy()
    df_cut_manual.loc[:, redist_var] = df_cut_manual.loc[:, redist_var] * df.loc[:, redist_var].sum() /  df_cut_manual.loc[:, redist_var].sum()
    assert_dataframes_close(df_cut, df_cut_manual)


def test_drop_isolated_sensors():
    size          = 20
    dist          = [10.1, 10.1]
    x             = random.choices(np.linspace(-200, 200, 41), k=size)
    y             = random.choices(np.linspace(-200, 200, 41), k=size)
    q             = np.random.uniform(0,  20, size)
    e             = np.random.uniform(0, 200, size)
    df            = pd.DataFrame({'X':x, 'Y':y, 'Q':q, 'E':e})
    drop_function = drop_isolated_sensors(dist, ['E'])
    df_cut        = drop_function(df)

    if len(df_cut) > 0:
        assert np.isclose(df_cut.E.sum(), df.E.sum())

    for row in df_cut.itertuples(index=False):
        n_neighbours = len(df_cut[in_range(df_cut.X, row.X - dist[0], row.X + dist[0]) &
                                  in_range(df_cut.Y, row.Y - dist[1], row.Y + dist[1])])
        assert n_neighbours > 1


def test_interpolate_signal():
    ref_interpolation = np.array([0.   , 0.   , 0.   , 0.   , 0.   , 0    , 0.   , 0.   , 0.   ,
                                  0.   , 0.   , 0.   , 0.   , 0.17 , 0.183, 0.188, 0.195, 0.202,
                                  0.201, 0.202, 0.188, 0.181, 0.168, 0.   , 0.   , 0.308, 0.328,
                                  0.344, 0.354, 0.362, 0.365, 0.357, 0.347, 0.326, 0.308, 0.   ,
                                  0.   , 0.5  , 0.531, 0.569, 0.585, 0.593, 0.592, 0.58 , 0.566,
                                  0.543, 0.514, 0.   , 0.   , 0.693, 0.751, 0.786, 0.825, 0.833,
                                  0.827, 0.816, 0.79 , 0.757, 0.703, 0.   , 0.   , 0.818, 0.886,
                                  0.924, 0.957, 0.965, 0.973, 0.963, 0.925, 0.882, 0.818, 0.   ,
                                  0.   , 0.82 , 0.884, 0.93 , 0.958, 0.969, 0.965, 0.954, 0.928,
                                  0.883, 0.818, 0.   , 0.   , 0.698, 0.752, 0.793, 0.819, 0.832,
                                  0.836, 0.826, 0.794, 0.752, 0.699, 0.   , 0.   , 0.509, 0.535,
                                  0.57 , 0.582, 0.6  , 0.592, 0.585, 0.568, 0.536, 0.51 , 0.   ,
                                  0.   , 0.311, 0.328, 0.347, 0.36 , 0.359, 0.361, 0.356, 0.343,
                                  0.328, 0.312, 0.0  , 0.0  , 0.169, 0.182, 0.196, 0.196, 0.201,
                                  0.197, 0.194, 0.192, 0.182, 0.169, 0.   , 0.   , 0.   , 0.   ,
                                  0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])

    g = multivariate_normal((0.5, 0.5), (0.05, 0.5))

    points          = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6)) # Coordinates where g is known
    points          = (points[0].flatten(), points[1].flatten())
    values          = g.pdf(list(zip(points[0], points[1]))) # Value of g at the known coordinates.
    n_interpolation = 12 # How many points to interpolate
    grid            = shift_to_bin_centers(np.linspace(-0.05, 1.05, n_interpolation + 1))# Grid for interpolation

    out_interpolation = interpolate_signal(values, points,
                                           [[-0.05, 1.05], [-0.05, 1.05]],
                                           [grid, grid],
                                           InterpolationMethod.cubic)
    inter_charge      = out_interpolation[0].flatten()
    inter_position    = out_interpolation[1]

    assert np.allclose(ref_interpolation, np.around(inter_charge, decimals=3))
    assert np.allclose(grid             , sorted(set(inter_position[0])))
    assert np.allclose(grid             , sorted(set(inter_position[1])))


def test_deconvolution_input(data_hdst, data_hdst_deconvolved):
    ref_interpolation = np.load(data_hdst_deconvolved)

    hdst   = load_dst(data_hdst, 'RECO', 'Events')
    h      = hdst[(hdst.event == 3021916) & (hdst.npeak == 0)]
    h      = h.groupby(['X', 'Y']).Q.sum().reset_index()
    h      = h[h.Q > 40]

    det_db   = DataSiPM('new', 0)
    det_grid = [np.arange(det_db[var].min() + bs/2, det_db[var].max() - bs/2 + np.finfo(np.float32).eps, bs)
               for var, bs in zip(['X', 'Y'], [1., 1.])]
    interpolator = deconvolution_input([10., 10.], det_grid, InterpolationMethod.cubic)
    inter        = interpolator((h.X, h.Y), h.Q)

    assert np.allclose(ref_interpolation['e_inter'], inter[0])
    assert np.allclose(ref_interpolation['x_inter'], inter[1][0])
    assert np.allclose(ref_interpolation['y_inter'], inter[1][1])


@mark.parametrize("interp_method", InterpolationMethod.__members__)
def test_deconvolution_input_interpolation_method(data_hdst, data_hdst_deconvolved, interp_method):
    with raises(ValueError):
        deconvolution_input([10., 10.], [1., 1.], interp_method)


def test_deconvolve(data_hdst, data_hdst_deconvolved, no_satellite_killer):
    ref_interpolation = np.load (data_hdst_deconvolved)

    hdst   = load_dst(data_hdst, 'RECO', 'Events')
    h      = hdst[(hdst.event == 3021916) & (hdst.npeak == 0)]
    z      = h.Z.mean()
    h      = h.groupby(['X', 'Y']).Q.sum().reset_index()
    h      = h[h.Q > 40]

    det_db   = DataSiPM('new', 0)
    det_grid = [np.arange(det_db[var].min() + bs/2, det_db[var].max() - bs/2 + np.finfo(np.float32).eps, bs)
               for var, bs in zip(['X', 'Y'], [1., 1.])]
    deconvolutor = deconvolve(15, 0.01, [10., 10.], det_grid, inter_method=InterpolationMethod.cubic)

    x, y   = np.linspace(-49.5, 49.5, 100), np.linspace(-49.5, 49.5, 100)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()

    psf           = {}
    psf['factor'] = multivariate_normal([0., 0.], [1.027 * np.sqrt(z/10)] * 2).pdf(list(zip(xx, yy)))
    psf['xr']     = xx
    psf['yr']     = yy
    psf['zr']     = [z] * len(xx)
    psf           = pd.DataFrame(psf)

    deco          = deconvolutor((h.X, h.Y), h.Q, psf,
                                 **no_satellite_killer)

    assert np.allclose(ref_interpolation['e_deco'], deco[0].flatten())
    assert np.allclose(ref_interpolation['x_deco'], deco[1][0])
    assert np.allclose(ref_interpolation['y_deco'], deco[1][1])


def test_richardson_lucy(data_hdst, data_hdst_deconvolved, no_satellite_killer):
    ref_interpolation = np.load (data_hdst_deconvolved)

    hdst   = load_dst(data_hdst, 'RECO', 'Events')
    h      = hdst[(hdst.event == 3021916) & (hdst.npeak == 0)]
    z      = h.Z.mean()
    h      = h.groupby(['X', 'Y']).Q.sum().reset_index()
    h      = h[h.Q>40]

    det_db   = DataSiPM('new', 0)
    det_grid = [np.arange(det_db[var].min() + bs/2, det_db[var].max() - bs/2 + np.finfo(np.float32).eps, bs)
               for var, bs in zip(['X', 'Y'], [1., 1.])]

    interpolator = deconvolution_input([10., 10.], det_grid, InterpolationMethod.cubic)
    inter        = interpolator((h.X, h.Y), h.Q)

    x , y  = np.linspace(-49.5, 49.5, 100), np.linspace(-49.5, 49.5, 100)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()

    psf           = {}
    psf['factor'] = multivariate_normal([0., 0.], [1.027 * np.sqrt(z/10)] * 2).pdf(list(zip(xx, yy)))
    psf['xr']     = xx
    psf['yr']     = yy
    psf['zr']     = z
    psf           = pd.DataFrame(psf)

    deco = richardson_lucy(inter[0], psf.factor.values.reshape(psf.xr.nunique(), psf.yr.nunique()).T,
                           iterations=15, iter_thr=0.0001, **no_satellite_killer)

    assert np.allclose(ref_interpolation['e_deco'], deco.flatten())


def test_grid_binning(data_hdst, data_hdst_deconvolved, no_satellite_killer):
    hdst   = load_dst(data_hdst, 'RECO', 'Events')
    h      = hdst[(hdst.event == 3021916) & (hdst.npeak == 0)]
    z      = h.Z.mean()
    h      = h.groupby(['X', 'Y']).Q.sum().reset_index()
    h      = h[h.Q > 40]

    det_db   = DataSiPM('new', 0)
    det_grid = [np.arange(det_db[var].min() + bs/2, det_db[var].max() - bs/2 + np.finfo(np.float32).eps, bs)
               for var, bs in zip(['X', 'Y'], [9., 9.])]

    deconvolutor = deconvolve(15, 0.01, [10., 10.], det_grid, inter_method=InterpolationMethod.cubic)

    x, y   = np.linspace(-49.5, 49.5, 100), np.linspace(-49.5, 49.5, 100)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()

    psf           = {}
    psf['factor'] = multivariate_normal([0., 0.], [1.027 * np.sqrt(z/10)] * 2).pdf(list(zip(xx, yy)))
    psf['xr']     = xx
    psf['yr']     = yy
    psf['zr']     = [z] * len(xx)
    psf           = pd.DataFrame(psf)

    deco          = deconvolutor((h.X, h.Y), h.Q, psf,
                                 **no_satellite_killer)

    assert np.all((deco[1][0] - det_db['X'].min() + 9/2) % 9 == 0)
    assert np.all((deco[1][1] - det_db['Y'].min() + 9/2) % 9 == 0)


def test_nonexact_binning(data_hdst, data_hdst_deconvolved, no_satellite_killer):
    hdst   = load_dst(data_hdst, 'RECO', 'Events')
    h      = hdst[(hdst.event == 3021916) & (hdst.npeak == 0)]
    z      = h.Z.mean()
    h      = h.groupby(['X', 'Y']).Q.sum().reset_index()
    h      = h[h.Q > 40]

    det_db   = DataSiPM('new', 0)
    det_grid = [np.arange(det_db[var].min() + bs/2, det_db[var].max() - bs/2 + np.finfo(np.float32).eps, bs)
               for var, bs in zip(['X', 'Y'], [9., 9.])]

    deconvolutor = deconvolve(15, 0.01, [10., 10.], det_grid, inter_method=InterpolationMethod.cubic)
    x, y   = np.linspace(-49.5, 49.5, 100), np.linspace(-49.5, 49.5, 100)
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()

    psf           = {}
    psf['factor'] = multivariate_normal([0., 0.], [1.027 * np.sqrt(z/10)] * 2).pdf(list(zip(xx, yy)))
    psf['xr']     = xx
    psf['yr']     = yy
    psf['zr']     = [z] * len(xx)
    psf           = pd.DataFrame(psf)

    deco          = deconvolutor((h.X, h.Y), h.Q, psf,
                                 **no_satellite_killer)

    check_x = np.diff(np.sort(np.unique(deco[1][0]), axis=None))
    check_y = np.diff(np.sort(np.unique(deco[1][1]), axis=None))

    assert(np.all(check_x % 9 == 0))
    assert(np.all(check_y % 9 == 0))


@mark.parametrize("cut_type", CutType)
def test_removing_satellites(sat_arr, cut_type):
    '''
    Test case for removing the satellite in the top-right of the given array

                                  The satellite value to be removed
                                                  |
                                                  V
    hdst = np.array([[1.   , 1.   , 0.3  , 0.1  , 1.   ],
                     [1.   , 1.   , 0.1  , 0.1  , 0.1  ],
                     [0.1  , 0.1  , 0.1  , 0.1  , 0.1  ],
                     [0.2  , 0.1  , 1.   , 1.   , 1.   ],
                     [0.1  , 0.1  , 1.   , 1.   , 1.   ]])
    
    Test uses relative and absolute cuts as both are equivalent here (normalised to 1)
    '''
    # produce array with satellite removed
    hdst_nosat              = np.array(sat_arr)
    hdst_nosat[0][-1]       = 0

    e_cut                   = 0.5
    satellite_max_size      = 3

    # create and check mask
    mask = generate_satellite_mask(sat_arr, satellite_max_size, e_cut, cut_type)
    sat_arr[mask] = 0

    assert np.allclose(sat_arr, hdst_nosat)


@mark.parametrize("cut_type", CutType)
def test_satellite_ecut_minimum(sat_arr, cut_type):
    '''
    Test case in which e_cut is set to zero, and so no modification to the
    original array shoud occur (post-ecut array is all 1s).

    Test uses relative and absolute cuts as both are equivalent here (normalised to 1)
    '''
    
    # set e_cut to minimum (zero)
    e_cut                    = 0
    
    hdst                     = np.array(sat_arr)
    satellite_max_size       = 3
    # create and check mask
    mask = generate_satellite_mask(hdst, satellite_max_size, e_cut, cut_type)
    hdst[mask] = 0
    assert np.allclose(hdst, sat_arr)


@mark.parametrize("cut_type", CutType)
def test_satellite_ecut_maximum(sat_arr, cut_type):
    '''
    Test case in which e_cut is set above largest value (1), 
    and so no modification to the original array shoud occur 
    (post-ecut array is all 0s).

    Test uses relative and absolute cuts as both are equivalent here (normalised to 1)
    '''
    
    # set e_cut to large number (999)
    e_cut                    = 999
    
    hdst                     = np.array(sat_arr)
    satellite_max_size       = 3
    # create and check mask
    mask = generate_satellite_mask(hdst, satellite_max_size, e_cut, cut_type)
    hdst[mask] = 0
    assert np.allclose(hdst, sat_arr)


def test_satellite_size_minimum(sat_arr):
    '''
    Test case in which satellite size is set to zero, and so no modification
    to the original array should occur (every cluster of 1s is > 0).
    '''

    # set maximum satellite size to minimum (zero)
    satellite_max_size      = 0

    hdst                    = np.array(sat_arr)
    e_cut                   = 0.5
    cut_type                = CutType.abs

    # create and check mask
    mask = generate_satellite_mask(hdst, satellite_max_size, e_cut, cut_type)
    hdst[mask] = 0
    assert np.allclose(hdst, sat_arr)


def test_satellite_size_maximum(sat_arr):
    '''
    Test case in which satellite size is set to 999 (larger than feasible), 
    and so all values that surpass the energy cut should be removed.
    '''

    # set maximum satellite size to maximum (999)
    satellite_max_size = 999

    # produce modified array with 1s -> 0s (as expected)
    hdst_novals              = np.array(sat_arr)
    hdst_novals[hdst_novals == 1] = 0

    e_cut                   = 0.5
    cut_type                = CutType.abs

    # create and check mask    
    mask = generate_satellite_mask(sat_arr, satellite_max_size, e_cut, cut_type)
    sat_arr[mask] = 0
    assert np.allclose(sat_arr, hdst_novals)


@mark.parametrize("e_cut, expected_size", [(0, 2), (0.2, 3), (0.4, 2), (0.6, 1)])
def test_component_sizes(compsize_array, e_cut, expected_size):
    '''
    Given an array, with differing applied e_cuts will
    provide expected component sizes.

    array = ([[0.5, 0.3, 0.1, 0.3],
              [0.3, 0.5, 0.1, 0.1],
              [0.1, 0.3, 0.5, 0.1],
              [0.1, 0.1, 0.1, 0.5])
    
    ecuts:
    0     ->  array is all 1s, 2 components (zero 0s, sixteen 1s)
    0.2   ->  array is no longer uniform, 3 components (eight 0s, eight 1s -> two clusters (seven 1s, one 2))
    0.4   ->  array is identity matrix, 2 components (twelve 0s, four 1s)
    0.6   ->  array is all 0s, 1 component (sixteen 0s)
    '''

    bool_mask = np.where(compsize_array < e_cut, 0, 1)

    # check number of components in array match what is expected
    labels, no_components = collect_component_sizes(bool_mask)
    assert len(no_components) == expected_size


def test_component_elements(compsize_array):
    '''
    Given an array, with differing applied e_cuts will
    provide expected number of elements in each component.
    
    The exact logic of the array is described above in test_component_sizes()
    '''
    # 0.2 cut, provides 3 components with lengths seven 0s, seven 1s, and one 2
    e_cut = 0.2
    
    # generate labelling
    bool_mask = np.where(compsize_array < e_cut, 0, 1)
    labels, no_components = collect_component_sizes(bool_mask)

    # array to compare against
    expected_elements = np.array([8, 7, 1])
    assert np.allclose(no_components, expected_elements)

    
def test_satellite_zero_protection():
    '''
    This test checks that the protection in place to avoid cases in which
    the number of 0s within the boolean mask from `generate_satellite_mask()` 
    are less than the satellite_max_size, and so are flagged incorrectly as satellites.
    '''

    # set size to be just above total number of zeros in given array
    satellite_max_size = 10
    
    e_cut = 0.5
    cut_type = CutType.abs
    
    # provide array with very few 0s that would usually be mislabelled as satellites
    inverted_array = np.array([[0, 0, 1, 1, 0],
                               [0, 0, 1, 1, 1],     
                               [1, 1, 1, 1, 1],     
                               [1, 1, 1, 0, 0],     
                               [1, 1, 1, 0, 0]])

    # mask with no satellites, as we don't want zeros to be considered as satellites
    no_satellite_mask = np.full(inverted_array.shape, False)
    
    # generate mask
    mask = generate_satellite_mask(inverted_array, satellite_max_size, e_cut, cut_type)

    # ensure that no satellites are flagged (as there are none)
    assert np.allclose(no_satellite_mask, mask)


def test_generate_satellite_mask_doesnt_modify_input(sat_arr):
    '''
    Test that ensures that the applied functions don't modify the provided
    z-slice array in any way.
    '''
    copy_hdst = np.array(sat_arr)
    satellite_max_size = 3
    e_cut = 0.5
    cut_type = CutType.abs

    mask = generate_satellite_mask(copy_hdst, satellite_max_size, e_cut, cut_type)

    assert np.allclose(sat_arr, copy_hdst)
