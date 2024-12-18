import os
import numpy  as np
import pandas as pd

from pytest                      import mark
from pytest                      import raises

from ..     reco.psf_functions   import hdst_psf_processing
from ..     reco.psf_functions   import add_variable_weighted_mean
from ..     reco.psf_functions   import add_empty_sensors_and_normalize_q
from ..     reco.psf_functions   import create_psf

from .. database                 import load_db
from ..       io.dst_io          import load_dst
from ..     core.testing_utils   import assert_dataframes_close


def test_add_variable_weighted_mean(ICDATADIR):
    PATH_IN = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")

    hdst    = load_dst(PATH_IN, 'RECO', 'Events')
    x_mean  = np.average(hdst.loc[:, 'X'], weights=hdst.loc[:, 'E'])
    y_mean  = np.average(hdst.loc[:, 'Y'], weights=hdst.loc[:, 'E'])

    add_variable_weighted_mean(hdst, 'X', 'E', 'Xpeak')
    add_variable_weighted_mean(hdst, 'Y', 'E', 'Ypeak')

    assert np.allclose(x_mean, hdst.Xpeak.unique())
    assert np.allclose(y_mean, hdst.Ypeak.unique())


def test_add_empty_sensors_and_normalize_q(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    hdst           = load_dst(PATH_IN, 'RECO', 'Events')
    group          = hdst.groupby('event time npeak'.split())
    hdst_processed = group.apply(add_empty_sensors_and_normalize_q    ,
                                 var      = ['X', 'Y']                ,
                                 ranges   = [[-50, 50], [-50, 50]]    ,
                                 database = load_db.DataSiPM('new', 0),
                                 include_groups=False)
    hdst_processed.reset_index(level=3, inplace=True, drop=True )
    hdst_processed.reset_index(         inplace=True, drop=False)

    assert np.allclose(hdst_processed.groupby('event').NormQ.sum().values, 1.0)
    assert np. isclose(hdst_processed.E.sum(), hdst.E.sum())
    assert np. isclose(hdst_processed.Q.sum(), hdst.Q.sum())


@mark.skip(reason="test uses bad input data")
def test_add_empty_sensors_and_normalize_q_file(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    PATH_TEST      = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf_empty_sensors.h5")
    hdst           = load_dst(PATH_IN, 'RECO', 'Events')
    group          = hdst.groupby('event time npeak'.split())
    hdst_processed = group.apply(add_empty_sensors_and_normalize_q    ,
                                 var      = ['X', 'Y']                ,
                                 ranges   = [[-50, 50], [-50, 50]]    ,
                                 database = load_db.DataSiPM('new', 0),
                                 include_groups=False)
    hdst_processed.reset_index(level=3, inplace=True, drop=True )
    hdst_processed.reset_index(         inplace=True, drop=False)
    hdst_psf       = pd.read_hdf(PATH_TEST)
    hdst_psf       = hdst_psf.astype(hdst_processed.dtypes) # since pandas update (1.0.3->1.3.4)

    assert_dataframes_close(hdst_psf, hdst_processed)


@mark.skip(reason="test uses bad input data")
def test_hdst_psf_processing(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    PATH_TEST      = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")

    hdst           = load_dst(PATH_IN, 'RECO', 'Events')
    hdst_processed = hdst_psf_processing(hdst                                 ,
                                         ranges   = [[-50, 50], [-50, 50]]    ,
                                         database = load_db.DataSiPM('new', 0))
    hdst_psf       = pd.read_hdf(PATH_TEST)
    hdst_psf       = hdst_psf.astype(hdst_processed.dtypes) # since pandas update (1.0.3->1.3.4)

    assert_dataframes_close(hdst_psf, hdst_processed)


def test_create_psf(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")
    PATH_TEST      = os.path.join(ICDATADIR, "test_psf.npz")

    hdst           = pd.read_hdf(PATH_IN)
    psf            = np.load(PATH_TEST)

    bin_edges = [np.linspace(-50, 50, 101) for i in range(2)]
    psf_val, entries, binss = create_psf((hdst.RelX, hdst.RelY), hdst.NormQ, bin_edges)

    assert np.allclose(psf['psf'    ], psf_val)
    assert np.allclose(psf['entries'], entries)
    assert np.allclose(psf['bins'   ],   binss)

@mark.parametrize('ndim', (1, 3))
def test_create_psf_fails_param_dim(ICDATADIR, ndim):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")
    hdst           = pd.read_hdf(PATH_IN)
    bin_edges      = [np.linspace(-50, 50, 101) for i in range(ndim)]

    with raises(ValueError):
        psf_val, entries, binss = create_psf((hdst.RelX, hdst.RelY), hdst.NormQ, bin_edges)


def test_create_psf_fails_ndim(ICDATADIR):
    PATH_IN        = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_psf.h5")
    hdst           = pd.read_hdf(PATH_IN)
    bin_edges      = [np.linspace(-50, 50, 101) for i in range(3)]

    with raises(NotImplementedError):
        psf_val, entries, binss = create_psf((hdst.RelX, hdst.RelY, hdst.RelY), hdst.NormQ, bin_edges)
