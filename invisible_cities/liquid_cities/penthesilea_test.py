import os
import numpy  as np
import tables as tb

from pytest import mark

from .. core.core_functions    import in_range
from .. core.system_of_units_c import units
from .. core.testing_utils     import assert_dataframes_close
from .. core.configure         import configure
from .. io                     import dst_io as dio
from .. io.mchits_io           import load_mchits

from .  penthesilea            import penthesilea


def test_penthesilea_KrMC(KrMC_pmaps_filename, KrMC_hdst, config_tmpdir):
    PATH_IN   = KrMC_pmaps_filename
    PATH_OUT  = os.path.join(config_tmpdir,'Kr_HDST.h5')
    conf      = configure('dummy invisible_cities/config/liquid_penthesilea.conf'.split())
    nevt_req  = 10

    DF_TRUE =  KrMC_hdst.true

    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     event_range   = (nevt_req,),
                     **KrMC_hdst.config))

    cnt = penthesilea(**conf)
    assert cnt.events_in  == nevt_req
    assert cnt.events_out == len(set(DF_TRUE.event))

    df_penthesilea = dio.load_dst(PATH_OUT , 'RECO', 'Events')
    assert_dataframes_close(df_penthesilea, DF_TRUE, check_types=False)


def test_penthesilea_filter_events(config_tmpdir, Kr_pmaps_run4628_filename):
    PATH_IN =  Kr_pmaps_run4628_filename

    PATH_OUT = os.path.join(config_tmpdir, 'KrDST_4628.h5')
    nrequired = 50
    conf = configure('dummy invisible_cities/config/liquid_penthesilea.conf'.split())
    conf.update(dict(run_number = 4628,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT,

                     drift_v     =      2 * units.mm / units.mus,
                     s1_nmin     =      1,
                     s1_nmax     =      1,
                     s1_emin     =      1 * units.pes,
                     s1_emax     =     30 * units.pes,
                     s1_wmin     =    100 * units.ns,
                     s1_wmax     =    300 * units.ns,
                     s1_hmin     =      1 * units.pes,
                     s1_hmax     =      5 * units.pes,
                     s1_ethr     =    0.5 * units.pes,
                     s2_nmin     =      1,
                     s2_nmax     =      2,
                     s2_emin     =    1e3 * units.pes,
                     s2_emax     =    1e4 * units.pes,
                     s2_wmin     =      2 * units.mus,
                     s2_wmax     =     20 * units.mus,
                     s2_hmin     =    1e3 * units.pes,
                     s2_hmax     =    1e5 * units.pes,
                     s2_ethr     =      1 * units.pes,
                     s2_nsipmmin =      5,
                     s2_nsipmmax =     30,
                     event_range = (0, nrequired)))

    events_pass = ([ 1]*21 + [ 4]*15 + [10]*16 + [19]*17 +
                   [20]*19 + [21]*15 + [26]*23 + [29]*22 +
                   [33]*14 + [41]*18 + [43]*18 + [45]*13 +
                   [46]*18)
    peak_pass   = [int(in_range(i, 119, 126))
                   for i in range(229)]

    cnt      = penthesilea(**conf)
    nevt_in  = cnt.events_in
    nevt_out = cnt.events_out
    assert nrequired    == nevt_in
    assert nevt_out     == len(set(events_pass))

    dst = dio.load_dst(PATH_OUT, "RECO", "Events")
    assert len(set(dst.event.values)) ==   nevt_out
    assert  np.all(dst.event.values   == events_pass)
    assert  np.all(dst.npeak.values   ==   peak_pass)


@mark.serial
def test_penthesilea_produces_tracks_when_require(KrMC_pmaps_filename, KrMC_hdst, config_tmpdir):
    PATH_IN   = KrMC_pmaps_filename
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_HDST_with_MC.h5")
    conf      = configure('dummy invisible_cities/config/liquid_penthesilea.conf'.split())
    nevt_req  = 10

    conf.update(dict(files_in        = PATH_IN,
                     file_out        = PATH_OUT,
                     event_range     = (nevt_req,),
                     **KrMC_hdst.config))

    penthesilea(**conf)

    with tb.open_file(PATH_OUT) as h5out:
        assert "MC"          in h5out.root
        assert "MC/MCTracks" in h5out.root


@mark.serial
def test_penthesilea_true_hits_are_correct(KrMC_true_hits, config_tmpdir):
    penthesilea_output_path = os.path.join(config_tmpdir,'Kr_HDST_with_MC.h5')
    penthesilea_evts        = load_mchits(penthesilea_output_path)
    true_evts               = KrMC_true_hits.hdst

    assert sorted(penthesilea_evts) == sorted(true_evts)
    for evt_no, true_hits in true_evts.items():
        penthesilea_hits = penthesilea_evts[evt_no]

        assert len(penthesilea_hits) == len(true_hits)
        assert all(p_hit == t_hit for p_hit, t_hit in zip(penthesilea_hits, true_hits))