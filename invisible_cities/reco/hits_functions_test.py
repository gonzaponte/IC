import os

from pytest        import mark
from numpy.testing import assert_almost_equal

from   .. core.configure       import configure
from   .. evm.event_model      import Hit
from   .. evm.event_model      import Cluster
from   .. types.ic_types       import xy
from   .. core                 import system_of_units as units
from   .. core.testing_utils   import assert_hit_equality
from   .. types.ic_types       import NN
from   .. cities.penthesilea   import penthesilea
from   .. io                   import hits_io          as hio
from   .  hits_functions       import merge_NN_hits
from   .  hits_functions       import threshold_hits
from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import lists
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers
from copy                      import deepcopy
from hypothesis                import assume
from hypothesis.strategies     import composite

@composite
def hit(draw, min_value=1, max_value=100):
    x      = draw(floats  (  1,   5))
    y      = draw(floats  (-10,  10))
    xvar   = draw(floats  (.01,  .5))
    yvar   = draw(floats  (.10,  .9))
    Q      = draw(floats  (-10, 100).map(lambda x: NN if x<=0 else x))
    nsipm  = draw(integers(  1,  20))
    npeak  = 0
    z      = draw(floats  ( 50, 100))
    E      = draw(floats  ( 50, 100))
    x_peak = draw(floats  (  1,   5))
    y_peak = draw(floats  (-10,  10))
    Qc     = draw(floats  (  0, 100).map(lambda x: -1 if Q==NN else x))
    assume(abs(Qc - Q) > 1e-3)
    return Hit(npeak,Cluster(Q, xy(x, y), xy(xvar, yvar), nsipm,  Qc=Qc), z, E, xy(x_peak, y_peak))

@composite
def list_of_hits(draw):
    list_of_hits = draw(lists(hit(), min_size=2, max_size=10))
    assume(sum((h.Q > 0 for h in list_of_hits)) >= 1)
    return list_of_hits

@composite
def thresholds(draw, min_value=1, max_value=1):
    th1 = draw (integers(  10   ,  20))
    th2 = draw (integers(  th1+1,  30))
    return th1, th2

@given(list_of_hits())
def test_merge_NN_does_not_modify_input(hits):
    hits_org    = deepcopy(hits)
    before_len  = len(hits)

    merge_NN_hits(hits)

    after_len   = len(hits)

    assert before_len == after_len
    for h1, h2 in zip(hits, hits_org):
        assert_hit_equality(h1, h2)

@given(list_of_hits())
def test_merge_hits_energy_conserved(hits):
    hits_merged = merge_NN_hits(hits)
    assert_almost_equal(sum((h.E for h in hits)), sum((h.E for h in hits_merged)))

def test_merge_NN_hits_exact(TlMC_hits, TlMC_hits_merged):
    for ev, hitc in TlMC_hits.items():
        hits_test   = TlMC_hits_merged[ev].hits
        hits_merged = merge_NN_hits(hitc.hits)
        assert len(hits_test) == len(hits_merged)
        for h1, h2 in zip(hits_test, hits_merged):
            print(h1.Xrms, h1.Yrms, h2.Xrms, h2.Yrms)
            assert_hit_equality(h1, h2)

@given(threshs=thresholds(1, 1))
@settings(deadline=None, max_examples = 1)
@mark.slow
def test_threshold_hits_with_penthesilea(config_tmpdir, Kr_pmaps_run4628_filename, threshs):
    th1, th2     = threshs
    PATH_IN      = Kr_pmaps_run4628_filename
    nrequired    = 1
    conf         = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    PATH_OUT_th1 = os.path.join(config_tmpdir, 'KrDST_4628_th1.h5')
    conf.update(dict(run_number        = 4628,
                     files_in          = PATH_IN,
                     file_out          = PATH_OUT_th1,
                     event_range       = (0, nrequired),
                     slice_reco_params = dict(
                         Qthr          = th1 * units.pes,
                         Qlm           = 0 * units.pes,
                         lm_radius     = 0 * units.mm ,
                         new_lm_radius = 0 * units.mm ,
                         msipm         = 1      )))

    penthesilea (**conf)

    PATH_OUT_th2 = os.path.join(config_tmpdir, 'KrDST_4628_th2.h5')
    conf.update(dict(run_number = 4628,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT_th2,
                     event_range = (0, nrequired),
                     slice_reco_params = dict(
                         Qthr          = th2 * units.pes,
                         Qlm           = 0 * units.pes,
                         lm_radius     = 0 * units.mm ,
                         new_lm_radius = 0 * units.mm ,
                         msipm         = 1      )))

    penthesilea (**conf)

    hits_pent_th1 = hio.load_hits(PATH_OUT_th1)
    hits_pent_th2 = hio.load_hits(PATH_OUT_th2)
    ev_num = 1
    hits_thresh  = threshold_hits (hits_pent_th1[ev_num].hits, th = th2)
    assert len(hits_pent_th2[ev_num].hits)==len(hits_thresh)
    for h1, h2 in zip(hits_pent_th2[ev_num].hits, hits_thresh):
        assert_hit_equality(h1, h2)


@given(list_of_hits(), floats())
def test_threshold_hits_does_not_modify_input(hits, th):
    hits_org    = deepcopy(hits)
    before_len  = len(hits)

    threshold_hits(hits,th)

    after_len   = len(hits)
    assert before_len == after_len
    for h1, h2 in zip(hits, hits_org):
        assert_hit_equality(h1, h2)


@mark.parametrize("on_corrected", (False, True))
@given(hits=list_of_hits(), th=floats())
def test_threshold_hits_energy_conserved(hits, th, on_corrected):
    hits_thresh = threshold_hits(hits, th, on_corrected=on_corrected)
    assert_almost_equal(sum((h.E  for h in hits)), sum((h.E  for h in hits_thresh)))


@mark.parametrize("on_corrected", (False, True))
@given(hits=list_of_hits(), th=floats())
def test_threshold_hits_all_larger_than_th(hits, th, on_corrected):
    hits_thresh  = threshold_hits(hits, th, on_corrected = on_corrected)
    assert (h.Q  > th or h.Q  == NN for h in hits_thresh)
    assert (h.Qc > th or h.Qc == NN for h in hits_thresh)
