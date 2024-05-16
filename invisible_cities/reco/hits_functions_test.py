import numpy as np
import pandas as pd

from pytest        import mark
from numpy.testing import assert_almost_equal

from   .. core.testing_utils   import assert_dataframes_close
from   .. types.ic_types       import NN
from   .  hits_functions       import make_nn_hit
from   .  hits_functions       import merge_NN_hits
from   .  hits_functions       import threshold_hits
from hypothesis                import given
from hypothesis                import settings
from hypothesis                import assume
from hypothesis.strategies     import lists
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers
from hypothesis.strategies     import composite
from hypothesis.strategies     import one_of
from hypothesis.strategies     import just

qs         = floats(0, 100)
q_or_nns   = one_of(qs, just(NN))
thresholds = one_of(floats(90, 110), floats(0, 10))

@composite
def hit(draw):
    x      = draw(floats  (   1,   5))
    y      = draw(floats  ( -10,  10))
    z      = draw(floats  (  50, 100))
    e      = draw(floats  (  50, 100))
    ec     = draw(floats  (  50, 100))
    q      = draw(q_or_nns           )
    qc     = draw(qs                 ) if q != NN else NN
    peak_x = draw(floats  (   1,   5))
    peak_y = draw(floats  ( -10,  10))
    if q != NN:
        assume(abs(qc - q) > 1e-3)

    hit = dict( npeak = 0
              , Xpeak = peak_x
              , Ypeak = peak_y
              , X     = x
              , Y     = y
              , Z     = z
              , Q     = q
              , Qc    = qc
              , E     = e
              , Ec    = ec)
    return pd.DataFrame(hit, index=[0])

@composite
def hits(draw):
    hits = draw(lists(hit(), min_size=2, max_size=10))
    hits = pd.concat(hits, ignore_index=True)
    assume(hits.Q [hits.Q  != NN].sum() >= 1)
    assume(hits.Qc[hits.Qc != NN].sum() >= 1)
    return hits


@composite
def threshold_pairs(draw, min_value=1, max_value=1):
    th1 = draw (integers(  10   ,  20))
    th2 = draw (integers(  th1+1,  30))
    return th1, th2


@mark.parametrize("ec", (None, 0))
def test_make_nn_hit_only_adds_corrected_fields_when_ec_is_not_None(ec):
    hit = make_nn_hit(0, 0, 0, 0, 0, ec)
    columns_exist = ("Ec" in hit.columns) and ("Qc" in hit.columns)
    assert (ec is None) ^ columns_exist


@mark.parametrize("ec", (None, 0))
def test_make_nn_hit_has_nn_fields(ec):
    hit = make_nn_hit(0, 0, 0, 0, 0, ec).iloc[0]
    assert hit.X == NN
    assert hit.Y == NN
    assert hit.Q == NN
    if ec is not None:
        assert hit.Qc == NN


@given(hits())
def test_merge_NN_does_not_modify_input(hits):
    hits_org   = hits.copy()
    before_len = len(hits)

    merge_NN_hits(hits)

    after_len   = len(hits)

    assert before_len == after_len
    assert_dataframes_close(hits, hits_org)


@given(hits())
def test_merge_hits_energy_conserved(hits):
    hits_merged = merge_NN_hits(hits)
    assert_almost_equal(hits.E .sum(), hits_merged.E .sum())
    assert_almost_equal(hits.Ec.sum(), hits_merged.Ec.sum())


@given(hits())
def test_merge_nn_hits_does_not_leave_nn_hits(hits):
    hits_merged = merge_NN_hits(hits)
    assert not np.isclose(hits_merged.Q, NN).any()


@given(hits(), thresholds)
def test_threshold_hits_does_not_modify_input(hits, th):
    hits_org   = hits.copy()
    before_len = len(hits)

    threshold_hits(hits, th)

    after_len   = len(hits)

    assert before_len == after_len
    assert_dataframes_close(hits, hits_org)


@mark.parametrize("on_corrected", (False, True))
@given(hits=hits(), th=thresholds)
def test_threshold_hits_energy_conserved(hits, th, on_corrected):
    hits_thresh = threshold_hits(hits, th, on_corrected=on_corrected)

    assert_almost_equal(hits_thresh.E .sum(), hits.E .sum())
    assert_almost_equal(hits_thresh.Ec.sum(), hits.Ec.sum())


@mark.parametrize("on_corrected", (False, True))
@given(hits=hits(), th=thresholds)
def test_threshold_hits_all_larger_than_th(hits, th, on_corrected):
    hits_thresh = threshold_hits(hits, th, on_corrected = on_corrected)

    column = "Qc" if on_corrected else "Q"
    qs     = hits_thresh[column]
    assert np.all( (qs > th) | np.isclose(qs, NN) )
