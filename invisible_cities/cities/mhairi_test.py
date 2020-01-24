import os

import numpy  as np
import tables as tb

from pytest import fixture

from .  mhairi             import                 mhairi
from .. core.configure     import              configure
from .. core.testing_utils import assert_tables_equality


@fixture
def accepted_events():
    return np.loadtxt('invisible_cities/database/test_data/events.txt',
                      dtype=int)


def test_mhairi_events_correct(config_tmpdir, accepted_events):

    PATH_OUT = os.path.join(config_tmpdir, "test_mhairi.h5")
    conf     = configure('dummy invisible_cities/config/mhairi.conf'.split())

    conf.update(dict(file_out = PATH_OUT))

    cnt = mhairi(**conf)

    with tb.open_file(PATH_OUT) as h5out:
        assert 'RECO'    in h5out.root
        assert 'Run'     in h5out.root
        assert 'Events'  in h5out.root.RECO
        assert 'events'  in h5out.root.Run
        assert 'runInfo' in h5out.root.Run

        evt_gen = (row[0] for row in h5out.root.RECO.Events[:])
        evt_nos = np.unique(np.fromiter(evt_gen, int))

        assert len(evt_nos) == len(accepted_events)
        assert np.all(evt_nos in accepted_events)


def test_mhairi_output_equal(ICDATADIR, config_tmpdir, accepted_events):

    PATH_IN  = os.path.join(ICDATADIR, "hdst.h5")
    PATH_OUT = os.path.join(config_tmpdir, "test_mhairi.h5")
    conf     = configure('dummy invisible_cities/config/mhairi.conf'.split())

    conf.update(dict(file_out = PATH_OUT))

    cnt = mhairi(**conf)

    evt_tup = tuple(accepted_events)

    with tb.open_file(PATH_IN)  as h5in,\
         tb.open_file(PATH_OUT) as h5out:

        inp_tab        = h5in .root.RECO.Events
        read_condition = '(event == {}) | (event == {})'.format(*evt_tup)
        filt_tab       = inp_tab.read_where(read_condition)
        out_tab = h5out.root.RECO.Events

        assert_tables_equality(out_tab, filt_tab)
