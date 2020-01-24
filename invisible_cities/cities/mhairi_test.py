import os

import numpy  as np
import tables as tb

from pytest import fixture

from .  mhairi         import    mhairi
from .. core.configure import configure


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
        assert 'RECO'   in h5out.root
        assert 'Run'    in h5out.root
        assert 'Events' in h5out.root.RECO

        evt_gen = (row[0] for row in h5out.root.RECO.Events[:])
        evt_nos = np.unique(np.fromiter(evt_gen, int))

        assert len(evt_nos) == len(accepted_events)
        assert np.all(evt_nos in accepted_events)
