import pytest

import os
import numpy  as np
import tables as tb
import pandas as pd

from .. core.configure import configure
from .. core.testing_utils import assert_dataframes_equal
from . paola import paola

@pytest.fixture(scope = 'module')
def paola_file_input_data(ICDATADIR):
    return os.path.join(ICDATADIR, 'hdst_paola_test.h5')


@pytest.fixture(scope = 'module')
def paola_sel_event_file(ICDATADIR):
    return os.path.join(ICDATADIR, 'events_paola.txt')


@pytest.fixture(scope = 'module')
def evts_to_sel(ICDATADIR, paola_sel_event_file):
    return np.loadtxt(paola_sel_event_file, dtype=int)


def test_selection_is_correct(paola_file_input_data, paola_sel_event_file, evts_to_sel, output_tmpdir):

    file_out = os.path.join(output_tmpdir,'hdst_paola_out.h5')
    conf = configure('dummy invisible_cities/config/paola.conf'.split())
    conf.update(dict(files_in = paola_file_input_data,
                     file_out = file_out,
                     sel_event_file = paola_sel_event_file))
    res = paola(**conf)

    with tb.open_file(file_out) as h5out:
        selected_evts = h5out.root.Run.events[:]['evt_number']

        assert np.all(evts_to_sel == selected_evts)


def test_content_is_the_same(paola_file_input_data, paola_sel_event_file, evts_to_sel, output_tmpdir):

    file_out = os.path.join(output_tmpdir,'hdst_paola_out.h5')
    conf = configure('dummy invisible_cities/config/paola.conf'.split())
    conf.update(dict(files_in = paola_file_input_data,
                     file_out = file_out,
                     sel_event_file = paola_sel_event_file))
    res = paola(**conf)

    with tb.open_file(paola_file_input_data) as h5in:

        in_table = getattr(getattr(h5in.root, 'RECO'), 'Events').read()
        in_hits = pd.DataFrame.from_records(in_table)
        sel_in_hits = in_hits[in_hits.event.isin(evts_to_sel)].reset_index(drop=True)

        in_table = getattr(getattr(h5in.root, 'DST'), 'Events').read()
        in_kdsts = pd.DataFrame.from_records(in_table)
        sel_in_kdsts = in_kdsts[in_kdsts.event.isin(evts_to_sel)].reset_index(drop=True)

        in_table = getattr(getattr(h5in.root, 'Run'), 'events').read()
        in_times = pd.DataFrame.from_records(in_table)
        sel_in_times = in_times[in_times.evt_number.isin(evts_to_sel)].reset_index(drop=True)

        with tb.open_file(file_out) as h5out:

            out_table = getattr(getattr(h5out.root, 'RECO'), 'Events').read()
            out_hits = pd.DataFrame.from_records(out_table).reset_index(drop=True)

            out_table = getattr(getattr(h5out.root, 'DST'), 'Events').read()
            out_kdsts = pd.DataFrame.from_records(out_table).reset_index(drop=True)

            out_table = getattr(getattr(h5out.root, 'Run'), 'events').read()
            out_times = pd.DataFrame.from_records(out_table).reset_index(drop=True)

            assert_dataframes_equal(sel_in_hits, out_hits)
            assert_dataframes_equal(sel_in_kdsts, out_kdsts)
            assert_dataframes_equal(sel_in_times, out_times)
