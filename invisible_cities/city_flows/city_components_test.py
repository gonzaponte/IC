import os

from operator    import attrgetter
from operator    import itemgetter
from operator    import add

from functools   import reduce
from collections import namedtuple


import numpy as np

import tables as tb

from pytest import fixture

from .. dataflow import dataflow      as df
from .. reco     import tbl_functions as tbl
from .. database import load_db




def count_events_in_file(path):
    # Used in test_count_events_in_multiple_files to verify that the
    # stream approach can count events correctly.
    with tb.open_file(path, 'r') as h5in:
        events_info = h5in.root.Run.events
        return len(events_info)

def events_numbers_in_file(path):
    # Used in test_count_events_in_multiple_files to verify that the
    # stream approach can count events correctly.
    with tb.open_file(path, 'r') as h5in:
        events_info = h5in.root.Run.events
        return list(map(itemgetter(0),events_info[:]))


def get_mc_tracks(h5in, *, monte_carlo=False):
    if monte_carlo: return tbl.get_mc_tracks(h5in)
    else:           return None


def events_from_files(paths, *, waveform_type):
    for path in paths:
        print("Opening", path, end="... ")
        with tb.open_file(path, "r") as h5in:

            events_info = h5in.root.Run.events
            # mc_tracks   = get_mc_tracks(h5in, monte_carlo=True)

            # # TODO: replace RWF and MCRD strings with symbols
            # if   waveform_type == 'RWF' : fn = tbl.get_rwf_vectors
            # elif waveform_type == 'MCRD': fn = tbl.get_rd_vectors
            # else: raise UnknownRWF
            # NEVT, pmt, sipm = fn(h5in)[:3]

            for event_info in events_info:
                yield event_info


def dummy_events_from_files(paths, *, waveform_type):
    # For illustrative purposes only: A simple source which produces
    # one dummy object for each event contained in the files specified
    # in paths.

    for path in paths:
        with tb.open_file(path, "r") as h5in:
            for event_info in h5in.root.Run.events:
                yield "these will be counted so we don't care what's inside"


def test_count_events_in_multiple_files(input_files):
    # Simply count the dummy objects produced by the source: this
    # amounts to counting the events in the input_files.

    count = df.count()
    df.push(source = dummy_events_from_files(input_files, waveform_type='RWF'),
            pipe   = count.sink)
    total_N_events = sum(map(count_events_in_file, input_files))
    assert count.future.result() == total_N_events


def event_infos_from_files(paths, *, waveform_type):
    # A slightly more interesting example: A simple source which
    # produces the event info for every event contained in the files
    # specified in paths.

    for path in paths:
        with tb.open_file(path, "r") as h5in:
            for event_info in h5in.root.Run.events:
                yield event_info


def test_event_numbers_in_multiple_files(input_files):
    count = df.count()
    spied_event_numbers = []

    def collect_event_numbers(event_info):
        spied_event_numbers.append(event_info['evt_number'])

    df.push(source = events_from_files(input_files, waveform_type='RWF'),
            pipe   = df.pipe(df.spy(collect_event_numbers), count.sink))

    all_event_numbers = reduce(add, map(events_numbers_in_file, input_files))
    assert spied_event_numbers == all_event_numbers




RawWaveforms = namedtuple('RawWaveforms', 'rwf blr')

def rwfs_from_files(paths):
    for path in paths:
        with tb.open_file(path, "r") as h5in:
            _, pmtrwfs, _, blrs = tbl.get_rwf_vectors(h5in)
            # yield from zip(pmtrwfs, blrs)
            for pmtrwf, blr in zip(pmtrwfs, blrs):
                yield RawWaveforms(rwf=pmtrwf, blr=blr)


def test_rwfs_in_multiple_files(input_files):

    DataPMT    = load_db.DataPMT(run_number = 0)
    pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist()
    coeff_c    = DataPMT.coeff_c  .values.astype(np.double)
    coeff_blr  = DataPMT.coeff_blr.values.astype(np.double)

    rwf_to_cwf = df.map(deconv_pmt(coeff_c               = coeff_c,
                                   coeff_blr             = coeff_blr,
                                   pmt_active            = pmt_active,
                                   n_baseline            = 28000,
                                   thr_trigger           =     5,
                                   acum_discharge_length =  5000))

    compare = df.join2(lambda CWF, BLR: np.mean([np.abs(pmt - cwf) for pmt, cwf in zip(BLR, CWF)]) < 1)

    def assert_(x): assert x
    assert_pipe = compare(df.sink(assert_))

    rwf_to_cwf_branch = df.pipe(df.map(attrgetter('rwf')), rwf_to_cwf, assert_pipe)
    blr_branch        = df.pipe(df.map(attrgetter('blr')),             assert_pipe)

    df.push(source = rwfs_from_files(input_files),
            pipe   = df.fork(rwf_to_cwf_branch, blr_branch))




@fixture
def input_files(ICDIR):
    TEST_DATA_DIR = os.path.join(ICDIR, 'database/test_data/')
    return (os.path.join(TEST_DATA_DIR, 'electrons_40keV_z250_RWF.h5'),
            os.path.join(TEST_DATA_DIR, 'electrons_40keV_z250_RWF.h5'),
            # os.path.join(TEST_DATA_DIR, 'electrons_1250keV_z250_RWF.h5'),
    )
