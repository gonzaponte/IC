import os

import pandas as pd
import tables as tb

from ..core.testing_utils import assert_dataframes_close

from . hits_io import hits_writer


def test_hits_writer(Th228_hits, config_tmpdir):
    output_file = os.path.join(config_tmpdir, "hits_writer.h5")

    hits_true = pd.read_hdf(Th228_hits, "/RECO/Events")

    with tb.open_file(output_file, 'w') as h5out:
        write = hits_writer(h5out, "RECO", "Events")
        write(hits_true)

    hits_got = pd.read_hdf(output_file, "/RECO/Events")
    assert_dataframes_close(hits_got, hits_true)
