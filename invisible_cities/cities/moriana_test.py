import os

import tables as tb
import numpy  as np

from pytest import mark

from .  moriana        import moriana
from .. core.configure import configure


@mark.slow
def test_moriana_electrons_40keV(config_tmpdir, ICDATADIR):
    PATH_IN  = os.path.join(ICDATADIR    , 'electrons_40keV_z250_RWF.h5' )
    PATH_OUT = os.path.join(config_tmpdir, 'electrons_40keV_z250_HIST.h5')

    nrequired  = 2

    conf = configure('dummy invisible_cities/config/liquid_moriana.conf'.split())
    conf.update(dict(run_number   = 0,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range  = (0, nrequired)))

    out = moriana(**conf)

    if nrequired > 0:
        assert nrequired == out.event_count

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            nrow = 0

            # check events numbers & timestamps
            evts_in  = h5in .root.Run.events[:out.event_count]
            evts_out = h5out.root.Run.events[:out.event_count]
            np.testing.assert_array_equal(evts_in, evts_out)
