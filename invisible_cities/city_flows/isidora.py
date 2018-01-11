from operator  import attrgetter
from functools import partial

import numpy  as np
import tables as tb

from .. reco                import tbl_functions as tbl
from .. database            import load_db
from .. sierpe              import blr
from .. io.          rwf_io import           rwf_writer
from .. io.run_and_event_io import run_and_event_writer

from .. dataflow            import dataflow      as df
from .. dataflow.dataflow   import     sink
from .. dataflow.dataflow   import starsink
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import pick
from .. dataflow.dataflow   import fork

from .  city_components import city
from .  city_components import deconv_pmt
from .  city_components import event_data_from_files
from .  city_components import sensor_data
from .  city_components import make_write_mc


@city
def isidora(files_in, file_out, event_range, run_number, n_baseline,
            raw_data_type, compression, print_mod, verbosity):
    """
    The city of ISIDORA performs a fast processing from raw data
    (pmtrwf and sipmrwf) to BLR wavefunctions.

    """
    sd = sensor_data(files_in[0])

    rwf_to_cwf = df.map(deconv_pmt(n_baseline))

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        RWF = partial(rwf_writer, h5out, group_name='BLR')
        write_pmt   = sink(RWF(table_name='pmtcwf' , n_sensors=sd.NPMT , waveform_length=sd.PMTWL))
        write_sipm  = sink(RWF(table_name='sipmrwf', n_sensors=sd.NSIPM, waveform_length=sd.SIPMWL))
        write_event_info = starsink(run_and_event_writer(h5out))

        write_mc = make_write_mc(h5out)

        event_count = df.count()

        return push(
            source = event_data_from_files(files_in),
            pipe   = pipe(df.slice(*event_range),
             fork((pick('pmt'       ), rwf_to_cwf, write_pmt       ),
                  (pick('sipm'      ),             write_sipm      ),
                  (pick('mc'        ),             write_mc        ),
                  (pick('event_info'),             write_event_info),
                  event_count.sink)),
            result = (event_count.future,))
