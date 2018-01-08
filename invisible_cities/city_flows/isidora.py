from operator import attrgetter

import numpy  as np
import tables as tb


from .. dataflow            import dataflow      as df
from .. reco                import tbl_functions as tbl
from .. database            import load_db
from .. sierpe              import blr

from .. io.           mc_io import      mc_track_writer
from .. io.          rwf_io import           rwf_writer
from .. io.run_and_event_io import run_and_event_writer

from city_components import deconv_pmt
from city_components import event_data_from_files
from city_components import sensor_data

################################################################################

def isidora(input_paths, output_path, run_number=0, n_baseline=30000, compression='ZLIB4'):
    """
    The city of ISIDORA performs a fast processing from raw data
    (pmtrwf and sipmrwf) to BLR wavefunctions.

    """
    sd = sensor_data(input_paths[0])

    h5out = tb.open_file(output_path, "w", filters = tbl.filters(compression))
    rwf_to_cwf = df.map(deconv_pmt(n_baseline))

    RWF = partial(rwf_writer, h5out, group_name='BLR')
    writer_pmt   = RWF(table_name='pmtcwf' , n_sensors=sd.NPMT , waveform_length=sd.PMTWL),
    writer_sipm  = RWF(table_name='sipmrwf', n_sensors=sd.NSIPM, waveform_length=sd.SIPMWL),
    writer_mc         = df.sink(mc_track_writer     (h5out))
    writer_event_info = df.sink(run_and_event_writer(h5out))

    event_count = df.count()

    return df.push(source  = event_data_from_files(input_paths),
                   pipe    = df.fork((df.map(attrgetter('pmt'       )), rwf_to_cwf, writer_pmt       ),
                                     (df.map(attrgetter('sipm'      )),             writer_sipm      ),
                                     (df.map(attrgetter('mc'        )),             writer_mc        ),
                                     (df.map(attrgetter('event_info')),             writer_event_info),
                                     event_count.sink),
                   futures = (event_count.future,))

    return df.push(source  = event_data_from_files(input_paths),
                   pipe    = df.fork((df.pick('pmt'       ), rwf_to_cwf, writer_pmt       ),
                                     (df.pick('sipm'      ),             writer_sipm      ),
                                     (df.pick('mc'        ),             writer_mc        ),
                                     (df.pick('event_info'),             writer_event_info),
                                     event_count.sink),
                   futures = (event_count.future,))
