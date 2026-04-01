import os
import numpy as np
import tables as tb

from .. dataflow import dataflow as fl

from .  components import city
from .  components import hits_and_kdst_from_files

from .. io.hits_io import hits_writer
from .. io.dst_io  import _store_pandas_as_tables
from .. io.run_and_event_io  import run_and_event_writer


def kdst_from_df_writer(h5out, compression='ZLIB4', group_name='DST', table_name='Events', descriptive_string='KDST Events', str_col_length=32):
    """
    For a given open table returns a writer for KDST dataframe info
    """
    def write_kdst(df):
        return _store_pandas_as_tables(h5out=h5out, df=df, compression=compression, group_name=group_name, table_name=table_name, descriptive_string=descriptive_string, str_col_length=str_col_length)
    return write_kdst

@city
def paola(files_in, file_out, event_range, sel_event_file):

    selected_evts = np.loadtxt(sel_event_file, dtype=int)

    def event_sel(evt):
        return evt in selected_evts
        
    evt_selector = fl.filter(event_sel, args='event_number')

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()
    
    with tb.open_file(file_out, "w") as h5out:
        write_kdst = fl.sink(kdst_from_df_writer(h5out), args='kdst')
        write_hits = fl.sink(hits_writer(h5out), args='hits')
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))

        result = fl.push(source = hits_and_kdst_from_files(files_in),
                    pipe = fl.pipe(#fl.slice(*event_range),
                                   event_count_in.spy,
                                   evt_selector,
                                   event_count_out.spy,
                                   fl.fork(write_kdst,
                                           write_hits,
                                           write_event_info)
                                   ),
                    result = dict (events_in = event_count_in.future,
                                   events_out = event_count_out.future ))

        return result
