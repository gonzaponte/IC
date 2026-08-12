"""
-----------------------------------------------------------------------
                              Clarice
-----------------------------------------------------------------------
"""
import tables as tb
import numpy  as np

from .. dataflow   import dataflow                 as fl
from .  components import city
from .  components import hits_and_kdst_from_files
from .  esmeralda  import kdst_from_df_writer



@city
def clarice(files_in, file_out, file_events, event_range, run_number):

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out,    "w") as h5out  ,\
            open     (file_events, "r") as file_ev:

        sel_events       = np.array([int(ev) for ev in file_ev.read().split('\n')])
        filter_events    = fl.filter(lambda x: x in sel_events, args="event_number")
        write_kdst_table = fl.sink(kdst_from_df_writer(h5out) , args="kdst")

        return fl.push(source = hits_and_kdst_from_files(files_in),
                    pipe   = fl.pipe(event_count_in .spy,
                                     filter_events      ,
                                     event_count_out.spy,
                                     write_kdst_table  ),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future))

    print(f"{result.events_in} input numbers")
    print(f"{result.events_out} output numbers after slicing")
