import os
import tables as tb
import numpy  as np
import pandas as pd

from invisible_cities.cities.components import hits_and_kdst_from_files
from invisible_cities.cities.components import print_every
from invisible_cities.cities.components import city
from invisible_cities.cities.esmeralda  import kdst_from_df_writer

from invisible_cities. io . hits_io          import          hits_writer
from invisible_cities. io . kdst_io          import            kr_writer
from invisible_cities. io . run_and_event_io import run_and_event_writer

from invisible_cities.reco                   import        tbl_functions as tbl

from invisible_cities.dataflow          import dataflow as fl

@city
def selection_city(files_in, file_out, compression, txt_file, event_range, print_mod, run_number):


    def events_to_select(file_sel):
        with open(file_sel, "r") as file_s:
            events = []
            for index, line in enumerate(file_s):
                event = int(float(line))
                events.append(event)
        return events

    def selecting_events(events):
        def selection(x):
            return x.event in events
        return selection

    count_input = fl.spy_count()
    count_sel   = fl.spy_count()

    events_list = events_to_select(txt_file)
    filtering_events  = fl.filter(selecting_events(events_list), args=("hits"))

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        write_event_info      = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_hits            = fl.sink(         hits_writer(h5out), args="hits")
        write_kdst_table      = fl.sink( kdst_from_df_writer(h5out), args="kdst"               )

        result= fl.push(source = hits_and_kdst_from_files(files_in),
                    pipe   = fl.pipe(
                             fl.slice(*event_range, close_all=True),
                             print_every(print_mod)                ,
                             count_input           .spy            ,
                             filtering_events                      ,
                             count_sel             .spy            ,
                             fl.fork(write_hits                    ,
                                     write_kdst_table              ,
                                     write_event_info)),
                    result = dict(events_in  = count_input.future  ,
                                  events_out = count_sel.future   ))


        print("Number of input events: {0}".format(result.events_in))
        print("Number of events after selection: {0}".format(result.events_out))


        return result
