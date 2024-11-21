from invisible_cities.cities.components import hits_and_kdst_from_files
from invisible_cities.cities.components import city
from invisible_cities.cities.components import print_every



from invisible_cities.dataflow             import dataflow      as fl
from invisible_cities.reco                 import tbl_functions as tbl


from invisible_cities.io.mcinfo_io        import mc_info_writer
from invisible_cities.io.run_and_event_io import run_and_event_writer
from invisible_cities.io.hits_io          import hits_writer
from invisible_cities.io.dst_io           import _store_pandas_as_tables



import numpy  as np
import tables as tb


def kdst_from_df_writer(h5out, compression='ZLIB4', 
                        group_name='DST', 
                        table_name='Events', 
                        descriptive_string='KDST Events', 
                        str_col_length=32):
    """
    For a given open table returns a writer for KDST dataframe info
    """
    def write_kdst(df):
        return _store_pandas_as_tables(h5out=h5out, 
                                       df=df, 
                                       compression=compression, 
                                       group_name=group_name, 
                                       table_name=table_name, 
                                       descriptive_string=descriptive_string, 
                                       str_col_length=str_col_length)
    return  write_kdst

def txt_read(filename):
        return np.loadtxt(filename, dtype=int)


@city
def crashcity(files_in, file_out, compression, event_range, print_mod, run_number,
              global_reco_params = dict(),
              crashcityparam     = dict()):
    
    
    
    good_events = txt_read(crashcityparam['filterfile'])
    
    def filter_events(event_n):
        return event_n  in good_events

    event_filter = fl.filter(filter_events, args="event_number")
    
    
    count_in  = fl.spy_count()
    count_out = fl.spy_count()
    
    
    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        
        write_mc_        = mc_info_writer(h5out) if run_number <= 0 else (lambda *_: None)
        write_mc         = fl.sink(write_mc_ , 
                                   args=("mc", "event_number"))

        write_event_info = fl.sink(run_and_event_writer(h5out), 
                                   args=("run_number", "event_number", "timestamp"))
        write_kdst_table = fl.sink(kdst_from_df_writer(h5out),
                                   args="kdst")
        write_hits       = fl.sink(hits_writer(h5out, 
                                               group_name='HITS', 
                                               table_name='Events'), 
                                   args="hits")
        
        result = fl.push(source = hits_and_kdst_from_files(files_in),
                         pipe   = fl.pipe( 
                             fl.slice(*event_range, close_all=True),
                             print_every(print_mod)                ,
                             count_in.spy                          ,
                             event_filter                          ,
                             count_out.spy                         ,
                             fl.fork(write_kdst_table   ,
                                     write_mc           ,
                                     write_event_info   ,
                                     write_hits          )          ),
                         
                         result = dict(
                             count_in  = count_in .future,
                             count_out = count_out.future)
                         )
        print(f"for {result.count_in} events in we got {result.count_out}") 
        
        return result