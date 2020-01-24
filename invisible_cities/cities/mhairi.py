import numpy  as np
import tables as tb

from typing import Callable
from typing import     List
from typing import    Tuple

from .. reco                import tbl_functions        as tbl
from .. dataflow            import dataflow             as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import hits_and_kdst_from_files

from .. io.         hits_io  import          hits_writer
from .. io.run_and_event_io  import run_and_event_writer


def select_events(event_file: str) -> Callable:
    good_events = np.loadtxt(event_file, dtype=int)
    def check_event_good(evt_no: int) -> bool:
        return evt_no in good_events
    return check_event_good


@city
def mhairi(files_in   : List[str],
           file_out   :      str ,
           compression:      str ,
           event_range:     Tuple,
           event_file :      str ):
    """
    Simple dataflow which takes hits from a file
    as well as a text file with events of interest
    which is used to filter the events.
    """

    evt_count_in   = fl.spy_count()
    evt_count_filt = fl.spy_count()

    event_filter = fl.filter(select_events(event_file), args="event_number")


    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:

        write_hits       = fl.sink(hits_writer(h5out), args="hits")
        write_event_info = fl.sink(run_and_event_writer(h5out),
                                   args=("run_number"  ,
                                         "event_number",
                                         "timestamp"   ))

        return push(source = hits_and_kdst_from_files(files_in)   ,
                    pipe   = pipe(evt_count_in.spy         ,
                                  event_filter             ,
                                  evt_count_filt.spy       ,
                                  fl.fork(write_hits      ,
                                          write_event_info))      ,
                    result = dict(evt_in  = evt_count_in  .future,
                                  evt_out = evt_count_filt.future))
