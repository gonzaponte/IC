from operator import attrgetter

import tables as tb

from .. reco                  import tbl_functions        as tbl
from .. io  .         hits_io import          hits_writer
from .. io  .           mc_io import      mc_track_writer
from .. io  .run_and_event_io import run_and_event_writer

from .. dataflow            import dataflow as df
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every
from .  components import peak_classifier
from .  components import compute_xy_position
from .  components import pmap_from_files
from .  components import hit_builder


@city
def penthesilea(files_in, file_out, compression, event_range, print_mod, run_number,
                drift_v, rebin,
                s1_nmin, s1_nmax, s1_emin, s1_emax, s1_wmin, s1_wmax, s1_hmin, s1_hmax, s1_ethr,
                s2_nmin, s2_nmax, s2_emin, s2_emax, s2_wmin, s2_wmax, s2_hmin, s2_hmax, s2_ethr, s2_nsipmmin, s2_nsipmmax,
                qthr, qlm, lm_radius, new_lm_radius, msipm):

    classify_peaks = df.map(peak_classifier(**locals()),
                            args = "pmap",
                            out  = "selector_output")

    empty_s1       = df.count_filter(attrgetter("s1s"   ), args="pmap"           )
    empty_s2       = df.count_filter(attrgetter("s2s"   ), args="pmap"           )
    pmap_select    = df.count_filter(attrgetter("passed"), args="selector_output")


    reco_algo     = compute_xy_position(qthr, qlm, lm_radius, new_lm_radius, msipm)
    build_hits    = df.map(hit_builder(run_number, drift_v, reco_algo, rebin),
                           args = ("pmap", "selector_output", "event_number", "timestamp"),
                           out  = "hits"                                                  )

    time_execution  = df.spy_clock()
    event_count_in  = df.spy_count()
    event_count_out = df.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_mc_        = mc_track_writer(h5out) if run_number <= 0 else (lambda *_: None)

        write_mc         = df.sink(write_mc_                  , args=(        "mc", "event_number"             ))
        write_event_info = df.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_hits       = df.sink(         hits_writer(h5out), args="hits")

        return push(source = pmap_from_files(files_in),
                    pipe   = pipe(time_execution       .spy             ,
                                  df.slice(*event_range, close_all=True),
                                  print_every(print_mod)                ,
                                  event_count_in       .spy             ,
                                  empty_s1             .filter          ,
                                  empty_s2             .filter          ,
                                  classify_peaks                        ,
                                  pmap_select          .filter          ,
                                  event_count_out      .spy             ,
                                  build_hits                            ,
                                  df.fork(write_hits                    ,
                                          write_mc                      ,
                                          write_event_info              )),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future,
                                  empty_s1   = empty_s1       .future,
                                  empty_s2   = empty_s2       .future,
                                  selection  = pmap_select    .future,
                                  total_time = time_execution .future))
