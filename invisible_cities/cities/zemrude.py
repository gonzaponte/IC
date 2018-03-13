from operator  import add
from functools import partial

import numpy  as np
import tables as tb

from .. io   .         hist_io import          hist_writer
from .. io   .run_and_event_io import run_and_event_writer
from .. icaro.hst_functions    import shift_to_bin_centers
from .. reco                   import           tbl_functions as tbl
from .. reco                   import calib_sensors_functions as csf
from .. database               import load_db

from .. dataflow import dataflow as fl

from .  components import city
from .  components import WfType
from .  components import print_every
from .  components import sensor_data
from .  components import wf_from_files
from .  components import waveform_binner


@city
def zemrude(files_in, file_out, compression, event_range, print_mod, run_number,
            raw_data_type,
            min_bin, max_bin, bin_width):
    raw_data_type_ = getattr(WfType, raw_data_type.lower())

    bin_edges   = np.arange(min_bin, max_bin, bin_width)
    bin_centres = shift_to_bin_centers(bin_edges)
    nsipm       = sensor_data(files_in[0], raw_data_type_).NSIPM
    shape       = nsipm, len(bin_centres)

    calibrate_with_mode   = fl.map(mode_calibrator  (run_number))
    calibrate_with_median = fl.map(median_calibrator(run_number))
    bin_waveforms         = fl.map(waveform_binner  (bin_edges ))
    sum_histograms        = fl.reduce(add, np.zeros(shape, dtype=np.int))
    accumulate_mode       = sum_histograms()
    accumulate_median     = sum_histograms()

    time_execution = fl.spy_clock()
    event_count    = fl.count()

    with tb.open_file(file_out, 'w', filters=tbl.filters(compression)) as h5out:
        write_event_info    = run_and_event_writer(h5out)
        write_run_and_event = fl.sink(write_event_info, args=("run_number", "event_number", "timestamp"))

        write_hist = partial(hist_writer,
                             h5out,
                             group_name  = 'HIST',
                             n_sensors   = nsipm,
                             bin_centres = bin_centres)

        out = fl.push(
            source = wf_from_files(files_in, raw_data_type_),
            pipe   = fl.pipe(time_execution.spy,
                             fl.slice(*event_range, close_all=True),
                             print_every(print_mod),
                             fl.fork(("sipm", calibrate_with_mode  , bin_waveforms, accumulate_mode    .sink),
                                     ("sipm", calibrate_with_median, bin_waveforms, accumulate_median  .sink),
                                                                                    write_run_and_event      ,
                                                                                    event_count        .sink )),
            result = dict(mode        = accumulate_mode  .future,
                          median      = accumulate_median.future,
                          event_count =       event_count.future,
                          total_time  =    time_execution.future))

        write_hist(table_name = "mode"  )(out.mode  )
        write_hist(table_name = "median")(out.median)

    return out


def mode_calibrator(run_number):
    adc_to_pes = load_db.DataSiPM(run_number).adc_to_pes.values
    def calibrate_with_mode(wfs):
        return csf.sipm_subtract_mode_and_calibrate(wfs, adc_to_pes)
    return calibrate_with_mode


def median_calibrator(run_number):
    adc_to_pes = load_db.DataSiPM(run_number).adc_to_pes.values
    def calibrate_with_median(wfs):
        return csf.sipm_subtract_median_and_calibrate(wfs, adc_to_pes)
    return calibrate_with_median
