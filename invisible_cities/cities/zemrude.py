import sys

from operator import add
from argparse import Namespace

from functools import partial

import numpy  as np
import tables as tb

from .. io.         hist_io import          hist_writer
from .. io.run_and_event_io import run_and_event_writer
from .. icaro.hst_functions import shift_to_bin_centers
from .. reco                import         tbl_functions   as tbl
from .. reco                import calib_sensors_functions as csf

from .. dataflow import dataflow as fl

from .  components import city
from .  components import WfType
from .  components import print_every
from .  components import sensor_data
from .  components import wf_from_files
from .  components import calibrate_with_mau
from .  components import calibrate_with_mean


@city
def zemrude(files_in, file_out, compression, event_range, print_mod, run_number,
            raw_data_type,
            n_mau_sipm,
            min_bin, max_bin, bin_width):

    raw_data_type_ = getattr(WfType, raw_data_type.lower())

    bin_edges   = np.arange(min_bin, max_bin, bin_width)
    bin_centres = shift_to_bin_centers(bin_edges)
    sd          = sensor_data(files_in[0], raw_data_type_)
    shape       = sd.NSIPM, len(bin_edges) - 1

    def waveform_to_histogram(waveform):
        return np.histogram(waveform, bin_edges)[0]

    def waveform_array_to_histogram_array(waveforms):
        return np.apply_along_axis(waveform_to_histogram, 1, waveforms)

    def empty_waveform_array():
        return np.zeros(shape, dtype=np.int)

    make_mau_stream  = fl.map(calibrate_with_mau (run_number, n_mau_sipm) , args='sipm', out= 'mau_stream')
    make_mean_stream = fl.map(calibrate_with_mean(run_number)             , args='sipm', out='mean_stream')

    rebin = fl.map(waveform_array_to_histogram_array)

    sum_histograms = fl.reduce(add, empty_waveform_array())
    accumulate_mau_sipms  = sum_histograms()
    accumulate_mean_sipms = sum_histograms()

    time_execution = fl.spy_clock()
    event_count    = fl.count()

    with tb.open_file(file_out, 'w', filters=tbl.filters(compression)) as h5out:

        write_event_info    = run_and_event_writer(h5out)
        write_run_and_event = fl.sink(write_event_info, args=("run_number", "event_number", "timestamp"))

        write_hist = partial(hist_writer,
                             h5out,
                             group_name  = 'HIST',
                             n_sensors   = sd.NSIPM,
                             bin_centres = bin_centres)

        out = fl.push(
            source = wf_from_files(files_in, raw_data_type_),
            pipe   = fl.pipe(time_execution.spy,
                             fl.slice(*event_range, close_all = True),
                             print_every(print_mod),
                             make_mau_stream,
                             make_mean_stream,
                             fl.fork(( 'mau_stream', rebin, accumulate_mau_sipms .sink),
                                     ('mean_stream', rebin, accumulate_mean_sipms.sink),
                                                            write_run_and_event,
                                                            event_count          .sink)),
            result = dict(mau         = accumulate_mau_sipms .future,
                          mean        = accumulate_mean_sipms.future,
                          event_count =           event_count.future,
                          total_time  =        time_execution.future))

        write_hist(table_name  = 'sipmMAU')(out.mau )
        write_hist(table_name  = 'sipm'   )(out.mean)

    return out


def accumulate_and_write_sipms(h5out, shape):
    sipm_hist = np.zeros(shape, dtype=np.int)
    def accumulate_and_write_sipms(wfm):
        sipm_hist += bin_waveforms(wfm)



# class Zemrude(CalibratedCity):
#     """
#     Generates binned spectra of sipm rwf - mean
#     and (rwf - mean)-mau
#     Reads: Raw data waveforms.
#     Produces: Histograms of pedestal subtracted waveforms.
#     """

#     def __init__(self, **kwds):
#         """
#         zemrude Init:
#         1. inits base city
#         2. inits counters
#         3. gets sensor parameters
#         """

#         self.cnt.init(n_events_tot = 0)
#         self.sp = self.get_sensor_params(self.input_files[0])


#     def event_loop(self, NEVT, dataVectors):
#         """
#         actions:
#         1. loops over all the events in each file.
#         2. write event/run to file
#         3. write histogram info to file (to reduce memory usage)
#         """
#         write       = self.writers
#         sipmrwf     = dataVectors.sipm
#         events_info = dataVectors.events

#         ## Where we'll be saving the binned info for each channel
#         shape    = sipmrwf.shape[1], len(self.histbins) - 1
#         bsipmzs  = np.zeros(shape, dtype=np.int)
#         bsipmmzs = np.zeros(shape, dtype=np.int)

#         for evt in range(NEVT):
#             self.conditional_print(evt, self.cnt.n_events_tot)

#             what_next = self.event_range_step()
#             if what_next is EventLoop.skip_this_event: continue
#             if what_next is EventLoop.terminate_loop : break
#             self.cnt.n_events_tot += 1

#             ## Zeroed sipm waveforms in pe
#             sipmzs   = self.calibrate_with_mean(sipmrwf[evt])
#             bsipmzs += self.bin_waveforms(sipmzs)

#             ## Difference from the MAU
#             sipmmzs   = self.calibrate_with_mau(sipmrwf[evt])
#             bsipmmzs += self.bin_waveforms(sipmmzs)

#             # write stuff
#             event, timestamp = self.event_and_timestamp(evt, events_info)
#             write.run_and_event(self.run_number, event, timestamp)

#         write.sipm (bsipmzs)
#         write.mausi(bsipmmzs)


#     def bin_waveforms(self, waveforms):
#         """
#         Bins the current event data and adds it
#         to the file level bin array
#         """
#         bin_waveform = lambda x: np.histogram(x, self.histbins)[0]
#         return np.apply_along_axis(bin_waveform, 1, waveforms)


#     def get_writers(self, h5out):
#         bin_centres = shift_to_bin_centers(self.histbins)
#         HIST        = partial(hist_writer,
#                               h5out,
#                               group_name  = 'HIST',
#                               n_sensors   = self.sp.NSIPM,
#                               n_bins      = len(bin_centres),
#                               bin_centres = bin_centres)

#         writers = Namespace(
#             run_and_event = run_and_event_writer(h5out),
#             sipm          = HIST(table_name  = 'sipm'),
#             mausi         = HIST(table_name  = 'sipmMAU'))

#         return writers

#     def write_parameters(self, h5out):
#         pass

#     def display_IO_info(self):
#         super().display_IO_info()
#         print(self.sp)

#     def _copy_sensor_table(self, h5in):
#         # Copy sensor table if exists (needed for GATE)
#         if 'Sensors' not in h5in.root: return

#         group = self.output_file.create_group(self.output_file.root,
#                                               "Sensors")
#         datapmt  = h5in.root.Sensors.DataPMT
#         datasipm = h5in.root.Sensors.DataSiPM
#         datapmt .copy(newparent=group)
#         datasipm.copy(newparent=group)
