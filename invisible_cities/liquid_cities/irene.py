import tables as tb

from .. types.ic_types         import minmax
from .. database               import load_db

from .. reco                import tbl_functions   as tbl
from .. reco                import  peak_functions as pkf
from .. io.        pmaps_io import          pmap_writer
from .. io.           mc_io import      mc_track_writer
from .. io.run_and_event_io import run_and_event_writer

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import sink

from .  components import city
from .  components import print_every
from .  components import deconv_pmt
from .  components import calibrate_pmts
from .  components import calibrate_sipms
from .  components import zero_suppress_wfs
from .  components import WfType
from .  components import   wf_from_files
from .  components import check_nonempty_indices


@city
def irene(files_in, file_out, compression, event_range, print_mod, run_number,
          n_baseline, raw_data_type,
          n_mau, n_mau_sipm, thr_mau, thr_sipm, thr_sipm_type,
          s1_lmin, s1_lmax, s1_tmin, s1_tmax, s1_rebin_stride, s1_stride, thr_csum_s1,
          s2_lmin, s2_lmax, s2_tmin, s2_tmax, s2_rebin_stride, s2_stride, thr_csum_s2, thr_sipm_s2):

    #### Define data transformations

    # Raw WaveForm to Corrected WaveForm
    rwf_to_cwf       = fl.map(deconv_pmt(n_baseline, run_number),
                              args = "pmt",
                              out  = "cwf")

    # Corrected WaveForm to Calibrated Corrected WaveForm
    cwf_to_ccwf      = fl.map(calibrate_pmts(n_mau, thr_mau, run_number),
                              args = "cwf",
                              out  = ("ccwfs", "ccwfs_mau", "cwf_sum", "cwf_sum_mau"))

    # Find where waveform is above threshold
    zero_suppress    = fl.map(zero_suppress_wfs(thr_csum_s1, thr_csum_s2),
                              args = ("cwf_sum", "cwf_sum_mau"),
                              out  = ("s1_indices", "s2_indices", "s2_energies"))

    # Remove baseline and calibrate SiPMs
    sipm_rwf_to_cal  = fl.map(calibrate_sipms(thr_sipm, n_mau_sipm, run_number),
                              item = "sipm")

    # Build the PMap
    compute_pmap     = fl.map(build_pmap(s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                                         s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2,
                                         run_number),
                              args = ("ccwfs", "s1_indices", "s2_indices", "sipm"),
                              out  = "pmap")

    ### Define data filters

    # Filter events with zero peaks
    empty_indices_s1 = fl.count_filter(check_nonempty_indices,
                                       args = "s1_indices")
    # Filter events with zero peaks
    empty_indices_s2 = fl.count_filter(check_nonempty_indices,
                                       args = "s2_indices")

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info_ = run_and_event_writer(h5out)
        write_mc_         = mc_track_writer(h5out) if run_number <= 0 else (lambda *_: None)
        write_pmap_       = pmap_writer(h5out, compression=compression)

        # ... and make them sinks
        write_event_info = sink(write_event_info_, args=("run_number", "event_number", "timestamp"))
        write_mc         = sink(write_mc_        , args=(        "mc", "event_number"             ))
        write_pmap       = sink(write_pmap_      , args=(      "pmap", "event_number"             ))

        return push(source = wf_from_files(files_in, WfType.rwf),
                    pipe   = pipe(
                                fl.slice(*event_range, close_all = True),
                                print_every(print_mod),
                                event_count_in.spy,
                                rwf_to_cwf,
                                cwf_to_ccwf,
                                zero_suppress,
                                empty_indices_s1.filter,
                                empty_indices_s2.filter,
                                sipm_rwf_to_cal,
                                compute_pmap,
                                event_count_out.spy,
                                fl.fork(write_pmap,
                                        write_mc,
                                        write_event_info)),
                    result = dict(events_in  = event_count_in  .future,
                                  events_out = event_count_out .future,
                                  empty_s1   = empty_indices_s1.future,
                                  empty_s2   = empty_indices_s2.future))



def build_pmap(s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
               s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2,
               run_number):
    # TODO: fill dicts
    s1_params = dict(time  = minmax(min = s1_tmin,
                                    max = s1_tmax),
                    length = minmax(min = s1_lmin,
                                    max = s1_lmax),
                    stride       = s1_stride,
                    rebin_stride = s1_rebin_stride)
    s2_params = dict(time  = minmax(min = s2_tmin,
                                    max = s2_tmax),
                    length = minmax(min = s2_lmin,
                                    max = s2_lmax),
                    stride        = s2_stride,
                    rebin_stride  = s2_rebin_stride)

    DataPMT   = load_db.DataPMT(run_number)
    pmt_ids   = DataPMT.SensorID[DataPMT.Active.astype(bool)].values

    def build_pmap(ccwf, s1_indx, s2_indx, sipmzs): # -> PMap
        return pkf.get_pmap(ccwf, s1_indx, s2_indx, sipmzs,
                            s1_params, s2_params, thr_sipm_s2, pmt_ids)

    return build_pmap