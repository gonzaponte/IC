"""
-----------------------------------------------------------------------
                                 Hypathia
-----------------------------------------------------------------------

From ancient Greek ‘Υπατια: highest, supreme.

This city reads true waveforms from detsim and compute pmaps from them
without simulating the electronics. This includes:
    - Rebin 1-ns waveforms to 25-ns waveforms to match those produced
      by the detector.
    - Produce a PMT-summed waveform.
    - Apply a threshold to the PMT-summed waveform.
    - Find pulses in the PMT-summed waveform.
    - Match the time window of the PMT pulse with those in the SiPMs.
    - Build the PMap object.
"""
import numpy  as np
import tables as tb

from functools import partial

from .. reco                  import sensor_functions     as sf
from .. reco                  import tbl_functions        as tbl
from .. reco                  import peak_functions       as pkf
from .. core. random_sampling import NoiseSampler         as SiPMsNoiseSampler
from .. core                  import system_of_units      as units
from .. io  .run_and_event_io import run_and_event_writer
from .. io  .      trigger_io import       trigger_writer

from liquidata import pipe
from liquidata import Slice
from liquidata import get
from liquidata import put
from liquidata import out
from liquidata import use
from liquidata import star

from .  components import city
from .  components import counter
from .  components import print_every
from .  components import copy_mc_info
from .  components import zero_suppress_wfs
from .  components import WfType
from .  components import sensor_data
from .  components import wf_from_files
from .  components import get_number_of_active_pmts
from .  components import compute_and_write_pmaps
from .  components import sipm_response_simulator
from .  components import calibrate_sipms


@city
def hypathia(files_in, file_out, compression, event_range, print_mod, detector_db, run_number,
             sipm_noise_cut, filter_padding, thr_sipm, thr_sipm_type, pmt_wfs_rebin, pmt_pe_rms,
             s1_lmin, s1_lmax, s1_tmin, s1_tmax, s1_rebin_stride, s1_stride, thr_csum_s1,
             s2_lmin, s2_lmax, s2_tmin, s2_tmax, s2_rebin_stride, s2_stride, thr_csum_s2, thr_sipm_s2,
             pmt_samp_wid=25*units.ns, sipm_samp_wid=1*units.mus):
    if   thr_sipm_type.lower() == "common":
        # In this case, the threshold is a value in pes
        sipm_thr = thr_sipm

    elif thr_sipm_type.lower() == "individual":
        # In this case, the threshold is a percentual value
        noise_sampler = SiPMsNoiseSampler(detector_db, run_number)
        sipm_thr      = noise_sampler.compute_thresholds(thr_sipm)

    else:
        raise ValueError(f"Unrecognized thr type: {thr_sipm_type}. "
                          "Only valid options are 'common' and 'individual'")

    #### Define data transformations
    sd = sensor_data(files_in[0], WfType.mcrd)

    # Raw WaveForm to Corrected WaveForm
    mcrd_to_rwf      = rebin_pmts(pmt_wfs_rebin)
    # Add single pe fluctuation to pmts
    simulate_pmt     = partial(sf.charge_fluctuation, single_pe_rms=pmt_pe_rms)

    # Find where waveform is above threshold
    zero_suppress    = zero_suppress_wfs(thr_csum_s1, thr_csum_s2)
    # SiPMs simulation
    simulate_sipm_response  = sipm_response_simulator(detector_db, run_number,
                                                      sd.SIPMWL, sipm_noise_cut,
                                                      filter_padding)
    # SiPMs calibration
    sipm_rwf_to_cal  = calibrate_sipms(detector_db, run_number, sipm_thr)

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        # Define writers...
        write_event_info   = run_and_event_writer(h5out)
        write_trigger_info = trigger_writer      (h5out, get_number_of_active_pmts(detector_db, run_number))

        compute_pmaps = compute_and_write_pmaps(
                            detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                            s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                            s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2,
                            h5out, compression, None)

        hypathia = pipe(Slice(*event_range, close_all=True),
                        print_every(print_mod),
                        [out.events_in(counter, 0)],
                        get.pmt   * (mcrd_to_rwf, simulate_pmt) >> put.ccwfs,
                        get.ccwfs * pmts_sum >> put.pmt, 
                        get.pmt.pmt * zero_suppress >> put.s1_indices.s2_indices.s2_energies,
                        get.sipm * (simulate_sipm_response, np.round, use(np.ndarray.astype, np.int16), sipm_rwf_to_cal) >> put.sipm,
                        *compute_pmaps,
                        [out.events_out(counter, 0)],
                        [get.event_number, out.evtnum_list],
                        [get.run_number.event_number.timestamp, star(write_event_info)],
                        [get.trigger_type.trigger_channels    , star(write_trigger_info)])

        result = hypathia(wf_from_files(files_in, WfType.mcrd))
        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)


def rebin_pmts(rebin_stride):
    def rebin_pmts(rwf):
        rebinned_wfs = rwf
        if rebin_stride > 1:
            # dummy data for times and widths
            times     = np.zeros(rwf.shape[1])
            widths    = times
            waveforms = rwf
            _, _, rebinned_wfs = pkf.rebin_times_and_waveforms(times, widths, waveforms, rebin_stride=rebin_stride)
        return rebinned_wfs
    return rebin_pmts


def pmts_sum(rwfs):
    return rwfs.sum(axis=0)
