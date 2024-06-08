"""
code: peak_functions.py

description: functions related to the pmap creation.

credits: see ic_authors_and_legal.rst in /doc

last revised: @abotas & @gonzaponte. Dec 1st 2017
"""

import numpy        as np
import pandas       as pd

from .. core               import system_of_units as units
from .. evm .ic_containers import ZsWf
from .. evm .pmaps         import S1
from .. evm .pmaps         import S2
from .. evm .pmaps         import PMap
from .. evm .pmaps         import PMTResponses
from .. evm .pmaps         import SiPMResponses


def indices_and_wf_above_threshold(wf, thr):
    indices_above_thr = np.where(wf > thr)[0]
    wf_above_thr      = wf[indices_above_thr]
    return ZsWf(indices_above_thr, wf_above_thr)


def select_wfs_above_time_integrated_thr(wfs, thr):
    selected_ids = np.where(np.sum(wfs, axis=1) >= thr)[0]
    selected_wfs = wfs[selected_ids]
    return selected_ids, selected_wfs


def split_in_peaks(indices, stride):
    where = np.where(np.diff(indices) > stride)[0]
    return np.split(indices, where + 1)


def select_peaks(peaks, time, length, pmt_samp_wid=25*units.ns):
    def is_valid(indices):
        return (time  .contains(indices[ 0] * pmt_samp_wid) and
                time  .contains(indices[-1] * pmt_samp_wid) and
                length.contains(indices[-1] + 1 - indices[0]))
    return tuple(filter(is_valid, peaks))


def pick_slice_and_rebin(indices, times, widths,
                         wfs, rebin_stride, pad_zeros=False,
                         sipm_pmt_bin_ratio=40):
    slice_ = slice(indices[0], indices[-1] + 1)
    times_  = times [   slice_]
    widths_ = widths[   slice_]
    wfs_    = wfs   [:, slice_]
    if pad_zeros:
        n_miss = indices[0] % sipm_pmt_bin_ratio
        n_wfs  = wfs.shape[0]
        times_  = np.concatenate([np.zeros(        n_miss) ,  times_])
        widths_ = np.concatenate([np.zeros(        n_miss) , widths_])
        wfs_    = np.concatenate([np.zeros((n_wfs, n_miss)),    wfs_], axis=1)

    if rebin_stride < 2:
        return times_, widths_, wfs_

    indices = np.arange(0, len(times_), rebin_stride)
    (times ,
     widths,
     wfs   ) = rebin_times_and_waveforms(times_, widths_, wfs_, indices)
    return times, widths, wfs


def build_pmt_responses(indices, times, widths, ccwf,
                        pmt_ids, rebin_stride, pad_zeros,
                        sipm_pmt_bin_ratio):
    (pk_times ,
     pk_widths,
     pmt_wfs  ) = pick_slice_and_rebin(indices, times, widths,
                                       ccwf   , rebin_stride,
                                       pad_zeros          = pad_zeros,
                                       sipm_pmt_bin_ratio = sipm_pmt_bin_ratio)

    pk_sum   = pd.DataFrame(dict( time   = pk_times
                                , bwidth = pk_widths
                                , ene    = pmt_wfs.sum(axis=0)))

    pk_split = pd.DataFrame(dict( npmt = np.repeat(pmt_ids, pk_times.size)
                                , ene  = pmt_wfs.flatten()))
    return pk_sum, pk_split


def build_sipm_responses(indices, times, widths,
                         sipm_wfs, rebin_stride, thr_sipm_s2):
    _, _, sipm_wfs_ = pick_slice_and_rebin(indices , times, widths,
                                           sipm_wfs, rebin_stride,
                                           pad_zeros = False)
    (sipm_ids,
     sipm_wfs)   = select_wfs_above_time_integrated_thr(sipm_wfs_,
                                                        thr_sipm_s2)

    sipm_ids = np.repeat(sipm_ids, sipm_wfs.shape[1])
    return pd.DataFrame(dict( nsipm = sipm_ids
                            , ene   = sipm_wfs.flatten()))


def build_peak(indices, times,
               widths, ccwf, pmt_ids,
               rebin_stride,
               with_sipms, Pk,
               pmt_samp_wid  = 25 * units.ns,
               sipm_samp_wid =  1 * units.mus,
               sipm_wfs      = None,
               thr_sipm_s2   = 0):
    sipm_pmt_bin_ratio = int(sipm_samp_wid/pmt_samp_wid)
    pmt_sum, pmt_split = build_pmt_responses(indices, times, widths,
                                             ccwf, pmt_ids,
                                             rebin_stride, pad_zeros = with_sipms,
                                             sipm_pmt_bin_ratio = sipm_pmt_bin_ratio)
    if with_sipms:
       sipm  = build_sipm_responses(indices // sipm_pmt_bin_ratio,
                                    times // sipm_pmt_bin_ratio,
                                    widths * sipm_pmt_bin_ratio,
                                    sipm_wfs,
                                    rebin_stride // sipm_pmt_bin_ratio,
                                    thr_sipm_s2)
    else:
        sipm = pd.DataFrame()

    return pmt_sum, pmt_split, sipm


def find_peaks(ccwfs, index,
               time, length,
               stride, rebin_stride,
               Pk, pmt_ids,
               pmt_samp_wid = 25*units.ns,
               sipm_samp_wid = 1*units.mus,
               sipm_wfs=None, thr_sipm_s2=0):
    ccwfs = np.array(ccwfs, ndmin=2)

    peaks           = []
    times           = np.arange     (0, ccwfs.shape[1] * pmt_samp_wid, pmt_samp_wid)
    widths          = np.full       (ccwfs.shape[1],   pmt_samp_wid)
    indices_split   = split_in_peaks(index, stride)
    selected_splits = select_peaks  (indices_split, time, length, pmt_samp_wid)
    with_sipms      = Pk is S2 and sipm_wfs is not None

    pmt_sums   = []
    pmt_splits = []
    sipms      = []
    for peak_no, indices in enumerate(selected_splits):
        pmt_sum, pmt_split, sipm = build_peak(indices, times,
                                              widths, ccwfs, pmt_ids,
                                              rebin_stride,
                                              with_sipms, Pk,
                                              pmt_samp_wid, sipm_samp_wid,
                                              sipm_wfs, thr_sipm_s2)
        pmt_sum  .insert(0, "peak", peak_no)
        pmt_split.insert(0, "peak", peak_no)
        sipm     .insert(0, "peak", peak_no)
        pmt_sums  .append(pmt_sum  )
        pmt_splits.append(pmt_split)
        sipms     .append(sipm     )

    pmt_sums   = pd.concat(pmt_sums  , ignore_index=True)
    pmt_splits = pd.concat(pmt_splits, ignore_index=True)
    sipms      = pd.concat(sipms     , ignore_index=True)
    return pmt_sums, pmt_splits, sipms


def get_pmap(event, ccwf, s1_indx, s2_indx, sipm_zs_wf,
             s1_params, s2_params, thr_sipm_s2, pmt_ids,
             pmt_samp_wid, sipm_samp_wid):
    s1, s1pmt, _ = find_peaks(ccwf, s1_indx, Pk=S1, pmt_ids=pmt_ids,
                              pmt_samp_wid=pmt_samp_wid,
                              **s1_params)
    s2, s2pmt, s2si = find_peaks(ccwf, s2_indx, Pk=S2, pmt_ids=pmt_ids,
                                 sipm_wfs      = sipm_zs_wf,
                                 thr_sipm_s2   = thr_sipm_s2,
                                 pmt_samp_wid  = pmt_samp_wid,
                                 sipm_samp_wid = sipm_samp_wid,
                                 **s2_params)
    for df in (s1, s1pmt, s2, s2pmt, s2si):
        df.insert(0, "event", event)

    return s1, s1pmt, s2, s2pmt, s2si

def rebin_times_and_waveforms(times, widths, waveforms, indices):
    """
    Groups together consecutive samples and aggregates them together
    into a single one. Times are averaged using the waveform samples
    as weights, while widths and waveform samples are summed up. The
    number of consecutive slices taken is determined by `indices`,
    which mark the start of each slice.

    Parameters
    ----------
    times : np.ndarray (n_samples,)
        Array of buffer times

    widths : np.ndarray (n_samples,)
        Array of sample widths

    waveforms : np.ndarray (n_sensors, n_samples)
        Waveform values

    rebin_stride: np.ndarray (n_new_samples,)
        Indices that mark the lower bound of each new sample

    Returns
    -------
    times : np.ndarray (n_new_samples,)
        Array of resampled buffer times

    widths : np.ndarray (n_new_samples,)
        Array of resampled sample widths

    waveforms : np.ndarray (n_sensors, n_new_samples)
        Resampled waveform values
    """
    # Runtime optimized
    # - Slice loop moved to numpy via np.add.reduceat
    # - Weighted average performed by hand inseat of using np.average,
    #   not because it's faster per se, but because it allows us to
    #   skip the external loop over slices
    # - indices are now given as arguments, since they don't need to
    #   be computed every time
    enes      = waveforms.sum(axis=0)
    widths    = np.add.reduceat(widths    , indices)
    waveforms = np.add.reduceat(waveforms , indices, axis=1)
    times     = np.add.reduceat(times*enes, indices) / waveforms.sum(axis=0)
    return times, widths, waveforms
