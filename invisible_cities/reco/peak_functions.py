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



def prepend_zeros(arr, n):
    """
    Prepend zeros to the last axis of an array.

    Parameters
    ----------
    arr : np.ndarray
        Array to which prepend zeros

    n : int
        Number of zeros to prepend

    Returns
    -------
    parr : np.ndarray
        Array with prepended zeros.
    """
    n    = (*arr.shape[:-1], n)
    axis = len(arr.shape) - 1
    return np.concatenate([np.zeros(n), arr], axis=axis) if n else arr


def build_pmt_responses(indices, times, widths, ccwfs,
                        pmt_ids, rebin_stride, pad_zeros,
                        sipm_pmt_bin_ratio):
    low  = indices[0]
    high = indices[-1] + 1
    n_mismatch = low % sipm_pmt_bin_ratio

    times  = prepend_zeros(times [   low:high], n_mismatch)
    widths = prepend_zeros(widths[   low:high], n_mismatch)
    wfs    = prepend_zeros(ccwfs [:, low:high], n_mismatch)

    if rebin_stride > 1:
        splits  = np.arange(0, wfs.shape[1], rebin_stride)
        weights = wfs.sum(axis=0)
        times   = rebin_average( times, splits, weights)
        widths  = rebin_add    (widths, splits)
        wfs     = rebin_add    (wfs   , splits, axis=1)

    pk_sum   = pd.DataFrame(dict( time   = times
                                , bwidth = widths
                                , ene    = wfs.sum(axis=0)))

    pk_split = pd.DataFrame(dict( npmt = np.repeat(pmt_ids, times.size)
                                , ene  = wfs.flatten()))
    return pk_sum, pk_split


def build_sipm_responses(indices, sipm_wfs, rebin_stride, thr_sipm_s2):
    wfs = sipm_wfs[:, indices[0] : indices[-1] + 1]

    if rebin_stride > 1:
        splits = np.arange(0, wfs.shape[1], rebin_stride)
        wfs    = rebin_add(wfs, splits, axis=1)

    sipm_ids, sipm_wfs = select_wfs_above_time_integrated_thr(wfs, thr_sipm_s2)

    sipm_ids = np.repeat(sipm_ids, sipm_wfs.shape[1])
    return pd.DataFrame(dict( nsipm = sipm_ids
                            , ene   = sipm_wfs.flatten()))


def build_peak(indices, times,
               widths, ccwf, pmt_ids,
               rebin_stride,
               with_sipms,
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
                                    sipm_wfs,
                                    rebin_stride // sipm_pmt_bin_ratio,
                                    thr_sipm_s2)
    else:
        sipm = pd.DataFrame()

    return pmt_sum, pmt_split, sipm


def find_peaks(ccwfs, index,
               time, length,
               stride, rebin_stride,
               pmt_ids,
               pmt_samp_wid = 25*units.ns,
               sipm_samp_wid = 1*units.mus,
               sipm_wfs=None, thr_sipm_s2=0):
    times           = np.arange     (0, ccwfs.shape[1] * pmt_samp_wid, pmt_samp_wid)
    widths          = np.full       (ccwfs.shape[1], pmt_samp_wid)
    indices_split   = split_in_peaks(index, stride)
    selected_splits = select_peaks  (indices_split, time, length, pmt_samp_wid)
    with_sipms      = sipm_wfs is not None

    pmt_sums   = []
    pmt_splits = []
    sipms      = []
    for peak_no, indices in enumerate(selected_splits):
        pmt_sum, pmt_split, sipm = build_peak(indices, times,
                                              widths, ccwfs, pmt_ids,
                                              rebin_stride,
                                              with_sipms,
                                              pmt_samp_wid, sipm_samp_wid,
                                              sipm_wfs, thr_sipm_s2)
        pmt_sum  .insert(0, "peak", peak_no)
        pmt_split.insert(0, "peak", peak_no)
        sipm     .insert(0, "peak", peak_no)

        pmt_sums  .append(pmt_sum  )
        pmt_splits.append(pmt_split)
        sipms     .append(sipm     )

    pmt_sums   = pd.concat(pmt_sums  , ignore_index=True, copy=False)
    pmt_splits = pd.concat(pmt_splits, ignore_index=True, copy=False)
    sipms      = pd.concat(sipms     , ignore_index=True, copy=False)
    return pmt_sums, pmt_splits, sipms


def get_pmap(event, ccwf, s1_indx, s2_indx, sipm_zs_wf,
             s1_params, s2_params, thr_sipm_s2, pmt_ids,
             pmt_samp_wid, sipm_samp_wid):
    s1, s1pmt, _ = find_peaks(ccwf, s1_indx, pmt_ids=pmt_ids,
                              pmt_samp_wid=pmt_samp_wid,
                              **s1_params)
    s2, s2pmt, s2si = find_peaks(ccwf, s2_indx, pmt_ids=pmt_ids,
                                 sipm_wfs      = sipm_zs_wf,
                                 thr_sipm_s2   = thr_sipm_s2,
                                 pmt_samp_wid  = pmt_samp_wid,
                                 sipm_samp_wid = sipm_samp_wid,
                                 **s2_params)
    for df in (s1, s1pmt, s2, s2pmt, s2si):
        df.insert(0, "event", event)

    return s1, s1pmt, s2, s2pmt, s2si


def rebin_add(*args, **kwargs):
    """
    Groups together consecutive samples and adds them up into a single
    one. It's a convenient alias to np.add.reduceat.
    """
    return np.add.reduceat(*args, **kwargs)


def rebin_average(arr, indices, weights=None, **kwargs):
    """
    Groups together consecutive samples and averages them into a single
    one.

    Parameters
    ----------
    arr : np.ndarray (n_initial_samples,)
        Array to rebin

    indices : np.array (n_final_samples,)
        Array of lower bin edges. The last bin will contain the
        aggregation of indices[-1] till the end.

    weights : np.array (n_initial_samples,), optional
        Weights to perform average.

    **kwargs : optional arguments to np.add.reduceat

    Returns
    -------
    resampled : np.ndarray (n_final_samples,)
        Resampled waveform values
    """
    if len(arr.shape) > 1:
        raise NotImplementedError("rebin_average is only implemented for 1D arrays")

    if weights is None: return rebin_add(arr        , indices, **kwargs) / arr.size
    else              : return rebin_add(arr*weights, indices, **kwargs) / weights.sum()
