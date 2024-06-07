import numpy        as np
import scipy.signal as signal
import scipy.stats  as stats

from functools import wraps

from .. core.core_functions import to_col_vector

from ..types.symbols import BlsMode
from ..types.symbols import SiPMCalibMode


def scipy_mode(x, axis=0):
    """
    Scipy implementation of the mode (runs very slow).
    Returns a column vector.
    """
    m, c = stats.mode(x, axis=axis)
    return m


def mode(wfs, axis=0, keepdims=False):
    """
    A fast calculation of the mode: it runs 10 times
    faster than the SciPy version but only applies to
    positive waveforms.
    """
    # Runtime optimized
    # - Avoid discarding zeros (which implies looping over the waveform)
    # - Count zeros, but ignore them to find the max
    # - This can result in an empty sequence if the waveform is all
    #   zeros, so we deal with it explicitly.
    def wf_mode(wf):
        try              : return np.bincount(wf)[1:].argmax() + 1
        except ValueError: return 0 # if all zeros
    out = np.apply_along_axis(wf_mode, axis, wfs).astype(float)
    if keepdims:
        # mimic keepdims behaviour in other np functions
        shape = list(wfs.shape)
        shape[axis] = 1
        out = out.reshape(*shape)
    return out


def zero_masked(fn):
    """
    protection for mean and median
    so that we get the correct answer in case of
    zero suppressed data
    """
    @wraps(fn)
    def proxy(wfs, *args, **kwds):
        mask_wfs = np.ma.masked_where(wfs == 0, wfs)
        return fn(mask_wfs, *args, **kwds).filled(0)
    proxy.__doc__ = "Masked version to protect ZS mode \n\n" + proxy.__doc__
    return proxy

median = zero_masked(np.ma.median)
mean   = zero_masked(np.ma.mean)

def means  (wfs): return mean  (wfs, axis=1, keepdims=True)
def medians(wfs): return median(wfs, axis=1, keepdims=True)
def modes  (wfs): return mode  (wfs, axis=1, keepdims=True)


def subtract_baseline(wfs, *, bls_mode=BlsMode.mean):
    """
    Subtract the baseline to all waveforms in the input
    with a specific algorithm.

    Parameters
    ----------
    wfs: np.ndarray with shape (n, m)
        Waveforms with baseline.

    Keyword-only parameters
    -----------------------
    bls_mode: BlsMode
        Algorithm to be used.

    Returns
    -------
    bls: np.ndarray with shape (n, m)
        Baseline-subtracted waveforms.
    """

    if   bls_mode is BlsMode.mean     : return wfs - means     (wfs)
    elif bls_mode is BlsMode.median   : return wfs - medians   (wfs)
    elif bls_mode is BlsMode.mode     : return wfs - modes     (wfs)
    elif bls_mode is BlsMode.scipymode: return wfs - scipy_mode(wfs, axis=1)
    else:
        raise TypeError(f"Unrecognized baseline subtraction option: {bls_mode}")


def calibrate_wfs(wfs, adc_to_pes):
    """
    Convert waveforms in adc to pes. Masked channels (determined by
    their adc_to_pes value) are ignored.
    """
    # Runtime optimzed
    # - fast copy using astype, which we need to do anyway
    # - adc_to_pes reshaped which does not make a copy
    # - division in place to avoid making a copy
    # - setting to 0 masked channels in place to avoid making a copy
    wfs  = wfs.astype(float)
    wfs /= adc_to_pes.reshape(adc_to_pes.size, 1)
    wfs[adc_to_pes <= 0, :] = 0
    return wfs


def subtract_baseline_and_calibrate(sipm_wfs, adc_to_pes, *, bls_mode=BlsMode.mean):
    bls = subtract_baseline(sipm_wfs, bls_mode=bls_mode)
    return calibrate_wfs(bls, adc_to_pes)


def calibrate_pmts(cwfs, adc_to_pes, n_maw=100, thr_maw=3):
    """
    This function is called for PMT waveforms that have
    already been baseline restored and pedestal subtracted.
    It computes the calibrated waveforms and its sensor sum.
    It also computes the calibrated waveforms and sensor
    sum for elements of the waveforms above some value
    (thr_maw) over a MAW that follows the waveform. These
    are useful to suppress oscillatory noise and thus can
    be applied for S1 searches (the calibrated version
    without the MAW should be applied for S2 searches).
    """
    window      = np.full(n_maw, 1 / n_maw)
    maw         = signal.lfilter(window, 1, cwfs, axis=1)

    # ccwfs stands for calibrated corrected waveforms
    ccwfs       = calibrate_wfs(cwfs, adc_to_pes)
    ccwfs_maw   = np.where(cwfs >= maw + thr_maw, ccwfs, 0)

    cwf_sum     = np.sum(ccwfs    , axis=0)
    cwf_sum_maw = np.sum(ccwfs_maw, axis=0)
    return ccwfs, ccwfs_maw, cwf_sum, cwf_sum_maw


def pmt_subtract_maw(cwfs, n_maw=100):
    """
    Subtract a MAW from the input waveforms.
    """
    window = np.full(n_maw, 1 / n_maw)
    maw    = signal.lfilter(window, 1, cwfs, axis=1)

    return cwfs - maw


def calibrate_sipms(sipm_wfs, adc_to_pes, thr, *, bls_mode=BlsMode.mode):
    """
    Subtracts the baseline, calibrates waveforms to pes
    and suppresses values below `thr` (in pes).
    """
    thr  = to_col_vector(np.full(sipm_wfs.shape[0], thr))
    bls  = subtract_baseline(sipm_wfs, bls_mode=bls_mode)
    cwfs = calibrate_wfs(bls, adc_to_pes)
    return np.where(cwfs > thr, cwfs, 0)


def subtract_mean  (wfs): return subtract_baseline(wfs, bls_mode=BlsMode.mean  )
def subtract_median(wfs): return subtract_baseline(wfs, bls_mode=BlsMode.median)
def subtract_mode  (wfs): return subtract_baseline(wfs, bls_mode=BlsMode.mode  )


def sipm_subtract_mode_and_calibrate  (sipm_wfs, adc_to_pes): return calibrate_wfs(subtract_mode  (sipm_wfs), adc_to_pes)
def sipm_subtract_mean_and_calibrate  (sipm_wfs, adc_to_pes): return calibrate_wfs(subtract_mean  (sipm_wfs), adc_to_pes)
def sipm_subtract_median_and_calibrate(sipm_wfs, adc_to_pes): return calibrate_wfs(subtract_median(sipm_wfs), adc_to_pes)


# Dict of functions for SiPM processing
sipm_processing = {
    SiPMCalibMode.subtract_mode            :      subtract_mode                ,# For gain extraction
    SiPMCalibMode.subtract_median          :      subtract_median              ,# For gain extraction
    SiPMCalibMode.subtract_mode_calibrate  : sipm_subtract_mode_and_calibrate  ,# For PDF calculation
    SiPMCalibMode.subtract_mean_calibrate  : sipm_subtract_mean_and_calibrate  ,# For PDF calculation
    SiPMCalibMode.subtract_median_calibrate: sipm_subtract_median_and_calibrate,# For PDF calculation
    SiPMCalibMode.subtract_mode_zs         : calibrate_sipms                    # For data processing
}
