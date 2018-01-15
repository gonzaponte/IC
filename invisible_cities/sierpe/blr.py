from functools import partial

import numpy        as np
import scipy.signal as sgn


def compute_accumulator(signal_daq,
                        coeff_blr,
                        trigger_line,
                        thr_accum,
                        accum_discharge_length):
    accum = np.zeros_like(signal_daq, dtype=np.double)

    j = 0
    for k in range(1, signal_daq.size):
        accum[k] = accum[k-1] + signal_daq[k]

        if (signal_daq[k] < trigger_line) and (accum[k-1] < thr_accum):
            # discharge accumulator

            if accum[k-1] > 1:
                accum[k] = accum[k-1] * (1 - coeff_blr)
                j        = min(j + 1, accum_discharge_length - 1)
            else:
                accum[k] = 0
                j        = 0

    return accum


def subtract_baseline_and_invert(signal_daq, n_baseline):
    baseline = np.mean(signal_daq[:n_baseline])
    return baseline - signal_daq


def compute_baseline_noise(signal_daq, n_samples=400): # 400 samples is 10 mus
    return np.mean(signal_daq[:n_samples]**2)**0.5


def apply_filter(signal_daq, coeff_clean):
    b_cf, a_cf = sgn.butter(1, coeff_clean, 'high', analog=False);
    return sgn.lfilter(b_cf, a_cf, signal_daq)


def integrate_signal(signal_daq, coeff_blr, accumulator):
    signal_int    = signal_daq * (1 + coeff_blr / 2) + coeff_blr * np.roll(accumulator, 1)
    signal_int[0] = signal_daq[0]
    return signal_int


def deconvolve_signal(signal_daq,
                      coeff_clean            = 2.905447E-06,
                      coeff_blr              = 1.632411E-03,
                      n_baseline             = 28000,
                      thr_trigger            =     5,
                      accum_discharge_length =  5000):
    """
    The accumulator approach by Master VHB
    decorated and cythonized  by JJGC
    Current version using memory views

    In this version the recovered signal and the accumulator are
    always being charged. At the same time, the accumulator is being
    discharged when there is no signal. This avoids runoffs
    The baseline is computed using a window of 700 mus (by default)
    which should be good for Na and Kr
    """
    signal_daq  = subtract_baseline_and_invert(signal_daq, n_baseline)
    noise_rms   = compute_baseline_noise(signal_daq)
    signal_daq  = apply_filter(signal_daq, coeff_clean)
    accumulator = compute_accumulator(signal_daq,
                                      coeff_blr,
                                      thr_trigger * noise_rms,
                                      thr_trigger / coeff_blr,
                                      accum_discharge_length)
    return integrate_signal(signal_daq, coeff_blr, accumulator)


def deconv_pmts(rwfs,
                coeff_c,
                coeff_blr,
                pmt_active             =    [],
                n_baseline             = 28000,
                thr_trigger            =     5,
                accum_discharge_length =  5000):
    """
    Deconvolve all the PMTs in the event.
    :param pmtrwf: array of PMTs holding the raw waveform
    :param coeff_c:     cleaning coefficient
    :param coeff_blr:   deconvolution coefficient
    :param pmt_active:  list of active PMTs (by id number). An empt list
                        implies that all PMTs are active
    :param n_baseline:  number of samples taken to compute baseline
    :param thr_trigger: threshold to start the BLR process

    :returns: an array with deconvoluted PMTs. If PMT is not active
              the array is filled with zeros.
    """
    deconvolute = partial(deconvolve_signal,
                          n_baseline             = n_baseline,
                          thr_trigger            = thr_trigger,
                          accum_discharge_length = accum_discharge_length)

    if len(pmt_active):
        rwfs      = rwfs     [pmt_active]
        coeff_c   = coeff_c  [pmt_active]
        coeff_blr = coeff_blr[pmt_active]

    cwfs = map(deconvolute, rwfs, coeff_c, coeff_blr)
    return np.array(list(cwfs))


deconv_pmt = deconv_pmts
