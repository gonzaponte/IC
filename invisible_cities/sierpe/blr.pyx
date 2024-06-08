import  numpy as np
cimport numpy as np
cimport cython
from scipy import signal as SGN

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
cpdef deconvolve_signal(double [:] signal_daq,
                        double coeff_clean            = 2.905447E-06,
                        double coeff_blr              = 1.632411E-03,
                        double thr_trigger            =     5,
                        int    accum_discharge_length =  5000):
    """
    The accumulator approach by Master VHB
    decorated and cythonized  by JJGC
    Current version using memory views

    In this version the recovered signal and the accumulator are
    always being charged. At the same time, the accumulator is being
    discharged when there is no signal. This avoids runoffs
    """

    cdef double thr_acum     = thr_trigger / coeff_blr
    cdef int len_signal_daq  = len(signal_daq)

    cdef double [:] signal_r = np.zeros(len_signal_daq, dtype=np.double)
    cdef double [:] acum     = np.zeros(len_signal_daq, dtype=np.double)


    # compute noise
    cdef int nn = 400 # fixed at 10 mus
    cdef sample
    cdef double noise_rms = 0
    for i in range(nn):
        sample     = signal_daq[i]
        noise_rms += sample*sample
    noise_rms = np.sqrt(noise_rms/nn)

    # trigger line
    cdef double trigger_line = thr_trigger * noise_rms

    # cleaning signal
    cdef double [:] b_cf
    cdef double [:] a_cf

    b_cf, a_cf = SGN.butter(1, coeff_clean, 'high', analog=False);
    signal_daq = SGN.lfilter(b_cf, a_cf, signal_daq)

    cdef int k
    cdef int jmax = accum_discharge_length - 1
    cdef double factor1 = 1 + coeff_blr/2
    cdef double factor2 = 1 - coeff_blr

    cdef double sdaq
    cdef double acc
    cdef int j = 0

    signal_r[0] = signal_daq[0]
    for k in range(1, len_signal_daq):
        # search for values only once
        sdaq = signal_daq[k]
        acc  = acum[k-1]

        # always update signal and accumulator
        signal_r[k] = sdaq * factor1 + coeff_blr * acc
        acum    [k] = sdaq + acc

        if (sdaq < trigger_line) and (acc < thr_acum):
            # discharge accumulator

            if acc > 1:
                acum[k] = acc * factor2
                if j < jmax: j = j + 1
                else       : j = jmax
            else:
                acum[k] = 0
                j = 0

    # return recovered signal
    return np.asarray(signal_r)