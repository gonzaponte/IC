from operator  import attrgetter
from functools import partial

import numpy  as np
import tables as tb

from .. reco                import tbl_functions as tbl
from .. database            import load_db
from .. sierpe              import blr
from .. io.          rwf_io import           rwf_writer
from .. io.run_and_event_io import run_and_event_writer

from .. dataflow            import dataflow      as df
from .. dataflow.dataflow   import     sink
from .. dataflow.dataflow   import starsink
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import pick
from .. dataflow.dataflow   import fork

from .  city_components import city
from .  city_components import deconv_pmt
from .  city_components import event_data_from_files
from .  city_components import sensor_data
from .  city_components import make_write_mc
from .  city_components import calibrate_pmts


@city
def isidora(files_in, file_out, event_range, run_number, n_baseline,
            raw_data_type, compression, print_mod, verbosity):
    """
    The city of ISIDORA performs a fast processing from raw data
    (pmtrwf and sipmrwf) to BLR wavefunctions.

    """
    sd = sensor_data(files_in[0])

    rwf_to_cwf = df.map(deconv_pmt(n_baseline))

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        RWF = partial(rwf_writer, h5out, group_name='BLR')
        write_pmt   = sink(RWF(table_name='pmtcwf' , n_sensors=sd.NPMT , waveform_length=sd.PMTWL))
        write_sipm  = sink(RWF(table_name='sipmrwf', n_sensors=sd.NSIPM, waveform_length=sd.SIPMWL))
        write_event_info = starsink(run_and_event_writer(h5out))

        write_mc = make_write_mc(h5out)

        event_count = df.count()

        return push(
            source = event_data_from_files(files_in),
            pipe   = pipe(df.slice(*event_range),
             fork((pick('pmt'       ), rwf_to_cwf, write_pmt       ),
                  (pick('sipm'      ),             write_sipm      ),
                  (pick('mc'        ),             write_mc        ),
                  (pick('event_info'),             write_event_info),
                  event_count.sink)),
            result = (event_count.future,))


@city
def irene(files_in, file_out, event_range, run_number, compression, compute_ipmt_pmaps, config_file, n_baseline,
          n_mau, n_mau_sipm, print_mod,
          raw_data_type,
          run_number,
          s1_lmax,
          s1_lmin,
          s1_rebin_stride,
          s1_stride,
          s1_tmax,
          s1_tmin,
          s2_lmax,
          s2_lmin,
          s2_rebin_stride,
          s2_stride,
          s2_tmax,
          s2_tmin,
          thr_csum_s1,
          thr_csum_s2,
          thr_mau,
          thr_sipm,
          thr_sipm_s2,
          thr_sipm_type,
          verbosity):

    rwf_to_cwf    = df.map(deconv_pmt(n_baseline))
    cwf_to_ccwf   = df.map(calibrate_pmts(n_mau, thr_mau, run_number))
    ccwf_to_zs_s1 = df.map(zero_suppress_wf_s1(thr_csum_s1))
    ccwf_to_zs_s2 = df.map(zero_suppress_wf_s2(thr_csum_s2))

    filter_empty_wf = df.filter(check_wf_sum)
    sipm_rwf_to_cal = df.map(calibrate_sipms(thr_sipm, n_mau_sipm, run_number))

    pmap_builder = build_pmap(s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                              s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2, thr_sipm_type,
                              run_number))

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        pmap_writer       = sink(pmap_writer(h5out, compression=compression))
        write_event_info  = starsink(run_and_event_writer(h5out))
        write_mc          = make_write_mc(h5out)
        JOIN_PMT_AND_SIPM_BUILD_PMAP_AND_WRITE = join(2, gather_pmt_and_sipm_data)(pmap_builder, PMP_writer)
        JOIN_PMT          = join(3, gather_pmt_data         )(JOIN_PMT_AND_SIPM)

        event_count_in             = df.count()
        event_count_out            = df.count()
        empty_events               = df.count()
        empty_events_s2_ene_eq_0   = df.count()
        empty_events_s1_indx_empty = df.count()
        empty_events_s2_indx_empty = df.count()

        class JOIN:
            pass

        FAT_PIPE = fork((pick('pmt' ), rwf_to_cwf, cwf_to_ccwf, filter_empty_wf, fork(                                  JOIN,
                                                                                      (pick('csum_mau'), ccwf_to_zs_s1, JOIN),
                                                                                      (pick('csum'    ), ccwf_to_zs_s2, JOIN)), JOIN),
                        (pick('sipm'), sipm_rwf_to_cal,                                                                         JOIN)), pmap_builder, pmap_writer


        return push(source = event_data_from_files(files_in),
                    pipe   = pipe(df.slice(*event_range),
                                  fork(FAT_PIPE,
                                       (pick('mc'        ), write_mc        ),
                                       (pick('event_info'), write_event_info)),
                                       event_count.sink),
                    result = (event_count.future,))
