from functools import partial

import numpy  as np
import tables as tb

from .. reco                 import    tbl_functions as tbl
from .. reco                 import sensor_functions as sf
from .. reco                 import    wfm_functions as wfm
from .. sierpe               import fee              as FE
from .. io.rwf_io            import           rwf_writer
from .. io.           mc_io  import      mc_track_writer
from .. io.run_and_event_io  import run_and_event_writer
from .. database             import load_db
from .. core.random_sampling import NoiseSampler as SiPMsNoiseSampler


from .. dataflow          import dataflow as fl
from .. dataflow.dataflow import push
from .. dataflow.dataflow import pipe
from .. dataflow.dataflow import fork

from .  components import city
from .  components import print_every
from .  components import sensor_data
from .  components import WfType
from .  components import   wf_from_files


@city
def diomira(files_in, file_out, compression, event_range, print_mod, run_number,
            sipm_noise_cut):

    sd = sensor_data(files_in[0], WfType.mcrd)


    simulate_pmt_response_  = fl.map(simulate_pmt_response (run_number)                           , args="pmt" , out= ("pmt_sim", "blr_sim"))
    simulate_sipm_response_ = fl.map(simulate_sipm_response(run_number, sd.SIPMWL, sipm_noise_cut), args="sipm", out="sipm_sim"             )

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        RWF        = partial(rwf_writer, h5out, group_name='RD')
        write_pmt  = fl.sink(RWF(table_name='pmtrwf' , n_sensors=sd.NPMT , waveform_length=sd. PMTWL // int(FE.t_sample)), args= "pmt_sim")
        write_blr  = fl.sink(RWF(table_name='pmtblr' , n_sensors=sd.NPMT , waveform_length=sd. PMTWL // int(FE.t_sample)), args= "blr_sim")
        write_sipm = fl.sink(RWF(table_name='sipmrwf', n_sensors=sd.NSIPM, waveform_length=sd.SIPMWL                    ), args="sipm_sim")

        write_event_info_ = run_and_event_writer(h5out)
        write_mc_         = mc_track_writer(h5out) if run_number <= 0 else (lambda *_: None)

        write_event_info = fl.sink(write_event_info_, args=("run_number", "event_number", "timestamp"))
        write_mc         = fl.sink(write_mc_        , args=(        "mc", "event_number"             ))

        event_count_in = fl.spy_count()

        return push(
            source = wf_from_files(files_in, WfType.mcrd),
            pipe   = pipe(fl.slice(*event_range, close_all = True) ,
                          event_count_in    .spy ,
                          print_every(print_mod) ,
                          simulate_pmt_response_ ,
                          simulate_sipm_response_,
                          fork(write_pmt         ,
                               write_blr         ,
                               write_sipm        ,
                               write_mc          ,
                               write_event_info) ),
            result = dict(events_in = event_count_in.future))


def simulate_pmt_response(run_number):
    datapmt    = load_db.DataPMT(run_number)
    adc_to_pes = datapmt.adc_to_pes.values
    def simulate_pmt_response(pmtrd):
        rwf, blr = sf.simulate_pmt_response(0, pmtrd[np.newaxis], adc_to_pes, run_number)
        return rwf.astype(np.int16), blr.astype(np.int16)
    return simulate_pmt_response


def simulate_sipm_response(run_number, wf_length, noise_cut):
    datasipm      = load_db.DataSiPM(run_number)
    adc_to_pes    = datasipm.adc_to_pes.values
    noise_sampler = SiPMsNoiseSampler(run_number, wf_length, True)
    thresholds    = noise_cut * adc_to_pes
    def simulate_sipm_response(sipmrd):
        wfs = sf.simulate_sipm_response(0, sipmrd[np.newaxis], noise_sampler, adc_to_pes)
        return wfm.noise_suppression(wfs, thresholds)
    return simulate_sipm_response
