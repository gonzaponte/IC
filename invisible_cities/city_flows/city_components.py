from .. evm.ic_containers import SensorData
from .. evm.ic_containers import EventData
from .. reco              import tbl_functions as tbl
from .. database          import load_db
from .. sierpe            import blr


def event_data_from_files(paths):
    for path in paths:
        with tb.open_file(path, "r") as h5in:
            _, pmtrwfs, sipmrwfs, _ = tbl.get_rwf_vectors(h5in)
            mc_tracks               = get_mc_tracks      (h5in)
            events_infos            = h5in.root.Run.events
            for pmt, sipm, mc, event_info in zip(pmtrwfs, sipmrwfs, mc_tracks, events_info):
                yield EventData(pmt=pmt, sipm=sipm, mc=mc, event_info=event_info)



def deconv_pmt(n_baseline):
    DataPMT    = load_db.DataPMT(run_number = run_number)
    pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist()
    coeff_c    = DataPMT.coeff_c  .values.astype(np.double)
    coeff_blr  = DataPMT.coeff_blr.values.astype(np.double)

    def deconv_pmt(RWF):
        return blr.deconv_pmt(RWF,
                              coeff_c,
                              coeff_blr,
                              pmt_active = pmt_active,
                              n_baseline = n_baseline)
    return deconv_pmt


def sensor_data(path):
    with tb.open_file(path[0], "r") as h5in:
        _, pmtrwfs, sipmrwfs, _ = tbl.get_rwf_vectors(h5in)
        _, NPMT,   PMTWL = pmtrwfs .shape
        _, NSIPM, SIPMWL = sipmrwfs.shape
    return SensorData(NPMT=NPMT, PMTWL=PMTWL, NSIPM=NSIPM, SIPMWL=SIPMWL)
