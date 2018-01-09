from collections import Sequence
from functools   import wraps
from glob        import glob
from os.path     import expandvars

from .. evm.ic_containers import SensorData
from .. evm.ic_containers import EventData
from .. reco              import tbl_functions as tbl
from .. database          import load_db
from .. sierpe            import blr

from .. core.exceptions         import NoInputFiles
from .. core.exceptions         import NoOutputFile


def city(city_function):
    @wraps(city_function)
    def proxy(**kwds):
        conf = Namespace(**kwds)

        # TODO Check raw_data_type in parameters for RawCity

        if 'files_in'  not in kwds: raise NoInputFiles
        if 'files_out' not in kwds: raise NoOutputFile

        conf.input_files  = sorted(glob(expandvars(conf.files_in)))
        conf.output_file  =             expandvars(conf.file_out)

        conf.event_range  = _event_range(conf)
        # TODO There were deamons! self.daemons = tuple(map(summon_daemon, kwds.get('daemons', [])))

        return city_function(**vars(conf))
    return proxy

def _event_range(conf):
    if not hasattr(conf, 'event_range'):        return None, 1
    er = conf.event_range
    if not isinstance(er, Sequence): er = (er,)
    if len(er) == 1:                            return None, er[0]
    if len(er) == 2:                            return tuple(er)
    if len(er) == 0:
        raise ValueError('event_range needs at least one value')
    if len(er) >  2:
        raise ValueError('event_range accepts at most 2 values but was given {}'
                        .format(er))

############################

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
