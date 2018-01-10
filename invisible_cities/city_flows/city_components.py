from argparse    import Namespace
from collections import Sequence
from functools   import wraps
from glob        import glob
from os.path     import expandvars
from itertools   import count

import numpy  as np
import tables as tb

from .. dataflow          import dataflow      as df
from .. evm.ic_containers import SensorData
from .. evm.ic_containers import EventData
from .. reco              import tbl_functions as tbl
from .. database          import load_db
from .. sierpe            import blr
from .. io.mc_io          import mc_track_writer

from .. core.exceptions   import NoInputFiles
from .. core.exceptions   import NoOutputFile


def city(city_function):
    @wraps(city_function)
    def proxy(**kwds):
        conf = Namespace(**kwds)

        # TODO: remove these in the config parser itself, before
        # they ever gets here
        del conf.config_file
        # TODO: these will disappear once hierarchical config files
        # are removed
        del conf.print_config_only
        del conf.hide_config
        del conf.no_overrides
        del conf.no_files
        del conf.full_files

        # TODO Check raw_data_type in parameters for RawCity

        if 'files_in' not in kwds: raise NoInputFiles
        if 'file_out' not in kwds: raise NoOutputFile

        conf.files_in  = sorted(glob(expandvars(conf.files_in)))
        conf.file_out  =             expandvars(conf.file_out)

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
            mc_tracks               = tbl.get_mc_tracks  (h5in)
            event_infos             = h5in.root.Run.events
            run_number              = h5in.root.Run.RunInfo[0]['run_number']
            for pmt, sipm, _, event_info in zip(pmtrwfs, sipmrwfs, mc_tracks, event_infos):
                ei = (run_number, event_info['evt_number'], event_info['timestamp'])
                yield EventData(pmt=pmt, sipm=sipm, mc=mc_tracks, event_info=ei)
            # NB, the monte_carlo writer is different from the others:
            # it needs to be given the WHOLE TABLE (rather than a
            # single event) at a time. THIS NEEDS TO BE IMPROVED.


def deconv_pmt(n_baseline, run_number=0):
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
    with tb.open_file(path, "r") as h5in:
        _, pmtrwfs, sipmrwfs, _ = tbl.get_rwf_vectors(h5in)
        _, NPMT,   PMTWL = pmtrwfs .shape
        _, NSIPM, SIPMWL = sipmrwfs.shape
    return SensorData(NPMT=NPMT, PMTWL=PMTWL, NSIPM=NSIPM, SIPMWL=SIPMWL)


@df.coroutine
def make_write_mc(h5out):
    write_mc = mc_track_writer(h5out)
    for event_number in count():
        mctrack = yield
        write_mc(mctrack, event_number)
