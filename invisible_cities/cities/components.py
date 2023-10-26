from functools       import wraps
from functools       import partial
from collections.abc import Sequence
from argparse        import Namespace
from glob            import glob
from os.path         import expandvars
from itertools       import count
from itertools       import repeat
from typing          import Callable
from typing          import Iterator
from typing          import Mapping
from typing          import Generator
from typing          import List
from typing          import Dict
from typing          import Tuple
from typing          import Union
from typing          import Optional
from typing          import Any
import tables as tb
import numpy  as np
import pandas as pd
import inspect
import warnings
import math

from .. dataflow                  import                  dataflow as  fl
from .. dataflow.dataflow         import                      sink
from .. dataflow.dataflow         import                      pipe
from .. evm    .ic_containers     import                SensorData
from .. evm    .event_model       import                   KrEvent
from .. evm    .event_model       import                       Hit
from .. evm    .event_model       import                   Cluster
from .. evm    .event_model       import             HitCollection
from .. evm    .event_model       import                    MCInfo
from .. core                      import           system_of_units as units
from .. core   .exceptions        import                XYRecoFail
from .. core   .exceptions        import           MCEventNotFound
from .. core   .exceptions        import              NoInputFiles
from .. core   .exceptions        import              NoOutputFile
from .. core   .exceptions        import InvalidInputFileStructure
from .. core   .configure         import          event_range_help
from .. core   .configure         import         check_annotations
from .. core   .random_sampling   import              NoiseSampler
from .. detsim                    import          buffer_functions as  bf
from .. detsim .sensor_utils      import             trigger_times
from .. reco                      import           calib_functions as  cf
from .. reco                      import          sensor_functions as  sf
from .. reco                      import   calib_sensors_functions as csf
from .. reco                      import            peak_functions as pkf
from .. reco                      import           pmaps_functions as pmf
from .. reco                      import            hits_functions as hif
from .. reco                      import             wfm_functions as wfm
from .. reco                      import         paolina_functions as plf
from .. reco   .xy_algorithms     import                    corona
from .. reco   .xy_algorithms     import                barycenter
from .. filters.s1s2_filter       import               S12Selector
from .. filters.s1s2_filter       import               pmap_filter
from .. database                  import                   load_db
from .. sierpe                    import                       blr
from .. io                        import                 mcinfo_io
from .. io     .pmaps_io          import                load_pmaps
from .. io     .dst_io            import                  load_dst
from .. io     .event_filter_io   import       event_filter_writer
from .. io     .pmaps_io          import               pmap_writer
from .. io     .rwf_io            import             buffer_writer
from .. io     .mcinfo_io         import            load_mchits_df
from .. io     .dst_io            import                 df_writer
from .. types  .ic_types          import                  NoneType
from .. types  .ic_types          import                        xy
from .. types  .ic_types          import                        NN
from .. types  .ic_types          import                       NNN
from .. types  .ic_types          import                    minmax
from .. types  .ic_types          import        types_dict_summary
from .. types  .ic_types          import         types_dict_tracks
from .. types  .symbols           import                    WfType
from .. types  .symbols           import                   BlsMode
from .. types  .symbols           import             SiPMThreshold
from .. types  .symbols           import                EventRange
from .. types  .symbols           import                 HitEnergy
from .. types  .symbols           import                SiPMCharge
from .. types  .symbols           import                    XYReco



def city(city_function):
    @wraps(city_function)
    def proxy(**kwds):
        conf = Namespace(**kwds)

        # TODO: remove these in the config parser itself, before
        # they ever gets here
        if hasattr(conf, 'config_file'):       del conf.config_file
        # TODO: these will disappear once hierarchical config files
        # are removed
        if hasattr(conf, 'print_config_only'): del conf.print_config_only
        if hasattr(conf, 'hide_config'):       del conf.hide_config
        if hasattr(conf, 'no_overrides'):      del conf.no_overrides
        if hasattr(conf, 'no_files'):          del conf.no_files
        if hasattr(conf, 'full_files'):        del conf.full_files

        # TODO: we have decided to remove verbosity.
        # Needs to be removed form config parser
        if hasattr(conf, 'verbosity'):         del conf.verbosity

        # TODO Check raw_data_type in parameters for RawCity

        if 'files_in' not in kwds: raise NoInputFiles
        if 'file_out' not in kwds: raise NoOutputFile

        # For backward-compatibility we set NEW as the default DB in
        # case it is not defined in the config file
        if 'detector_db' in inspect.getfullargspec(city_function).args and \
           'detector_db' not in kwds:
            conf.detector_db = 'new'

        conf.files_in  = sorted(glob(expandvars(conf.files_in)))
        conf.file_out  =             expandvars(conf.file_out)

        conf.event_range  = event_range(conf)
        # TODO There were deamons! self.daemons = tuple(map(summon_daemon, kwds.get('daemons', [])))

        result = check_annotations(city_function)(**vars(conf))
        index_tables(conf.file_out)
        return result
    return proxy


@check_annotations
def create_timestamp(rate: float) -> float:
    """
    Get rate value safely: It raises warning if rate <= 0 and
    it sets a physical rate value in Hz.

    Parameters
    ----------
    rate : float
           Value of the rate in Hz.

    Returns
    -------
    Function to calculate timestamp for the given rate with
    event_number as parameter.
    """

    if rate == 0:
        warnings.warn("Zero rate is unphysical, using default "
                      "rate = 0.5 Hz instead", stacklevel=2)
        rate = 0.5 * units.hertz
    elif rate < 0:
        warnings.warn(f"Negative rate is unphysical, using "
                      f"rate = {abs(rate) / units.hertz} Hz instead",
                      stacklevel=2)
        rate = abs(rate)

    def create_timestamp_(event_number: Union[int, float]) -> float:
        """
        Calculates timestamp for a given Event Number and Rate.

        Parameters
        ----------
        event_number : Union[int, float]
                       ID value of the current event.

        Returns
        -------
        Calculated timestamp : float
        """

        period = 1. / rate
        timestamp = abs(event_number * period) + np.random.uniform(0, period)
        return timestamp

    return create_timestamp_


def index_tables(file_out):
    """
    -finds all tables in output_file
    -checks if any columns in the tables have been marked to be indexed by writers
    -indexes those columns
    """
    with tb.open_file(file_out, 'r+') as h5out:
        for table in h5out.walk_nodes(classname='Table'):        # Walk over all tables in h5out
            if 'columns_to_index' not in table.attrs:  continue  # Check for columns to index
            for colname in table.attrs.columns_to_index:         # Index those columns
                table.colinstances[colname].create_index()


def _check_invalid_event_range_spec(er):
    return (len(er) not in (1, 2)                   or
            (len(er) == 2 and EventRange.all in er) or
            er[0] is EventRange.last                )


def event_range(conf):
    # event_range not specified
    if not hasattr(conf, 'event_range')           : return None, 1
    er = conf.event_range

    if not isinstance(er, Sequence): er = (er,)
    if _check_invalid_event_range_spec(er):
        message = "Invalid spec for event range. Only the following are accepted:\n" + event_range_help
        raise ValueError(message)

    if   len(er) == 1 and er[0] is EventRange.all : return (None,)
    elif len(er) == 2 and er[1] is EventRange.last: return (er[0], None)
    else                                          : return er


def print_every(N):
    counter = count()
    return fl.branch(fl.map  (lambda _: next(counter), args="event_number", out="index"),
                     fl.slice(None, None, N),
                     fl.sink (lambda data: print(f"events processed: {data['index']}, event number: {data['event_number']}")))


def print_every_alternative_implementation(N):
    @fl.coroutine
    def print_every_loop(target):
        with fl.closing(target):
            for i in count():
                data = yield
                if not i % N:
                    print(f"events processed: {i}, event number: {data['event_number']}")
                target.send(data)
    return print_every_loop


def get_actual_sipm_thr(thr_sipm_type, thr_sipm, detector_db, run_number):
    if   thr_sipm_type is SiPMThreshold.common:
        # In this case, the threshold is a value in pes
        sipm_thr = thr_sipm

    elif thr_sipm_type is SiPMThreshold.individual:
        # In this case, the threshold is a percentual value
        noise_sampler = NoiseSampler(detector_db, run_number)
        sipm_thr      = noise_sampler.compute_thresholds(thr_sipm)

    else:
        raise ValueError(f"Unrecognized thr type: {thr_sipm_type}. "
                          "Only valid options are `common` and `individual`")

    return sipm_thr


def collect():
    """Return a future/sink pair for collecting streams into a list."""
    def append(l,e):
        l.append(e)
        return l
    return fl.reduce(append, initial=[])()


@check_annotations
def copy_mc_info(files_in     : List[str],
                 h5out        : tb.File  ,
                 event_numbers: List[int],
                 db_file      :      str ,
                 run_number   :      int ) -> None:
    """
    Copy to an output file the MC info of a list of selected events.

    Parameters
    ----------
    files_in     : List of strings
                   Name of the input files.
    h5out        : tables.File
                   The output h5 file.
    event_numbers: List[int]
                   List of event numbers for which the MC info is copied
                   to the output file.
    db_file      : str
                   Name of database to be used where necessary to
                   read the MC info (used for pre-2020 format files)
    run_number   : int
                   Run number for database (used for pre-2020 format files)
    """

    writer = mcinfo_io.mc_writer(h5out)

    copied_events   = []
    for f in files_in:
        if mcinfo_io.check_mc_present(f):
            evt_copied  = mcinfo_io.safe_copy_nexus_eventmap(h5out        ,
                                                             event_numbers,
                                                             f            )
            mcinfo_io.copy_mc_info(f, writer, evt_copied['nexus_evt'],
                                   db_file, run_number)
            copied_events.extend(evt_copied['evt_number'])
        else:
            warnings.warn('File does not contain MC tables.\
             Use positve run numbers for data', UserWarning)
            continue
    if len(np.setdiff1d(event_numbers, copied_events)) != 0:
        raise MCEventNotFound('Some events not found in MC tables')


@check_annotations
def wf_binner(max_buffer: float) -> Callable:
    """
    Returns a function to be used to convert the raw
    input MC sensor info into data binned according to
    a set bin width, effectively
    padding with zeros inbetween the separate signals.

    Parameters
    ----------
    max_buffer : float
                 Maximum event time to be considered in nanoseconds
    """
    def bin_sensors(sensors  : pd.DataFrame,
                    bin_width: float       ,
                    t_min    : float       ,
                    t_max    : float       ) -> Tuple[np.ndarray, pd.Series]:
        return bf.bin_sensors(sensors, bin_width, t_min, t_max, max_buffer)
    return bin_sensors


@check_annotations
def signal_finder(buffer_len   : float,
                  bin_width    : float,
                  bin_threshold:   int) -> Callable:
    """
    Decides where there is signal-like
    charge according to the configuration
    and the PMT sum in order to give
    a useful position for buffer selection.
    Currently simple threshold on binned charge.

    Parameters
    ----------
    buffer_len    : float
                    Configured buffer length in mus
    bin_width     : float
                    Sampling width for sensors
    bin_threshold : int
                    PE threshold for selection
    """
    # The stand_off is the minumum number of samples
    # necessary between candidate triggers.
    stand_off = int(buffer_len / bin_width)
    def find_signal(wfs: pd.Series) -> List[int]:
        return bf.find_signal_start(wfs, bin_threshold, stand_off)
    return find_signal


# TODO: consider caching database
def deconv_pmt(dbfile, run_number, n_baseline,
               selection=None, pedestal_function=csf.means):
    DataPMT    = load_db.DataPMT(dbfile, run_number = run_number)
    pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist() if selection is None else selection
    coeff_c    = DataPMT.coeff_c  .values.astype(np.double)
    coeff_blr  = DataPMT.coeff_blr.values.astype(np.double)

    def deconv_pmt(RWF):
        CWF = pedestal_function(RWF[:, :n_baseline]) - RWF
        return np.array(tuple(map(blr.deconvolve_signal, CWF[pmt_active],
                                  coeff_c              , coeff_blr      )))
    return deconv_pmt


def get_run_number(h5in):
    if   "runInfo" in h5in.root.Run: return h5in.root.Run.runInfo[0]['run_number']
    elif "RunInfo" in h5in.root.Run: return h5in.root.Run.RunInfo[0]['run_number']

    raise tb.exceptions.NoSuchNodeError(f"No node runInfo or RunInfo in file {h5in}")


def get_pmt_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD.pmtrwf
    elif wf_type is WfType.mcrd: return h5in.root.   pmtrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")

def get_sipm_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD.sipmrwf
    elif wf_type is WfType.mcrd: return h5in.root.   sipmrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")


def get_trigger_info(h5in):
    group            = h5in.root.Trigger if "Trigger" in h5in.root else ()
    trigger_type     = group.trigger if "trigger" in group else repeat(None)
    trigger_channels = group.events  if "events"  in group else repeat(None)
    return trigger_type, trigger_channels


def get_event_info(h5in):
    return h5in.root.Run.events


def get_number_of_active_pmts(detector_db, run_number):
    datapmt = load_db.DataPMT(detector_db, run_number)
    return np.count_nonzero(datapmt.Active.values.astype(bool))


def check_nonempty_indices(s1_indices, s2_indices):
    return s1_indices.size and s2_indices.size


def check_empty_pmap(pmap):
    return bool(pmap.s1s) or bool(pmap.s2s)


def length_of(iterable):
    if   isinstance(iterable, tb.table.Table  ): return iterable.nrows
    elif isinstance(iterable, tb.earray.EArray): return iterable.shape[0]
    elif isinstance(iterable, np.ndarray      ): return iterable.shape[0]
    elif isinstance(iterable, NoneType        ): return None
    elif isinstance(iterable, Iterator        ): return None
    elif isinstance(iterable, Sequence        ): return len(iterable)
    elif isinstance(iterable, Mapping         ): return len(iterable)
    else:
        raise TypeError(f"Cannot determine size of type {type(iterable)}")


def check_lengths(*iterables):
    lengths  = map(length_of, iterables)
    nonnones = filter(lambda x: x is not None, lengths)
    if np.any(np.diff(list(nonnones)) != 0):
        raise InvalidInputFileStructure("Input data tables have different sizes")


@check_annotations
def mcsensors_from_file(paths     : List[str],
                        db_file   :      str ,
                        run_number:      int ,
                        rate      :    float ) -> Generator:
    """
    Loads the nexus MC sensor information into
    a pandas DataFrame using the IC function
    load_mcsensor_response_df.
    Returns info event by event as a
    generator in the structure expected by
    the dataflow.

    paths      : List of strings
                 List of input file names to be read
    db_file    : string
                 Name of detector database to be used
    run_number : int
                 Run number for database
    rate       : float
                 Rate value in base unit (ns^-1) to generate timestamps
    """

    timestamp = create_timestamp(rate)

    pmt_ids  = load_db.DataPMT(db_file, run_number).SensorID

    for file_name in paths:
        sns_resp = mcinfo_io.load_mcsensor_response_df(file_name              ,
                                                       return_raw = False     ,
                                                       db_file    = db_file   ,
                                                       run_no     = run_number)

        for evt in mcinfo_io.get_event_numbers_in_file(file_name):

            try:
                ## Assumes two types of sensor, all non pmt
                ## assumed to be sipms. NEW, NEXT100 and DEMOPP safe
                ## Flex with this structure too.
                pmt_indx  = sns_resp.loc[evt].index.isin(pmt_ids)
                pmt_resp  = sns_resp.loc[evt][ pmt_indx]
                sipm_resp = sns_resp.loc[evt][~pmt_indx]
            except KeyError:
                pmt_resp = sipm_resp = pd.DataFrame(columns=sns_resp.columns)

            yield dict(event_number = evt      ,
                       timestamp    = timestamp(evt),
                       pmt_resp     = pmt_resp ,
                       sipm_resp    = sipm_resp)


def wf_from_files(paths, wf_type):
    for path in paths:
        with tb.open_file(path, "r") as h5in:
            try:
                event_info  = get_event_info  (h5in)
                run_number  = get_run_number  (h5in)
                pmt_wfs     = get_pmt_wfs     (h5in, wf_type)
                sipm_wfs    = get_sipm_wfs    (h5in, wf_type)
                (trg_type ,
                 trg_chann) = get_trigger_info(h5in)
            except tb.exceptions.NoSuchNodeError:
                continue

            check_lengths(pmt_wfs, sipm_wfs, event_info, trg_type, trg_chann)

            for pmt, sipm, evtinfo, trtype, trchann in zip(pmt_wfs, sipm_wfs, event_info, trg_type, trg_chann):
                event_number, timestamp         = evtinfo.fetch_all_fields()
                if trtype  is not None: trtype  = trtype .fetch_all_fields()[0]

                yield dict(pmt=pmt, sipm=sipm, run_number=run_number,
                           event_number=event_number, timestamp=timestamp,
                           trigger_type=trtype, trigger_channels=trchann)


def pmap_from_files(paths):
    for path in paths:
        try:
            pmaps = load_pmaps(path)
        except tb.exceptions.NoSuchNodeError:
            continue

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
            except tb.exceptions.NoSuchNodeError:
                continue
            except IndexError:
                continue

            check_lengths(event_info, pmaps)

            for evtinfo in event_info:
                event_number, timestamp = evtinfo.fetch_all_fields()
                yield dict(pmap=pmaps[event_number], run_number=run_number,
                           event_number=event_number, timestamp=timestamp)


@check_annotations
def hits_and_kdst_from_files( paths : List[str]
                            , group : str
                            , node  : str ) -> Iterator[Dict[str, Union[pd.DataFrame, int, float]]]:
    """
        Reader of hits files. Yields a dictionary with
        - hits         : a DataFrame with hits
        - kdst         : a DataFrame with a pointlike event summary
        - run_number   : the run number
        - event_number : the event number
        - timestamp    : the timestamp of the event
    """
    for path in paths:
        try:
            hits_df = load_dst (path, group, node)
            kdst_df = load_dst (path, 'DST' , 'Events')
        except tb.exceptions.NoSuchNodeError as e:
            print(f"Error on file {path}:")
            print(e)
            print("----------------------")
            continue

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue

            check_lengths(event_info, hits_df.event.unique())

            for evtinfo in event_info:
                event_number, timestamp = evtinfo.fetch_all_fields()
                this_event = lambda df: df.event == event_number
                yield dict(hits         = hits_df.loc[this_event],
                           kdst         = kdst_df.loc[this_event],
                           run_number   = run_number,
                           event_number = event_number,
                           timestamp    = timestamp)


@check_annotations
def dst_from_files(paths: List[str], group: str, node:str) -> Iterator[Dict[str,Union[pd.DataFrame, int, np.ndarray]]]:
    """
    Reader for a generic dst.
    """
    for path in paths:
        try:
            df = load_dst(path, group, node)
            with tb.open_file(path, "r") as h5in:
                run_number  = get_run_number(h5in)
                evt_numbers = get_event_info(h5in).col("evt_number")
                timestamps  = get_event_info(h5in).col("timestamp")
        except (tb.exceptions.NoSuchNodeError, IndexError):
            continue

        yield dict( dst           = df
                  , run_number    = run_number
                  , event_numbers = evt_numbers
                  , timestamps    = timestamps
                  )


@check_annotations
def MC_hits_from_files(files_in : List[str], rate: float) -> Generator:
    timestamp = create_timestamp(rate)
    for filename in files_in:
        try:
            hits_df = load_mchits_df(filename)
        except tb.exceptions.NoSuchNodeError:
            continue
        for evt, hits in hits_df.groupby(level=0):
            yield dict(event_number = evt,
                       x            = hits.x     .values,
                       y            = hits.y     .values,
                       z            = hits.z     .values,
                       energy       = hits.energy.values,
                       time         = hits.time  .values,
                       label        = hits.label .values,
                       timestamp    = timestamp(evt))


def sensor_data(path, wf_type):
    with tb.open_file(path, "r") as h5in:
        if   wf_type is WfType.rwf :   (pmt_wfs, sipm_wfs) = (h5in.root.RD .pmtrwf,   h5in.root.RD .sipmrwf)
        elif wf_type is WfType.mcrd:   (pmt_wfs, sipm_wfs) = (h5in.root.    pmtrd ,   h5in.root.    sipmrd )
        else                       :   raise TypeError(f"Invalid WfType: {type(wf_type)}")
        _, NPMT ,  PMTWL =  pmt_wfs.shape
        _, NSIPM, SIPMWL = sipm_wfs.shape
        return SensorData(NPMT=NPMT, PMTWL=PMTWL, NSIPM=NSIPM, SIPMWL=SIPMWL)

####### Transformers ########

def build_pmap(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
               s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
               s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2):
    s1_params = dict(time        = minmax(min = s1_tmin,
                                          max = s1_tmax),
                    length       = minmax(min = s1_lmin,
                                          max = s1_lmax),
                    stride       = s1_stride,
                    rebin_stride = s1_rebin_stride)

    s2_params = dict(time        = minmax(min = s2_tmin,
                                          max = s2_tmax),
                    length       = minmax(min = s2_lmin,
                                          max = s2_lmax),
                    stride       = s2_stride,
                    rebin_stride = s2_rebin_stride)

    datapmt = load_db.DataPMT(detector_db, run_number)
    pmt_ids = datapmt.SensorID[datapmt.Active.astype(bool)].values

    def build_pmap(ccwf, s1_indx, s2_indx, sipmzs): # -> PMap
        return pkf.get_pmap(ccwf, s1_indx, s2_indx, sipmzs,
                            s1_params, s2_params, thr_sipm_s2, pmt_ids,
                            pmt_samp_wid, sipm_samp_wid)

    return build_pmap


def calibrate_pmts(dbfile, run_number, n_maw, thr_maw):
    DataPMT    = load_db.DataPMT(dbfile, run_number = run_number)
    adc_to_pes = np.abs(DataPMT.adc_to_pes.values)
    adc_to_pes = adc_to_pes[adc_to_pes > 0]

    def calibrate_pmts(cwf):# -> CCwfs:
        return csf.calibrate_pmts(cwf,
                                  adc_to_pes = adc_to_pes,
                                  n_maw      = n_maw,
                                  thr_maw    = thr_maw)
    return calibrate_pmts


def calibrate_sipms(dbfile, run_number, thr_sipm):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)

    def calibrate_sipms(rwf):
        return csf.calibrate_sipms(rwf,
                                   adc_to_pes = adc_to_pes,
                                   thr        = thr_sipm,
                                   bls_mode   = BlsMode.mode)

    return calibrate_sipms


def calibrate_with_mean(dbfile, run_number):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)
    def calibrate_with_mean(wfs):
        return csf.subtract_baseline_and_calibrate(wfs, adc_to_pes)
    return calibrate_with_mean

def calibrate_with_maw(dbfile, run_number, n_maw_sipm):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)
    def calibrate_with_maw(wfs):
        return csf.subtract_baseline_maw_and_calibrate(wfs, adc_to_pes, n_maw_sipm)
    return calibrate_with_maw


def zero_suppress_wfs(thr_csum_s1, thr_csum_s2):
    def ccwfs_to_zs(ccwf_sum, ccwf_sum_maw):
        return (pkf.indices_and_wf_above_threshold(ccwf_sum_maw, thr_csum_s1).indices,
                pkf.indices_and_wf_above_threshold(ccwf_sum    , thr_csum_s2).indices)
    return ccwfs_to_zs


def compute_pe_resolution(rms, adc_to_pes):
    return np.divide(rms                              ,
                     adc_to_pes                       ,
                     out   = np.zeros_like(adc_to_pes),
                     where = adc_to_pes != 0          )


def simulate_sipm_response(detector, run_number, wf_length, noise_cut, filter_padding):
    datasipm      = load_db.DataSiPM (detector, run_number)
    baselines     = load_db.SiPMNoise(detector, run_number)[-1]
    noise_sampler = NoiseSampler(detector, run_number, wf_length, True)

    adc_to_pes    = datasipm.adc_to_pes.values
    thresholds    = noise_cut * adc_to_pes + baselines
    single_pe_rms = datasipm.Sigma.values.astype(np.double)
    pe_resolution = compute_pe_resolution(single_pe_rms, adc_to_pes)

    def simulate_sipm_response(sipmrd):
        wfs = sf.simulate_sipm_response(sipmrd, noise_sampler, adc_to_pes, pe_resolution)
        return wfm.noise_suppression(wfs, thresholds, filter_padding)
    return simulate_sipm_response


####### Filters ########

def peak_classifier(**params):
    selector = S12Selector(**params)
    return partial(pmap_filter, selector)


def compute_xy_position(dbfile, run_number, algo, **reco_params):
    if algo is XYReco.corona:
        datasipm    = load_db.DataSiPM(dbfile, run_number)
        reco_params = dict(all_sipms = datasipm, **reco_params)
        algorithm   = corona
    else:
        algorithm   = barycenter

    def compute_xy_position(xys, qs):
        return algorithm(xys, qs, **reco_params)

    return compute_xy_position


def compute_z_and_dt(t_s2, t_s1, drift_v):
    dt  = t_s2 - np.array(t_s1)
    z   = dt * drift_v
    dt *= units.ns / units.mus
    return z, dt


def build_pointlike_event(dbfile, run_number, drift_v,
                          reco, charge_type):
    datasipm   = load_db.DataSiPM(dbfile, run_number)
    sipm_xs    = datasipm.X.values
    sipm_ys    = datasipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)

    sipm_noise = NoiseSampler(dbfile, run_number).signal_to_noise

    def build_pointlike_event(pmap, selector_output, event_number, timestamp):
        evt = KrEvent(event_number, timestamp * 1e-3)

        evt.nS1 = 0
        for passed, peak in zip(selector_output.s1_peaks, pmap.s1s):
            if not passed: continue

            evt.nS1 += 1
            evt.S1w.append(peak.width)
            evt.S1h.append(peak.height)
            evt.S1e.append(peak.total_energy)
            evt.S1t.append(peak.time_at_max_energy)

        evt.nS2 = 0

        for passed, peak in zip(selector_output.s2_peaks, pmap.s2s):
            if not passed: continue

            evt.nS2 += 1
            evt.S2w.append(peak.width / units.mus)
            evt.S2h.append(peak.height)
            evt.S2e.append(peak.total_energy)
            evt.S2t.append(peak.time_at_max_energy)

            xys = sipm_xys[peak.sipms.ids           ]
            qs  = peak.sipm_charge_array(sipm_noise, charge_type,
                                         single_point = True)
            try:
                clusters = reco(xys, qs)
            except XYRecoFail:
                c    = NNN()
                Z    = tuple(NN for _ in range(0, evt.nS1))
                DT   = tuple(NN for _ in range(0, evt.nS1))
                Zrms = NN
            else:
                c = clusters[0]
                Z, DT = compute_z_and_dt(evt.S2t[-1], evt.S1t, drift_v)
                Zrms  = peak.rms / units.mus

            evt.Nsipm.append(c.nsipm)
            evt.S2q  .append(c.Q)
            evt.X    .append(c.X)
            evt.Y    .append(c.Y)
            evt.Xrms .append(c.Xrms)
            evt.Yrms .append(c.Yrms)
            evt.R    .append(c.R)
            evt.Phi  .append(c.Phi)
            evt.DT   .append(DT)
            evt.Z    .append(Z)
            evt.Zrms .append(Zrms)
            evt.qmax .append(max(qs))

        return evt

    return build_pointlike_event


def get_s1_time(pmap, selector_output):
    # in order to compute z one needs to define one S1
    # for time reference. By default the filter will only
    # take events with exactly one s1. Otherwise, the
    # convention is to take the first peak in the S1 object
    # as reference.
    if np.any(selector_output.s1_peaks):
        first_s1 = np.where(selector_output.s1_peaks)[0][0]
        s1_t     = pmap.s1s[first_s1].time_at_max_energy
    else:
        first_s2 = np.where(selector_output.s2_peaks)[0][0]
        s1_t     = pmap.s2s[first_s2].times[0]

    return s1_t


def try_global_reco(reco, xys, qs):
    try              : cluster = reco(xys, qs)[0]
    except XYRecoFail: return xy.empty()
    else             : return xy(cluster.X, cluster.Y)


def sipm_positions(dbfile, run_number):
    datasipm = load_db.DataSiPM(dbfile, run_number)
    sipm_xs  = datasipm.X.values
    sipm_ys  = datasipm.Y.values
    sipm_xys = np.stack((sipm_xs, sipm_ys), axis=1)
    return sipm_xys


def hit_builder(dbfile, run_number, drift_v,
                rebin_slices, rebin_method,
                global_reco, slice_reco,
                charge_type):
    sipm_xys   = sipm_positions(dbfile, run_number)
    sipm_noise = NoiseSampler(dbfile, run_number).signal_to_noise

    def build_hits(pmap, selector_output, event_number, timestamp):
        event_number = np.int64(event_number)
        hits = []
        hitc = HitCollection(event_number, timestamp * 1e-3)
        s1_t = get_s1_time(pmap, selector_output)

        # here hits are computed for each peak and each slice.
        # In case of an exception, a hit is still created with a NN cluster.
        # (NN cluster is a cluster where the energy is an IC not number NN)
        # this allows to keep track of the energy associated to non reonstructed hits.
        for peak_no, (passed, peak) in enumerate(zip(selector_output.s2_peaks,
                                                     pmap.s2s)):
            if not passed: continue

            peak = pmf.rebin_peak(peak, rebin_slices, rebin_method)

            xys  = sipm_xys[peak.sipms.ids]
            qs   = peak.sipm_charge_array(sipm_noise, charge_type,
                                          single_point = True)
            xy_peak = try_global_reco(global_reco, xys, qs)

            sipm_charge = peak.sipm_charge_array(sipm_noise        ,
                                                 charge_type       ,
                                                 single_point=False)

            df_peak = pd.DataFrame(dict( event = event_number
                                       , time  = timestamp
                                       , npeak = peak_no
                                       , Xpeak = xy_peak[0]
                                       , Ypeak = xy_peak[1]
                                       ), index=[0])
            for slice_no, (t_slice, qs) in enumerate(zip(peak.times ,
                                                         sipm_charge)):
                z_slice = (t_slice - s1_t) * units.ns * drift_v
                e_slice = peak.pmts .sum_over_sensors[slice_no]
                try:
                    xys      = sipm_xys[peak.sipms.ids]
                    clusters = slice_reco(xys, qs)
                    qs       = [c.Q for c in clusters]
                    es       = [q/sum(qs) * e_slice for q in qs]
                    for c, e in zip(clusters, es):
                        hit = df_peak.assign( nsipm = c.nsipm
                                            , X     = c.X
                                            , Y     = c.Y
                                            , Xrms  = c.Xrms
                                            , Yrms  = c.Yrms
                                            , Z     = z_slice
                                            , E     = e
                                            , Q     = c.Q
                                            , Qc    = c.Qc)
                        hits.append(hit)
                except XYRecoFail:
                    hit = df_peak.assign( nsipm = 0
                                        , X     = NN
                                        , Y     = NN
                                        , Xrms  = 0
                                        , Yrms  = 0
                                        , Z     = z_slice
                                        , E     = e_slice
                                        , Q     = NN
                                        , Qc    = -1)

                    hits.append(hit)

        hits = pd.concat(hits, ignore_index=True)
        return hits
    return build_hits


def sipms_as_hits(dbfile, run_number, drift_v,
                  rebin_slices, rebin_method,
                  q_thr,
                  global_reco, charge_type):
    sipm_xys   = sipm_positions(dbfile, run_number)
    sipm_noise = NoiseSampler(dbfile, run_number).signal_to_noise
    epsilon    = np.finfo(np.float64).eps

    def build_hits(pmap, selector_output, event_number, timestamp):
        s1_t = get_s1_time(pmap, selector_output)

        hits = []
        for peak_no, (passed, peak) in enumerate(zip(selector_output.s2_peaks,
                                                     pmap.s2s)):
            if not passed: continue

            peak = pmf.rebin_peak(peak, rebin_slices, rebin_method)

            xys  = sipm_xys[peak.sipms.ids]
            qs   = peak.sipm_charge_array(sipm_noise, charge_type,
                                          single_point = True)

            peak_x, peak_y = try_global_reco(global_reco, xys, qs).XY

            sipm_charge = peak.sipm_charge_array(sipm_noise        ,
                                                 charge_type       ,
                                                 single_point=False)

            slice_zs = (peak.times - s1_t) * units.ns * drift_v
            slice_es = peak.pmts.sum_over_sensors
            xys      = sipm_xys[peak.sipms.ids]

            for (slice_z, slice_e, sipm_qs) in zip(slice_zs, slice_es, sipm_charge):
                over_thr = sipm_qs >= q_thr
                if np.any(over_thr):
                    sipm_qs  = sipm_qs[over_thr]
                    sipm_xy  = xys[over_thr]
                    sipm_es  = slice_e * sipm_qs / (np.sum(sipm_qs) + epsilon)

                    for q, e, (x, y) in zip(sipm_qs, sipm_es, sipm_xy):
                        hit = dict( npeak = peak_no
                                  , Xpeak = peak_x
                                  , Ypeak = peak_y
                                  , X     = x
                                  , Y     = y
                                  , Z     = slice_z
                                  , Q     = q
                                  , E     = e)
                        hits.append(pd.DataFrame(hit, index=[0]))

                else:
                    hit = hif.make_nn_hit( peak_no, peak_x, peak_y
                                         , slice_z, slice_e, np.nan)
                    hits.append(hit)

        hits = pd.concat(hits, ignore_index=True)
        hits = hits.assign( event    = event_number
                          , time     = timestamp * 1e-3
                          , nsipm    = 1
                          , Xrms     = 0.
                          , Yrms     = 0.
                          , track_id = -1
                          , Qc       = -1.
                          , Ep       = -1.)
        return hits

    return build_hits


def waveform_binner(bins):
    def bin_waveforms(wfs):
        return cf.bin_waveforms(wfs, bins)
    return bin_waveforms


def waveform_integrator(limits):
    def integrate_wfs(wfs):
        return cf.spaced_integrals(wfs, limits)[:, ::2]
    return integrate_wfs


# Compound components
def compute_and_write_pmaps(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                  s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                  s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2,
                  h5out, sipm_rwf_to_cal=None):

    # Filter events without signal over threshold
    indices_pass    = fl.map(check_nonempty_indices,
                             args = ("s1_indices", "s2_indices"),
                             out = "indices_pass")
    empty_indices   = fl.count_filter(bool, args = "indices_pass")

    # Build the PMap
    compute_pmap     = fl.map(build_pmap(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                                         s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                                         s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2),
                              args = ("ccwfs", "s1_indices", "s2_indices", "sipm"),
                              out  = "pmap")

    # Filter events with zero peaks
    pmaps_pass      = fl.map(check_empty_pmap, args = "pmap", out = "pmaps_pass")
    empty_pmaps     = fl.count_filter(bool, args = "pmaps_pass")

    # Define writers...
    write_pmap_         = pmap_writer        (h5out,              )
    write_indx_filter_  = event_filter_writer(h5out, "s12_indices")
    write_pmap_filter_  = event_filter_writer(h5out, "empty_pmap" )

    # ... and make them sinks
    write_pmap         = sink(write_pmap_        , args=(        "pmap", "event_number"))
    write_indx_filter  = sink(write_indx_filter_ , args=("event_number", "indices_pass"))
    write_pmap_filter  = sink(write_pmap_filter_ , args=("event_number",   "pmaps_pass"))

    fn_list = (indices_pass,
               fl.branch(write_indx_filter),
               empty_indices.filter,
               sipm_rwf_to_cal,
               compute_pmap,
               pmaps_pass,
               fl.branch(write_pmap_filter),
               empty_pmaps.filter,
               fl.branch(write_pmap))

    # Filter out simp_rwf_to_cal if it is not set
    compute_pmaps = pipe(*filter(None, fn_list))

    return compute_pmaps, empty_indices, empty_pmaps


@check_annotations
def check_max_time(max_time: float, buffer_length: float) -> Union[int, float]:
    """
    `max_time` must be greater than `buffer_length`. If not, raise warning
        and set `max_time` == `buffer_length`.

    :param max_time: Maximal length of the event that will be taken into
        account starting from the first detected signal, all signals after
        that are simply lost.
    :param buffer_length: Length of buffers.
    :return: `max_time` if `max_time` >= `buffer_length`, else `buffer_length`.
    """
    if max_time % units.mus:
        message = "Invalid value for max_time, it has to be a multiple of 1 mus"
        raise ValueError(message)

    if max_time < buffer_length:
        warnings.warn("`max_time` shorter than `buffer_length`, "
                      "setting `max_time` to `buffer_length`",
                      stacklevel=2)
        return buffer_length
    else:
        return max_time


@check_annotations
def calculate_and_save_buffers(buffer_length    : float        ,
                               max_time         : float        ,
                               pre_trigger      : float        ,
                               pmt_wid          : float        ,
                               sipm_wid         : float        ,
                               trigger_threshold: int          ,
                               h5out            : tb.File      ,
                               run_number       : int          ,
                               npmt             : int          ,
                               nsipm            : int          ,
                               nsamp_pmt        : int          ,
                               nsamp_sipm       : int          ,
                               order_sensors    : Union[NoneType, Callable]):
    find_signal       = fl.map(signal_finder(buffer_length, pmt_wid,
                                             trigger_threshold     ),
                               args = "pmt_bin_wfs"                 ,
                               out  = "pulses"                      )

    filter_events_signal = fl.map(lambda x: len(x) > 0,
                                  args= 'pulses',
                                  out = 'passed_signal')
    events_passed_signal = fl.count_filter(bool, args='passed_signal')
    write_signal_filter  = fl.sink(event_filter_writer(h5out, "signal"),
                                   args=('event_number', 'passed_signal'))

    event_times       = fl.map(trigger_times                             ,
                               args = ("pulses", "timestamp", "pmt_bins"),
                               out  = "evt_times"                        )

    calculate_buffers = fl.map(bf.buffer_calculator(buffer_length, pre_trigger,
                                                    pmt_wid     ,    sipm_wid),
                               args = ("pulses",
                                       "pmt_bins" ,  "pmt_bin_wfs",
                                       "sipm_bins", "sipm_bin_wfs")        ,
                               out  = "buffers"                            )

    saved_buffers = "buffers" if order_sensors is None else "ordered_buffers"
    max_subevt    =  math.ceil(max_time / buffer_length)
    buffer_writer_    = sink(buffer_writer( h5out
                                          , run_number = run_number
                                          , n_sens_eng = npmt
                                          , n_sens_trk = nsipm
                                          , length_eng = nsamp_pmt
                                          , length_trk = nsamp_sipm
                                          , max_subevt = max_subevt),
                             args = ("event_number", "evt_times"  ,
                                     saved_buffers                ))

    find_signal_and_write_buffers = ( find_signal
                                    , filter_events_signal
                                    , fl.branch(write_signal_filter)
                                    , events_passed_signal.filter
                                    , event_times
                                    , calculate_buffers
                                    , order_sensors
                                    , fl.branch(buffer_writer_))

    # Filter out order_sensors if it is not set
    buffer_definition = pipe(*filter(None, find_signal_and_write_buffers))
    return buffer_definition


@check_annotations
def Efield_copier(energy_type: HitEnergy):
    def copy_Efield(hitc : HitCollection) -> HitCollection:
        mod_hits = []
        for hit in hitc.hits:
            hit = Hit(hit.npeak,
                      Cluster(hit.Q, xy(hit.X, hit.Y), hit.var, hit.nsipm),
                      hit.Z,
                      hit.E,
                      xy(hit.Xpeak, hit.Ypeak),
                      s2_energy_c=getattr(hit, energy_type.value),
                      Ep=getattr(hit, energy_type.value))
            mod_hits.append(hit)
        mod_hitc = HitCollection(hitc.event, hitc.time, hits=mod_hits)
        return mod_hitc
    return copy_Efield


@check_annotations
def make_event_summary(event_number : int         ,
                       tracks       : pd.DataFrame,
                       hits         : pd.DataFrame,
                       out_of_map   : bool
                       ) -> pd.DataFrame:
    """
    For a given event number, timestamp, topology info dataframe, paolina hits and kdst information returns a
    dataframe with the whole event summary.

    Parameters
    ----------
    event_number  : int
    tracks        : DataFrame
        Dataframe containing track information,
        output of track_blob_info_creator_extractor
    hits          : DataFrame
        Hits table passed through paolina functions,
        output of track_blob_info_creator_extractor
    out_of_map    : bool
        Whether there are out_of_map hits


    Returns
    ----------
    DataFrame containing relevant per event information.
    """
    es = pd.DataFrame(columns=list(types_dict_summary.keys()))
    if len(hits) == 0: return es

    ntrks = len(tracks)
    nhits = len(hits)

    S2ec = hits.Ec.sum()
    S2qc = -1 #not implemented yet

    x, y, z, e = hits.loc[:, "X Y Z Ec".split()].values.T
    r          = np.sqrt(x**2 + y**2)
    ave        = lambda arg: np.average(arg, weights=e, axis=0)

    list_of_vars  = [event_number, S2ec, S2qc, ntrks, nhits,
                     ave(x), ave(y), ave(z), ave(r),
                     min(x), min(y), min(z), min(r),
                     max(x), max(y), max(z), max(r),
                     out_of_map]

    es.loc[0] = list_of_vars
    #change dtype of columns to match type of variables
    es = es.apply(lambda x : x.astype(types_dict_summary[x.name]))
    return es



def track_writer(h5out):
    """
    For a given open table returns a writer for topology info dataframe
    """
    def write_tracks(df):
        return df_writer(h5out              = h5out              ,
                         df                 = df                 ,
                         group_name         = 'Tracking'         ,
                         table_name         = 'Tracks'           ,
                         descriptive_string = 'Track information',
                         columns_to_index   = ['event']          )
    return write_tracks


def summary_writer(h5out):
    """
    For a given open table returns a writer for summary info dataframe
    """
    def write_summary(df):
        return df_writer(h5out              = h5out                      ,
                         df                 = df                         ,
                         group_name         = 'Summary'                  ,
                         table_name         = 'Events'                   ,
                         descriptive_string = 'Event summary information',
                         columns_to_index   = ['event']                  )
    return write_summary


@check_annotations
def track_blob_info_creator_extractor(vox_size         : Tuple[float, float, float],
                                      strict_vox_size  : bool                      ,
                                      energy_threshold : float                     ,
                                      min_voxels       : int                       ,
                                      blob_radius      : float                     ,
                                      max_num_hits     : int
                                     ) -> Callable:
    """
    For a given paolina parameters returns a function that extract tracks / blob information from a HitCollection.

    Parameters
    ----------
    vox_size         : [float, float, float]
        (maximum) size of voxels for track reconstruction
    strict_vox_size  : bool
        if `False` allows per event adaptive voxel size,
        smaller of equal thatn vox_size
    energy_threshold : float
        if energy of end-point voxel is smaller
        the voxel will be dropped and energy redistributed to the neighbours
    min_voxels       : int
        after `min_voxels` number of voxels is reached no dropping will happen.
    blob_radius      : float
        radius of blob

    Returns
    ----------
    A function that from a given DataFrame returns another DataFrame with per track information.
    """
    def create_extract_track_blob_info(hits : pd.DataFrame) -> pd.DataFrame:
        tracks_df = pd.DataFrame(columns=list(types_dict_tracks.keys()))
        if len(hits) > max_num_hits:
            return tracks_df, hits, True

        hits = hits.assign(Ep = hits.Ec, out_of_map = hits.Ec.isna(), track=-1)
        out_of_map = hits.out_of_map.any()

        if len(hits) > 0 and (hits.Ep > 0).any():
            hits, voxels, vox_size = plf.voxelize_hits(hits, vox_size, strict_vox_size, HitEnergy.Ep)
            hits, voxels, dropped_hits, dropped_voxels = plf.drop_end_point_voxels(voxels, energy_threshold, min_voxels)

            hits = pd.concat([hits, dropped_hits])

            tracks = plf.make_track_graphs(voxels)
            tracks = sorted(tracks, key=plf.get_track_energy, reverse=True)

            numb_of_tracks = len(tracks)
            for track_id, track in enumerate(tracks):
                track_energy   = plf.get_track_energy(track)
                hits_in_track  = hits.loc[hits.voxel.in1d(t.nodes())]
                numb_of_hits   = len(hits_in_track)
                numb_of_voxels = len(track.nodes())
                x, y, z, e = hits_in_track.loc[:, "X Y Z Ep".split()].values.T
                r          = np.sqrt(x**2 + y**2)
                ave         = lambda arg: np.average(arg, weights=e, axis=0)

                distances            = plf.shortest_paths(t)
                extr1, extr2, length = plf.find_extrema(distances)
                extr1_pos = voxels.loc[extr1, list("XYZ")]
                extr2_pos = voxels.loc[extr2, list("XYZ")]

                blob1, blob2 = plf.find_blobs(hits, voxels, vox_size, track, blob_radius, HitEnergy.Ep)
                (E1, pos1, hits1, _) = blob1
                (E2, pos2, hits2, _) = blob2

                overlap = hits1.index.in1d(hits2.index).any()
                list_of_vars = [event, track_id, track_energy, length,
                                numb_of_voxels, numb_of_hits, numb_of_tracks,
                                min(x), min(y), min(z), min(r),
                                max(x), max(y), max(z), max(r),
                                ave(x), ave(y), ave(z), ave(r),
                                *extr1_pos, *extr2_pos,
                                *pos1, *pos2, E1, E2,
                                overlap, *vox_size]

                tracks_df.loc[track_id] = list_of_vars

                hits.loc[hits_in_track.index, "track"] = track_id

            #change dtype of columns to match type of variables
            tracks_df = tracks_df.apply(lambda x : x.astype(types_dict_tracks[x.name]))
        return tracks_df, hits, out_of_map

    return create_extract_track_blob_info


@check_annotations
def compute_and_write_tracks_info( paolina_params : Dict[str, Any]
                                 , h5out          : tb.File
                                 , hit_type       : HitEnergy
                                 , write_hits     : Optional[Generator] = None):

    enough_hits = fl.map( lambda x : len(x) > 0
                        , args = 'hits'
                        , out  = 'enough_hits')
    hits_passed = fl.count_filter(bool, args="enough_hits")

    # Create tracks and compute topology-related information
    create_extract_track_blob_info = fl.map(track_blob_info_creator_extractor(**paolina_params),
                                            args = 'hits',
                                            out  = ('tracks', 'hits', 'out_of_map'))

    # Filter empty topology events
    filter_events_topology         = fl.map(lambda x : len(x) > 0,
                                            args = 'tracks',
                                            out  = 'topology_passed')
    events_passed_topology         = fl.count_filter(bool, args="topology_passed")

    # Create table with summary information
    make_final_summary             = fl.map(make_event_summary,
                                            args = ('event_number', 'tracks', 'hits', 'out_of_map'),
                                            out  = 'event_info')


    # Define writers and make them sinks
    write_tracks          = fl.sink(   track_writer     (h5out=h5out)             , args="topology_info")
    write_summary         = fl.sink( summary_writer     (h5out=h5out)             , args="event_info"   )
    write_topology_filter = fl.sink( event_filter_writer(h5out, "topology_select"), args=("event_number", "topology_passed"))
    write_no_hits_filter  = fl.sink( event_filter_writer(h5out,     "hits_select"), args=("event_number",     "enough_hits"))


    make_and_write_summary  = make_final_summary, write_summary
    select_and_write_tracks = events_passed_topology.filter, write_tracks

    fork_pipes = filter(None, ( make_and_write_summary
                              , write_topology_filter
                              , write_hits
                              , select_and_write_tracks))

    return pipe( enough_hits
               , fl.branch(write_no_hits_filter)
               , hits_passed.filter
               , create_extract_track_blob_info
               , filter_events_topology
               , fl.branch(fl.fork(*fork_pipes))
               )


@check_annotations
def hits_thresholder(threshold_charge : float, same_peak : bool) -> Callable:
    """
    Applies a threshold to hits and redistributes the charge/energy.

    Parameters
    ----------
    threshold_charge : float
        minimum charge of a hit
    same_peak        : bool
        whether to reassign NN hits' energy only to the hits in the same peak

    Returns
    ----------
    A function that takes DataFrame as input and returns another one with
    only non NN hits of charge above `threshold_charge`.
    The energy of NN hits is redistributed among neighbors.
    """

    def threshold_hits(hits: pd.DataFrame) -> pd.DataFrame:
        hits = hif.threshold_hits(hits, threshold_charge)
        hits = hif.merge_NN_hits (hits,        same_peak)
        return hits

    return threshold_hits
