import sqlite3
import numpy as np
import pandas as pd
import os
from operator import itemgetter

from .. evm.ic_containers import SensorList

DATABASE_LOCATION =  os.environ['ICTDIR'] + '/invisible_cities/database/localdb.sqlite3'

def tmap(*args):
    return tuple(map(*args))

# Run to take always the same calibration constant, etc for MC files
# 3012 was the first SiPM calibration after remapping.
runNumberForMC = 3012

def get_min_run_values(db, run_number, sensors):
    ''' Due to ART database design MaxRun is not being used currently,
    so we need to get the exact MinRun value for a given run number to
    get only one row per sensor...
    db is sqlite3 connection and sensors pmt or sipm.'''

    if sensors == 'pmt':  bound = '<' #pmt id's are < 100
    if sensors == 'sipm': bound = '>' #sipms id's are > 100

    sql = '''select Max(MinRun) from ChannelGain
where SensorID {} 100 and MinRun < {}'''.format(bound, abs(run_number))
    cursor = db.execute(sql)
    minrun_gain = cursor.fetchone()[0]

    sql = '''select Max(MinRun) from ChannelPosition
where SensorID {} 100 and MinRun < {}'''.format(bound, abs(run_number))
    cursor = db.execute(sql)
    minrun_position = cursor.fetchone()[0]

    sql = '''select Max(MinRun) from ChannelMapping
where SensorID {} 100 and MinRun < {}'''.format(bound, abs(run_number))
    cursor = db.execute(sql)
    minrun_map = cursor.fetchone()[0]

    return minrun_gain, minrun_position, minrun_map


def DataPMT(run_number=1e5):
    if run_number == 0:
        run_number = runNumberForMC
    dbfile = DATABASE_LOCATION
    conn = sqlite3.connect(dbfile)

    minrun_gain, minrun_position, minrun_map = \
            get_min_run_values(conn, run_number, 'pmt')

    sql = '''select pos.SensorID, map.ElecID "ChannelID", Label "PmtID",
case when msk.SensorID is NULL then 1 else 0 end "Active",
X, Y, coeff_blr, coeff_c, abs(Centroid) "adc_to_pes", noise_rms, Sigma
from ChannelPosition as pos INNER JOIN ChannelMapping
as map ON pos.SensorID = map.SensorID LEFT JOIN
(select * from PmtNoiseRms where MinRun < {3} and (MaxRun >= {3} or MaxRun is NULL))
as noise on map.ElecID = noise.ElecID LEFT JOIN
(select * from ChannelMask where MinRun < {3} and {3} < MaxRun)
as msk ON pos.SensorID = msk.SensorID LEFT JOIN
(select * from ChannelGain where MinRun={1})
as gain ON pos.SensorID = gain.SensorID LEFT JOIN
(select * from PmtBlr where MinRun < {3} and (MaxRun >= {3} or MaxRun is NULL))
as blr ON map.ElecID = blr.ElecID
where pos.SensorID < 100 and pos.MinRun={0} and map.MinRun={2} and pos.Label LIKE 'PMT%'
order by Active desc, pos.SensorID'''\
    .format(minrun_position, minrun_gain, minrun_map, abs(run_number))
    data = pd.read_sql_query(sql, conn)
    data.fillna(0, inplace=True)
    conn.close()
    return data

def DataSiPM(run_number=1e5):
    if run_number == 0:
        run_number = runNumberForMC
    dbfile = DATABASE_LOCATION
    conn = sqlite3.connect(dbfile)

    minrun_gain, minrun_position, minrun_map = \
            get_min_run_values(conn, run_number, 'sipm')

    sql = '''select pos.SensorID, map.ElecID "ChannelID",
case when msk.SensorID is NULL then 1 else 0 end "Active",
X, Y, Centroid "adc_to_pes"
from ChannelPosition as pos INNER JOIN ChannelGain as gain
ON pos.SensorID = gain.SensorID INNER JOIN ChannelMapping as map
ON pos.SensorID = map.SensorID LEFT JOIN
(select * from ChannelMask where MinRun < {3} and {3} < MaxRun) as msk
ON pos.SensorID = msk.SensorID
where pos.SensorID > 100 and pos.MinRun={0} and gain.MinRun={1}
and map.MinRun={2} order by pos.SensorID'''\
    .format(minrun_position, minrun_gain, minrun_map, abs(run_number))
    data = pd.read_sql_query(sql, conn)
    conn.close()
    return data

def DetectorGeo():
    dbfile = DATABASE_LOCATION
    conn = sqlite3.connect(dbfile)
    sql = 'select * from DetectorGeo'
    data = pd.read_sql_query(sql, conn)
    conn.close()
    return data

def SiPMNoise(run_number=1e5, dbfile=DATABASE_LOCATION):
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sqlbaseline = '''select Energy from SipmBaseline
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by SensorID;'''.format(abs(run_number))
    cursor.execute(sqlbaseline)
    baselines = np.array(tmap(itemgetter(0), cursor.fetchall()))
    nsipms = baselines.shape[0]

    sqlnoisebins = '''select distinct(BinEnergyPes) from SipmNoisePDF
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by BinEnergyPes;'''.format(abs(run_number))
    cursor.execute(sqlnoisebins)
    noise_bins = np.array(tmap(itemgetter(0), cursor.fetchall()))
    nbins = noise_bins.shape[0]

    sqlnoise = '''select Probability from SipmNoisePDF
where MinRun <= {0} and (MaxRun >= {0} or MaxRun is NULL)
order by SensorID, BinEnergyPes;'''.format(abs(run_number))
    cursor.execute(sqlnoise)
    data = tmap(itemgetter(0), cursor.fetchall())
    noise = np.array(data).reshape(nsipms, nbins)

    return noise, noise_bins, baselines


def position_table(table_name = "ELPointsPosition"):
    dbfile = os.environ['ICTDIR'] + DATABASE_LOCATION
    conn   = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sql = "select posID, X, Y from {table_name};"
    cursor.execute(sql)
    pos_ID, X, Y = np.array(cursor.fetchall()).T

    x_pitch  = np.diff(X)[0]
    y_pitch  = np.diff(Y)[0]
    pos_dict = dict(zip(pos_ID, zip(X, Y)))
    return pos_dict, x_pitch, y_pitch


def PMT_light_table(table_name = "ELProductionCathode"):
    dbfile = os.environ['ICTDIR'] + DATABASE_LOCATION
    conn   = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sql      = "select * order by PosID from {table_name};"
    cursor.execute(sql)
    data     = np.array(cursor.fetchall()).T
    pos_ID   = data[1]
    probs    = data[3:]

    prob_dict = {}
    for i in np.unique(pos_ID):
        where = pos_ID == i
        prob_dict[i] = probs[where].sum(axis=0)
    return prob_dict


def SiPM_light_table(table_name = "ELProductionAnode"):
    dbfile = os.environ['ICTDIR'] + DATABASE_LOCATION
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sql   = "select * from {table_name};"
    data  = np.array(cursor.fetchall()).T
    cursor.execute(sql)
    pos_ID    = data[1]
    sensor_ID = data[2]
    probs     = data[3:]

    prob_dict = {}
    for i in np.unique(pos_ID):
        where = pos_ID == i
        prob_dict[i] = SensorList(sensor_ID[where],
                                  probs    [where].sum(axis=0))
    return prob_dict
