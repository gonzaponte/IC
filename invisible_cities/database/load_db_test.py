import numpy as np

from . import load_db as DB

import sqlite3
import numpy as np
from pytest  import fixture
from os.path import join

def test_pmts_pd():
    """Check that we retrieve the correct number of PMTs."""
    pmts = DB.DataPMT()
    columns =['SensorID', 'ChannelID', 'PmtID', 'Active', 'X', 'Y',
              'coeff_blr', 'coeff_c', 'adc_to_pes', 'noise_rms', 'Sigma']
    assert columns == list(pmts)
    assert pmts['PmtID'].str.startswith('PMT').all()
    assert pmts.shape[0] == 12

def test_pmts_MC_pd():
    """Check that we retrieve the correct number of PMTs."""
    mc_run = 0
    pmts = DB.DataPMT(mc_run)
    columns =['SensorID', 'ChannelID', 'PmtID', 'Active', 'X', 'Y',
              'coeff_blr', 'coeff_c', 'adc_to_pes', 'noise_rms', 'Sigma']
    assert columns == list(pmts)
    assert pmts['PmtID'].str.startswith('PMT').all()
    assert pmts.shape[0] == 12

def test_sipm_pd():
    """Check that we retrieve the correct number of SiPMs."""
    sipms = DB.DataSiPM()
    columns = ['SensorID', 'ChannelID', 'Active', 'X', 'Y', 'adc_to_pes']
    assert columns == list(sipms)
    assert sipms.shape[0] == 1792

def test_SiPMNoise():
    """Check we have noise for all SiPMs and energy of each bin."""
    noise, energy, baseline = DB.SiPMNoise()
    assert noise.shape[0] == baseline.shape[0]
    assert noise.shape[0] == 1792
    assert noise.shape[1] == energy.shape[0]


def test_DetectorGeometry():
    """Check Detector Geometry."""
    geo = DB.DetectorGeo()
    assert geo['XMIN'][0] == -198
    assert geo['XMAX'][0] ==  198
    assert geo['YMIN'][0] == -198
    assert geo['YMAX'][0] ==  198
    assert geo['ZMIN'][0] ==    0
    assert geo['ZMAX'][0] ==  532
    assert geo['RMAX'][0] ==  198


def test_mc_runs_equal_data_runs():
    assert (DB.DataPMT (-3550).values == DB.DataPMT (3550).values).all()
    assert (DB.DataSiPM(-3550).values == DB.DataSiPM(3550).values).all()


@fixture(scope='module')
def test_db(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp('output_files')
    dbfile = join(temp_dir, 'db.sqlite3')
    connSql3 = sqlite3.connect(dbfile)
    cursorSql3 = connSql3.cursor()

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmBaseline` (
`MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `Energy` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmNoisePDF` (
    `MinRun` integer NOT NULL
    ,  `MaxRun` integer DEFAULT NULL
    ,  `SensorID` integer NOT NULL
    ,  `BinEnergyPes` float NOT NULL
    ,  `Probability` float NOT NULL
);''')

    #Insert sample data
    sql = 'INSERT INTO SipmBaseline (MinRun, MaxRun, SensorID, Energy) VALUES ({})'
    cursorSql3.execute(sql.format('0,NULL,1,0'))
    sql = 'INSERT INTO SipmNoisePDF (MinRun, MaxRun, SensorID, BinEnergyPes, Probability) VALUES ({})'
    cursorSql3.execute(sql.format('0,NULL,1,5,0.1'))
    cursorSql3.execute(sql.format('0,NULL,1,3,0.3'))
    cursorSql3.execute(sql.format('0,NULL,1,4,0.2'))
    cursorSql3.execute(sql.format('0,NULL,1,1,0.5'))
    cursorSql3.execute(sql.format('0,NULL,1,2,0.4'))
    connSql3.commit()
    connSql3.close()

    noise_true     = np.array([[ 0.5,  0.4,  0.3,  0.2,  0.1]])
    bins_true      = np.array([ 1.,  2.,  3.,  4.,  5.])
    baselines_true = np.array([ 0.])
    sipm_noise = noise_true, bins_true, baselines_true

    return dbfile, sipm_noise


def test_sipm_noise_order(test_db):
    #Read from DB
    dbfile = test_db[0]
    noise, bins, baselines = DB.SiPMNoise(1, dbfile)

    #'True' values
    sipm_noise     = test_db[1]
    noise_true     = sipm_noise[0]
    bins_true      = sipm_noise[1]
    baselines_true = sipm_noise[2]

    np.testing.assert_allclose(noise,     noise_true)
    np.testing.assert_allclose(bins,      bins_true)
    np.testing.assert_allclose(baselines, baselines_true)


def test_position_table():
    _, x, y, z = DB.position_table()

    x = np.sort(np.unique(x))
    y = np.sort(np.unique(y))

    pitch_x = np.diff(x)
    pitch_y = np.diff(y)
    pitch_z = np.diff(z)

    assert np.all(pitch_x == pitch_x[0]) # fixed pitch
    assert np.all(pitch_y == pitch_y[0]) # fixed pitch
    assert np.all(pitch_z == 0         ) # fixed z


def test_pmt_light_table():
    data_pmt   = DB.pmt_light_table()
#    pos_ID     = DB. position_table()[0].astype(int)
    sens_ID    = DB.DataPMT(0).SensorID.values
    n_pmt      = sens_ID.size

#    pmt_pos_id  = np.unique(data_pmt[1 ].astype(int))
    pmt_sens_id = np.unique(data_pmt[2 ].astype(int))
    pmt_probs   =           data_pmt[3:]

    assert data_pmt.shape[1] % n_pmt == 0        # all pmts are present for each point
#    assert np.all(np.in1d(pmt_pos_id ,  pos_ID)) #  pos_ids are valid
    assert np.all(np.in1d(pmt_sens_id, sens_ID)) # sens_ids are valid
    assert np.all((pmt_probs >= 0   ) &
                  (pmt_probs <= 5e-5))           # probs in expected range


def test_sipm_light_table():
    data_sipm  = DB.sipm_light_table()
#    pos_ID     = DB.  position_table()[0].astype(int)
    sens_ID    = DB.DataSiPM(0).SensorID.values

#    sipm_pos_id  = np.unique(data_sipm[1 ].astype(int))
    sipm_sens_id = np.unique(data_sipm[2 ].astype(int))
    sipm_probs   =           data_sipm[3:]

#    assert np.all(np.in1d(sipm_pos_id ,  pos_ID)) #  pos_ids are valid
    assert np.all(np.in1d(sipm_sens_id, sens_ID)) # sens_ids are valid
    assert np.all((sipm_probs >= 0   ) &
                  (sipm_probs <= 2e-4))           # probs in expected range
