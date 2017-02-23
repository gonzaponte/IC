"""
An useful class for accessing light detection probabilities.
"""

import invisible_cities.database.load_db as DB


class LightTables:
    def __init__(self):
        self._load()

    def PMT(self, x, y):
        xbin = int((x - self.xmin) // self.pmt_pitch)
        ybin = int((y - self.ymin) // self.pmt_pitch)
        return self._pmt[xbin, ybin]

    def SiPM(self, x, y):
        xbin = int((x - self.xmin) // self.pmt_pitch)
        ybin = int((y - self.ymin) // self.pmt_pitch)
        return self._sipm[xbin, ybin]

    def _load(self):
        detector_geo = DB.DetectorGeo()

        self.xmin = detector_geo.XMIN[0]
        self.xmax = detector_geo.XMAX[0]
        self.ymin = detector_geo.YMIN[0]
        self.ymax = detector_geo.YMAX[0]

        self.pmt_pitch,  self._pmt  = DB. PMT_light_table()
        self.sipm_pitch, self._sipm = DB.SiPM_light_table()
