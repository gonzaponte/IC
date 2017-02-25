"""Defines a class for accessing light detection probabilities."""

from __future__ import print_function, division, absolute_import

import math
import numpy as np

import invisible_cities.database.load_db as DB


class LightParametrization(LT.LightTables):
    def __init__(self, version=1, max_range_sipm=20.):
        self.version = version
        self.max_range_sipm = max_range_sipm
        self._load()

    def SiPM(self, x, y):
        closest = self.closest_sipms[int(x), int(y)]
        p = np.zeros(1792)
        dx = x - self.sens.X[self.sens.SensorID==closest].values
        dy = y - self.sens.Y[self.sens.SensorID==closest].values
        bins = ((dx**2 + dy**2)**0.5 // self.sipm_pitch).astype(int)
        p[closest] = self._sipm[bins]
        return p

    def _load(self):
        self.sens = DB.DataSiPM()
        self.geo  = DB.DataDetector()
        self. pmt_pitch, self._pmt  = DB.TablePMT("ParamPMT")
        self.sipm_pitch, self._sipm = DB.TableSiPM()

        self.xmin, self.xmax, self.ymin, self.ymax = DB.load_detector_dims()
        self.pmt_pitch,  self._pmt  = DB.load_light_table_pmt()
        self.sipm_pitch, self._sipm = DB.load_light_table_sipm()

        self.closest_sipms = {}
        for x in range(int(-self.xmin), int(self.xmax)):
            for y in range(int(-self.ymin), int(self.ymax)):
                dx = x - self.SiPMdb.X
                dy = y - self.SiPMdb.Y
                dr = (dx**2 + dy**2)**0.5
                closest = dr <= self.max_range_sipm
                self.closest_sipms[x, y] = self.SiPMdb.SensorID.values[closest]
