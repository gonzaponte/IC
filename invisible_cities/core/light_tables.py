"""
Defines a class for accessing light detection probabilities.

GML Feb 2017
"""

from __future__ import print_function, division, absolute_import

import math
import numpy as np

import invisible_cities.database.load_db as DB


class LightTables:
    """
    Interface for accessing light detection probabilities
    stored in tables.
    """
    def __init__(self):
        self._load()

    def PMT(self, x, y):
        xbin = int((x-self.xmin)//self.pmt_pitch)
        ybin = int((y-self.ymin)//self.pmt_pitch)
        return self._pmt[xbin, ybin]

    def SiPM(self, x, y):
        xbin = int((x-self.xmin)//self.pmt_pitch)
        ybin = int((y-self.ymin)//self.pmt_pitch)
        return self._sipm[xbin][ybin]

    def _load(self):
        self.xmin, self.xmax, self.ymin, self.ymax = DB.DataDetector()
        self. pmt_pitch, self._pmt  = DB.TablePMT()
        self.sipm_pitch, self._sipm = DB.TableSiPM()
