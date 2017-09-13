"""
An useful class for accessing light detection probabilities.
"""

cimport numpy as np
import numpy as np

from .. database import load_db as DB

from .. evm.ic_containers import SensorCollection


cdef class LightTables:
    def __init__(self, bool load_pmt_table  = True,
                       bool load_sipm_table = True):
        cdef np.ndarray[double, ndim=2]  pos_table
        cdef np.ndarray[double, ndim=2]  pmt_table
        cdef np.ndarray[double, ndim=2] sipm_table

        self.load_pmt_table  = load_pmt_table
        self.load_sipm_table = load_sipm_table

        pos_table  = DB.  position_table()
        pmt_table  = DB. pmt_light_table()
        sipm_table = DB.sipm_light_table()

        (self. pmt_dict,
         self.sipm_dict) = self._create_probability_dicts(pos_table, pmt_table, sipm_table)

        self.min_x, self.pitch_x, self.pos_x = self._grid_info(pos_table[1]) # X
        self.min_y, self.pitch_y, self.pos_y = self._grid_info(pos_table[2]) # Y

    cdef pmt_probabilities(self, np.ndarray[double, ndim=1] x,
                                 np.ndarray[double, ndim=1] y):
        assert self.load_pmt_table, "PMT table not loaded"
        return self.get_probability(x, y, self.pmt_dict)

    cdef sipm_probabilities(self, np.ndarray[double, ndim=1] x,
                                  np.ndarray[double, ndim=1] y):
        assert self.load_sipm_table, "SiPM table not loaded"
        return self.get_probability(x, y, self.sipm_dict)

    cdef get_probability(self, np.ndarray[double, ndim=1] x,
                               np.ndarray[double, ndim=1] y,
                               dict                       from_dict):
        cdef np.ndarray[int, ndim=1] index_x
        cdef np.ndarray[int, ndim=1] index_y

        cdef tuple grid_point

        index_x = ((x - self.min_x) // self.pitch_x).astype(int)
        index_y = ((y - self.min_y) // self.pitch_y).astype(int)

        grid_point = (self.pos_x[index_x],
                      self.pos_y[index_y])

        if not grid_point in from_dict:
            raise KeyError("Grid point not in table")
        return from_dict[grid_point]

    cdef _create_probability_dicts(self, np.ndarray[double, ndim=2] pos_table,
                                         np.ndarray[double, ndim=2] pmt_table,
                                         np.ndarray[double, ndim=2] sipm_table):
        cdef np.ndarray[double, ndim=1]      pos_ID, X, Y
        cdef np.ndarray[double, ndim=1]  pmt_pos_ID,  pmt_sensor_ID
        cdef np.ndarray[double, ndim=1] sipm_pos_ID, sipm_sensor_ID
        cdef np.ndarray[double, ndim=2]  pmt_probs , sipm_probs

        cdef dict  pos_dict
        cdef dict  pmt_dict
        cdef dict sipm_dict

        cdef np.ndarray[double, ndim=1] where
        cdef np.ndarray[double, ndim=1] order

        pos_ID, X, Y, _ = pos_table
        pos_dict        = dict(zip(pos_ID, zip(X, Y)))

        _,  pmt_pos_ID,  pmt_sensor_ID, * _pmt_probs =  pmt_table
        _, sipm_pos_ID, sipm_sensor_ID, *_sipm_probs = sipm_table
        pmt_probs  = np.stack( _pmt_probs)
        sipm_probs = np.stack(_sipm_probs)

        pmt_dict  = {}
        sipm_dict = {}
        for i, xy in pos_dict.items():
            if self.load_pmt_table:
                where =  pmt_pos_ID == i
                order = np.argsort(pmt_sensor_ID[where])
                pmt_dict [xy] = pmt_probs[:, where].sum(axis=0)[order]

            if self.load_sipm_table:
                where = sipm_pos_ID == i
                sipm_dict[xy] = SensorCollection(sipm_sensor_ID[   where],
                                                 sipm_probs    [:, where].sum(axis=0))

        return pmt_dict, sipm_dict

    cdef _grid_info(self, np.ndarray[double, ndim=1] pos):
        pos_array = np.unique(pos)
        pos_array.sort()
        min_value = pos_array[0]
        pitch     = np.diff(pos_array)[0]
        return min_value, pitch, pos_array
