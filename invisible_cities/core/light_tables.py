"""
An useful class for accessing light detection probabilities.
"""

import numpy as np

from .. database import load_db as DB

from .. evm.ic_containers import SensorCollection


class LightTables:
    def __init__(self):
        pos_table  = DB.  position_table()
        pmt_table  = DB. pmt_light_table()
        sipm_table = DB.sipm_light_table()

        (self. pmt_dict,
         self.sipm_dict) = self._create_probability_dicts(pos_table, pmt_table, sipm_table)

        (self.min_x, self.pitch_x, self.pos_x,
         self.min_y, self.pitch_y, self.pos_y) = self._grid_info(pos_table)

    def pmt_probabilities(self, x, y):
        return self.get_probability(x, y, self.pmt_dict)

    def sipm_probabilities(self, x, y):
        return self.get_probability(x, y, self.sipm_dict)

    def get_probability(self, x, y, from_dict):
        index_x = ((x - self.min_x) // self.pitch_x).astype(int)
        index_y = ((y - self.min_y) // self.pitch_y).astype(int)

        grid_point = (self.pos_x[index_x],
                      self.pos_y[index_y])

        if not grid_point in from_dict:
            raise KeyError("Grid point not in table")
        return from_dict[grid_point]

    def _create_probability_dicts(self, pos_table, pmt_table, sipm_table):
        pos_ID, X, Y, _ = pos_table
        pos_dict        = dict(zip(pos_ID, zip(X, Y)))

        _, pos_ID, sensor_ID, *probs = pmt_table
        probs     = np.stack(probs)

        pmt_dict = {}
        for i in np.unique(pos_ID):
            where = pos_ID == i
            order = np.argsort(sensor_ID[where])
            pmt_dict[pos_dict[i]] = probs[where].sum(axis=0)[order]

        _, pos_ID, sensor_ID, *probs = sipm_table
        probs = np.stack(probs)

        sipm_dict = {}
        for i in np.unique(pos_ID):
            where = pos_ID == i
            sipm_dict[pos_dict[i]] = SensorCollection(sensor_ID[where],
                                                      probs    [where].sum(axis=0))

        return pmt_dict, sipm_dict

    def _grid_info(self, pos_table):
        def extract_info(pos):
            pos_array = np.unique(pos)
            pos_array.sort()
            min_value = pos_array[0]
            pitch     = np.diff(pos_array)[0]
            return min_value, pitch, pos_array

        _, X, Y, _ = pos_table
        return extract_info(X), extract_info(Y)
