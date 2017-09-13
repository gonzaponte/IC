cimport numpy as np
import numpy as np

cdef class LightTables:
    """
    docstring
    """
    cdef public dict  pmt_dict
    cdef public dict sipm_dict

    cdef public double   min_x,   min_y
    cdef public double pitch_x, pitch_y

    cdef public np.ndarray[double, ndim=1] pos_x
    cdef public np.ndarray[double, ndim=1] pos_y

    cdef public bool load_pmt_table
    cdef public bool load_sipm_table

    cdef  pmt_probabilities       (self, np.ndarray[double, ndim=1] x,
                                         np.ndarray[double, ndim=1] y)
    cdef sipm_probabilities       (self, np.ndarray[double, ndim=1] x,
                                         np.ndarray[double, ndim=1] y)
    cdef  get_probability         (self, np.ndarray[double, ndim=1] x,
                                         np.ndarray[double, ndim=1] y,
                                         dict                       from_dict)
    cdef _create_probability_dicts(self, np.ndarray[double, ndim=2] pos_table,
                                         np.ndarray[double, ndim=2] pmt_table,
                                         np.ndarray[double, ndim=2] sipm_table)
    cdef _grid_info               (self, np.ndarray[double, ndim=1] pos)