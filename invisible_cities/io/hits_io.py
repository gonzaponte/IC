from functools import partial

from . dst_io              import df_writer


def hits_writer( hdf5_file
               , group_name='RECO'
               , table_name='Events'
               , *
               , compression=None):
    return partial( df_writer
                  , hdf5_file
                  , group_name         = group_name
                  , table_name         = table_name
                  , descriptive_string = "Hits"
                  , columns_to_index   = ["event"]
                  , compression        = compression)
