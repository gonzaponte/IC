import os
import tables as tb

from invisible_cities.core.configure         import configure
from invisible_cities.cities.selection_city  import selection_city

def test_tables_are_the_same():
    PATH_IN  = '/home/ausonandres/IC-crash-course/data/hdst.h5'
    PATH_OUT = '/home/ausonandres/IC-crash-course/data/output_sel_city/test.h5'
    conf      = configure('dummy /home/ausonandres/IC/invisible_cities/config/selection_city.conf'.split())
    conf.update(dict(files_in                  = PATH_IN ,
                     file_out                  = PATH_OUT))
    selection_city(**conf)

    def get_groupnames_from_file(file):
        names = []
        with tb.open_file(PATH_OUT) as h5out:
            for group in h5out.root:
                group_name = str(group)
                group_name = group_name[1:-11]
                if group_name != 'Filters':
                    for table in getattr(h5out.root, group_name):
                        table_name = str(table)
                        table_name = table_name.split()[0]
                        names.append(table_name)
            return names

    tables_in  = get_groupnames_from_file(PATH_IN)
    tables_out = get_groupnames_from_file(PATH_OUT)
    assert tables_in, tables_out
