import pytest
import os
import numpy  as np
import tables as tb

from hypothesis import strategies as st

from invisible_cities.cities.crashcity import crashcity
from invisible_cities.cities.crashcity import txt_read

from invisible_cities.core.configure import configure

def test_output_length(config_tmpdir):
    
    PATH_OUT  = os.path.join(config_tmpdir, "crash_thing.h5")
    
    
    conf      = configure('dummy invisible_cities/config/crashcity.conf'.split())
    
    
    
    conf.update(dict(file_out                  = PATH_OUT,
                     print_mod = 10000000       ))
    
    result = crashcity(**conf)
    f = '$ICDIR/database/test_data/events_crash.txt'
    len_in = len( txt_read(os.path.expandvars(f)))
    
    assert result.count_out == len_in
    
    with tb.open_file(PATH_OUT) as file_out:
        assert len(file_out.root.Run.events) == len_in
    
def test_contents(config_tmpdir):
    
    PATH_OUT  = os.path.join(config_tmpdir, "crash_thing.h5")
    
    
    conf      = configure('dummy invisible_cities/config/crashcity.conf'.split())
    
    
    
    conf.update(dict(file_out                  = PATH_OUT,
                     print_mod = 10000000       ))
    
    crashcity(**conf)
    
    with tb.open_file(PATH_OUT) as h5out:
        assert "DST"          in h5out.root
        assert "HITS"         in h5out.root
        assert "Run"          in h5out.root
        assert "Run/events"   in h5out.root
        assert "Run/runInfo"  in h5out.root
        assert "DST/Events"   in h5out.root
        assert "HITS/Events"  in h5out.root