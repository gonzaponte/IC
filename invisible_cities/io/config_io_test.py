import os

import numpy  as np
import pandas as pd
import tables as tb

from . config_io import write_config
from . config_io import copy_config


def test_write_config(config_tmpdir):
    output_file = os.path.join(config_tmpdir, "test_write_config.h5")

    variables = dict( variable_1   = 12
                    , flooooooat   = 1e-12
                    , a_string_var = "a_string"
                    , withatuple   = (1, 1.0, "tuple")
                    , a_list       = ["hi", 4, True]
                    , a_dict       = {1 : "one", "2" : 2, 3 : np.pi}
                    , an_array     = np.logspace(-2, 10, 12, base=2)
                    )

    write_config(output_file, variables, "test")

    df = pd.read_hdf(output_file, "/Config/test")
    for (_, item) in df.iterrows():
        assert str(variables[item.parameter]) == item.value


def test_copy_config(config_tmpdir):
    filename1 = os.path.join(config_tmpdir, "test_copy_config1.h5")
    filename2 = os.path.join(config_tmpdir, "test_copy_config2.h5")

    write_config(filename1, locals(), "test")
    copy_config (filename1, filename2)

    with tb.open_file(filename2) as file2:
        assert "Config" in file2.root
        assert "test"   in file2.root.Config
