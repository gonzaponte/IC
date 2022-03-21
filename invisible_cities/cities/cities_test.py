import os

import tables as tb

from importlib import import_module
from pytest import mark
from .. core.configure      import configure

all_cities = ("beersheba berenice buffy detsim diomira"
              "dorothea esmeralda eutropia hypathia irene"
              "isaura isidora penthesilea phyllis trude").split()

lsc_cities = "irene dorothea penthesilea esmeralda beersheba".split()

@mark.filterwarnings("ignore::UserWarning")
@mark.parametrize("city", lsc_cities)
def test_city_empty_input_file(config_tmpdir, ICDATADIR, city):
    # All cities run in Canfranc must run on an empty file
    # without raising any exception

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_file.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_output.h5')

    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)
    conf = configure(config_file.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT))

    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    city_function(**conf)


@mark.filterwarnings("ignore::UserWarning")
@mark.parametrize("city", all_cities)
def test_city_contains_config(config_tmpdir, ICDATADIR, city):
    output_file = os.path.join(config_tmpdir, f"test_{city}_contains_config.h5")
    config_file = 'dummy invisible_cities/config/{}.conf'.format(city)

    conf = configure(config_file.split())
    conf.update(dict(file_out = output_file))

    module_name   = f'invisible_cities.cities.{city}'
    city_function = getattr(import_module(module_name), city)

    city_function(**conf)

    with tb.open_file(output_file) as file:
        assert "Config" in file.root
        assert    city  in file.root.Config
