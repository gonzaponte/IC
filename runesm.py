import traceback
from sys       import argv
from importlib import import_module
from invisible_cities.cities import esmeralda
from invisible_cities.core.configure import configure

args  = argv
city_name = esmeralda

esmeralda.esmeralda(**configure(["esmeralda", "/home/gonzalo/sw/git/IC/esm.conf", "-v"]))
