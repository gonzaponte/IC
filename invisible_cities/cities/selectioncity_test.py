import os
import numpy  as np
import tables as tb

from pytest import fixture, mark

from invisible_cities.core.configure import configure

from .selectioncity import selectioncity


@fixture(params="rwfs pmaps".split())
def testfile(request, ICDATADIR):
    filetype = request.param
    if   filetype == "rwfs":
        return os.path.join(ICDATADIR, "run_7775_0120_trigger1_waveforms.h5")
    elif filetype == "pmaps":
        return os.path.join(ICDATADIR, "pmaps_0000_7505_trigger1_v1.1.0_20190801_krbg1600.h5")


@fixture
def selectionfile(testfile, config_tmpdir, n_evts=2):
    with tb.open_file(testfile, "r") as h5file:
        events_node = h5file.get_node("/Run/events")
        events = events_node.read()["evt_number"]
    selevents = np.sort(np.random.choice(events, n_evts))

    selfile = os.path.join(config_tmpdir, "selected_events.txt")
    with open(selfile, "w") as file:
        for event in selevents:
            file.write(f"{event}\n")
    return selfile


def test_selectioncity_contain_structure(testfile, selectionfile, output_tmpdir):
    ## run selection city ###
    PATH_IN  = testfile
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selectionfile,
                     event_range = all ))
    selectioncity(**conf)

    ### test  ###
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         for node in h5in.walk_nodes():
             assert node in h5out


def test_selectioncity_contain_selected_events_uniquely(testfile, selectionfile, output_tmpdir):
    ## run selection city ###
    PATH_IN  = testfile
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selectionfile,
                     event_range = all ))
    selectioncity(**conf)

    ## test ##
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         inevents = h5in .get_node("/Run/events").read()["evt_number"]
         outevents= h5out.get_node("/Run/events").read()["evt_number"]

         assert (np.sort( np.loadtxt(selectionfile, dtype=int) ) == np.sort(outevents)).all()
         assert np.isin(outevents, inevents).all()


def test_selectioncity_contain_event_data(testfile, selectionfile, output_tmpdir):
    ## run selection city ###
    PATH_IN  = testfile
    PATH_OUT = os.path.join(output_tmpdir, "selected.h5")

    conf = configure("dummy /Users/gonzalo/Documents/NEXT/selectioncity/selectioncity.conf".split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT,
                     selected_events_filename = selectionfile,
                     event_range = all ))
    selectioncity(**conf)

    #### test ###
    with tb.open_file(PATH_IN , "r") as h5in, \
         tb.open_file(PATH_OUT, "r") as h5out:

         events = np.loadtxt(selectionfile, dtype=int)
         events_in = h5in .get_node("/Run/events").read()["evt_number"]

         for node in h5out.walk_nodes():
             if isinstance(node, tb.Table):
                 tableout = node.read()
                 tablein  = h5in.get_node(node._v_pathname).read()

                 for i, event in enumerate(events):
                     try:
                         selout = tableout["event"] == event
                         selin  = tablein ["event"] == event
                         assert (tablein[selin] == tabeout[selout]).all()
                     except:
                         pass

             if isinstance(node, tb.EArray):
                 earrayout = node
                 earrayin  = h5in.get_node(node._v_pathname)

                 for i, event in enumerate(events):
                     ii = np.argwhere(events_in == event).flatten()[0]
                     assert (earrayin[ii] == earrayout[i]).all()
