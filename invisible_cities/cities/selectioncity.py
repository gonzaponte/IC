"""
---------------------------------
        selectioncity
---------------------------------

This city selects the events and its data from
whichever IC data type.

"""

import numpy  as np
import tables as tb

from functools import partial

from invisible_cities.cities.components import city

from invisible_cities.dataflow import dataflow as fl


def get_file_structure(filename):
    """
    From a given filename, it returns a dictionary whose
    elements are the atributes of each of the nodes of the .h5,
    being the nodes of type tb.Table or tb.EArray.
    """
    d = dict()
    with tb.open_file(filename) as h5file:
        for node in h5file.walk_nodes():
            ####### Table #########
            if isinstance(node, tb.Table):
                d[node._v_pathname] = dict(nodetype = tb.Table               ,
                                           where = node._v_parent._v_pathname,
                                           name  = node.name                 ,
                                           description = node.description    ,
                                           title   = node.title              ,
                                           filters = node.filters)
            ####### EArray #######
            if isinstance(node, tb.EArray):
                shape = [*node.shape]
                shape[node.maindim] = 0
                d[node._v_pathname] = dict(nodetype = tb.EArray              ,
                                           where = node._v_parent._v_pathname,
                                           name  = node.name                 ,
                                           title = node.title                ,
                                           atom  = node.atom                 ,
                                           shape = shape)
    return d


# def create_file_from_structure(filename, structure):
#     with tb.open_file(filename, "w") as h5file:
#         for node in structure:
#             if structure[node]["nodetype"] is tb.Table:
#
#                 h5file.create_table(structure[node]["where"]                   ,
#                                     structure[node]["name"]                    ,
#                                     description= structure[node]["description"],
#                                     title      = structure[node]["title"]      ,
#                                     filters    = structure[node]["filters"]    ,
#                                     createparents=True)
#
#             if structure[node]["nodetype"] is tb.EArray:
#
#                 h5file.create_earray(structure[node]["where"]        ,
#                                      structure[node]["name"]         ,
#                                      atom  = structure[node]["atom"] ,
#                                      shape = structure[node]["shape"],
#                                      title = structure[node]["title"],
#                                      createparents = True)


def copy_file_structure(h5file, structure):
    for node in structure:
        if structure[node]["nodetype"] is tb.Table:
            h5file.create_table(structure[node]["where"]                   ,
                                structure[node]["name"]                    ,
                                description= structure[node]["description"],
                                title      = structure[node]["title"]      ,
                                filters    = structure[node]["filters"]    ,
                                createparents=True)

        if structure[node]["nodetype"] is tb.EArray:
            h5file.create_earray(structure[node]["where"]        ,
                                 structure[node]["name"]         ,
                                 atom  = structure[node]["atom"] ,
                                 shape = structure[node]["shape"],
                                 title = structure[node]["title"],
                                 createparents = True)


def general_source(files_in):
    for file in files_in:
        with tb.open_file(file, "r") as h5file:
            ######
            d_ = dict()
            for node in h5file.walk_nodes():
                if isinstance(node, (tb.Table, tb.EArray)):
                    d_[node._v_pathname] = node.read()
            ######
            d = dict()
            events = d_["/Summary/Events"]["event"]
            for i, event in enumerate( events ):
                selevent = np.eye(1, M=len(events), k=i, dtype=bool)[0]
                for node in d_:
                    if len(d_[node]) == len(events):
                        d[node] = d_[node][selevent]
                    else:
                        try:
                            sel = d_[node]["event"] == event
                            d[node] = d_[node][sel]
                        except (IndexError, ValueError):
                            d["g" + node] = d_[node]
                yield d


def general_writer(h5file, d):
    for nodename in d:
        # check if global node has been filled, if not, fill it
        if nodename[0] == "g":
            globalnode = h5file.get_node( nodename[1:] )
            if globalnode.nrows == 0:
                globalnode.append( d[nodename] )
                continue
            else:
                continue
        # fill event data
        node = h5file.get_node( nodename )
        node.append(d[nodename])
    h5file.flush()


def filter_event(selected_events, event_ts):
    sel = np.isin( event_ts["event"], selected_events)
    return bool(sel)


@city
def selectioncity(files_in, file_out,
                  selected_events_filename,
                  event_range):

    ###### get file structure and create empty file_out ####
    structure = get_file_structure(np.random.choice(files_in))
    # create_file_from_structure(file_out, structure)

    #### define filter #####
    selected_events = np.loadtxt(selected_events_filename, dtype=int)
    filter = fl.filter(partial(filter_event, selected_events), args="/Summary/Events")

    ###### define counters #####
    count_all  = fl.spy_count()
    count_pass = fl.spy_count()

    with tb.open_file(file_out, "w") as h5file:

        copy_file_structure(h5file, structure)

        writer = fl.sink(partial(general_writer, h5file ))

        return fl.push(source = general_source(files_in),
                       pipe   = fl.pipe(count_all  .spy,
                                        filter         ,
                                        count_pass.spy ,
                                        writer)        ,
                         result = dict(n_total = count_all .future,
                                       n_pass  = count_pass.future))
