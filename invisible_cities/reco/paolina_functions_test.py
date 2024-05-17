import os

from math      import sqrt
from functools import partial

import numpy    as np
import pandas   as pd
import networkx as nx

from itertools import combinations
from operator  import attrgetter

from numpy.testing import assert_almost_equal

from pytest import fixture
from pytest import mark
from pytest import approx
from pytest import raises

from hypothesis            import given
from hypothesis            import settings
from hypothesis            import HealthCheck
from hypothesis.strategies import composite
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import builds

from . paolina_functions import BOXEPS
from . paolina_functions import bounding_box
from . paolina_functions import digitize
from . paolina_functions import get_bin_edges
from . paolina_functions import distance_between_voxels
from . paolina_functions import energy_of_voxels_within_radius
from . paolina_functions import find_extrema
from . paolina_functions import find_blobs
from . paolina_functions import hits_in_blob
from . paolina_functions import voxelize_hits
from . paolina_functions import shortest_paths
from . paolina_functions import make_track_graphs
from . paolina_functions import drop_voxel_in_place
from . paolina_functions import drop_end_point_voxels
from . paolina_functions import make_tracks

from .. core               import system_of_units as units
from .. core.exceptions    import NoHits
from .. core.exceptions    import NoVoxels
from .. core.testing_utils import assert_dataframes_equal
from .. core.testing_utils import assert_dataframes_close

from .. io.mcinfo_io    import cast_mchits_to_dict
from .. io.mcinfo_io    import load_mchits_df

from .. types.ic_types import xy
from .. types.symbols  import Contiguity
from .. types.symbols  import HitEnergy

from .  hits_functions_test import hits


def sparse_enough(hits):
    ok_x = (hits.X.max() - hits.X.min()) > 0.1
    ok_y = (hits.Y.max() - hits.Y.min()) > 0.1
    ok_z = (hits.Z.max() - hits.Z.min()) > 0.1
    return ok_x & ok_y & ok_z


voxel_indices = "voxel_i voxel_j voxel_k".split()
sparse_hits   = lambda: hits(2, 5).filter(sparse_enough)


@composite
def p_hits(draw):
    hits  = draw(sparse_hits())
    track = draw(integers(0, 10))
    Ep    = draw(lists(floats( 50, 100), min_size=len(hits), max_size=len(hits)))
    return hits.assign( track_id = track
                      , Ep       = Ep)

@composite
def single_voxels(draw, index=0):
    voxel_i = draw(integers(min_value=0, max_value=10))
    voxel_j = draw(integers(min_value=0, max_value=10))
    voxel_k = draw(integers(min_value=0, max_value=10))
    size_x  = draw(  floats(min_value=1, max_value=10))
    size_y  = draw(  floats(min_value=1, max_value=10))
    size_z  = draw(  floats(min_value=1, max_value=10))
    e       = draw(  floats(min_value=1, max_value=1e3))
    ec      = draw(  floats(min_value=1, max_value=1e3))
    ep      = draw(  floats(min_value=1, max_value=1e3))
    nhits   = draw(integers(min_value=1, max_value=10))
    track   = draw(integers(min_value=0, max_value=10))

    voxel_x = voxel_i * size_x
    voxel_y = voxel_j * size_y
    voxel_z = voxel_k * size_z

    return pd.Series(dict(**locals()), name=index)

@composite
def many_voxels(draw):
    voxels = draw(lists(single_voxels, min_size=1, max_size=5))
    voxels = pd.concat(voxels, ignore_index=True)
    return voxels


box_dimensions    = floats  (min_value = 1, max_value =   5)
box_sizes         = builds  (np.array, lists(box_dimensions, min_size=3, max_size=3))
radius            = floats  (min_value = 1, max_value = 100)
fraction_zero_one = floats  (min_value = 0, max_value =   1)
min_n_of_voxels   = integers(min_value = 3, max_value =  10)


@given(sparse_hits())
def test_bounding_box_contains_all_hits(hits):
    if not len(hits): # TODO: deal with empty sequences
        return

    lo, hi = bounding_box(hits)

    mins = [float(' inf')] * 3
    maxs = [float('-inf')] * 3

    assert (hits.X >= lo[0]).all()
    assert (hits.X <= hi[0]).all()
    assert (hits.Y >= lo[1]).all()
    assert (hits.Y <= hi[1]).all()
    assert (hits.Z >= lo[2]).all()
    assert (hits.Z <= hi[2]).all()


@given(sparse_hits(), integers(1, 100))
def test_digitize_contains_all_hits(hits, nbins):
    if not len(hits): # TODO: deal with empty sequences
        return

    data    = np.concatenate([hits.X, hits.Y, hits.Z])
    bins    = np.linspace(data.min(), data.max(), nbins + 1)
    indices = digitize(data, bins)

    assert np.all(indices >=     0)
    assert np.all(indices <  nbins)


@mark.parametrize("strict", (False, True))
@given(hits=sparse_hits(), vox_size=box_sizes)
def test_get_bin_edges(hits, vox_size, strict):
    if not len(hits): # TODO: deal with empty sequences
        return

    bins_x, bins_y, bins_z = get_bin_edges(hits, vox_size, strict)

    assert (hits.X >= bins_x[ 0]).all()
    assert (hits.X <= bins_x[-1]).all()
    assert (hits.Y >= bins_y[ 0]).all()
    assert (hits.Y <= bins_y[-1]).all()
    assert (hits.Z >= bins_z[ 0]).all()
    assert (hits.Z <= bins_z[-1]).all()

    assert len(bins_x) >= 1
    assert len(bins_y) >= 1
    assert len(bins_z) >= 1


@mark.parametrize("strict", (False, True))
@given(hits=p_hits(), vox_size=box_sizes)
def test_voxelize_hits_does_not_modify_input(hits, vox_size, strict):
    hits_original = hits.copy()
    voxelize_hits(hits, vox_size, strict)
    assert_dataframes_equal(hits, hits_original)


@settings(max_examples=1)
@given(p_hits(), box_sizes)
def test_voxelize_hits_adds_columns_to_hits(hits, vox_size):
    updated_hits, _, _ = voxelize_hits(hits, vox_size, False)

    new_columns = "voxel_i voxel_j voxel_k voxel".split()
    for column in new_columns:
        assert column in updated_hits.columns


@given(p_hits(), box_sizes)
def test_voxelize_hits_creates_unique_voxels(hits, vox_size):
    _, voxels, _ = voxelize_hits(hits, vox_size, False)

    n_voxels_same_indices = voxels.groupby(voxel_indices).size()
    assert np.all(voxels.nhits          >= 1)
    assert np.all(n_voxels_same_indices == 1)


@given(p_hits(), box_sizes)
@settings(suppress_health_check=(HealthCheck.too_slow,))
def test_voxelize_hits_each_hit_has_a_voxel(hits, vox_size):
    hits, voxels, _ = voxelize_hits(hits, vox_size, False)

    for voxel_id, _ in hits.groupby("voxel"):
        assert voxel_id in voxels.index


@given(p_hits(), box_sizes)
def test_voxelize_hits_each_voxel_has_at_least_one_hit(hits, vox_size):
    hits, voxels, _ = voxelize_hits(hits, vox_size, False)

    for voxel_id, voxel in voxels.iterrows():
        sel = hits.voxel == voxel_id

        assert voxel.nhits >= 1
        assert np.count_nonzero(sel) >= 1


@given(p_hits(), box_sizes)
def test_voxelize_hits_does_not_lose_energy(hits, vox_size):
    updated_hits, voxels, _ = voxelize_hits(hits, vox_size, strict_voxel_size=False)

    assert np.isclose(hits.E .sum(), updated_hits.E .sum())
    assert np.isclose(hits.Ec.sum(), updated_hits.Ec.sum())
    assert np.isclose(hits.Ep.sum(), updated_hits.Ep.sum())
    assert np.isclose(hits.E .sum(),       voxels.E .sum())
    assert np.isclose(hits.Ec.sum(),       voxels.Ec.sum())
    assert np.isclose(hits.Ep.sum(),       voxels.Ep.sum())


@given(p_hits(), box_sizes)
def test_voxelize_hits_keeps_bounding_box(hits, vox_size):
    updated_hits, voxels, _ = voxelize_hits(hits, vox_size)

    hlo, hhi = bounding_box(hits)
    ulo, uhi = bounding_box(updated_hits)
    vlo, vhi = bounding_box(voxels)

    vlo -= 0.5 * vox_size
    vhi += 0.5 * vox_size

    assert (vlo <= hlo).all()
    assert (vlo <= ulo).all()
    assert (vhi >= hhi).all()
    assert (vhi >= uhi).all()


@given(p_hits(), box_sizes)
def test_voxelize_hits_respects_voxel_dimensions(hits, vox_size):
    hits, voxels, _ = voxelize_hits(hits, vox_size, strict_voxel_size=True)
    xyz = list("XYZ")
    for (_, v1), (_, v2) in combinations(voxels.iterrows(), 2):
        dxyz   = v1.loc[xyz].values - v2.loc[xyz].values
        off_by = dxyz % vox_size
        try:
            assert (np.isclose(off_by,        0) |
                    np.isclose(off_by, vox_size) ).all()
        except:
            print("dxyz", dxyz, hits.dtypes, voxels.dtypes, off_by.dtype, vox_size.dtype)
            raise


@given(p_hits(), box_sizes)
def test_voxelize_hits_gives_maximum_voxels_size(hits, vox_size):
    _, _, final_vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False)

    assert np.all(final_vox_size <= vox_size + 4 * BOXEPS)


@given(p_hits(), box_sizes)
def test_voxelize_hits_strict_gives_required_voxels_size(hits, vox_size):
    _, voxels, final_vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=True)

    assert np.allclose(final_vox_size, vox_size)


@given(p_hits(), box_sizes)
def test_voxelize_hits_flexible_gives_correct_voxels_size(hits, vox_size):
    _, voxels, final_vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False)

    def is_close_to_integer(n):
        return np.isclose(n, np.rint(n))

    for (_, v1), (_, v2) in combinations(voxels.iterrows(), 2):
        for size, axis in zip(final_vox_size, "XYZ"):
            separation_over_size = (v2[axis] - v1[axis]) / size
            assert is_close_to_integer(separation_over_size)


@given(p_hits(), box_sizes)
def test_voxelize_hits_single_hit(hits, vox_size):
    single_hit = hits.iloc[:1]
    _, voxels, _ = voxelize_hits(single_hit, vox_size)
    assert len(voxels) == 1


@given(floats(0, 100).filter(lambda x: x>0))
def test_distance_between_voxels(d0):
    d1 = d0 / 2**0.5
    d2 = d0 / 3**0.5
    v0 = pd.Series(dict(X= 0, Y= 0, Z= 0))
    v1 = pd.Series(dict(X=d0, Y= 0, Z= 0))
    v2 = pd.Series(dict(X= 0, Y=d0, Z= 0))
    v3 = pd.Series(dict(X= 0, Y= 0, Z=d0))
    v4 = pd.Series(dict(X=d1, Y=d1, Z= 0))
    v5 = pd.Series(dict(X=d2, Y=d2, Z=d2))

    assert np.isclose(distance_between_voxels(v0, v1), d0)
    assert np.isclose(distance_between_voxels(v0, v2), d0)
    assert np.isclose(distance_between_voxels(v0, v3), d0)
    assert np.isclose(distance_between_voxels(v0, v4), d0)
    assert np.isclose(distance_between_voxels(v0, v5), d0)


@given(p_hits(), box_sizes)
def test_hits_energy_in_voxel_is_equal_to_voxel_energy(hits, vox_size):
    updated_hits, voxels, _ = voxelize_hits(hits, vox_size, strict_voxel_size=False)

    for voxel_id, voxel in voxels.iterrows():
        hits_in_voxel = updated_hits.loc[updated_hits.voxel == voxel_id]

        assert np.isclose(hits_in_voxel.E .sum(), voxel.E )
        assert np.isclose(hits_in_voxel.Ec.sum(), voxel.Ec)
        assert np.isclose(hits_in_voxel.Ep.sum(), voxel.Ep)


def test_voxels_with_no_hits(ICDATADIR):
    hit_file = os.path.join(ICDATADIR, 'test_voxels_with_no_hits.h5')
    evt_number = 4803
    size = 15.
    vox_size = np.full(3, size, dtype=float)

    hits = (pd.read_hdf(hit_file, "/RECO/Events")
              .loc[lambda df: df.event == evt_number]
              .assign(Ep = lambda df: df.Ec))

    updated_hits, voxels, _ = voxelize_hits(hits, vox_size, strict_voxel_size=False)

    for voxel_id, voxel in voxels.iterrows():
        hits_in_voxel = updated_hits.loc[updated_hits.voxel == voxel_id]

        assert np.isclose(hits_in_voxel.E .sum(), voxel.E )
        assert np.isclose(hits_in_voxel.Ec.sum(), voxel.Ec)
        assert np.isclose(hits_in_voxel.Ep.sum(), voxel.Ep)


def test_hits_on_border_are_assigned_to_correct_voxel():
    hits = pd.DataFrame(dict( X     = [ 5., 15, 15, 15, 25]
                            , Y     = [15.,  5, 15, 25, 15]
                            , Z     = 10.
                            , E     =  0. # | these fields are necessary
                            , Ec    =  0. # | but they don't play a role
                            , Ep    =  0. # |
                            , Q     =  0. # |
                            , event =  0  # |
                            , npeak =  0  # |
                            ))

    vox_size = np.full(3, 15, dtype=np.int16)
    updated_hits, voxels, _ = voxelize_hits(hits, vox_size)

    assert len(voxels) == 3
    assert voxels.nhits.values.tolist() == [1, 1, 3]

    h0 = updated_hits.iloc[0]
    h1 = updated_hits.iloc[1]
    h2 = updated_hits.iloc[2]
    h3 = updated_hits.iloc[3]
    h4 = updated_hits.iloc[4]

    def indices(hit):
        return (hit.voxel_i, hit.voxel_j, hit.voxel_k)

    assert indices(h0) == (0, 1, 0)
    assert indices(h1) == (1, 0, 0)
    assert indices(h2) == (1, 1, 0)
    assert indices(h3) == (1, 1, 0)
    assert indices(h4) == (1, 1, 0)


@given(p_hits(), box_sizes)
def test_make_track_graphs_keeps_all_voxels(hits, vox_size):
    _, voxels, final_vox_size = voxelize_hits(hits, vox_size)

    tracks = make_track_graphs(voxels, final_vox_size, Contiguity.CORNER)

    voxels_in_tracks = set().union(*(set(t.nodes()) for t in tracks))
    assert set(voxels.index) == voxels_in_tracks


@given(p_hits(), box_sizes)
def test_make_track_graphs_all_voxels_are_assigned_to_exactly_one_track(hits, vox_size):
    _, voxels, final_vox_size = voxelize_hits(hits  , vox_size)

    tracks = make_track_graphs(voxels, final_vox_size, Contiguity.CORNER)

    voxels_in_tracks = [v for t in tracks for v in t.nodes()]

    # No repeats. i.e. each voxel is only assigned o one track
    assert len(voxels_in_tracks) == len(set(voxels_in_tracks))

    # All voxels are assigned
    assert np.in1d(voxels.index, voxels_in_tracks).all()


@given(single_voxels())
def test_find_extrema_single_voxel(voxel):
    g = nx.Graph()
    g.add_node(voxel.name)
    first, last, distance = find_extrema(g)
    assert first == last
    assert first == voxel.name
    assert distance == 0


def test_find_extrema_no_voxels():
    with raises(NoVoxels):
        find_extrema(nx.Graph())


# @fixture(scope='module')
# def voxels_without_hits():
#     voxel_spec = ((10,10,10,1),
#                   (10,10,11,2),
#                   (10,10,12,2),
#                   (10,10,13,2),
#                   (10,10,14,2),
#                   (10,10,15,2),
#                   (10,11,15,2),
#                   (10,12,15,2),
#                   (10,13,15,2),
#                   (10,14,15,2),
#                   (10,15,15,2)
#     )
#     voxels = [Voxel(x,y,z, E, np.array([1,1,1])) for (x,y,z,E) in voxel_spec]
#
#     return voxels
#
#
# def test_length(voxels_without_hits):
#     voxels = voxels_without_hits
#     tracks = make_track_graphs(voxels)
#
#     assert len(tracks) == 1
#     track_length = length(tracks[0])
#
#     expected_length = 8 + np.sqrt(2)
#
#     assert track_length == approx(expected_length)
#
#
@mark.parametrize('contiguity, expected_length',
                 ((Contiguity.FACE  , 4          ),
                  (Contiguity.CORNER, 2 * sqrt(2))))
def test_length_around_bend(contiguity, expected_length):
    # Make sure that we calculate the length along the track rather
    # that the shortcut. Shape:
    # X X
    #   X
    # X X
    fake_voxels = pd.DataFrame(dict( X = [0, 1, 1, 1, 0]
                                   , Y = [0, 0, 1, 2, 2]
                                   , Z = 0))
    vox_size = np.array([1,1,1])
    tracks = make_track_graphs(fake_voxels, vox_size, contiguity=contiguity)
    assert len(tracks) == 1

    _, _, track_length = find_extrema(tracks[0])
    assert track_length == approx(expected_length)


@mark.parametrize('contiguity, expected_length',
                 (# Face contiguity requires 3 steps, each parallel to an axis
                  (Contiguity.FACE,  1 + 1 + 1),
                  # Edge continuity allows to cut one corner
                  (Contiguity.EDGE,  1 + sqrt(2)),
                  # Corner contiguity makes it possible to do in a single step
                  (Contiguity.CORNER,    sqrt(3))))
def test_length_cuts_corners(contiguity, expected_length):
    "Make sure that we cut corners, if the contiguity allows"
    fake_voxels = pd.DataFrame(dict( X = [0, 1, 1, 1]
                                   , Y = [0, 0, 1, 1]
                                   , Z = [0, 0, 0, 1]))
    vox_size = np.array([1,1,1])
    tracks = make_track_graphs(fake_voxels, vox_size, contiguity=contiguity)

    assert len(tracks) == 1
    _, _, track_length = find_extrema(tracks[0])
    assert track_length == approx(expected_length)


FACE, EDGE, CORNER = Contiguity
@mark.parametrize('contiguity,  proximity,          are_neighbours',
                 ((FACE,      'share_face',            True),
                  (FACE,      'share_edge',            False),
                  (FACE,      'share_corner',          False),
                  (FACE,      'share_nothing',         False),
                  (FACE,      'share_nothing_aligned', False),

                  (EDGE,      'share_face',            True),
                  (EDGE,      'share_edge',            True),
                  (EDGE,      'share_corner',          False),
                  (EDGE,      'share_nothing',         False),
                  (EDGE,      'share_nothing_aligned', False),

                  (CORNER,    'share_face',            True),
                  (CORNER,    'share_edge',            True),
                  (CORNER,    'share_corner',          True),
                  (CORNER,    'share_nothing',         False),
                  (CORNER,    'share_nothing_aligned', False),))
def test_contiguity(proximity, contiguity, are_neighbours):
    voxels = pd.DataFrame(dict( X = [0, 0, 0, 0, 0, 1, 0, 2, 0, 2]
                              , Y = [0, 0, 0, 1, 0, 1, 0, 2, 0, 0]
                              , Z = [0, 1, 0, 1, 0, 1, 0, 2, 0, 0]
                              , P = ["share_face"  ]*2 + ["share_edge"   ]*2
                                  + ["share_corner"]*2 + ["share_nothing"]*2
                                  + ["share_nothing_aligned"]*2))

    voxels = voxels.groupby("P").get_group(proximity)
    expected_number_of_tracks = 1 if are_neighbours else 2
    vox_size = np.array([1, 1, 1])
    tracks = make_track_graphs(voxels, vox_size, contiguity=contiguity)

    assert len(tracks) == expected_number_of_tracks


@given(p_hits(), box_sizes, min_n_of_voxels, fraction_zero_one)
def test_energy_is_conserved_with_dropped_voxels(hits, vox_size, min_voxels, fraction_zero_one):
    etype = HitEnergy.E
    initial_energy = hits.E.sum()
    hits, voxels, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False, energy_type = etype)
    initial_tracks = make_track_graphs(voxels, vox_size)
    initial_track_energies = [sum(voxels.loc[i].E for i in track.nodes()) for track in initial_tracks]
    initial_track_energies.sort()

    e_thr = voxels.E.min() + fraction_zero_one * (voxels.E.max() - voxels.E.min())
    hits, voxels, dropped_hits, dropped_voxels = drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, etype, Contiguity.CORNER)

    final_energy = hits.E.sum()
    final_tracks =  make_track_graphs(voxels, vox_size)
    final_track_energies = [sum(voxels.loc[i].E for i in track.nodes()) for track in final_tracks]
    final_track_energies.sort()

    assert initial_energy == approx(final_energy)
    assert np.allclose(initial_track_energies, final_track_energies)


@mark.parametrize("energy_type", HitEnergy)
@given(hits              = p_hits(),
       vox_size          = box_sizes,
       min_voxels        = min_n_of_voxels,
       fraction_zero_one = fraction_zero_one)
def test_dropped_voxels_have_nan_energy(hits, vox_size, min_voxels, fraction_zero_one, energy_type):
    hits, voxels, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False, energy_type=energy_type)
    vox_e  = voxels[energy_type.value]
    e_thr  = vox_e.min() + fraction_zero_one * (vox_e.max() - vox_e.min())
    _, _, dropped_hits, dropped_voxels = drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, energy_type, Contiguity.CORNER)

    assert np.all(np.isnan(dropped_voxels[energy_type.value]))
    assert np.all(np.isnan(dropped_hits  [energy_type.value]))


@mark.parametrize("energy_type", HitEnergy)
@given(hits              = p_hits(),
       vox_size          = box_sizes,
       min_voxels        = min_n_of_voxels,
       fraction_zero_one = fraction_zero_one)
@settings(suppress_health_check=(HealthCheck.too_slow,))
def test_drop_end_point_voxels_doesnt_modify_other_energy_types(hits, vox_size, min_voxels, fraction_zero_one, energy_type):
    hits, voxels, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False, energy_type=energy_type)
    original_hits   =   hits.copy()
    original_voxels = voxels.copy()

    vox_e = voxels[energy_type.value]
    e_thr = vox_e.min() + fraction_zero_one * (vox_e.max() - vox_e.min())

    hits, voxels, dropped_hits, dropped_voxels = drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, energy_type, Contiguity.CORNER)
    new_voxels = pd.concat([voxels, dropped_voxels])
    new_hits   = pd.concat([  hits, dropped_hits  ])

    for e_type in HitEnergy:
        if e_type is energy_type: continue

        original_voxels = original_voxels.sort_values(e_type.value)
        original_hits   = original_hits  .sort_values(e_type.value)
        new_voxels      =      new_voxels.sort_values(e_type.value)
        new_hits        =      new_hits  .sort_values(e_type.value)
        assert np.allclose(original_voxels[e_type.value], new_voxels[e_type.value])
        assert np.allclose(original_hits  [e_type.value], new_hits  [e_type.value])


@mark.parametrize("energy_type", HitEnergy)
@given(hits              = p_hits(),
       vox_size          = box_sizes,
       min_voxels        = min_n_of_voxels,
       fraction_zero_one = fraction_zero_one)
@settings(suppress_health_check=(HealthCheck.too_slow,))
def test_drop_voxels_voxel_energy_is_sum_of_hits_general(hits, vox_size, min_voxels, fraction_zero_one, energy_type):
    hits, voxels, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False, energy_type=energy_type)

    vox_e = voxels[energy_type.value]
    e_thr = vox_e.min() + fraction_zero_one * (vox_e.max() - vox_e.min())

    hits, voxels, _, _ = drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, energy_type, Contiguity.CORNER)


    for i, v in voxels.iterrows():
        hits_e = hits.loc[hits.voxel==i, energy_type.value]
        assert np.isclose(v[energy_type.value], hits_e.sum())


@mark.parametrize("energy_type", HitEnergy)
@given(hits              = p_hits(),
       vox_size          = box_sizes,
       min_voxels        = min_n_of_voxels,
       fraction_zero_one = fraction_zero_one)
def test_drop_end_point_voxels_constant_number_of_voxels_and_hits(hits, vox_size, min_voxels, fraction_zero_one, energy_type):
    hits, voxels, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False, energy_type=energy_type)

    n_voxels_before = len(voxels)
    n_hits_before   = len(hits)

    vox_e = voxels[energy_type.value]
    e_thr = vox_e.min() + fraction_zero_one * (vox_e.max() - vox_e.min())

    hits, voxels, dropped_hits, dropped_voxels = drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, energy_type, Contiguity.CORNER)

    n_voxels_after = len(voxels)
    n_hits_after   = len(hits)

    assert n_voxels_before == n_voxels_after
    assert n_hits_before   == n_hits_after


@fixture(scope="module")
def tracks_from_data(ICDATADIR):
    hit_file   = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
    evt_number = 19
    e_thr      = 5867.92
    min_voxels = 3
    size       = 15.
    vox_size   = np.array([size]*3, dtype=np.float16)
    etype      = HitEnergy.E

    all_hits = pd.read_hdf(hit_file, "/RECO/Events")
    hits     = all_hits.groupby("event").get_group(evt_number).assign(Ep = lambda df: df.Ec)
    return hits, vox_size, e_thr, min_voxels, etype


def test_initial_voxels_are_the_same_after_dropping_voxels(tracks_from_data):
    hits, vox_size, e_thr, min_voxels, etype = tracks_from_data

    # This is the core of the test: collect data before/after ...
    hits, voxels, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False, energy_type = etype)
    original_hits   =   hits.copy()
    original_voxels = voxels.copy()

    drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, etype, Contiguity.CORNER)

    assert_dataframes_equal(original_hits  , hits)
    assert_dataframes_equal(original_voxels, voxels)


def test_tracks_with_dropped_voxels(tracks_from_data):
    hits, vox_size, e_thr, min_voxels, etype = tracks_from_data

    hits, voxels, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False)

    initial_tracks      = make_track_graphs(voxels, vox_size)
    initial_n_of_tracks = len(initial_tracks)

    initial_energies    = [voxels.loc[list(t.nodes()), etype.value].sum() for t in initial_tracks]
    initial_n_voxels    = np.array([len(t.nodes()) for t in initial_tracks])

    hits, voxels, dropped_hits, dropped_voxels = drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, etype, Contiguity.CORNER)

    final_tracks      = make_track_graphs(voxels, vox_size)
    final_n_of_tracks = len(final_tracks)
    final_energies    = [voxels.loc[list(t.nodes()), etype.value].sum() for t in final_tracks]
    final_n_voxels    = np.array([len(t.nodes()) for t in final_tracks])

    expected_diff_n_voxels = np.array([0, 0, 2])

    assert initial_n_of_tracks == final_n_of_tracks
    assert np.allclose(initial_energies, final_energies)
    assert np.all(initial_n_voxels - final_n_voxels == expected_diff_n_voxels)


def test_drop_voxels_deterministic(tracks_from_data):
    hits, vox_size, e_thr, min_voxels, etype = tracks_from_data

    hits0, voxels0, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=False)
    hits0   =   hits0.sort_values(etype.value, ascending=True)
    voxels0 = voxels0.sort_values(etype.value, ascending=True)
    hits1, voxels1, _, _ = drop_end_point_voxels(hits0.copy(), voxels0.copy(), vox_size, e_thr, min_voxels, etype, Contiguity.CORNER)

    hits0   =   hits0.sort_values(etype.value, ascending=False)
    voxels0 = voxels0.sort_values(etype.value, ascending=False)
    hits2, voxels2, _, _ = drop_end_point_voxels(hits0, voxels0, vox_size, e_thr, min_voxels, etype, Contiguity.CORNER)

    assert_dataframes_close(  hits1.sort_values(etype.value),   hits2.sort_values(etype.value))
    assert_dataframes_close(voxels1.sort_values(etype.value), voxels2.sort_values(etype.value))


def make_hits(event=0, npeak=0, X=0, Y=0, Z=0, E=0, Q=0, Ec=None, Ep=None, voxel=np.nan):
    return pd.DataFrame(dict( event = event
                            , npeak = npeak
                            , X     = X
                            , Y     = Y
                            , Z     = Z
                            , E     = E
                            , Q     = Q
                            , Ec    = Ec if Ec is not None else E
                            , Ep    = Ep if Ep is not None else E
                            , voxel = voxel
    ))


def test_voxel_drop_in_short_tracks():
    hits = make_hits( X = [10, 26]
                    , Y = [10, 10]
                    , Z = [10, 10]
                    , E = [ 1,  1])
    vox_size = [15]*3

    hits, voxels, vox_size = voxelize_hits(hits, vox_size, strict_voxel_size=True)

    e_thr      = voxels.E.sum() + 1
    min_voxels = 0

    _, voxels, _, _ = drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, HitEnergy.E, Contiguity.CORNER)

    assert len(voxels) >= 1


def test_drop_voxels_voxel_energy_is_sum_of_hits():
    hits   = make_hits( X     = [0  ,  5  ,  5  , 5  , 5  , 10  , 11  , 15  , 11  , 20  , 11  ]
                      , Y     = [0  , -5  , -8  , 5  , 8  ,  5  ,  5  ,  0  ,  0  ,  0  ,  0  ]
                      , E     = [0.1,  0.7,  0.3, 0.9, 0.6,  1.2,  0.8,  1.8,  0.7,  1.5,  1.5]
                      , voxel = [0  ,  1  ,  1  , 2  , 2  ,  3  ,  3  ,  4  ,  4  ,  5  ,  5  ])

    voxels   = pd.DataFrame(dict( X    = [0, 5, 5, 10, 15, 20]
                                , Y    = [0, -5, 5, 0, 0, 0]
                                , Z    = 0
                                , E    = [0.1, 1.0, 1.5, 2.0, 2.5, 3.0]
                                , nhits= [1, 2, 2, 2, 2, 2])
                           ).assign(Ec=lambda df: df.E, Ep=lambda df: df.E)

    e_thr      = 0.5
    e_type     = HitEnergy.Ep
    vox_size   = [5] * 3
    min_voxels = 1

    hits, voxels, _, _ = drop_end_point_voxels(hits, voxels, vox_size, e_thr, min_voxels, e_type, Contiguity.CORNER)

    for i, voxel in voxels.iterrows():
        hits_in_voxel = hits.loc[hits.voxel==i]
        assert voxel[e_type.value] == approx(hits_in_voxel[e_type.value].sum())


@mark.parametrize('radius expected'.split(),
                 ((10., ( 60,  20)),
                  (12., ( 60,  60)),
                  (14., (100,  60)),
                  (16., (120,  80)),
                  (18., (120,  80)),
                  (20., (140,  80)),
                  (22., (140, 100))))
def test_blobs(radius, expected):
    hits = make_hits( X = [105,  95,  95, 105, 105,  95,  95, 105, 105,  95,  95, 105, 115, 115]
                    , Y = [125, 125, 135, 135, 115, 115, 125, 125, 135, 135, 115, 115, 125, 125]
                    , Z = [77.7] * 6 + [79.5] * 7 + [85.2]
                    , E = 10)
    vox_size = [15.]*3

    hits, voxels, vox_size = voxelize_hits(hits, vox_size)
    tracks                 = make_track_graphs(voxels, vox_size)

    assert len(tracks) == 1

    blob1, blob2 = find_blobs(hits, voxels, vox_size, tracks[0], radius, HitEnergy.E)
    assert (blob1[0], blob2[0]) == expected


@given(p_hits(), box_sizes, radius)
def test_blob_hits_are_inside_radius(hits, vox_size, blob_radius):
    hits, voxels, vox_size = voxelize_hits(hits, vox_size)
    tracks                 = make_track_graphs(voxels, vox_size)
    e_type                 = HitEnergy.E
    for track in tracks:
        blob_a, blob_b = find_blobs(hits, voxels, vox_size, track, blob_radius, e_type)
        E_a, centre_a, hits_a, vox_a = blob_a
        E_b, centre_b, hits_b, vox_b = blob_b

        xyz = list("XYZ")
        assert np.all(np.linalg.norm(hits_a[xyz] - centre_a, axis=1) < blob_radius)
        assert np.all(np.linalg.norm(hits_b[xyz] - centre_b, axis=1) < blob_radius)


# @given(blob_radius=radius, min_voxels=min_n_of_voxels, fraction_zero_one=fraction_zero_one)
# def test_paolina_functions_with_voxels_without_associated_hits(blob_radius, min_voxels, fraction_zero_one, voxels_without_hits):
#     voxels = voxels_without_hits
#     tracks = make_track_graphs(voxels)
#     for t in tracks:
#         a, b = find_extrema(t)
#         hits_a = hits_in_blob(t, blob_radius, a)
#         hits_b = hits_in_blob(t, blob_radius, b)
#
#         assert len(hits_a) == len(hits_b) == 0
#
#         assert np.allclose(blob_centre(a), a.pos)
#         assert np.allclose(blob_centre(b), b.pos)
#
#         distances = shortest_paths(t)
#         Ea = energy_of_voxels_within_radius(distances[a], blob_radius)
#         Eb = energy_of_voxels_within_radius(distances[b], blob_radius)
#
#         if Ea < Eb:
#             assert np.allclose(blob_centres(t, blob_radius)[0], b.pos)
#             assert np.allclose(blob_centres(t, blob_radius)[1], a.pos)
#         else:
#             assert np.allclose(blob_centres(t, blob_radius)[0], a.pos)
#             assert np.allclose(blob_centres(t, blob_radius)[1], b.pos)
#
#         assert blob_energies(t, blob_radius) != (0, 0)
#
#         assert blob_energies_and_hits(t, blob_radius) != (0, 0, [], [])
#
#     energies = [v.E for v in voxels]
#     e_thr = min(energies) + fraction_zero_one * (max(energies) - min(energies))
#     mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
#
#     trks = make_track_graphs(mod_voxels)
#     for t in trks:
#         a, b = find_extrema(t)
#
#         assert np.allclose(blob_centre(a), a.pos)
#         assert np.allclose(blob_centre(b), b.pos)
#
#
@mark.parametrize("etype", (HitEnergy.Ec, HitEnergy.Ep))
@given(hits              = p_hits(),
       vox_size          = box_sizes,
       blob_radius       = radius,
       fraction_zero_one = fraction_zero_one)
def test_paolina_functions_with_hit_energy_different_from_default_value(hits, vox_size, blob_radius, fraction_zero_one, etype):
    hits  , voxels  , vox_size   = voxelize_hits(hits, vox_size, strict_voxel_size=False)
    hits_c, voxels_c, vox_size_c = voxelize_hits(hits, vox_size, strict_voxel_size=False, energy_type=etype)

    original_energy   = voxels_c.E           .sum()
    original_energy_c = voxels_c[etype.value].sum()

    for i, voxel in voxels_c.iterrows():
        hits_e = hits_c.loc[hits_c.voxel==i, etype.value]
        assert voxel[etype.value] == approx(hits_e.sum())

    vox_e = voxels_c[etype.value]
    e_thr = vox_e.min() + fraction_zero_one * (vox_e.max() - vox_e.min())

    # Test that this function doesn't fail
    min_voxels = 0
    hits_c, voxels_c, dropped_hits_c, _ = drop_end_point_voxels(hits_c, voxels_c, vox_size_c, e_thr, min_voxels, etype, Contiguity.CORNER)

    modified_energy   = voxels_c.E           .sum()
    modified_energy_c = voxels_c[etype.value].sum()
    assert original_energy_c == approx(modified_energy_c)

    # We don't want to modify the default energy of hits, if the voxels are made with energy_c
    if len(dropped_hits_c):
       assert original_energy > modified_energy


def test_make_tracks(tracks_from_data):
    all_hits, vox_size, e_thr, min_voxels, etype = tracks_from_data
    blob_radius = 21*units.mm

    # TODO: do it for all events
    for evt_number in range(1):
        evt_hits = all_hits
        evt_time = all_hits.time

        hits_g, voxels_g, vox_size_g = voxelize_hits(evt_hits, vox_size, strict_voxel_size=False, energy_type=etype)
        graphs                 = list(make_track_graphs(voxels_g, vox_size_g))

        hits, voxels, tracks = make_tracks(hits_g, voxels_g, vox_size, Contiguity.CORNER, blob_radius, etype)

        graphs.sort(key=lambda x : len(x.nodes()))
        tracks.sort_values("nvoxels")

        # Compare the two sets of tracks
        assert len(graphs) == len(tracks)

        for i, track in tracks.iterrows():
            graph   = graphs[i]
            indices = list(graph.nodes())
            voxels_in_graph = voxels_g.loc[indices]

            assert len(graph.nodes())                 == track.nvoxels
            assert voxels_in_graph[etype.value].sum() == approx(track.E)

            blob1, blob2 = find_blobs(hits_g, voxels_g, vox_size_g, graph, blob_radius, etype)

            assert blob1[0] == approx(track.Eblob1)
            assert blob2[0] == approx(track.Eblob2)

            assert blob1[1][0] == approx(track.Xblob1)
            assert blob1[1][1] == approx(track.Yblob1)
            assert blob1[1][2] == approx(track.Zblob1)
            assert blob2[1][0] == approx(track.Xblob2)
            assert blob2[1][1] == approx(track.Yblob2)
            assert blob2[1][2] == approx(track.Zblob2)


@given(p_hits(), box_sizes)
def test_make_voxel_graph_keeps_energy_consistence(hits, vox_size):
    hits, voxels, vox_size = voxelize_hits(hits, vox_size)
    tracks                 = make_track_graphs(voxels, vox_size)

    # assert sum of track energy equal to sum of hits energies
    hits_e   = hits.E.sum()
    tracks_e = sum(voxels.loc[list(track.nodes())].E.sum() for track in tracks)

    assert hits_e == approx(tracks_e)
