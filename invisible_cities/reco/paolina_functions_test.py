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
from hypothesis.strategies import composite
from hypothesis.strategies import lists
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import builds

from . paolina_functions import BOXEPS
from . paolina_functions import bounding_box
from . paolina_functions import digitize
from . paolina_functions import get_bins
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
def test_get_bins(hits, vox_size, strict):
    if not len(hits): # TODO: deal with empty sequences
        return

    bins_x, bins_y, bins_z = get_bins(hits, vox_size, strict)

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


# @parametrize('contiguity, expected_length',
#              (# Face contiguity requires 3 steps, each parallel to an axis
#               (Contiguity.FACE,  1 + 1 + 1),
#               # Edge continuity allows to cut one corner
#               (Contiguity.EDGE,  1 + sqrt(2)),
#               # Corner contiguity makes it possible to do in a single step
#               (Contiguity.CORNER,    sqrt(3))))
# def test_length_cuts_corners(contiguity, expected_length):
#     "Make sure that we cut corners, if the contiguity allows"
#     voxel_spec = ((0,0,0), # Extremum 1
#                   (1,0,0),
#                   (1,1,0),
#                   (1,1,1)) # Extremum 2
#     vox_size = np.array([1,1,1])
#     voxels = [Voxel(x,y,z, 1, vox_size) for x,y,z in voxel_spec]
#     tracks = make_track_graphs(voxels, contiguity=contiguity)
#
#     assert len(tracks) == 1
#     track_length = length(tracks[0])
#     assert track_length == approx(expected_length)
#
#
#
# FACE, EDGE, CORNER = Contiguity
# @parametrize('contiguity,  proximity,          are_neighbours',
#              ((FACE,      'share_face',            True),
#               (FACE,      'share_edge',            False),
#               (FACE,      'share_corner',          False),
#               (FACE,      'share_nothing',         False),
#               (FACE,      'share_nothing_algined', False),
#
#               (EDGE,      'share_face',            True),
#               (EDGE,      'share_edge',            True),
#               (EDGE,      'share_corner',          False),
#               (EDGE,      'share_nothing',         False),
#               (EDGE,      'share_nothing_algined', False),
#
#               (CORNER,    'share_face',            True),
#               (CORNER,    'share_edge',            True),
#               (CORNER,    'share_corner',          True),
#               (CORNER,    'share_nothing',         False),
#               (CORNER,    'share_nothing_algined', False),))
# def test_contiguity(proximity, contiguity, are_neighbours):
#     voxel_spec = dict(share_face            = ((0,0,0),
#                                                (0,0,1)),
#                       share_edge            = ((0,0,0),
#                                                (0,1,1)),
#                       share_corner          = ((0,0,0),
#                                                (1,1,1)),
#                       share_nothing         = ((0,0,0),
#                                                (2,2,2)),
#                       share_nothing_algined = ((0,0,0),
#                                                (2,0,0)) )[proximity]
#     expected_number_of_tracks = 1 if are_neighbours else 2
#     voxels = [Voxel(x,y,z, 1, np.array([1,1,1])) for x,y,z in voxel_spec]
#     tracks = make_track_graphs(voxels, contiguity=contiguity)
#
#     assert len(tracks) == expected_number_of_tracks
#
#
# @given(p_hits, box_sizes, min_n_of_voxels, fraction_zero_one)
# def test_energy_is_conserved_with_dropped_voxels(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one):
#     tot_initial_energy = sum(h.E for h in hits)
#     voxels = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
#     ini_trks = make_track_graphs(voxels)
#     ini_trk_energies = [sum(vox.E for vox in t.nodes()) for t in ini_trks]
#     ini_trk_energies.sort()
#
#     energies = [v.E for v in voxels]
#     e_thr = min(energies) + fraction_zero_one * (max(energies) - min(energies))
#     mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
#     tot_final_energy = sum(v.E for v in mod_voxels)
#     final_trks = make_track_graphs(mod_voxels)
#     final_trk_energies = [sum(vox.E for vox in t.nodes()) for t in final_trks]
#     final_trk_energies.sort()
#
#     assert tot_initial_energy == approx(tot_final_energy)
#     assert np.allclose(ini_trk_energies, final_trk_energies)
#
#
# @mark.parametrize("energy_type", HitEnergy)
# @given(hits                       = p_hits(),
#        requested_voxel_dimensions = box_sizes,
#        min_voxels                 = min_n_of_voxels,
#        fraction_zero_one          = fraction_zero_one)
# def test_dropped_voxels_have_nan_energy(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one, energy_type):
#     voxels            = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
#     energies          = [v.E for v in voxels]
#     e_thr             = min(energies) + fraction_zero_one * (max(energies) - min(energies))
#     _, dropped_voxels = drop_end_point_voxels(voxels, e_thr, min_voxels)
#     for voxel in dropped_voxels:
#         assert np.isnan(voxel.E)
#         for hit in voxel.hits:
#             assert np.isnan(getattr(hit, energy_type.value))
#
#
# @mark.parametrize("energy_type", HitEnergy)
# @given(hits                       = p_hits(),
#        requested_voxel_dimensions = box_sizes,
#        min_voxels                 = min_n_of_voxels,
#        fraction_zero_one          = fraction_zero_one)
# def test_drop_end_point_voxels_doesnt_modify_other_energy_types(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one, energy_type):
#     def energy_from_hits(voxel, e_type):
#         return [getattr(hit, e_type) for hit in voxel.hits]
#
#     voxels     = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
#     voxels     = sorted(voxels, key=attrgetter("xyz"))
#     energies   = [v.E for v in voxels]
#     e_thr      = min(energies) + fraction_zero_one * (max(energies) - min(energies))
#     mod, drop  = drop_end_point_voxels(voxels, e_thr, min_voxels)
#     new_voxels = sorted(mod + drop, key=attrgetter("xyz"))
#
#     for e_type in HitEnergy:
#         if e_type is energy_type: continue
#
#         for v_before, v_after in zip(voxels, new_voxels):
#             for h_before, h_after in zip(v_before.hits, v_after.hits):
#                 #assert sum(energy_from_hits(v_before, e_type.value)) == sum(energy_from_hits(v_after, e_type.value))
#                 assert np.isclose(getattr(h_before, e_type.value), getattr(h_after, e_type.value))
#
#
# @mark.parametrize("energy_type", HitEnergy)
# @given(hits                       = p_hits(),
#        requested_voxel_dimensions = box_sizes,
#        min_voxels                 = min_n_of_voxels,
#        fraction_zero_one          = fraction_zero_one)
# def test_drop_voxels_voxel_energy_is_sum_of_hits_general(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one, energy_type):
#     voxels        = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
#     energies      = [v.E for v in voxels]
#     e_thr         = min(energies) + fraction_zero_one * (max(energies) - min(energies))
#     mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
#     for v in mod_voxels:
#         assert np.isclose(v.E, sum(getattr(h, energy_type.value) for h in v.hits))
#
#
#
# @mark.parametrize("energy_type", HitEnergy)
# @given(hits                       = p_hits(),
#        requested_voxel_dimensions = box_sizes,
#        min_voxels                 = min_n_of_voxels,
#        fraction_zero_one          = fraction_zero_one)
# def test_drop_end_point_voxels_constant_number_of_voxels_and_hits(hits, requested_voxel_dimensions, min_voxels, fraction_zero_one, energy_type):
#     voxels           = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
#     energies         = [v.E for v in voxels]
#     e_thr            = min(energies) + fraction_zero_one * (max(energies) - min(energies))
#     new_voxels       = drop_end_point_voxels(voxels, e_thr, min_voxels)
#     (mod_voxels,
#      dropped_voxels) = new_voxels
#     assert len(mod_voxels) + len(dropped_voxels) == len(voxels)
#     assert sum(1 for vs in new_voxels for v in vs for h in v.hits) == len(hits)
#
#
# def test_initial_voxels_are_the_same_after_dropping_voxels(ICDATADIR):
#
#     # Get some test data: nothing interesting to see here
#     hit_file = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
#     evt_number = 19
#     e_thr = 5867.92
#     min_voxels = 3
#     size = 15.
#     vox_size = np.array([size,size,size], dtype=np.float16)
#     all_hits = load_hits(hit_file)
#     hits = all_hits[evt_number].hits
#     voxels = voxelize_hits(hits, vox_size, strict_voxel_size=False)
#
#     # This is the core of the test: collect data before/after ...
#     ante_energies  = [v.E   for v in voxels]
#     ante_positions = [v.XYZ for v in voxels]
#     mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
#     post_energies  = [v.E   for v in voxels]
#     post_positions = [v.XYZ for v in voxels]
#
#     ante_energies.sort()
#     post_energies.sort()
#     ante_positions.sort()
#     post_positions.sort()
#
#     # ... and make sure that nothing has changed
#     assert len(ante_energies)  == len(post_energies)
#     assert len(ante_positions) == len(post_positions)
#     assert np.allclose(ante_energies,  post_energies)
#     assert np.allclose(ante_positions, post_positions)
#
#
# def test_tracks_with_dropped_voxels(ICDATADIR):
#     hit_file = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
#     evt_number = 19
#     e_thr = 5867.92
#     min_voxels = 3
#     size = 15.
#     vox_size = np.array([size,size,size],dtype=np.float16)
#
#     all_hits = load_hits(hit_file)
#     hits = all_hits[evt_number].hits
#     voxels = voxelize_hits(hits, vox_size, strict_voxel_size=False)
#     ini_trks = make_track_graphs(voxels)
#     initial_n_of_tracks = len(ini_trks)
#     ini_energies = [sum(vox.E for vox in t.nodes()) for t in ini_trks]
#     ini_n_voxels = np.array([len(t.nodes()) for t in ini_trks])
#
#     mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
#
#     trks = make_track_graphs(mod_voxels)
#     n_of_tracks = len(trks)
#     energies = [sum(vox.E for vox in t.nodes()) for t in trks]
#     n_voxels = np.array([len(t.nodes()) for t in trks])
#
#     expected_diff_n_voxels = np.array([0, 0, 2])
#
#     ini_energies.sort()
#     energies.sort()
#
#     assert initial_n_of_tracks == n_of_tracks
#     assert np.allclose(ini_energies, energies)
#     assert np.all(ini_n_voxels - n_voxels == expected_diff_n_voxels)
#
#
# def test_drop_voxels_deterministic(ICDATADIR):
#     hit_file   = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
#     evt_number = 19
#     e_thr      = 5867.92
#     min_voxels = 3
#     vox_size   = [15.] * 3
#
#     all_hits        = load_hits(hit_file)
#     hits            = all_hits[evt_number].hits
#     voxels          = voxelize_hits(hits, vox_size, strict_voxel_size=False)
#     mod_voxels  , _ = drop_end_point_voxels(sorted(voxels, key = lambda v:v.E, reverse = False), e_thr, min_voxels)
#     mod_voxels_r, _ = drop_end_point_voxels(sorted(voxels, key = lambda v:v.E, reverse = True ), e_thr, min_voxels)
#
#     for v1, v2 in zip(sorted(mod_voxels, key = lambda v:v.E), sorted(mod_voxels_r, key = lambda v:v.E)):
#         assert np.isclose(v1.E, v2.E)
#
#
# def test_voxel_drop_in_short_tracks():
#     hits = [BHit(10, 10, 10, 1), BHit(26, 10, 10, 1)]
#     voxels = voxelize_hits(hits, [15,15,15], strict_voxel_size=True)
#     e_thr = sum(v.E for v in voxels) + 1.
#     min_voxels = 0
#
#     mod_voxels, _ = drop_end_point_voxels(voxels, e_thr, min_voxels)
#
#     assert len(mod_voxels) >= 1
#
#
# def test_drop_voxels_voxel_energy_is_sum_of_hits():
#     def make_hit(x, y, z, e):
#         return Hit(peak_number = 0,
#                    cluster     = Cluster(0, xy(x, y), xy(0, 0), 1),
#                    z           = z,
#                    s2_energy   = e,
#                    peak_xy     = xy(0, 0),
#                    Ep          = e)
#
#     # Create a track with an extreme to be dropped and two hits at the same
#     # distance from the barycenter of the the voxel to be dropped with
#     # different energies in the hits *and* in the voxels
#     voxels = [Voxel( 0,  0, 0, 0.1, size=5, e_type=HitEnergy.Ep, hits = [make_hit( 0,  0, 0, 0.1)                          ]),
#               Voxel( 5, -5, 0, 1.0, size=5, e_type=HitEnergy.Ep, hits = [make_hit( 5, -5, 0, 0.7), make_hit( 5, -8, 0, 0.3)]),
#               Voxel( 5,  5, 0, 1.5, size=5, e_type=HitEnergy.Ep, hits = [make_hit( 5,  5, 0, 0.9), make_hit( 5,  8, 0, 0.6)]),
#               Voxel(10,  0, 0, 2.0, size=5, e_type=HitEnergy.Ep, hits = [make_hit(10,  5, 0, 1.2), make_hit(11,  5, 0, 0.8)]),
#               Voxel(15,  0, 0, 2.5, size=5, e_type=HitEnergy.Ep, hits = [make_hit(15,  0, 0, 1.8), make_hit(11,  0, 0, 0.7)]),
#               Voxel(20,  0, 0, 3.0, size=5, e_type=HitEnergy.Ep, hits = [make_hit(20,  0, 0, 1.5), make_hit(11,  0, 0, 1.5)])]
#
#     modified_voxels, _ = drop_end_point_voxels(voxels, energy_threshold = 0.5, min_vxls = 1)
#     for v in modified_voxels:
#         assert np.isclose(v.E, sum(h.Ep for h in v.hits))
#
#
# @parametrize('radius, expected',
#              ((10., ( 60,  20)),
#               (12., ( 60,  60)),
#               (14., (100,  60)),
#               (16., (120,  80)),
#               (18., (120,  80)),
#               (20., (140,  80)),
#               (22., (140, 100))
#  ))
# def test_blobs(radius, expected):
#     hits = [BHit(105.0, 125.0, 77.7, 10),
#             BHit( 95.0, 125.0, 77.7, 10),
#             BHit( 95.0, 135.0, 77.7, 10),
#             BHit(105.0, 135.0, 77.7, 10),
#             BHit(105.0, 115.0, 77.7, 10),
#             BHit( 95.0, 115.0, 77.7, 10),
#             BHit( 95.0, 125.0, 79.5, 10),
#             BHit(105.0, 125.0, 79.5, 10),
#             BHit(105.0, 135.0, 79.5, 10),
#             BHit( 95.0, 135.0, 79.5, 10),
#             BHit( 95.0, 115.0, 79.5, 10),
#             BHit(105.0, 115.0, 79.5, 10),
#             BHit(115.0, 125.0, 79.5, 10),
#             BHit(115.0, 125.0, 85.2, 10)]
#     vox_size = np.array([15.,15.,15.],dtype=np.float16)
#     voxels = voxelize_hits(hits, vox_size)
#     tracks = make_track_graphs(voxels)
#
#     assert len(tracks) == 1
#     assert blob_energies(tracks[0], radius) == expected
#
#
# @given(p_hits, box_sizes, radius)
# def test_blob_hits_are_inside_radius(hits, voxel_dimensions, blob_radius):
#     voxels = voxelize_hits(hits, voxel_dimensions)
#     tracks = make_track_graphs(voxels)
#     for t in tracks:
#         Ea, Eb, hits_a, hits_b   = blob_energies_and_hits(t, blob_radius)
#         centre_a, centre_b       = blob_centres(t, blob_radius)
#
#         for h in hits_a:
#             assert np.linalg.norm(h.XYZ - centre_a) < blob_radius
#         for h in hits_b:
#             assert np.linalg.norm(h.XYZ - centre_b) < blob_radius
#
#
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
# @mark.parametrize("energy_type", (HitEnergy.Ec, HitEnergy.Ep))
# @given(hits                       = p_hits(),
#        requested_voxel_dimensions = box_sizes,
#        blob_radius                = radius,
#        fraction_zero_one          = fraction_zero_one)
# def test_paolina_functions_with_hit_energy_different_from_default_value(hits, requested_voxel_dimensions, blob_radius, fraction_zero_one, energy_type):
#     voxels   = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False)
#     voxels_c = voxelize_hits(hits, requested_voxel_dimensions, strict_voxel_size=False, energy_type=energy_type)
#
#     # The first assertion is needed for the test to keep being meaningful,
#     # in case we change the default value of energy_type to energy_c.
#     assert voxels[0].Etype   != voxels_c[0].Etype
#     assert voxels_c[0].Etype == energy_type.value
#
#     for voxel in voxels_c:
#         assert np.isclose(voxel.E, sum(getattr(h, energy_type.value) for h in voxel.hits))
#
#     energies_c = [v.E for v in voxels_c]
#     e_thr = min(energies_c) + fraction_zero_one * (max(energies_c) - min(energies_c))
#     # Test that this function doesn't fail
#     mod_voxels_c, _ = drop_end_point_voxels(voxels_c, e_thr, min_vxls=0)
#
#     tot_energy     = sum(getattr(h, energy_type.value) for v in voxels_c     for h in v.hits)
#     tot_mod_energy = sum(getattr(h, energy_type.value) for v in mod_voxels_c for h in v.hits)
#
#     assert np.isclose(tot_energy, tot_mod_energy)
#
#     tot_default_energy     = sum(h.E for v in voxels_c     for h in v.hits)
#     tot_mod_default_energy = sum(h.E for v in mod_voxels_c for h in v.hits)
#
#     # We don't want to modify the default energy of hits, if the voxels are made with energy_c
#     if len(mod_voxels_c) < len(voxels_c):
#         assert tot_default_energy > tot_mod_default_energy
#
#
# def test_make_tracks_function(ICDATADIR):
#
#     # Get some test data
#     hit_file    = os.path.join(ICDATADIR, 'tracks_0000_6803_trigger2_v0.9.9_20190111_krth1600.h5')
#     evt_number  = 19
#     size        = 15.
#     voxel_size  = np.array([size,size,size], dtype=np.float16)
#     blob_radius = 21*units.mm
#
#     # Read the hits and voxelize
#     all_hits = load_hits(hit_file)
#
#     for evt_number, hit_coll in all_hits.items():
#         evt_hits = hit_coll.hits
#         evt_time = hit_coll.time
#         voxels   = voxelize_hits(evt_hits, voxel_size, strict_voxel_size=False, energy_type=HitEnergy.E)
#
#         tracks   = list(make_track_graphs(voxels))
#
#         track_coll = make_tracks(evt_number, evt_time, voxels, voxel_size,
#                                  contiguity=Contiguity.CORNER,
#                                  blob_radius=blob_radius,
#                                  energy_type=HitEnergy.E)
#         tracks_from_coll = track_coll.tracks
#
#         tracks.sort          (key=lambda x : len(x.nodes()))
#         tracks_from_coll.sort(key=lambda x : x.number_of_voxels)
#
#         # Compare the two sets of tracks
#         assert len(tracks) == len(tracks_from_coll)
#         for i in range(len(tracks)):
#             t  = tracks[i]
#             tc = tracks_from_coll[i]
#
#             assert len(t.nodes())                   == tc.number_of_voxels
#             assert sum(v.E for v in t.nodes()) == tc.E
#
#             tc_blobs = list(tc.blobs)
#             tc_blobs.sort(key=lambda x : x.E)
#             tc_blob_energies = (tc.blobs[0].E, tc.blobs[1].E)
#
#             assert np.allclose(blob_energies(t, blob_radius), tc_blob_energies)
#
#
# @given(p_hits, box_sizes)
# def test_make_voxel_graph_keeps_energy_consistence(hits, voxel_dimensions):
#     voxels = voxelize_hits    (hits  , voxel_dimensions)
#     tracks = make_track_graphs(voxels)
#     # assert sum of track energy equal to sum of hits energies
#     assert_almost_equal(sum(get_track_energy(track) for track in tracks), sum(h.E for h in hits))
