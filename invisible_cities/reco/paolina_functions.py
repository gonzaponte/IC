from itertools   import combinations
from operator    import itemgetter
from collections import OrderedDict

import numpy    as np
import pandas   as pd
import networkx as nx

from networkx           import Graph
from .. core.exceptions import NoHits
from .. core.exceptions import NoVoxels
from .. core            import system_of_units as units
from .. types.symbols   import Contiguity
from .. types.symbols   import HitEnergy

from typing import Sequence
from typing import Tuple


Hits    = pd.DataFrame
Voxels  = pd.DataFrame
VoxSize = np.ndarray

MAX3D  = np.array([float(' inf')] * 3)
MIN3D  = np.array([float('-inf')] * 3)
BOXEPS = 3e-12


def _centre(edges):
    high = np.asarray(edges[1:  ])
    low  = np.asarray(edges[ :-1])
    return (high + low) / 2.


def _size(edges):
    high = np.asarray(edges[1:  ])
    low  = np.asarray(edges[ :-1])
    return (high - low)


def bounding_box(hits : pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
        Returns two arrays defining the coordinates of a box that
        bounds the voxels.
    """
    lower = [hits.X.min(), hits.Y.min(), hits.Z.min()]
    upper = [hits.X.max(), hits.Y.max(), hits.Z.max()]
    return np.array(lower), np.array(upper)


def digitize(variable : np.ndarray, edges : np.ndarray) -> np.ndarray:
    index  = np.digitize(variable, edges, right=False) - 1
    nvoxel = len(edges) - 1
    # Force hits on the rightmost edge to fall into the last bin
    index[index == nvoxel] = nvoxel - 1
    return index


def get_bins( hits             : pd.DataFrame
            , voxel_dimensions : np.ndarray
            , strict_vox_size  : bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hlo, hhi  = bounding_box(hits)
    bb_centre = _centre([hlo, hhi])[0]
    bb_size   = _size  ([hlo, hhi])[0]

    n_voxels = np.ceil(bb_size / voxel_dimensions).astype(int)
    n_voxels = np.clip(n_voxels, a_min=1, a_max=None)

    range = n_voxels * voxel_dimensions if strict_vox_size else bb_size
    half_range = range / 2

    voxel_edges_lo = bb_centre - half_range
    voxel_edges_hi = bb_centre + half_range

    # Expand the voxels a tiny bit, in order to include hits which
    # fall within the margin of error of the  bunch_of_correctedvoxel bounding box.
    eps = 3e-12 # geometric mean of range that seems to work
    voxel_edges_lo -= eps
    voxel_edges_hi += eps

    bins_x = np.linspace(voxel_edges_lo[0], voxel_edges_hi[0], n_voxels[0] + 1)
    bins_y = np.linspace(voxel_edges_lo[1], voxel_edges_hi[1], n_voxels[1] + 1)
    bins_z = np.linspace(voxel_edges_lo[2], voxel_edges_hi[2], n_voxels[2] + 1)

    return bins_x, bins_y, bins_z


def voxelize_hits(hits             : pd.DataFrame,
                  voxel_dimensions : np.ndarray,
                  strict_voxel_size: bool = False,
                  energy_type      : HitEnergy = HitEnergy.E) -> Tuple[Hits, Voxels, VoxSize]:
    """
        Groups hits in voxels.
    """
    # 1. Find bounding box of all hits.
    # 2. Allocate hits to regular sub-boxes within bounding box, using histogramdd.
    # 3. Calculate voxel energies by summing energies of hits within each sub-box.
    if not len(hits):
        raise NoHits

    bins_x, bins_y, bins_z = get_bins(hits, voxel_dimensions, strict_voxel_size)

    hits = hits.assign( voxel_i = digitize(hits.X, bins_x)
                      , voxel_j = digitize(hits.Y, bins_y)
                      , voxel_k = digitize(hits.Z, bins_z)
                      , voxel   = np.nan) # to be set later

    voxel_summary = dict( E     = "sum"
                        , Ec    = "sum"
                        , Ep    = "sum"
                        , event = "first"
                        , npeak = "first"
                        , Q     = "count")
    voxels = (hits.groupby("voxel_i voxel_j voxel_k".split())
                  .agg(voxel_summary)
                  .rename(columns = dict(Q = "nhits"))
                  .reset_index()
                  .assign( X = lambda df: _centre(bins_x)[df.voxel_i]
                         , Y = lambda df: _centre(bins_y)[df.voxel_j]
                         , Z = lambda df: _centre(bins_z)[df.voxel_k]
                         ))

    for ivoxel, voxel in voxels.iterrows():
        sel = ((hits.voxel_i == voxel.voxel_i) &
               (hits.voxel_j == voxel.voxel_j) &
               (hits.voxel_k == voxel.voxel_k) )
        hits.loc[sel, "voxel"] = ivoxel

    vox_size = np.array([ bins_x[1] - bins_x[0]
                        , bins_y[1] - bins_y[0]
                        , bins_z[1] - bins_z[0]]
                       , dtype=float)

    return hits, voxels, vox_size


def distance_between_voxels(v1 : pd.Series, v2 : pd.Series) -> float:
    xyz1 = v1["X Y Z".split()].values
    xyz2 = v2["X Y Z".split()].values
    return np.linalg.norm(xyz1 - xyz2)


def neighbours( v1         : pd.Series
              , v2         : pd.Series
              , vox_size   : np.array
              , contiguity : Contiguity = Contiguity.CORNER) -> bool:
    xyz1 = v1["X Y Z".split()].values
    xyz2 = v2["X Y Z".split()].values
    return np.linalg.norm((xyz1 - xyz2) / vox_size) < contiguity.value


def make_track_graphs( voxels     : pd.DataFrame
                     , vox_size   : np.array
                     , contiguity : Contiguity = Contiguity.CORNER) -> Sequence[Graph]:
    """
        Create a graph where the voxels are the nodes and the edges are any
        pair of neighbour voxel. Two voxels are considered to be
        neighbours if their distance normalized to their size is smaller
        than a contiguity factor.
    """
    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels.index)
    for (i1, v1), (i2, v2) in combinations(voxels.iterrows(), 2):
        if neighbours(v1, v2, vox_size, contiguity):
            d = distance_between_voxels(v1, v2)
            voxel_graph.add_edge(i1, i2, distance=d)

    return tuple(connected_component_subgraphs(voxel_graph))


def connected_component_subgraphs(G):
    return (G.subgraph(c).copy() for c in nx.connected_components(G))


# TODO: Update to Python 3.9 so this is valid
Distances        = "OrderedDict[int, float]"
PairsOfDistances = "OrderedDict[int, Distances]"

def shortest_paths(track_graph : Graph) -> PairsOfDistances:
    """Compute shortest path lengths between all nodes in a weighted graph."""
    distances = dict(nx.all_pairs_dijkstra_path_length(track_graph, weight='distance'))

    # sort the output so the result is reproducible
    distances = ((i1, OrderedDict(sorted(dmap.items())))
                 for i1, dmap in sorted(distances.items()))
    return OrderedDict(distances)



def find_extrema(track : Graph) -> Tuple[int, int, float]:
    """Find the extrema and the length of a track, given its dictionary of distances."""
    distances = shortest_paths(track)

    if not distances:
        raise NoVoxels

    if len(distances) == 1:
        only_voxel = next(iter(distances))
        return (only_voxel, only_voxel, 0.)

    first, last, max_distance = None, None, 0
    for (i1, dist_from_v1_to), (i2, _) in combinations(distances.items(), 2):
        d = dist_from_v1_to[i2]
        if d > max_distance:
            first, last, max_distance = i1, i2, d
    return first, last, max_distance


def voxels_within_radius( distances : Distances
                        , radius    : float) -> Sequence[int]:
    indices = [i for i, d in distances.items() if d<radius]
    return indices


def energy_of_voxels_within_radius( voxels    : pd.DataFrame
                                  , distances : Distances
                                  , radius    : float
                                  , e_type    : HitEnergy) -> float:
    indices = voxels_within_radius(distances, radius)
    return voxels.loc[indices, e_type].sum()


def blob_centre( hits  : pd.DataFrame
               , voxel : pd.Series
               , etype : HitEnergy) -> Tuple[float, float, float]:
    """Calculate the blob position, starting from the end-point voxel."""
    hits = hits.loc[hits.voxel == voxel.name]

    etype = etype.value
    if hits[etype].sum() > 0:
        pos    = hits.loc[:, list("XYZ")].values
        energy = hits.loc[:,       etype].values
        centre = np.average(pos, weights=energy, axis=0)
    else:
        centre = [voxel.X, voxel.Y, voxel.Z]

    return np.asarray(centre)


def hits_in_blob( hits          : pd.DataFrame
                , track_graph   : Graph
                , radius        : float
                , vox_size      : np.ndarray
                , extreme_voxel : pd.Series
                , etype         : HitEnergy) -> pd.DataFrame:
    """Returns the hits that belong to a blob."""
    distances         = shortest_paths(track_graph)
    dist_from_extreme = distances[extreme_voxel.name]
    blob_pos          = blob_centre(hits, extreme_voxel, etype)
    diag              = np.linalg.norm(vox_size)

    blob_hits = []
    # First, consider only voxels at a certain distance from the end-point, along the track.
    # We allow for 1 extra contiguity, because this distance is calculated between
    # the centres of the voxels, and not the hits. In the second step we will refine the
    # selection, using the euclidean distance between the blob position and the hits.
    for i in track_graph.nodes():
        voxel_distance = dist_from_extreme[i]
        if voxel_distance < radius + diag:
            hits_in_voxel = hits         .loc[hits.voxel == i]
            xyz           = hits_in_voxel.loc[:, list("XYZ")].values
            within_radius = np.linalg.norm(xyz - blob_pos, axis=1) < radius
            if np.count_nonzero(within_radius):
                blob_hits.extend(hits_in_voxel.loc[within_radius].index)

    return hits.loc[blob_hits]


#            energy centre     hits        voxel
Blob = Tuple[float, float, pd.DataFrame, pd.Series]
def find_blobs( hits        : pd.DataFrame
              , voxels      : pd.DataFrame
              , vox_size    : np.ndarray
              , track_graph : Graph
              , radius      : float
              , e_type      : HitEnergy) -> Tuple[Blob, Blob]:
    """Return the energies, the hits and the positions of the blobs.
       For each pair of observables, the one of the blob of largest energy is returned first."""
    distances = shortest_paths(track_graph)
    idx_a, idx_b, _   = find_extrema(track_graph)

    a = voxels.loc[idx_a]
    b = voxels.loc[idx_b]

    hits_a = hits_in_blob(hits, track_graph, radius, vox_size, a, e_type)
    hits_b = hits_in_blob(hits, track_graph, radius, vox_size, b, e_type)

    E_a = hits_a.loc[:, e_type.value].sum()
    E_b = hits_b.loc[:, e_type.value].sum()

    # Consider the case where voxels are built without associated hits
    if len(hits_a) == 0 and len(hits_b) == 0 :
        E_a = energy_of_voxels_within_radius(distances[a], radius)
        E_b = energy_of_voxels_within_radius(distances[b], radius)

    pos_a = blob_centre(hits, a, e_type)
    pos_b = blob_centre(hits, b, e_type)

    blob_a = (E_a, pos_a, hits_a, a)
    blob_b = (E_b, pos_b, hits_b, b)

    return sorted((blob_a, blob_b), key=itemgetter(0), reverse=True)


def make_tracks( hits        : pd.DataFrame
               , voxels      : pd.DataFrame
               , vox_size    : np.ndarray
               , contiguity  : Contiguity
               , blob_radius : float
               , etype       : HitEnergy
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    track_graphs = make_track_graphs(voxels, vox_size, contiguity)
    hits         = hits.assign  (track=np.nan, blob=np.nan)
    voxels       = voxels.assign(track=np.nan)
    tracks       = []
    for i, trk in enumerate(track_graphs):
        blob_a, blob_b = find_blobs(hits, voxels, vox_size, trk, blob_radius, etype)
        E_a, pos_a, hits_a, vox_a = blob_a
        E_b, pos_b, hits_b, vox_b = blob_b

        indices = list(trk.nodes())
        voxels.loc[indices, "track"] = i

        hits.loc[hits_a.index, "blob"] = 1
        hits.loc[hits_b.index, "blob"] = 2

        E     = voxels.loc[indices, etype.value].sum()
        track = dict( event   = hits.event.iloc[0]
                    , nvoxels = len(indices)
                    , E       = E
                    , Eblob1  = E_a
                    , Eblob2  = E_b
                    , Vblob1  = vox_a.name
                    , Vblob2  = vox_b.name
                    , Xblob1  = pos_a[0]
                    , Xblob2  = pos_b[0]
                    , Yblob1  = pos_a[1]
                    , Yblob2  = pos_b[1]
                    , Zblob1  = pos_a[2]
                    , Zblob2  = pos_b[2]
                    )
        tracks.append(pd.DataFrame(track, index=[0]))

    hits.loc[:, "track"] = voxels.track[hits.voxel.values].values

    tracks = pd.concat(tracks, ignore_index=True)
    return hits, voxels, tracks


def drop_voxel_in_place( hits     : pd.DataFrame
                       , voxels   : pd.DataFrame
                       , voxel_id : int
                       , vox_size : np.ndarray
                       , e_type   : HitEnergy) -> Tuple[pd.DataFrame, pd.Series]:
    """
        Eliminate an individual voxela and reassign its energy to
        the closest hits to the barycenter of the eliminated voxel hits, provided that it
        belongs to a neighbour voxel.
    """
    xyz = list("XYZ")
    e_type = e_type.value

    the_voxel      = voxels.loc[voxel_id].copy()
    the_voxel_hits = hits.loc[hits.voxel == voxel_id].copy()
    hits_ids       = the_voxel_hits.index

    neighbour_ids = [i for i, v in voxels.iterrows()
                       if i != voxel_id and neighbours(the_voxel, v, vox_size)]
    neighbour_hits = hits.loc[hits.voxel.isin(neighbour_ids)]

    if the_voxel.nhits > 0:
        vpos = np.average(           the_voxel_hits[xyz]
                         , weights = the_voxel_hits[e_type]
                         , axis    = 0)

        distances    = np.linalg.norm(neighbour_hits[xyz].values - vpos, axis=1)
        closest_idx  = np.argwhere(np.isclose(distances, distances.min())).flatten()
        closest_hits = neighbour_hits.iloc[closest_idx]

        hit_es  = closest_hits[e_type]
        delta_e = hit_es / hit_es.sum() * the_voxel[e_type]
        hits.loc[closest_hits.index, e_type] += delta_e

        for v, hs in hits.groupby("voxel"):
            voxels.loc[v, e_type] = hs[e_type].sum()

    else:
        vpos = the_voxel[xyz]
        neighbour_voxels  = voxels.loc[neighbour_vox_ids]

        distances      = np.linalg.norm(neighbour_voxels[xyz].values - vpos, axis=1)
        closest_idx    = np.argwhere(np.isclose(distances, distances.min())).flatten()
        closest_voxels = voxels.iloc[closest_idx]

        vox_es  = closest_voxels[e_type]
        delta_e = vox_es / vox_es.sum() * the_voxel[e_type]
        voxels.loc[closest_voxels.index, e_type] += delta_e

    voxels.drop(voxel_id, inplace=True)
    hits  .drop(hits_ids, inplace=True)
    the_voxel_hits.loc[:, e_type] = np.nan
    the_voxel     .loc[   e_type] = np.nan
    return the_voxel_hits, the_voxel


def drop_end_point_voxels( hits       : pd.DataFrame
                         , voxels     : pd.DataFrame
                         , vox_size   : np.ndarray
                         , threshold  : float
                         , min_voxels : int
                         , e_type     : HitEnergy) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Eliminate voxels at the end-points of a track, recursively,
        if their energy is lower than a threshold.
    """
    hits   =   hits.copy()
    voxels = voxels.copy()

    dropped_hits   = []
    dropped_voxels = []
    modified       = True
    while modified:
        modified = False
        tracks = make_track_graphs(voxels, vox_size)

        for i, t in enumerate(tracks):
            if len(t.nodes()) < min_voxels:
                continue

            for i_extreme in find_extrema(t)[:-1]: # last one is a distance
                extreme = voxels.loc[i_extreme]
                if extreme.E < threshold:
                    ### be sure that the voxel to be eliminated has at least one neighbour
                    ### beyond itself
                    n_neighbours = sum(neighbours(extreme, v, vox_size) for _, v in voxels.iterrows())
                    if n_neighbours > 1:
                        d_hits, d_voxel = drop_voxel_in_place(hits, voxels, i_extreme, vox_size, e_type)
                        dropped_hits.append(d_hits)
                        dropped_voxels.append(d_voxel.to_frame().T)
                        modified = True

    dropped_hits   = pd.concat(dropped_hits  ) if dropped_hits   else hits  .iloc[:0]
    dropped_voxels = pd.concat(dropped_voxels) if dropped_voxels else voxels.iloc[:0]
    return hits, voxels, dropped_hits, dropped_voxels


def track_energy_calculator(hits):
    def track_energy(track):
        sel = hits.voxel.in1d(track.nodes())
        return hits[sel].Ep.sum()
    return track_energy
