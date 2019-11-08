import numpy  as np
from itertools   import compress
from copy        import deepcopy
from typing      import List
from .. evm  import event_model as evm
from .. types.ic_types      import NN
from .. types.ic_types      import xy


def fetch_attributes(hits, *attributes):
    attributes = [[getattr(hit, attribute) for attribute in attributes] for hit in hits]
    arrays     = list(map(np.asarray, zip(*attributes)))
    return arrays if len(arrays) > 1 else arrays[0]


def copy_hit(hit, new_energy=None):
    new_hit = evm.Hit(peak_number = hit.peak_number                            ,
                      cluster     = evm.Cluster(hit.Q, hit._xy, hit.var,
                                                hit.nsipm, Qc=hit.Qc   )       ,
                      z           = hit.Z                                      ,
                      s2_energy   = hit.E if new_energy is None else new_energy,
                      peak_xy     = xy(hit.Xpeak, hit.Ypeak)                   ,
                      s2_energy_c = hit.Ec                                     ,
                      track_id    = hit.track_id                               )


def split_energy(total_e, clusters):
    if len(clusters) == 1:
        return [total_e]
    qs = np.array([c.Q for c in clusters])
    return total_e * qs / np.sum(qs)


def merge_NN_hits(hits      : List[evm.Hit],
                  same_peak : bool = True  ) -> List[evm.Hit]:
    """
    Returns a list of the hits where the energies of NN hits
    are distributed to the closest hits such that the added
    energy is proportional to the hit energy. If all the hits
    were NN the function returns empty list.
    """
    nn_hits     = [h for h in hits if h.Q==NN]
    non_nn_hits = [deepcopy(h) for h in hits if h.Q!=NN]
    passed = len(non_nn_hits)>0
    if not passed:
        return []
    hits_to_correct=[]
    for nn_h in nn_hits:
        peak_num = nn_h.npeak
        if same_peak:
            hits_to_merge = [h for h in non_nn_hits if h.npeak==peak_num]
        else:
            hits_to_merge = non_nn_hits
        try:
            z_closest  = min(abs(h.Z-nn_h.Z) for h in hits_to_merge)
        except ValueError:
            continue
        h_closest = [h for h in hits_to_merge if abs(h.Z-nn_h.Z)==z_closest]
        en_tot = sum([h.E for h in h_closest])
        for h in h_closest:
            hits_to_correct.append([h,nn_h.E*(h.E/en_tot)])

    for h, en in hits_to_correct:
        h.E += en
    return non_nn_hits


def threshold_hits(hits         : List[evm.Hit],
                   threshold    : float        ,
                   on_corrected : bool = False ) -> List[evm.Hit]:
    """
    List of the hits which charge is above the threshold.
    The energy of the hits below the threshold is distributed among
    the hits in the same time slice.
    """
    if not threshold: return hits

    new_hits = []
    z, e, q = fetch_attributes(hits, "Z", "E", "Qc" if on_corrected else "Q")

    for z_slice in np.unique(z):
        sel_z  = z  == z_slice
        sel_q  = q  >= threshold
        sel    = sel_z & sel_q

        e_slice = np.sum(e[sel_z])

        if np.count_nonzero(sel):
            hits_pass_th = list(compress(hits, sel))
            energies     = split_energy(e_slice, hits_pass_th)
            for hit, ei in zip(hits_pass_th, energies):
                # This is faster than deepcopy(hit)
                new_hit = copy_hit(hit, new_energy=e_slice)
                new_hits.append(hit)
        else:
            first   = next(compress(hits, sel_z))
            hit = evm.Hit(first.npeak,
                          evm.Cluster(NN, xy(0,0), xy(0,0), 0),
                          z_slice, e_slice,
                          xy(first.Xpeak, first.Ypeak))
            new_hits.append(hit)
    return new_hits
