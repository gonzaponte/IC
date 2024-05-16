import numpy  as np
import pandas as pd

from .. types.ic_types import NN

from typing import Optional


EPSILON = np.finfo(np.float64).eps


def make_nn_hit( peak_no : int
               , peak_x  : float
               , peak_y  : float
               , z       : float
               , e       : float
               , ec      : Optional[float] = None) -> pd.DataFrame:
    hit = dict( npeak = peak_no
              , Xpeak = peak_x
              , Ypeak = peak_y
              , X     = NN
              , Y     = NN
              , Z     = z
              , Q     = NN
              , E     = e)
    if ec is not None:
        hit.update(dict( Qc = NN
                       , Ec = ec))
    return pd.DataFrame(hit, index=[0])


def merge_NN_hits( hits      : pd.DataFrame
                 , same_peak : bool=True   ) -> pd.DataFrame:
    """
    Redistribute the energy of NN hits to the closest hits. The amount
    added is proportional to the receiving hit's energy.  The
    parameter `same_peak` controls whether the energy is shared among
    hits within the same peak or other peaks can be considered.  If
    all the hits are NN, an empty dataframe is returned.
    """
    are_nn  = hits.Q == NN
    nn_hits = hits.loc[ are_nn]
    ok_hits = hits.loc[~are_nn].reset_index(drop=True)

    if ok_hits.empty:
        return ok_hits

    # Temporary columns. Removed before returning
    ok_hits = ok_hits.assign(plus_E = 0., plus_Ec = 0.)

    for _, nn_h in nn_hits.iterrows():
        candidates = ok_hits.loc[ok_hits.npeak == nn_h.npeak] if same_peak else ok_hits
        dzs        = np.abs(candidates.Z - nn_h.Z)
        i_closest  = np.argwhere(np.isclose(dzs, dzs.min())).flatten()
        h_closest  = candidates.iloc[i_closest]

        plus_e  = nn_h.E  * h_closest.E  /  h_closest.E .sum()
        plus_ec = nn_h.Ec * h_closest.Ec / (h_closest.Ec.sum() + EPSILON)

        ok_hits.loc[h_closest.index, "plus_E" ] += plus_e
        ok_hits.loc[h_closest.index, "plus_Ec"] += plus_ec

    ok_hits.loc[:, "E" ] += ok_hits.plus_E
    ok_hits.loc[:, "Ec"] += ok_hits.plus_Ec

    return ok_hits.drop(columns="plus_E plus_Ec".split())


def threshold_hits( hits         : pd.DataFrame
                  , thr          : float
                  , on_corrected : bool=False  ) -> pd.DataFrame:
    """
    Apply a charge threshold to the hits.  The energy of the hits
    below the threshold is redistributed among the hits in the same
    time/z slice.  The parameter `on_corrected` controls whether `Q`
    or `Qc` is used for the threshold.
    """
    def process_slice(hits):
        slice_e  = np.   sum(hits.E )
        slice_ec = np.nansum(hits.Ec) + EPSILON

        qs  = hits.Qc if on_corrected else hits.Q
        sel = qs > thr
        out = hits.loc[sel].reset_index(drop=True)

        if out.empty:
            first_hit = hits.iloc[0]
            return make_nn_hit( first_hit.npeak
                              , first_hit.Xpeak
                              , first_hit.Ypeak
                              , first_hit.Z
                              , slice_e
                              , slice_ec)

        weights = out.Q / (out.Q .sum() + EPSILON)
        out.loc[:, "E" ] = slice_e  * weights
        out.loc[:, "Ec"] = slice_ec * weights
        return out


    if thr == 0:
        return hits

    return (hits.groupby("Z", as_index=False)
                .apply(process_slice))
