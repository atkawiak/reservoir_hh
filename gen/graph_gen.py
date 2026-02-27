"""Graph generation — ER or fixed-indegree."""
from __future__ import annotations

import numpy as np
from .config import GeneratorConfig

# Block IDs: EE=0, EI=1, IE=2, II=3
BLOCK_ID_MAP = {"EE": 0, "EI": 1, "IE": 2, "II": 3}


def generate_edges(cfg: GeneratorConfig, pop: dict, seed: int) -> dict:
    """Generate directed edges for the reservoir.

    Returns dict with:
        pre: int32[] — pre-synaptic neuron indices
        post: int32[] — post-synaptic neuron indices
        block_id: uint8[] — block type (0=EE, 1=EI, 2=IE, 3=II)
    """
    E_idx = pop["E_idx"]
    I_idx = pop["I_idx"]
    is_I = pop["is_I"]

    # Block definitions: (src_indices, tgt_indices, block_name, same_pop)
    blocks = [
        (E_idx, E_idx, "EE", True),
        (E_idx, I_idx, "EI", False),
        (I_idx, E_idx, "IE", False),
        (I_idx, I_idx, "II", True),
    ]

    rng = np.random.default_rng(seed)
    pre_all, post_all, bid_all = [], [], []

    for src, tgt, bname, same_pop in blocks:
        bid = BLOCK_ID_MAP[bname]

        if cfg.graph_type == "ER":
            _gen_er(src, tgt, cfg.p_conn, same_pop, cfg.allow_self,
                    bid, rng, pre_all, post_all, bid_all)
        else:
            _gen_fixed_indegree(src, tgt, cfg.k_in, same_pop, cfg.allow_self,
                                bid, rng, pre_all, post_all, bid_all)

    return {
        "pre": np.concatenate(pre_all).astype(np.int32) if pre_all else np.zeros(0, np.int32),
        "post": np.concatenate(post_all).astype(np.int32) if post_all else np.zeros(0, np.int32),
        "block_id": np.concatenate(bid_all).astype(np.uint8) if bid_all else np.zeros(0, np.uint8),
    }


def _gen_er(src, tgt, p, same_pop, allow_self, bid, rng, pre_all, post_all, bid_all):
    """ER random graph for one block (vectorized)."""
    n_src = len(src)
    n_tgt = len(tgt)
    if n_src == 0 or n_tgt == 0:
        return

    # Draw all random numbers at once
    conn = rng.random((n_src, n_tgt)) < p

    # Remove self-connections if same population and disallowed
    if same_pop and not allow_self:
        np.fill_diagonal(conn, False)

    si_idx, ti_idx = np.where(conn)
    if len(si_idx) == 0:
        return

    pre_all.append(src[si_idx].astype(np.int32))
    post_all.append(tgt[ti_idx].astype(np.int32))
    bid_all.append(np.full(len(si_idx), bid, dtype=np.uint8))


def _gen_fixed_indegree(src, tgt, k_in, same_pop, allow_self, bid, rng,
                        pre_all, post_all, bid_all):
    """Fixed in-degree graph: each target neuron gets exactly k_in inputs from src."""
    pre_list, post_list = [], []
    for ti in range(len(tgt)):
        candidates = list(range(len(src)))
        if same_pop and not allow_self:
            candidates = [c for c in candidates if c != ti]
        k = min(k_in, len(candidates))
        chosen = rng.choice(candidates, size=k, replace=False)
        for ci in chosen:
            pre_list.append(src[ci])
            post_list.append(tgt[ti])
    if pre_list:
        pre_all.append(np.array(pre_list, dtype=np.int32))
        post_all.append(np.array(post_list, dtype=np.int32))
        bid_all.append(np.full(len(pre_list), bid, dtype=np.uint8))
