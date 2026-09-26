"""QViD token scoring: per-head leverage scores of the query-visual affinity."""

from __future__ import annotations

import torch


def qvid_scores(
    q_txt: torch.Tensor,
    k_vis: torch.Tensor,
    num_heads: int,
    num_components: int = 16,
) -> torch.Tensor:
    """Score visual tokens by their leverage in the top singular subspace of
    the per-head text-visual affinity matrix.

    For each head h, M_h = Q_h K_h^T has shape (N_q, N_v) with N_q << N_v.
    Its right singular vectors are obtained without an SVD of M_h:

        M_h^T = Q R               (reduced QR, Q: N_v x N_q, R: N_q x N_q)
        R^T   = U S V^T           (SVD of a small N_q x N_q matrix)
        =>  M_h = U S (Q V)^T     so the right singular vectors of M_h are Q V.

    The leverage score of visual token j in head h is the squared norm of row j
    of the top-r right singular vectors; scores are averaged over heads.

    Args:
        q_txt: (N_q, D) query projections of the text tokens.
        k_vis: (N_v, D) key projections of the visual tokens.
        num_heads: number of attention heads (D must be divisible by it).
        num_components: rank r of the singular subspace (clipped to N_q).

    Returns:
        (N_v,) float32 scores; higher means more relevant to the text.
    """
    n_q, dim = q_txt.shape
    n_v = k_vis.shape[0]
    head_dim = dim // num_heads

    q = q_txt.view(n_q, num_heads, head_dim).transpose(0, 1).float()  # (H, N_q, d)
    k = k_vis.view(n_v, num_heads, head_dim).transpose(0, 1).float()  # (H, N_v, d)
    m = q @ k.transpose(-1, -2)                                       # (H, N_q, N_v)
    r = min(num_components, n_q, n_v)

    q_qr, r_qr = torch.linalg.qr(m.mT)                                # (H, N_v, N_q), (H, N_q, N_q)
    _, _, vh = torch.linalg.svd(r_qr.mT, full_matrices=False)         # (H, N_q, N_q)
    w = (q_qr @ vh.mT)[:, :, :r]                                      # (H, N_v, r)

    return (w * w).sum(dim=-1).mean(dim=0)                            # (N_v,)


def select_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k highest scores, in ascending order (keeps spatial order)."""
    return scores.topk(k, largest=True).indices.sort().values
