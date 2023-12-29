import torch
import torch.nn.functional as F


def gumbel_like(*args, **kwargs):
    return _gumbel(torch.rand_like(*args, **kwargs))


def gumbel(*args, **kwargs):
    return _gumbel(torch.rand(*args, **kwargs))


def _gumbel(u):
    return -torch.log(-torch.log(u))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return torch.where(
        x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
        -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
        torch.log(-torch.expm1(-torch.exp(-x)))  # Hope for the best
    )


def gumbel_with_maximum(phi, T, dim=-1):
    """
    Samples a set of gumbels which are conditioned on having a maximum along a dimension
    phi.max(dim)[0] should be broadcastable with the desired maximum T
    """

    # if ((-1e9 < phi) & (phi< -100)).any():
    #     print("habemus problemus")

    NEG_INF_THRESHOLD = -float('inf')

    # phi = phi.clamp(min=-1e10)
    inf_rows_mask = (phi == NEG_INF_THRESHOLD).all(dim=1)
    if inf_rows_mask.any():
        old_phi = phi.clone()
        phi = phi[~inf_rows_mask]
        old_T = T.clone()
        T = T[~inf_rows_mask]

    # Gumbel with location phi
    g_phi = phi + gumbel_like(phi)
    Z, argmax = g_phi.max(dim)
    g = _shift_gumbel_maximum(g_phi, T, dim, Z=Z)
    CHECK_VALIDITY = True
    if CHECK_VALIDITY:
        g_inv = _shift_gumbel_maximum(g, Z, dim)
        # Create a boolean mask where the condition fails
        mask = ~(((g_phi - g_inv) < 1e-3) | (g_phi == g_inv))
        # | torch.isinf(g_phi).all(dim).repeat_interleave(g_phi.shape[-1], dim).view(g_phi.shape))

        # Get indices where the mask is True (i.e., the assertion fails)
        fail_indices = torch.nonzero(mask, as_tuple=True)

        # Print the failed indices and corresponding values
        for ind in zip(*fail_indices):
            print(f"Index: {ind}, phi: {phi[ind]}, g_phi: {g_phi[ind]}, g_inv: {g_inv[ind]}")

        assert (((g_phi - g_inv) < 1e-3) | (g_phi == g_inv)).all()
        # | torch.isinf(g_phi).all(dim).repeat_interleave(g_phi.shape[-1], dim).view(g_phi.shape)).all()
        # if a row in g_phi is all -inf, set the corresponding row in g to -inf
        # g[torch.isinf(g_phi).all(dim)] = float('-inf')
    if inf_rows_mask.any():
        g_new = torch.full_like(old_phi, NEG_INF_THRESHOLD, device=old_phi.device)
        g_new[~inf_rows_mask] = g
        g = g_new
        argmax_new = torch.full_like(old_T, -1,dtype=torch.long, device=old_T.device)
        argmax_new[~inf_rows_mask] = argmax
        argmax = argmax_new
    return g, argmax


def _shift_gumbel_maximum(g_phi, T, dim=-1, Z=None):
    if Z is None:
        Z, _ = g_phi.max(dim)
    u = T.unsqueeze(dim) - g_phi + torch.log1p(-torch.exp(g_phi - Z.unsqueeze(dim)))
    return T.unsqueeze(dim) - F.relu(u) - torch.log1p(torch.exp(-u.abs()))
