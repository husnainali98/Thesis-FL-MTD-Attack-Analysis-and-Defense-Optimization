import numpy as np


def split_additive_multi(x: np.ndarray, num_node_shares: int, rng: np.random.Generator | None = None):
    """
    Multi-party Additive Secret Sharing.

    Creates:
        1 server share
        num_node_shares helper-node shares

    Such that:
        server_share + node_share_1 + ... + node_share_k = x

    Returns:
        server_share, node_shares_list
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random share for server
    server_share = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float64)

    # Special case: only one helper node
    if num_node_shares == 1:
        node_share = (x.astype(np.float64) - server_share).astype(np.float64)
        return server_share, [node_share]

    # Create random shares for first k-1 nodes
    node_shares = []
    running_sum = server_share.copy()

    for _ in range(num_node_shares - 1):
        sh = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float64)
        node_shares.append(sh)
        running_sum = running_sum + sh

    # Last node share makes the sum exact
    last_share = (x.astype(np.float64) - running_sum).astype(np.float64)
    node_shares.append(last_share)

    return server_share, node_shares