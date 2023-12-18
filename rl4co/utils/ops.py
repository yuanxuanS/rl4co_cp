from typing import Union

import torch

from tensordict import TensorDict
from torch import Tensor


def _batchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])  # expand把一个维度上的大小扩展为更大的； contiguous把张量在内存中的存储连续防止以便利view; 


def batchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x      # 把原来 batch_size大小，经过复制，变为batch_size*augment（s）大小。 其中 0+i 和 batch_size+i 处的值相同
    return x


def _unbatchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Undoes batchify operation for Tensordicts as well"""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.rearrange(x, '(r b) ... -> b r ...', r=repeats)` but ~2x faster and supports TensorDicts
    Repeats unbatchify operation `n` times as specified by each shape element
    If shape is a tuple, iterates over each element and unbatchifies that many times to match the tuple shape.

    Example:
    >>> x.shape: [a*b*c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a, b, c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x


def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    return src.gather(dim, idx).squeeze() if squeeze else src.gather(dim, idx)



@torch.jit.script
def get_distance(x: Tensor, y: Tensor):
    """Euclidean distance between two tensors of shape `[..., n, dim]`"""
    return (x - y).norm(p=2, dim=-1)


@torch.jit.script
def get_tour_length(ordered_locs):
    """Compute the total tour distance for a batch of ordered tours.
    Computes the L2 norm between each pair of consecutive nodes in the tour and sums them up.

    Args:
        ordered_locs: Tensor of shape [batch_size, num_nodes, 2] containing the ordered locations of the tour
    """
    ordered_locs_next = torch.roll(ordered_locs, 1, dims=-2)
    return get_distance(ordered_locs_next, ordered_locs).sum(-1)


def get_num_starts(td):
    """Returns the number of possible start nodes for the environment based on the action mask"""
    return td["action_mask"].shape[-1]


def select_start_nodes(td, env, num_nodes=None):
    """Node selection strategy as proposed in POMO (Kwon et al. 2020)
    and extended in SymNCO (Kim et al. 2022).
    Selects different start nodes for each batch element

    Args:
        td: TensorDict containing the data. We may need to access the available actions to select the start nodes
        env: Environment may determine the node selection strategy
        num_nodes: Number of nodes to select
    """
    num_nodes = get_num_starts(td) if num_nodes is None else num_nodes

    # Environments with depot: don't select the depot as start node
    if env.name in ["op", "pctsp", "spctsp", "mtsp"]:
        selected = torch.arange(1, num_nodes + 1, device=td.device).repeat_interleave(
            td.shape[0]
        )
    elif env.name == "pdp":
        # select only pickup nodes (until N//2 + 1). Note that this should be selected beforehand (e.g. for
        # 100 pickup and delivery nodes, we should select 50 as start nodes)
        selected = torch.arange(1, num_nodes + 1, device=td.device).repeat_interleave(
            td.shape[0]
        )
    else:
        selected = torch.arange(num_nodes, device=td.device).repeat_interleave(
            td.shape[0]
        )
    return selected


def get_best_actions(actions, max_idxs):
    actions = unbatchify(actions, max_idxs.shape[0])
    return actions.gather(0, max_idxs[..., None, None])
