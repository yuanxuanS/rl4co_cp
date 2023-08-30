import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor


class mTSPInitEmbedding(nn.Module):
    """
    mTSP init embedding module from Son et al., 2023.
    Reference: https://arxiv.org/abs/2306.02689
    https://github.com/kaist-silab/equity-transformer/blob/main/nets/attention_model.py
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        node_dim = 2  # x, y

        self.embedding_dim = embedding_dim

        self.init_embed_depot = nn.Linear(node_dim, embedding_dim, bias)
        self.init_embed_city = nn.Linear(node_dim, embedding_dim, bias)
        self.pos_emb_proj = nn.Linear(embedding_dim, embedding_dim, bias)
        self.alpha = nn.Parameter(torch.ones(1))
        self.pos_enc = PositionalEncoding(d_model=embedding_dim, max_len=10000)
        # max_len is hardcoded as 10000 also in https://github.com/kaist-silab/equity-transformer/blob/main/nets/attention_model.py#L76

    def forward(self, td: TensorDict) -> Tensor:
        depot_loc = td["locs"][..., 0:1, :]  # [batch, 1, node_dim]
        num_salesman = td["num_salesman"][0]  # assuming all salesman are the same in the bath!

        pos_emb = (
            self.alpha
            * self.pos_emb_proj(self.pos_enc(depot_loc.shape[0], num_salesman + 1))
            / num_salesman
        )  # [num_salesman + 1, embedding_dim]

        depot_emb = (
            self.init_embed_depot(depot_loc).expand(-1, num_salesman + 1, -1) + pos_emb[None, ...]
        )  # [batch, num_salesman + 1, embedding_dim]

        city_locs = td["locs"][..., 1:, :]  # [batch, N, node_dim], where N is the number of cities
        city_emb = self.init_embed_city(city_locs)  # [batch, num_cities, embedding_dim]

        return torch.cat([depot_emb, city_emb], dim=1)  # [batch, num_nodes, embedding_dim]


class mPDPInitEmbedding(nn.Module):
    """
    mPDP init embedding module from Son et al., 2023.
    Reference: https://arxiv.org/abs/2306.02689
    https://github.com/kaist-silab/equity-transformer/blob/main/nets/attention_model.py
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        node_dim = 2  # x, y

        self.embedding_dim = embedding_dim

        self.init_embed_depot = nn.Linear(node_dim, embedding_dim, bias)
        self.init_embed_pick = nn.Linear(node_dim, embedding_dim, bias)
        self.init_embed_delivery = nn.Linear(node_dim, embedding_dim, bias)
        self.pos_emb_proj = nn.Linear(embedding_dim, embedding_dim, bias)
        self.alpha = nn.Parameter(torch.ones(1))
        self.pos_enc = PositionalEncoding(d_model=embedding_dim, max_len=10000)
        # max_len is hardcoded as 10000 also in https://github.com/kaist-silab/equity-transformer/blob/main/nets/attention_model.py#L76

    def forward(self, td):
        depot, locs = td["locs"][..., 0:1, :], td["locs"][..., 1:, :]
        num_salesman = td["num_salesman"][0]  # assuming all salesman are the same in the bath!

        pos_emb = (
            self.alpha
            * self.pos_emb_proj(self.pos_enc(depot.shape[0], num_salesman + 1))
            / num_salesman
        )  # [num_salesman + 1, embedding_dim]

        depot_emb = (
            self.init_embed_depot(depot).expand(-1, num_salesman + 1, -1) + pos_emb[None, ...]
        )  # [batch, num_salesman + 1, embedding_dim]

        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)

        return torch.cat(
            [depot_emb, pick_embeddings, delivery_embeddings], dim=1
        )  # [batch, num_nodes, embedding_dim]


class PositionalEncoding(nn.Module):
    """Compute sinusoid encoding
    original code from Equity Transformer implementation
    https://github.com/kaist-silab/equity-transformer/blob/main/nets/positional_encoding.py#L5

    we made a few changes to make it work with our code, e.g.,
    including handling encoding as a buffer to support seamless device transfer and loading.
    """

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # Initialize encoding matrix
        encoding = torch.zeros(max_len, d_model)

        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        _2i = torch.arange(0, d_model, step=2).float()

        # Compute the positional encodings
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer("encoding", encoding)

    def forward(self, seq_len):
        # Return encoding matrix for the current sequence length
        return self.encoding[:seq_len, :]
