import torch
import torch.nn as nn


def do_nothing(x):
    return x


def bipartite_soft_matching_x_w(x, w, r, scaling_factors):
    # We can only reduce by a maximum of 50% channels
    # metric shape: [B * N, C]
    b = x.shape[0]
    t = x.shape[1]
    r = min(r, t // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # xa, xb shape: [B * N, C/2]
        xa, xb = x[..., ::2], x[..., 1::2]
        # wa, wb shape: [cout, C/2]
        wa, wb = w[..., ::2], w[..., 1::2]
        xa_c, xb_c = xa.shape[1], xb.shape[1]

        # shape: [C/2, C/2]
        # fast version
        # xdist = (xa.t().reshape(xa_c, b, 1) - xb.reshape(1, b, xb_c)).sum(1)
        # score_ij = wi (yi - yj) / 2 + wj (yj - yi) / 2
        # score_a[i, :] = w:i xdist:

        xdist = torch.cdist(xa.t(), xb.t(), p=2.0)
        scores_a = torch.zeros(xa_c, xb_c, device=x.device)
        scores_b = torch.zeros(xa_c, xb_c, device=x.device)
        scores_fast = torch.zeros(xa_c, xb_c, device=x.device)
        for i in range(xb_c):
            scores_a[i, :] = (wa[:, i].unsqueeze(1) * xdist[i]).sum(0)
        for j in range(xb_c):
            scores_b[:, j] = (wb[:, j].unsqueeze(1) * (xdist[:, j])).sum(0)
        scores_fast = (scores_a + scores_b).pow(2)
        scores = scores_fast

        if scaling_factors is not None:
            split_mask = scaling_factors != 1.0
            split_mask_a = split_mask[::2]
            split_index_a = split_mask_a.nonzero().squeeze()

            split_mask_b = split_mask[1::2]
            split_index_b = split_mask_b.nonzero().squeeze()
            scores.index_fill_(
                dim=0, index=split_index_a, value=torch.finfo(scores.dtype).max
            )
            scores.index_fill_(
                dim=1, index=split_index_b, value=torch.finfo(scores.dtype).max
            )

        # scores_a = torch.zeros(xa_c, xb_c, device=x.device)
        # for i in range(xb_c):
        #     scores_a[i, :] = (wa[:, i].unsqueeze(1) * xa[:, i]).mean(0)

        # slow version
        # scores = torch.zeros(t // 2, t // 2, device=x.device)
        # for i in range(t // 2):
        #     for j in range(t // 2):
        #         scores[i, j] = (
        #             (
        #                 (wa[..., i] * (xa[..., i] - xb[..., j]).sum())
        #                 + (wb[..., j] * (xb[..., j] - xa[..., i]).sum())
        #             )
        #             .mean(0)
        #             .pow(2)
        #         )

        # node max, node_idx shape: [C/2], index of b
        # Draw one edge from each token in A to its most similar token in B.
        node_min, node_idx = scores.min(dim=-1)
        # edge_idx shape: [C/2]
        # Keep the r most similar edges. index of a
        edge_idx = node_min.argsort(dim=-1, descending=False)

        # unm_idx shape: [C/2 -r]
        # unm_idx = edge_idx[r:]  # Unmerged Channels
        # src_idx shape: [r]
        src_idx = edge_idx[:r]  # Merged Channels
        dst_idx = node_idx[src_idx]
    return src_idx, dst_idx, scores[src_idx, dst_idx]


def merging(x, src_idx, dst_idx, r, mode="mean") -> torch.Tensor:
    # shape of src dst: [B, N, C]
    B, N, C = x.shape

    ori_src_idx = torch.arange(0, C, 2, device=x.device)
    ori_dst_idx = torch.arange(1, C, 2, device=x.device)
    src, dst = x[..., ori_src_idx], x[..., ori_dst_idx]
    src_C = src.shape[-1]
    dst_C = dst.shape[-1]

    # we set mask to 0 when channel is merged
    channel_mask = torch.ones(C, device=x.device, dtype=x.dtype)
    m_idx = ori_src_idx[src_idx]
    channel_mask[m_idx] = 0.0

    n, t1, c = src.shape
    sub_src = src.gather(dim=-1, index=src_idx.expand(n, t1, r))
    dst = dst.scatter_reduce(-1, dst_idx.expand(n, t1, r), sub_src, reduce=mode)
    src = src.view(B, N, src_C, 1)
    dst = dst.view(B, N, dst_C, 1)
    if src_C == dst_C:
        merged_x = torch.cat([src, dst], dim=-1).view(B, N, C)
    else:
        merged_x = torch.cat([src[..., :-1, :], dst], dim=-1).view(
            B, N, src_C + dst_C - 1
        )
        merged_x = torch.cat([merged_x, src[..., -1, :].reshape(B, N, 1)], dim=-1).view(
            B, N, src_C + dst_C
        )
    merged_x = merged_x.index_select(-1, (channel_mask != 0).nonzero().squeeze())
    # merged_x = merged_x.index_fill(-1, m_idx, 0.0)
    return merged_x


class CMModule(nn.Module):
    def __init__(self, num_merged_channels):
        super().__init__()
        self.num_merged_channels = num_merged_channels
        self.have_merge = False
        self.src_idx = None
        self.dst_idx = None
        self.num_split = None
        self.scaling_factors = None

    def find_similar_channels(self, x, fcs):
        B, N, C = x.shape
        x = x.view(B * N, C)

        fc_weight = []
        if not isinstance(fcs, list):
            fcs = [fcs]
        for fc in fcs:
            fc_weight.append(fc.weight)
        fc_weight = torch.cat(fc_weight, dim=0)

        x = x.float()
        src_idx, dst_idx, scores = bipartite_soft_matching_x_w(
            x, fc_weight, self.num_merged_channels, self.scaling_factors
        )
        del self.src_idx
        del self.dst_idx
        print("Score: {}".format(scores))
        self.register_buffer("src_idx", src_idx)
        self.register_buffer("dst_idx", dst_idx)
        self.have_merge = True

    def forward(self, x):
        # only perform merging after find_similar_channels
        if self.have_merge:
            B, N, C = x.shape
            # if size is None:
            #     size = torch.ones_like(x[0, 0])
            #     size = size.view(1, 1, C)

            # x = merging(
            #     x * size,
            #     self.src_idx,
            #     self.dst_idx,
            #     self.num_merged_channels,
            #     mode="sum",
            # )
            # size = merging(
            #     size,
            #     self.src_idx,
            #     self.dst_idx,
            #     self.num_merged_channels,
            #     mode="sum",
            # )
            # x = x / size
            x = merging(
                x,
                self.src_idx,
                self.dst_idx,
                self.num_merged_channels,
                mode="mean",
            )

            # x = masking_merged_channel(x, self.src_idx)
        return x
