"""A clean version of efficient moba implementation with flash-attn on Mindspore"""
import random
from functools import lru_cache

import mindspore as ms
from einops import rearrange
from mindformers.modules.flash_attention import FlashAttention
from mindspore import nn, ops
from mindspore.ops import Tensor
ms.set_context(device_target="CPU")


@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """计算需要处理的注意力块信息（缓存最多16次不同输入的计算结果）"""
    # 计算每个批次的序列长度（批次长度 = 当前累积长度 - 前一个累积长度）
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 计算每个批次的块数量（向上取整：块数 = (序列长度 + 块大小-1) // 块大小）
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size

    # 初始化累积块数量数组（初始全为1，后续用累积和填充）
    cu_num_chunk = ops.ones(
        batch_num_chunk.numel() + 1,
        dtype=batch_num_chunk.dtype
    )
    # 从第二个元素开始填入累积块数（例如：[1, 1+2, 1+2+3, ...]）
    cu_num_chunk[1:] = batch_num_chunk.cumsum(axis=0)

    # 总块数（所有批次块数总和）
    num_chunk = cu_num_chunk[-1]

    # 初始化所有块的大小为指定块大小（最后会修正最后一块的大小）
    chunk_sizes = ops.full(
        (num_chunk + 1,),
        moba_chunk_size,
        dtype=ms.int32,
    )
    chunk_sizes[0] = 0

    # 计算各批次最后一块的实际大小（总长度 - (块数-1)*块大小）
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    # 将各批次最后一块的大小写入对应位置（对应批次第一个块的位置）
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size

    # 计算每个块的起始位置（累积块大小）
    cu_chunk = chunk_sizes.cumsum(axis=-1, dtype=ms.int32)

    # 初始化块到批次的映射数组（初始全为0）
    chunk_to_batch = ops.zeros(
        num_chunk,
        dtype=ms.int32
    )
    chunk_to_batch[cu_num_chunk[0:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(axis=0, dtype=ms.int32)

    # 移除每个批次的最后一个块，生成需要处理的块索引
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = ops.ones(
        num_chunk,
        dtype=ms.bool_
    )
    chunk_to_remain[chunk_to_remove] = False
    indices = ops.nonzero(chunk_to_remain)
    filtered_chunk_indices = indices[:, 0]
    # filtered_chunk_indices = ops.nonzero(chunk_to_remain)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )


class MixedAttention(nn.Cell):
    def __init__(self, max_seqlen, moba_chunk_size):
        super().__init__()
        self.max_seqlen = max_seqlen
        self.moba_chunk_size = moba_chunk_size
        self.flash_attn = FlashAttention()

        # 初始化其他算子
        self.exp = ops.Exp()
        self.log = ops.Log()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.index_add = ops.index_add
        self.concat = ops.Concat()
        self.zeros = ops.Zeros()

    def construct(self,
                  k, q, v,
                  self_attn_cu_seqlen,  # [B+1]
                  moba_q, moba_kv,
                  moba_cu_seqlen_q, moba_cu_seqlen_kv,  # [B_moba + 1]
                  moba_q_sh_indices,  # 索引张量
                  ):
        # 计算 softmax_scale，倒数平方根，缩放注意力分数，避免数值过大导致 softmax 梯度消失或爆炸
        softmax_scale = q.shape[-1] ** (-0.5)

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                       Self Attention
        # ------------------------------------------------------------------------------------------------------------------ #
        self_attn_out_sh, self_attn_lse_hs = self.flash_attn(
            q, k, v,

        )

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                        MoBA Attention
        # ------------------------------------------------------------------------------------------------------------------ #
        moba_k, moba_v = moba_kv[:, 0], moba_kv[:, 1]
        moba_attn_out, moba_attn_lse_hs = self.flash_attn(
            query=moba_q, key=moba_k, value=moba_v,
            drop_mask=None,
            actual_seq_qlen=moba_cu_seqlen_q,
        )

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                        LSE 形状转换  [H,S] -> [S,H]
        # ------------------------------------------------------------------------------------------------------------------ #
        self_attn_lse_sh = self.transpose(self_attn_lse_hs, (1, 0)).contiguous()
        moba_attn_lse = self.transpose(moba_attn_lse_hs, (1, 0)).contiguous()

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                 混合逻辑，[S,H,D]全零张量 -> [vS,D]二维张量，方便索引和加权
        # ------------------------------------------------------------------------------------------------------------------ #
        output = ops.zeros((q.shape[0], q.shape[1], q.shape[2]), q.dtype)
        # output_2d = self.reshape(output, (-1, q.shape[2]))
        output_2d = output.view(-1, q.shape[2])

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                           1. 计算混合LSE
        # ------------------------------------------------------------------------------------------------------------------ #
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        # moba_attn_lse = moba_attn_lse - self._gather(max_lse_1d, moba_q_sh_indices)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                           2. 混合输出计算
        # ------------------------------------------------------------------------------------------------------------------ #
        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                           3. Attn输出融合
        # ------------------------------------------------------------------------------------------------------------------ #
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                           4. MoBA输出融合
        # ------------------------------------------------------------------------------------------------------------------ #
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)

        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)

        return output


def moba_attn_varlen(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        moba_chunk_size: int,
        moba_topk: int,
) -> Tensor:
    kv = ops.stack((k, v), axis=1)

    """ Basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ Prepare chunk meta """
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        # return flash_attn_varlen_func(
        #     q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        # )
        return q

    self_attn_cu_seqlen = cu_chunk

    """ Prepare filtered KV """
    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    # todo 这一步有问题，torch是[7,64]   ms是[7,448]
    filtered_kv_indices = ops.arange(
        0, moba_chunk_size, dtype=ms.int32
    )[None, :].repeat(num_filtered_chunk, 0)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    """ Calculate gate weights """
    # Convert to float32 for computation
    key_gate_weight = (filtered_kv[:, 0]
                       .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
                       .mean(axis=1)
                       .astype(ms.float32))
    q = q.astype(ms.float32)

    # Einsum operation
    gate = ops.einsum("nhd,shd->nhs", key_gate_weight, q)
    key_gate_weight = key_gate_weight.astype(k.dtype)
    q = q.astype(k.dtype)

    """ Apply masks """
    # Create sequence indices
    gate_seq_idx = (ops.arange(0, seqlen, dtype=ms.int32)[
                    None, :
                    ].repeat(num_filtered_chunk, 0))

    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    # todo 这一步会报错 RuntimeError: For 'Gather', the 'input_indices' should be in the range [0, 2), but got 2, error_code[58]
    gate.masked_fill(gate_inf_mask.unsqueeze(1), -float("inf"))
    # Apply mask using full-like tensor and where
    # gate = ops.masked_fill(gate, gate_inf_mask.expand_dims(1), -float('inf'))

    """ Find topk chunks """
    # Topk operation
    _, gate_top_k_idx = ops.topk(gate, moba_topk, dim=0, largest=True, sorted=False)
    gate_mask = ops.logical_not(ops.isinf(gate))

    # Create scatter mask
    gate_idx_mask = ops.zeros(gate_mask.shape, ms.bool_)
    gate_idx_mask = ops.scatter(gate_idx_mask, 0, gate_top_k_idx, ops.ones_like(gate_idx_mask))
    gate_mask = ops.logical_and(gate_mask, gate_idx_mask)

    """ Process MOBA Q indices """
    # Reshape and find indices
    moba_q_indices = ops.nonzero(gate_mask.reshape(gate_mask.shape[0], -1))[..., -1]
    moba_seqlen_q = ops.reduce_sum(gate_mask.astype(ms.int32), axis=-1).flatten()

    # Rearrange Q
    moba_q = rearrange(q, "s h d -> ( h s ) d").index_select(
        0, moba_q_indices
    )  # [ selected_S, D ]
    moba_q = moba_q.unsqueeze(1)
    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen

    """ Prepare MOBA KV """
    # Process KV chunks
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()

    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]

    moba_cu_seqlen_q = ops.concat((
        Tensor([0], dtype=ms.int32),
        ops.cumsum(moba_seqlen_q, axis=0)
    )).astype(ms.int32)

    # Reshape and split KV
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = ops.cat(moba_kv, axis=0)

    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlen_kv = (
            ops.arange(
                0,
                num_filtered_chunk * num_head + 1 - zero_expert_count,
                dtype=ms.int32,
            )
            * moba_chunk_size
    )

    # Shape check
    assert (
            moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

    """ Final attention computation """
    return MixedAttention()(
        q, k, v,
        self_attn_cu_seqlen,
        moba_q, moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices
    )


if __name__ == '__main__':
    batch, head, seqlen, head_dim, moba_chunk_size, moba_topk = 1, 1, 512, 128, 64, 2


    def generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype):
        random.seed(0)
        ms.manual_seed(0)
        # torch.cuda.manual_seed(0)
        # device = torch.cuda.current_device()

        # gen qkv
        q = ops.randn(
            (seqlen, num_q_head, headdim), dtype=dtype
        )
        k = ops.randn(
            (seqlen, num_kv_head, headdim), dtype=dtype
        )
        v = ops.randn(
            (seqlen, num_kv_head, headdim), dtype=dtype
        )

        # gen cu seqlen
        cu_seqlen = random.sample(range(1, seqlen - 1), batch - 1) if batch > 1 else []
        cu_seqlen.sort()
        cu_seqlen = [0] + cu_seqlen + [seqlen]
        cu_seqlen = ms.tensor(cu_seqlen, dtype=ms.int32)

        # max_seqlen
        max_seqlen = ops.amax(cu_seqlen[1:] - cu_seqlen[:-1])

        return q, k, v, cu_seqlen, max_seqlen.item()


    q, k, v, cu_seqlen, max_seqlen = generate_data(
        batch, seqlen, head, head, head_dim, ms.float16
    )

    o = moba_attn_varlen(
        q,
        k,
        v,
        cu_seqlen,
        max_seqlen,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )

    print(0)
