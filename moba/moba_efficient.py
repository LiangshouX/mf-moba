"""A clean version of efficient moba implementation with flash-attn on Mindspore"""

import mindspore as ms
from functools import lru_cache
from einops import rearrange
from mindspore.ops import zeros
from mindspore import nn, ops, Tensor
from mindspore.ops.auto_generate import FlashAttentionScoreGrad, FlashAttentionScore
from mindspore.ops.function.grad.grad_func import dtype


@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """计算需要处理的注意力块信息（缓存最多16次不同输入的计算结果）"""
    # 计算每个批次的序列长度（批次长度 = 当前累积长度 - 前一个累积长度）
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 计算每个批次的块数量（向上取整：块数 = (序列长度 + 块大小-1) // 块大小）
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size

    # 初始化累积块数量数组（初始全为1，后续用累积和填充）
    cu_num_chunk = ms.mint.ones(
        batch_num_chunk.numel() + 1,
        dtype=batch_num_chunk.dtype
    )
    # 从第二个元素开始填入累积块数（例如：[1, 1+2, 1+2+3, ...]）
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)

    # 总块数（所有批次块数总和）
    num_chunk = cu_num_chunk[-1].item()

    # 初始化所有块的大小为指定块大小（最后会修正最后一块的大小）
    chunk_sizes = ms.mint.full(
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
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=ms.int32)

    # 初始化块到批次的映射数组（初始全为0）
    chunk_to_batch = ms.mint, zeros(
        (num_chunk,),
        dtype=ms.int32
    )

    chunk_to_batch = ms.mint.zeros(
        (num_chunk,),
        dtype=ms.int32
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=ms.int32)

    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = ms.mint.ones(
        (num_chunk,),
        dtype=ms.bool_
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
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
        self.flash_attn = FlashAttentionScore()

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
        softmax_scale = q.shape[-1] ** (-0.5)

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                        Self Attention 分支
        # ------------------------------------------------------------------------------------------------------------------ #
        self_attn_out, self_attn_lse_hs = self.flash_attn(
            q, k, v,

        )
        # LSE 形状转换  [H,S] -> [S,H]
        self_attn_lse_sh = self.transpose(self_attn_lse_hs, (1, 0))

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                        MoBA Attention 分支
        # ------------------------------------------------------------------------------------------------------------------ #
        moba_k, moba_v = moba_kv[:, 0], moba_kv[:, 1]
        moba_attn_out, moba_attn_lse_hs = self.flash_attn(
            query=q, key=k, value=v,
            drop_mask=None,
            actual_seq_qlen=moba_cu_seqlen_q,
        )
        moba_attn_lse = self.transpose(moba_attn_lse_hs, (1, 0))

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                              混合逻辑
        # ------------------------------------------------------------------------------------------------------------------ #
        output = self.zeros((q.shape[0], q.shape[1], q.shape[2]), q.dtype)
        output_2d = self.reshape(output, (-1, q.shape[2]))

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                           1. 计算混合LSE
        # ------------------------------------------------------------------------------------------------------------------ #
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
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
        #                                            3. 融合输出
        # ------------------------------------------------------------------------------------------------------------------ #
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # ------------------------------------------------------------------------------------------------------------------ #
        #                                           4. MoBA输出融合
        # ------------------------------------------------------------------------------------------------------------------ #

    def _index_reduce_amax(self, target, indices, src):
        # 手动实现类似 pytorch 的index_reduce(amax)
        gathered = self._gather(target, indices, 0)

    def _gather(self, tensor, indices):
        return ops.Gather()(tensor, indices, 0)

    def _index_add(self, target, indices, src):
        return None