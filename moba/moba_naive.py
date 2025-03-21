"""A Clean Version of MoBA implementation using Mindspore"""

import math
import random
import mindspore as ms
from mindspore import ops, Tensor


def moba_attn_varlen_naive(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seq_lens: Tensor,
        max_seq_len: int,
        moba_chunk_size: int,
        moba_topk: int,
) -> Tensor:
    """MOBA attention implementation with variable sequence length for MindSpore."""

    batch = cu_seq_lens.shape[0] - 1  # 批次大小
    softmax_scale = q.shape[-1] ** (-0.5)  # 缩放因子

    o = ops.zeros_like(q)  # 初始化输出
    cu_seq_lens_np = cu_seq_lens.asnumpy()  # 转换为numpy便于索引

    for batch_idx in range(batch):
        # 获取当前批次的序列范围
        batch_start = int(cu_seq_lens_np[batch_idx])
        batch_end = int(cu_seq_lens_np[batch_idx + 1])
        q_ = q[batch_start:batch_end]
        k_ = k[batch_start:batch_end]
        v_ = v[batch_start:batch_end]
        o_ = o[batch_start:batch_end]

        batch_size = batch_end - batch_start
        num_block = math.ceil(batch_size / moba_chunk_size)  # 计算块数

        # 计算每个块的键门控权重
        key_gate_weight = []
        for block_idx in range(num_block):
            block_start = block_idx * moba_chunk_size
            block_end = min(block_start + moba_chunk_size, batch_size)
            k_block = k_[block_start:block_end]
            key_gate_weight.append(ops.mean(k_block, axis=0, keep_dims=True))
        key_gate_weight = ops.concat(key_gate_weight, axis=0)

        # 计算门控矩阵
        q_float = q_.astype(ms.float32)
        key_gate_float = key_gate_weight.astype(ms.float32)
        gate = ops.einsum("shd,nhd->hsn", q_float, key_gate_float)

        # 应用位置掩码
        H, S, N = gate.shape
        for i in range(num_block):
            # 生成下三角掩码
            s_mask = ops.arange(S).reshape(1, S, 1) < (i + 1) * moba_chunk_size
            n_mask = ops.arange(N).reshape(1, 1, N) == i
            mask = ops.logical_and(s_mask, n_mask).broadcast_to(gate.shape)
            gate = ops.where(mask, -float('inf'), gate)

            # 生成当前块掩码
            s_start = i * moba_chunk_size
            s_end = min((i + 1) * moba_chunk_size, S)
            block_mask = ops.logical_and(
                ops.arange(S) >= s_start,
                ops.arange(S) < s_end
            ).reshape(1, S, 1)
            gate = ops.where(block_mask & n_mask, float('inf'), gate)

        # 选取topK门控
        k_topk = min(moba_topk, num_block)
        gate_topk_val, gate_topk_idx = ops.topk(gate, k_topk, dim=-1, sorted=False)
        threshold = ops.min(gate_topk_val, axis=-1, keepdims=True)
        need_attend = gate > threshold

        # 创建索引掩码
        indices = ops.stack([
            ops.broadcast_to(ops.arange(H).reshape(H, 1, 1), gate_topk_idx.shape),
            ops.broadcast_to(ops.arange(S).reshape(1, S, 1), gate_topk_idx)
        ], axis=-1)
        gate_idx_mask = ops.scatter_nd(indices.reshape(-1, 3),
                                       ops.ones(indices.shape[:-1], ms.bool_).flatten(),
                                       gate.shape)
        need_attend = ops.logical_and(need_attend, gate_idx_mask)

        # 应用最终门控
        gate = ops.where(need_attend, 0.0, -float('inf'))
        gate = ops.repeat_elements(gate, moba_chunk_size, axis=-1)[:, :, :batch_size]

        # 添加因果掩码
        tril_mask = ops.tril(ops.ones((batch_size, batch_size), ms.bool_))
        gate = ops.where(tril_mask[None, :, :], gate, -float('inf'))

        # 计算注意力
        qk = ops.einsum("xhd,yhd->hxy",
                        q_.astype(ms.float32),
                        k_.astype(ms.float32))
        qk = qk + gate.astype(ms.float32)
        qk *= softmax_scale
        attn = ops.softmax(qk, axis=-1)

        # 更新输出
        o_update = ops.einsum("hxy,yhd->xhd", attn, v_.astype(ms.float32))
        o_ += o_update.astype(q.dtype)

        # 回写结果
        o = ops.tensor_scatter_elements_update(
            o, ops.arange(batch_start, batch_end).reshape(-1, 1),
            o_, axis=0
        )

    return o


def generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype):
    random.seed(0)
    ms.set_seed(0)

    # Generate q, k, v using standard normal distribution
    std_normal = ops.StandardNormal()
    q = std_normal((seqlen, num_q_head, headdim)).astype(dtype)
    k = std_normal((seqlen, num_kv_head, headdim)).astype(dtype)
    v = std_normal((seqlen, num_kv_head, headdim)).astype(dtype)

    # Wrap tensors as Parameters to enable gradients
    q = ms.Parameter(q, requires_grad=True)
    k = ms.Parameter(k, requires_grad=True)
    v = ms.Parameter(v, requires_grad=True)

    # Generate cu_seqlen
    if batch > 1:
        cu_seqlen = random.sample(range(1, seqlen - 1), batch - 1)
        cu_seqlen.sort()
    else:
        cu_seqlen = []
    cu_seqlen = [0] + cu_seqlen + [seqlen]
    cu_seqlen = ms.Tensor(cu_seqlen, dtype=ms.int32)

    # Calculate max sequence length
    diff = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seqlen = ops.reduce_max(diff).asnumpy().item()

    return q, k, v, cu_seqlen, max_seqlen


if __name__ == '__main__':
    batch, seqlen, num_q_head, num_kv_head, headdim, dtype = 1, 32, 2, 2, 128, ms.float16
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype)

    print(q, k, v, cu_seqlen, max_seqlen)

    o = moba_attn_varlen_naive(q, k, v, cu_seqlen, max_seqlen, 32, 3)

    print(o.shape)
    print(o)
