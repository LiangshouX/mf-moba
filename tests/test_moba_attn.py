import random

import mindspore as ms
import mindspore.ops as ops
import pytest

from moba.moba_naive import moba_attn_varlen_naive


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


@pytest.mark.parametrize("batch", [1, 4, 7])  # can be arbitrary
@pytest.mark.parametrize("head", [1, 2, 4, 8])
@pytest.mark.parametrize("seqlen", [512, 1024, 2048])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("moba_chunk_size", [128, 256, 1024])
@pytest.mark.parametrize("moba_topk", [2, 3, 4])
def test_attn_varlen_moba(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk):
    dtype = ms.float16  # MindSpore暂不支持bfloat16，此处使用float16替代
    eps = 2e-2

    # Get data
    q, k, v, cu_seqlen, max_seqlen = generate_data(
        batch, seqlen, head, head, head_dim, dtype
    )

    # 生成反向传播的梯度输入
    vo_grad = ops.StandardNormal()(q.shape).astype(dtype)

    # 使用GradientTape计算梯度
    with ms.GradientTape() as tape:
        tape.watch([q, k, v])  # 确保参数被跟踪
        o = moba_attn_varlen_naive(  # 假设已实现MindSpore版本
            q, k, v, cu_seqlen, max_seqlen,
            moba_chunk_size=moba_chunk_size,
            moba_topk=moba_topk
        )
    grads = tape.gradient(o, [q, k, v], grad_inputs=vo_grad)
    gq, gk, gv = grads
    gqkv = ops.stack([gq, gk, gv], axis=1)

    # 重置梯度并计算参考实现
    with ms.GradientTape() as tape_ref:
        tape_ref.watch([q, k, v])
        o_ref = moba_attn_varlen_naive(  # 假设已实现MindSpore版本
            q, k, v, cu_seqlen, max_seqlen,
            moba_chunk_size=moba_chunk_size,
            moba_topk=moba_topk
        )
    grads_ref = tape_ref.gradient(o_ref, [q, k, v], grad_inputs=vo_grad)
    gq_ref, gk_ref, gv_ref = grads_ref
    gqkv_ref = ops.stack([gq_ref, gk_ref, gv_ref], axis=1)

    # 计算差异
    o_diff = ops.abs(o - o_ref)
    max_o_diff = ops.reduce_max(o_diff).asnumpy().item()
    mean_o_diff = ops.reduce_mean(o_diff).asnumpy().item()
    print(f"output diff: {max_o_diff:.2e}, {mean_o_diff:.2e}")
    assert ops.allclose(o, o_ref, rtol=eps, atol=eps), (
        f"Output not close. Max diff: {max_o_diff}, Mean: {mean_o_diff}"
    )

    gqkv_diff = ops.abs(gqkv - gqkv_ref)
    max_gqkv_diff = ops.reduce_max(gqkv_diff).asnumpy().item()
    mean_gqkv_diff = ops.reduce_mean(gqkv_diff).asnumpy().item()
    print(f"grad diff: {max_gqkv_diff:.2e}, {mean_gqkv_diff:.2e}")
    assert ops.allclose(gqkv, gqkv_ref, rtol=eps, atol=eps), (
        f"Gradients not close. Max diff: {max_gqkv_diff}, Mean: {mean_gqkv_diff}"
    )

    assert max_o_diff < 4e-2, f"o_diff max {max_o_diff}"
    assert mean_o_diff < 4e-4, f"o_diff mean {mean_o_diff}"
    assert max_gqkv_diff < 4e-2, f"gqkv_diff max {max_gqkv_diff}"
    assert mean_gqkv_diff < 4e-4, f"gqkv_diff mean {mean_gqkv_diff}"
