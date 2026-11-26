# SPDX-License-Identifier: Apache-2.0
"""A5 Copy-SFA with registered MemFabric packed-C8 host memory."""

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
offload = pytest.importorskip("memfabric_hybrid").offload

from vllm_ascend.utils import bootstrap_custom_op_env  # noqa: E402

from .fused_quant_scatter_copy_sparse_flash_attention_golden import NOPE_DIM, TOPK, make_test_data  # noqa: E402
from .test_fused_quant_scatter_copy_sparse_flash_attention import check_attention  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def load_a5_custom_operator() -> None:
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401,PLC0415

    torch.npu.set_device(torch.device("npu:0"))
    config = offload.OffloadConfig()
    config.device_id = torch_npu.npu.current_device()
    config.reserve_size = 64 * 1024 * 1024
    config.alloc_size = config.reserve_size
    config.world_size = 1
    config.rank_id = 0
    config.scene = offload.Scene.SHARED
    config.store_url = "tcp://127.0.0.1:18583"
    assert offload.initialize(config) == 0


@pytest.mark.parametrize("miss_count", [8, 513], ids=["fused_miss", "union_copy_fallback"])
def test_copy_sfa_reads_packed_c8_from_memfabric(miss_count: int) -> None:
    device = torch.device("npu:0")
    data = make_test_data(miss_count, TOPK)
    hbm = data.initial_hbm.to(device)
    # This is a non-owning CPU view backed by registered MemFabric memory, not
    # an ordinary torch.empty CPU allocation and not swapped NPU memory.
    host = offload.empty(tuple(data.dram.shape), dtype=torch.int8, pin_memory=True)
    host.copy_(data.dram)
    inputs = [
        data.query,
        data.hbm_block_table,
        data.dram_block_table,
        data.topk_dst_slots,
        data.topk_src_ids,
        data.topk_miss_counts,
        data.actual_seq_lengths_query,
        data.actual_seq_lengths_kv,
        data.num_cache_tokens,
        data.miss_src_ids,
        data.miss_dst_slots,
        data.miss_counts,
    ]
    device_inputs = [tensor.to(device) for tensor in inputs]
    attention_out = torch.empty((*data.query.shape[:-1], NOPE_DIM), dtype=data.query.dtype, device=device)
    torch.ops._C_ascend.npu_fused_quant_scatter_copy_sparse_flash_attention.default(
        device_inputs[0],
        hbm,
        host,
        *device_inputs[1:],
        float(data.scale_value),
        attention_out,
    )
    torch.npu.synchronize()
    assert torch.equal(hbm.cpu(), data.expected_hbm)
    check_attention(attention_out, data.attention_golden)
