from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import CopySfaTailDest
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import MembPullReadThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import (
    BACKEND_MEMFABRIC,
    SFAPDRD2HConsumerWorker,
    SFAPDRD2HProducerWorker,
)


def _state():
    return SimpleNamespace(
        num_blocks=1,
        tp_size=1,
        layer_metadata={},
        main_name_to_idx={"layer": 0},
        cpu_pools=[(object(), None)],
        main_gva_bases=[(3000, 0)],
        main_block_lens=[(128 * 656, 0)],
        indexer_tensors=[None],
        indexer_scale_tensors=[None],
        dest_blocks_by_req={"req": (list(range(24)), [])},
        get_offload_layer_id=lambda _: 0,
        copy_sfa_tail_by_req={},
        topk_k_bases=[10000],
        topk_v_bases=[],
        topk_row_tokens=2304,
        topk_hot_tokens=2048,
        block_size=128,
        dspark_context_receiver=None,
        dspark_draft_kv_metadata={},
        dspark_draft_blocks_by_req={},
    )


def _meta(count=1):
    return {
        "layer": {
            "base_addrs": [1000] if count == 1 else [1000, 2000],
            "block_len": [128 * 656] if count == 1 else [128 * 656, 128],
            "block_size_scale": [1] if count == 1 else [1, 1],
            "main_tensor_count": count,
            "has_indexer": False,
        }
    }


def _thread():
    thread = MembPullReadThread.__new__(MembPullReadThread)
    thread._state = _state()
    thread.tp_rank = 0
    return thread


def test_resolve_packed_c8_layer_uses_single_main_tensor():
    thread = _thread()
    layer = thread._resolve_read_layer("layer", _meta())
    assert layer["packed_c8"] is True
    assert layer["p_k_base"] == 1000
    assert layer["p_v_base"] is None
    assert layer["p_k_len"] == 128 * 656
    assert layer["p_v_len"] == 0
    with pytest.raises(RuntimeError, match="one packed-C8 tensor"):
        thread._resolve_read_layer("layer", _meta(count=2))


def test_packed_c8_dense_descriptors_do_not_include_v_plane():
    thread = _thread()
    thread._state.copy_sfa_tail_by_req["req"] = CopySfaTailDest(
        pool_slot=0,
        tail_tokens=0,
        tail_block_index=0,
        dense=True,
        kv_tokens=300,
    )
    layer = thread._resolve_read_layer("layer", _meta())
    local, peer, lengths, _ = thread._build_req_descriptors(
        layer,
        "req",
        [0, 1, 2],
        [],
        False,
    )
    assert peer == [1000, 1000]
    assert local == [3000, 10000]
    assert lengths == [3 * 128 * 656, 300 * 656]


def test_packed_c8_tail_descriptor_uses_circular_tail_without_v_plane():
    thread = _thread()
    thread._state.copy_sfa_tail_by_req["req"] = CopySfaTailDest(
        pool_slot=0,
        tail_tokens=69,
        tail_block_index=23,
        dense=False,
        kv_tokens=3013,
    )
    layer = thread._resolve_read_layer("layer", _meta())
    local, peer, lengths, _ = thread._build_req_descriptors(
        layer,
        "req",
        list(range(24)),
        [],
        False,
    )
    assert peer[-1] == 1000 + 23 * 128 * 656
    assert local[-1] == 10000 + (2048 + 128) * 656
    assert lengths[-1] == 69 * 656


def test_consumer_binds_packed_hot_bases_without_v():
    worker = SFAPDRD2HConsumerWorker.__new__(SFAPDRD2HConsumerWorker)
    worker.offload_manager = SimpleNamespace(
        use_fused_copy_sfa=True,
        topk_buffers_k=[torch.zeros(1, 2304, 1, 656, dtype=torch.int8)],
        topk_buffers_v=[None],
        block_size=128,
    )
    worker._bind_copy_sfa_tail_destinations()
    assert worker._topk_k_bases
    assert worker._topk_v_bases == []
    assert worker._topk_row_tokens == 2304
    assert worker._topk_hot_tokens == 2048


def test_producer_filters_none_and_publishes_single_main_tensor():
    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    group = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(cache_sparse_sfa_c8=True),
        layer_names=["model.layers.0.self_attn.attn"],
    )
    worker._backend = BACKEND_MEMFABRIC
    worker.kv_cache_config = SimpleNamespace(num_blocks=1, kv_cache_groups=[group])
    worker.kv_cache_specs = [group.kv_cache_spec]
    worker.layer_metadata = {}
    worker.dspark_draft_layer_names = ()
    worker.dspark_draft_kv_metadata = {}
    worker.last_layer_idx = 0
    worker.total_layers = 1
    worker.stage_layer_names = []
    worker._build_producer_send_state = MagicMock(return_value=object())
    packed = torch.zeros(1, 128, 1, 656, dtype=torch.int8)
    hot = torch.zeros(1, 2304, 1, 656, dtype=torch.int8)

    class FakeSendThread:
        def __init__(self, ready_event, state, device_id):
            ready_event.set()
        startup_error = None

        def start(self):
            return None
        def stop(self):
            return None

    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker.collect_storage_merged_register_regions",
            return_value=SimpleNamespace(ptrs=[], lengths=[]),
        ),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker.validate_register_region_count"),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker.global_memfabric_te") as engine,
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker.MembPullSendingThread", FakeSendThread),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker.torch.npu.current_device", return_value=0),
    ):
        worker.register_kv_caches({"model.layers.0.self_attn.attn": (packed, None)})

    engine.register_buffer.assert_called_once_with([], [])
    metadata = worker.layer_metadata["model.layers.0.self_attn.attn"]
    assert metadata.main_tensor_count == 1
    assert len(metadata.kv_caches_base_addr) == 1
    assert metadata.block_len == [128 * 656]
