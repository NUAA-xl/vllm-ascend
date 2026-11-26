"""Regression tests for SFA KV-offload attention metadata."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.attention.attention_v1 import AscendAttentionState  # noqa: E402
from vllm_ascend.attention.sfa_kv_offload import (  # noqa: E402
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
    AscendSFAOffloadMetadata,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    FSA_EXTERNAL_PLAN_READY_MARKER,
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
    SparseKVOffloadManager,
    reshape_kv_cache_tensors_for_sparse_kv_offload,
)


@pytest.mark.parametrize("defer_wait", [False, True])
def test_exec_kv_offload_always_completes_compute_and_cache_write(defer_wait):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    k_nope, k_pe = torch.ones(2, 4), torch.ones(2, 2)
    impl._is_decode_only = MagicMock(return_value=True)
    impl.enable_sparse_sfa_c8 = False
    impl._compute_kv_only = MagicMock(return_value=(k_nope, k_pe))
    impl._offload_layer_name = MagicMock(return_value="layer")
    impl._cpu_cache_pair = MagicMock(return_value=(torch.empty(1), torch.empty(1)))
    impl._in_graph_runtime = MagicMock(return_value=False)
    slots = torch.arange(2)
    metadata = SimpleNamespace(fused_copy_sfa_enabled=False)
    with patch("vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager") as get_manager:
        manager = get_manager.return_value
        result = impl.exec_kv(k_nope, k_pe, k_pe, (), slots, metadata, defer_wait=defer_wait)
        assert isinstance(result, tuple)
        assert result[0] is k_pe and result[1] is k_nope
        impl._compute_kv_only.assert_called_once_with(k_nope, k_pe, k_pe)
        manager.offload_new_kv.assert_called_once()
        assert manager.offload_new_kv.call_args.kwargs["slot_mapping"] is slots
        assert manager.offload_new_kv.call_args.kwargs["k"] is k_nope
        assert manager.offload_new_kv.call_args.kwargs["v"] is k_pe


def _make_boundary_decode_metadata():
    return SimpleNamespace(
        max_query_len=1,
        num_reqs=1,
        num_actual_tokens=1,
        query_start_loc_cpu=torch.tensor([0, 1]),
        is_prefilling=torch.tensor([True]),
        req_ids_tensor=torch.tensor([7]),
        token_to_req=torch.tensor([0]),
    )


@pytest.mark.parametrize(
    ("kv_transfer_config", "expected"),
    [
        (None, False),
        (SimpleNamespace(is_kv_consumer=False, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=True), False),
        (SimpleNamespace(is_kv_consumer=True, is_kv_producer=False), True),
    ],
)
def test_pd_decode_consumer_is_derived_from_kv_role(kv_transfer_config, expected):
    vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    with (
        patch.object(AscendSFAMetadataBuilder, "__init__", return_value=None) as init,
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_ascend_config",
            return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(use_fused_copy_sfa=False)),
        ),
    ):
        builder = AscendSFAKVOffloadMetadataBuilder(
            kv_cache_spec=None,
            layer_names=[],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )

    assert builder.is_pd_decode_consumer is expected
    assert init.call_args.args[4] is AscendSFAOffloadMetadata
    assert "copy_sfa_seq_lens" in AscendSFAOffloadMetadata.__dataclass_fields__
    assert "copy_sfa_seq_lens" not in AscendSFAMetadata.__dataclass_fields__


@pytest.mark.parametrize(
    ("is_pd_decode_consumer", "expected_decodes", "expected_prefills"),
    [
        (True, 1, 0),
        (False, 0, 1),
    ],
)
def test_boundary_token_classification_depends_on_pd_decode_role(
    is_pd_decode_consumer,
    expected_decodes,
    expected_prefills,
):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_fused_copy_sfa = False
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = is_pd_decode_consumer
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly)

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, _make_boundary_decode_metadata())

    assert metadata.num_decodes == expected_decodes
    assert metadata.num_prefills == expected_prefills
    assert metadata.num_decode_tokens == expected_decodes
    assert metadata.req_ids_tensor.tolist() == [7]
    assert metadata.token_to_req.tolist() == [0]
    assert AscendSFAKVOffloadImpl._is_decode_only(metadata) is is_pd_decode_consumer


def test_pd_decode_consumer_still_rejects_long_prefill_classification():
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_fused_copy_sfa = False
    builder.decode_threshold = 1
    builder.is_pd_decode_consumer = True
    metadata = SimpleNamespace()
    common_metadata = _make_boundary_decode_metadata()
    common_metadata.max_query_len = 2
    common_metadata.num_actual_tokens = 2
    common_metadata.query_start_loc_cpu = torch.tensor([0, 2])

    with patch(
        "vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled",
        return_value=False,
    ):
        builder._populate_offload_metadata(metadata, common_metadata)

    assert metadata.num_decodes == 0
    assert metadata.num_prefills == 1
    assert metadata.num_decode_tokens == 0


def _make_fused_overlap_impl() -> AscendSFAKVOffloadImpl:
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.block_size = 4
    impl.local_num_heads = 2
    impl.lru_resident_capacity = 8
    impl.sfa_sparse_topk = 4
    impl.max_num_topk_rows = 4
    impl.scale = 0.5
    impl.selection_kv_block_table = None
    impl.selection_kv_block_status = None
    impl.selection_membership_map = None
    impl.fused_overlap_last_req_ids = None
    impl._fused_overlap_selection_capacity = None
    impl._fused_overlap_decode_logged = True
    impl.skip_topk = False
    return impl


def test_mtp_rewrite_invalidates_membership_slot_map():
    impl = _make_fused_overlap_impl()
    topk_count = 4
    selection_status = torch.full((4, 1, 8), -1, dtype=torch.int32)
    selection_status[:, 0, :topk_count] = torch.tensor([8, 9, 10, 11])
    selection_status[:, 0, topk_count] = topk_count
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    control = membership[
        :, FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + 8
    ]
    control[:, 0] = 0x5A4D
    last_req_ids = torch.full((4,), 7, dtype=torch.int64)
    metadata = SimpleNamespace(
        token_to_req=torch.zeros(4, dtype=torch.int32),
        req_ids_tensor=torch.tensor([7], dtype=torch.int64),
    )

    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
            return_value=SimpleNamespace(capturing=False),
        ),
        patch("vllm_ascend.attention.sfa_kv_offload._EXTRA_CTX", SimpleNamespace(capturing=False)),
    ):
        impl._invalidate_fused_overlap_selection_rows(
            selection_status,
            membership,
            last_req_ids,
            metadata,
            num_tokens=4,
            num_reqs=1,
            topk_count=topk_count,
            seq_lens=torch.tensor([12], dtype=torch.int32),
            cum_query_lens=torch.tensor([4], dtype=torch.int32),
        )

    assert bool((selection_status[:, 0, :topk_count] == -1).all())
    assert bool((control[:, 0] == -1).all())


def test_fused_overlap_external_plan_passes_raw_topk_and_full_selection_state():
    """
    Test that the fused overlap external plan receives the raw topk indices and the full selection state,
    and that the fused op receives the correct inputs.
    """
    impl = _make_fused_overlap_impl()
    ql_nope = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    q_pe = torch.ones((2, 2, 1), dtype=torch.float32)
    topk = torch.tensor([[0, -1, 2, 5], [3, 7, 1, -1]], dtype=torch.int64)
    full_kv_cpu = torch.zeros((3, 4, 1, 3), dtype=torch.float32)
    full_rope_cpu = torch.zeros((3, 4, 1, 1), dtype=torch.float32)
    metadata = SimpleNamespace(
        num_decodes=2,
        token_to_req=torch.tensor([0, 1], dtype=torch.int32),
        req_ids_tensor=torch.tensor([101, 202], dtype=torch.int64),
        block_table=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
    )
    call_order = []
    fused_inputs = {}
    plan_inputs = {}

    def allocate_membership(row_capacity):
        membership = torch.full(
            (row_capacity, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
            -1,
            dtype=torch.int16,
        )
        control = membership[
            :,
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + 8,
        ]
        control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
        control[:, 2] = 4
        control[:, 3] = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - 4
        return membership

    def prepare_plan(**kwargs):
        call_order.append("plan")
        plan_inputs.update(kwargs)
        return True

    def inject_current(**kwargs):
        call_order.append("inject")
        assert kwargs["layer_name"] == "model.layers.0.self_attn"
        assert kwargs["num_tokens"] == 2
        assert kwargs["selection_kv_cache"].shape == (8, 4, 3)
        assert kwargs["selection_k_rope"].shape == (8, 4, 1)
        assert kwargs["capturing"] is False

    def wait_for_writeback(capturing):
        call_order.append("wait_writeback")
        assert capturing is False

    def fused_op(**kwargs):
        call_order.append("fused")
        fused_inputs.update(kwargs)
        return kwargs["query"] + 10

    manager = SimpleNamespace(
        topk_buffers_k=[torch.zeros((4, 8, 1, 3), dtype=torch.float32)],
        topk_buffers_v=[torch.zeros((4, 8, 1, 1), dtype=torch.float32)],
        _get_offload_layer_id=lambda _: 0,
        get_fused_overlap_cpu_kv_inputs=lambda _: (full_kv_cpu, full_rope_cpu),
        allocate_fused_overlap_membership_map=allocate_membership,
        prepare_fused_overlap_external_plan=prepare_plan,
        inject_current_kv_into_selection=inject_current,
        wait_for_current_kv_writeback=wait_for_writeback,
    )

    with (
        patch.object(impl, "_require_custom_op", return_value=fused_op),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
            return_value=manager,
        ),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
            return_value=SimpleNamespace(capturing=False),
        ),
        patch("vllm_ascend.attention.sfa_kv_offload._EXTRA_CTX", SimpleNamespace(capturing=False)),
    ):
        output = impl._execute_fused_overlap_offload_decode(
            ql_nope,
            q_pe,
            topk,
            metadata,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int32),
            "model.layers.0.self_attn",
        )

    torch.testing.assert_close(output, ql_nope + 10)
    assert call_order == [
        "plan",
        "inject",
        "fused",
        "wait_writeback",
    ]
    assert fused_inputs["selection_kv_cache"].shape == (8, 4, 3)
    assert fused_inputs["selection_k_rope"].shape == (8, 4, 1)
    assert fused_inputs["selection_kv_block_table"].shape == (4, 2)
    assert fused_inputs["selection_kv_block_status"].shape == (4, 1, 8)
    assert fused_inputs["selection_membership_map"].shape == (4, 16400)
    assert fused_inputs["selection_membership_map"].dtype == torch.int16
    torch.testing.assert_close(
        fused_inputs["query"],
        torch.cat([ql_nope, q_pe], dim=-1),
    )
    assert plan_inputs["selection_membership_map"].data_ptr() == fused_inputs["selection_membership_map"].data_ptr()
    expected_topk = torch.tensor(
        [[[0, -1, 2, 5]], [[3, 7, 1, -1]]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(
        fused_inputs["selection_topk_indices"],
        expected_topk,
    )
    torch.testing.assert_close(
        plan_inputs["topk_indices_npu"],
        expected_topk.squeeze(1),
    )
    torch.testing.assert_close(
        plan_inputs["req_ids_npu"],
        torch.tensor([101, 202], dtype=torch.int64),
    )
    torch.testing.assert_close(
        plan_inputs["stable_prefix_lens_npu"],
        torch.tensor([3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        plan_inputs["visible_seq_lens_npu"],
        torch.tensor([4, 4], dtype=torch.int32),
    )
    assert plan_inputs["capturing"] is False


def test_fused_overlap_common_inputs_are_reused_only_within_one_forward():
    impl = _make_fused_overlap_impl()
    metadata = SimpleNamespace(
        num_decodes=2,
        token_to_req=torch.tensor([0, 1], dtype=torch.int32),
        req_ids_tensor=torch.tensor([101, 202], dtype=torch.int64),
        block_table=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
    )
    topk = torch.tensor(
        [[[0, 1, 2, 3]], [[3, 2, 1, 0]]],
        dtype=torch.int32,
    )
    query_lens = torch.tensor([1, 2], dtype=torch.int32)
    kv_lens = torch.tensor([4, 5], dtype=torch.int32)
    first_forward = SimpleNamespace(capturing=False)
    second_forward = SimpleNamespace(capturing=False)
    current_forward = [first_forward]

    with (
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_forward_context",
            side_effect=lambda: current_forward[0],
        ),
        patch("vllm_ascend.attention.sfa_kv_offload._EXTRA_CTX", SimpleNamespace(capturing=False)),
    ):
        first = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )
        reused = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )
        kv_lens.add_(1)
        current_forward[0] = second_forward
        refreshed = impl._prepare_fused_overlap_decode_common_inputs(
            metadata,
            num_tokens=2,
            num_reqs=2,
            topk_indices_decode=topk,
            actual_seq_lengths_query_decode=query_lens,
            actual_seq_lengths_key_decode=kv_lens,
        )

    assert reused is first
    assert refreshed is not first
    torch.testing.assert_close(
        first.seq_len_thresholds.reshape(-1),
        torch.tensor([4, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        refreshed.seq_len_thresholds.reshape(-1),
        torch.tensor([5, 6], dtype=torch.int32),
    )


def test_c8_quant_lim():
    n_head, head_dim, num_decodes, tokens, blocks = 32, 128, 2, 2, 8
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.lim_slot_map = torch.full((num_decodes * 2, blocks * head_dim), -(1 << 31), dtype=torch.int32)
    impl.lim_topk_src = torch.zeros((tokens, 1, 2048), dtype=torch.int32)
    impl.lim_topk_dst = torch.zeros_like(impl.lim_topk_src)
    impl.lim_topk_misses = torch.zeros(tokens, dtype=torch.int32)
    impl.lim_miss_src = torch.empty((num_decodes, 32768), dtype=torch.int32)
    impl.lim_miss_dst = torch.empty_like(impl.lim_miss_src)
    impl.lim_misses = torch.zeros(num_decodes, dtype=torch.int32)
    impl.lim_reuse_request_count = 0
    impl._copy_sfa_metadata = SimpleNamespace(
        copy_sfa_pool_entries=torch.arange(num_decodes, dtype=torch.int32),
        num_decode_tokens=tokens,
        copy_sfa_prefix_lens=torch.full((num_decodes,), 4096, dtype=torch.int32),
        copy_sfa_cache_tokens=torch.full((num_decodes,), 2048, dtype=torch.int32),
        copy_sfa_query_ends=torch.tensor([1, 2], dtype=torch.int32),
        copy_sfa_seq_lens=torch.full((num_decodes,), 8192, dtype=torch.int32),
        lim_request_state=torch.tensor([-2, -3], dtype=torch.int32),
        copy_sfa_reuse_logical_lens=None,
    )
    query = torch.randint(-8, 8, (tokens * n_head, head_dim), dtype=torch.int8)
    query_scale = torch.ones(tokens * n_head, dtype=torch.float16)
    weights = torch.randn(tokens, n_head, dtype=torch.bfloat16)
    indexer = SimpleNamespace(
        enable_sparse_li_c8=True,
        n_head=n_head,
        head_dim=head_dim,
        k_cache=SimpleNamespace(
            kv_cache=(
                torch.zeros((blocks, head_dim, 1, head_dim), dtype=torch.int8),
                torch.zeros((blocks, head_dim, 1, 1), dtype=torch.float16),
            )
        ),
    )
    indexer_metadata = SimpleNamespace(block_table=torch.zeros((num_decodes, blocks), dtype=torch.int32))
    m_c8 = MagicMock()
    m_bf16 = MagicMock()
    with patch(
        "vllm_ascend.attention.sfa_kv_offload.torch.ops._C_ascend",
        SimpleNamespace(npu_fused_quant_lightning_indexer_manage=m_c8, npu_fused_lightning_indexer_manage=m_bf16),
    ):
        result = impl._lim_select(query, weights, indexer, indexer_metadata, query_scale)

    assert m_c8.called
    assert not m_bf16.called
    op_weights, q_scale, op_query, key_scale, index_key = m_c8.call_args.args[:5]
    assert op_weights.dtype == torch.bfloat16
    assert tuple(q_scale.shape) == (tokens, n_head) and q_scale.dtype == torch.float16
    assert tuple(op_query.shape) == (tokens, n_head, head_dim) and op_query.dtype == torch.int8
    assert tuple(key_scale.shape) == (blocks, head_dim, 1)
    assert index_key.dtype == torch.int8
    assert result.data_ptr() == impl.lim_topk_src.data_ptr()
    torch.testing.assert_close(result, impl.lim_topk_src[:tokens])


@pytest.mark.parametrize("mode", ["copy_sfa", "overlap", "mixed_debug", "prefill_debug"])
def test_offload_attention_dispatch_after_removing_unfused_mode(mode):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl.use_fused_overlap = mode == "overlap"
    impl._copy_sfa_attention = MagicMock(return_value=torch.ones(2, 1, 4))
    impl._execute_fused_overlap_offload_decode = MagicMock(return_value=torch.ones(2, 1, 4))
    query = torch.zeros(2, 1, 4)
    cache = (torch.zeros(1, 128, 1, 4), torch.zeros(1, 128, 1, 4))
    table = torch.tensor([[0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        fused_copy_sfa_enabled=mode == "copy_sfa",
        num_decodes=0 if mode == "prefill_debug" else 1,
        num_decode_tokens=0 if mode == "prefill_debug" else (1 if mode == "mixed_debug" else 2),
        num_prefills=1 if mode.endswith("debug") else 0,
        req_ids_tensor=torch.tensor([0]),
        token_to_req=torch.tensor([0, 0]),
    )
    paged_output = torch.full_like(query, 2)
    with (
        patch("vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager"),
        patch("vllm_ascend.attention.sfa_kv_offload._check_device_kv_cache_exist") as check_cache,
        patch.object(
            AscendSFAImpl,
            "_execute_sparse_flash_attention_process",
            return_value=paged_output,
        ) as paged_attention,
    ):
        output = impl._execute_sparse_flash_attention_process(
            query,
            query,
            cache,
            torch.zeros(2, 1, dtype=torch.int32),
            metadata,
            torch.tensor([2]),
            torch.tensor([2]),
            block_table=table,
        )
    if mode.endswith("debug"):
        check_cache.assert_called_once()
        paged_attention.assert_called_once()
        assert paged_attention.call_args.args[2] is cache
        assert paged_attention.call_args.kwargs["block_table"] is table
        assert output is paged_output
    else:
        check_cache.assert_not_called()
        paged_attention.assert_not_called()
    assert impl._copy_sfa_attention.call_count == (mode == "copy_sfa")
    assert impl._execute_fused_overlap_offload_decode.call_count == (mode == "overlap")


def test_pack_c8_kv_preserves_component_byte_order():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    nope = torch.arange(512, dtype=torch.int8).view(torch.float8_e4m3fn)
    rope_bf16 = torch.arange(64, dtype=torch.bfloat16)
    rope = rope_bf16.view(torch.float8_e4m3fn)
    scale = torch.arange(16, dtype=torch.int8).view(torch.float8_e4m3fn)

    packed = impl._pack_c8_kv(rope, nope, scale)

    assert packed.dtype == torch.int8
    assert tuple(packed.shape) == (1, 656)
    torch.testing.assert_close(packed[0, :512], nope.view(torch.int8))
    torch.testing.assert_close(packed[0, 512:640], rope_bf16.view(torch.int8))
    torch.testing.assert_close(packed[0, 640:], scale.view(torch.int8))


def test_c8_prefill_store_runs_at_post_projection_hook():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.enable_sparse_sfa_c8 = True
    impl._is_decode_only = MagicMock(return_value=False)
    impl._offload_layer_name = MagicMock(return_value="layer")
    impl._cpu_cache_pair = MagicMock(return_value=(object(), None))
    impl._in_graph_runtime = MagicMock(return_value=False)
    impl._copy_prefill_kv_to_copy_sfa_row = MagicMock()
    result = (object(), object(), object())
    cache = (torch.zeros(1, 128, 1, 656, dtype=torch.int8), None)
    slots = torch.tensor([0])
    metadata = SimpleNamespace()
    with (
        patch.object(AscendSFAImpl, "exec_kv", return_value=result),
        patch.object(AscendSFAImpl, "_store_packed_kv_cache") as native_store,
        patch("vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager") as get_manager,
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_ascend_config",
            return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(keep_device_kv_cache=True)),
        ),
    ):
        assert impl.exec_kv(None, None, None, cache, slots, metadata) is result
        get_manager.return_value.offload_new_kv.assert_not_called()
        impl._store_packed_kv_cache(result[0], result[1], result[2], cache, slots, metadata)

    native_store.assert_called_once_with(result[0], result[1], result[2], cache, slots, metadata)
    get_manager.return_value.offload_new_kv.assert_called_once()
    assert get_manager.return_value.offload_new_kv.call_args.kwargs["has_prefill"] is True
    impl._copy_prefill_kv_to_copy_sfa_row.assert_called_once()


def test_c8_decode_store_hook_does_not_duplicate_current_kv_write():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl._is_decode_only = MagicMock(return_value=True)
    with (
        patch.object(AscendSFAImpl, "_store_packed_kv_cache") as native_store,
        patch("vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager") as get_manager,
    ):
        impl._store_packed_kv_cache(None, None, None, (None,), torch.tensor([0]), SimpleNamespace())
    native_store.assert_not_called()
    get_manager.assert_not_called()


@pytest.mark.parametrize(("kv_len", "tail_rows"), [(3013, 197), (3072, 256)])
def test_long_prefill_hot_fill_copies_both_circular_tail_blocks(kv_len, tail_rows):
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.use_fused_copy_sfa = True
    impl.enable_sparse_sfa_c8 = True
    impl.copy_sfa_hot_tokens = 2048
    block_count = (kv_len + 127) // 128
    paged = (torch.arange(block_count * 128 * 656, dtype=torch.int32) % 256).to(torch.int8).view(block_count, 128, 1, 656)
    hot = torch.zeros(1, 2048 + 256, 1, 656, dtype=torch.int8)
    manager = SimpleNamespace(
        topk_buffers_k=[hot],
        _get_offload_layer_id=MagicMock(return_value=0),
    )
    metadata = SimpleNamespace(
        copy_sfa_prefill_pool_slots=torch.tensor([0]),
        num_decodes=0,
        num_prefills=1,
        cum_query_lens=torch.tensor([kv_len]),
        seq_lens=torch.tensor([kv_len]),
        block_table=torch.arange(block_count, dtype=torch.int32).view(1, -1),
    )
    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(keep_device_kv_cache=True)),
    ):
        impl._copy_prefill_kv_to_copy_sfa_row((paged, None), metadata, manager, "layer")

    start = ((kv_len - 1) // 128 - 1) * 128
    positions = torch.arange(start, kv_len)
    expected_hot = torch.zeros(256, 656, dtype=torch.int8)
    expected_hot[positions % 256] = paged.view(-1, 656)[positions]
    torch.testing.assert_close(hot.view(-1, 656)[2048 : 2048 + 256], expected_hot)


def test_copy_sfa_c8_uses_packed_memfabric_abi():
    tokens, count, heads = 1, 1, 8
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.skip_topk = False
    impl.lim_indexer_owner = impl
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.scale = 1.0
    impl.lim_topk_src = torch.zeros((tokens, 1, 2048), dtype=torch.int32)
    impl.lim_topk_dst = torch.zeros_like(impl.lim_topk_src)
    impl.lim_topk_misses = torch.zeros(tokens, dtype=torch.int32)
    impl.lim_miss_src = torch.zeros((count, 32768), dtype=torch.int32)
    impl.lim_miss_dst = torch.zeros_like(impl.lim_miss_src)
    impl.lim_misses = torch.zeros(count, dtype=torch.int32)
    manager = SimpleNamespace(
        topk_buffers_k=[torch.zeros(1, 256, 1, 656, dtype=torch.int8)],
        k_caches_cpu=[torch.zeros(1, 128, 1, 656, dtype=torch.int8)],
        _get_offload_layer_id=MagicMock(return_value=0),
    )
    metadata = SimpleNamespace(
        num_decode_tokens=tokens,
        copy_sfa_pool_entries=torch.zeros(count, dtype=torch.int32),
        copy_sfa_cache_tokens=torch.tensor([2048], dtype=torch.int32),
        copy_sfa_logical_lens=torch.tensor([2048], dtype=torch.int32),
        copy_sfa_query_ends=torch.tensor([1], dtype=torch.int32),
        copy_sfa_hbm_block_table=torch.zeros((1, 2), dtype=torch.int32),
        copy_sfa_source_block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    op = MagicMock()
    with patch(
        "vllm_ascend.attention.sfa_kv_offload.torch.ops._C_ascend",
        SimpleNamespace(npu_fused_quant_scatter_copy_sparse_flash_attention=op),
    ):
        output = impl._copy_sfa_attention(
            torch.zeros(tokens, heads, 512, dtype=torch.bfloat16),
            torch.zeros(tokens, heads, 64, dtype=torch.bfloat16),
            impl.lim_topk_src,
            metadata,
            manager,
            "layer",
        )

    assert tuple(output.shape) == (tokens, heads, 512)
    args = op.call_args.args
    assert tuple(args[0].shape) == (tokens, heads, 576)
    assert tuple(args[1].shape) == (1 * 256 // 128, 128, 1, 656)
    assert args[2].device.type == "cpu"
    assert tuple(args[2].shape) == (1, 128, 1, 656)
    assert args[5].data_ptr() == impl.lim_topk_dst.data_ptr()
    assert args[6].data_ptr() == impl.lim_topk_src.data_ptr()
    assert tuple(args[11].shape) == (count, 1, 32768)
    assert tuple(args[12].shape) == (count, 1, 32768)
    assert tuple(args[15].shape) == (tokens, heads, 512)


def test_c8_mtp_reuse_requires_step0_selection():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.skip_topk = True
    impl.has_indexer = True
    impl.lim_indexer_owner = impl
    impl.lim_reuse_request_count = 0
    with pytest.raises(RuntimeError, match="draft-step-0"):
        impl._copy_sfa_attention(
            None,
            None,
            None,
            SimpleNamespace(num_decode_tokens=1, copy_sfa_pool_entries=torch.zeros(1, dtype=torch.int32)),
            SimpleNamespace(),
            "layer",
        )


def test_c8_mtp_reuses_step0_lengths_and_miss_counts():
    tokens, count, heads = 1, 1, 8
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.skip_topk = True
    impl.has_indexer = True
    impl.lim_indexer_owner = impl
    impl.lim_reuse_request_count = 1
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.scale = 1.0
    impl.lim_topk_src = torch.zeros((tokens, 1, 2048), dtype=torch.int32)
    impl.lim_topk_dst = torch.zeros_like(impl.lim_topk_src)
    impl.lim_miss_src = torch.zeros((count, 32768), dtype=torch.int32)
    impl.lim_miss_dst = torch.zeros_like(impl.lim_miss_src)
    impl.copy_sfa_reuse_cache_tokens = torch.tensor([1024], dtype=torch.int32)
    impl.copy_sfa_reuse_logical_lens = torch.tensor([1025], dtype=torch.int32)
    impl.lim_reuse_topk_misses = torch.tensor([7], dtype=torch.int32)
    impl.lim_reuse_misses = torch.tensor([8], dtype=torch.int32)
    manager = SimpleNamespace(
        topk_buffers_k=[torch.zeros(1, 256, 1, 656, dtype=torch.int8)],
        k_caches_cpu=[torch.zeros(1, 128, 1, 656, dtype=torch.int8)],
        _get_offload_layer_id=MagicMock(return_value=0),
    )
    metadata = SimpleNamespace(
        num_decode_tokens=tokens,
        copy_sfa_pool_entries=torch.zeros(count, dtype=torch.int32),
        copy_sfa_cache_tokens=torch.tensor([2048], dtype=torch.int32),
        copy_sfa_logical_lens=torch.tensor([2049], dtype=torch.int32),
        copy_sfa_query_ends=torch.tensor([1], dtype=torch.int32),
        copy_sfa_hbm_block_table=torch.zeros((1, 2), dtype=torch.int32),
        copy_sfa_source_block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    op = MagicMock()
    with patch(
        "vllm_ascend.attention.sfa_kv_offload.torch.ops._C_ascend",
        SimpleNamespace(npu_fused_quant_scatter_copy_sparse_flash_attention=op),
    ):
        impl._copy_sfa_attention(
            torch.zeros(tokens, heads, 512, dtype=torch.bfloat16),
            torch.zeros(tokens, heads, 64, dtype=torch.bfloat16),
            impl.lim_topk_src,
            metadata,
            manager,
            "layer",
        )

    args = op.call_args.args
    assert args[7].data_ptr() == impl.lim_reuse_topk_misses.data_ptr()
    assert args[9].item() == 1025
    assert args[10].item() == 1024
    assert args[13].data_ptr() == impl.lim_reuse_misses.data_ptr()


def test_c8_mtp_compacts_rejected_draft_rows():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.lim_reuse_request_count = 2
    impl.lim_topk_src = torch.tensor([[[11], [22]], [[33], [44]]], dtype=torch.int32)
    impl.lim_topk_dst = torch.tensor([[[55], [66]], [[77], [88]]], dtype=torch.int32)
    impl.compact_lim_topk_metadata(torch.tensor([1, 0]))
    assert impl.lim_topk_src[0, 0].item() == 33
    assert impl.lim_topk_dst[0, 0].item() == 77
    assert impl.lim_topk_src[1, 0].item() == 11
    assert impl.lim_topk_dst[1, 0].item() == 55


def test_c8_mtp_exec_kv_writes_each_current_packed_row():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl._is_decode_only = MagicMock(return_value=True)
    impl._offload_layer_name = MagicMock(return_value="layer")
    impl._cpu_cache_pair = MagicMock(return_value=(torch.zeros(1, 128, 1, 656, dtype=torch.int8), None))
    impl._in_graph_runtime = MagicMock(return_value=False)
    nope = torch.arange(2 * 512, dtype=torch.int8).view(torch.float8_e4m3fn)
    rope = torch.arange(2 * 128, dtype=torch.int8).view(torch.float8_e4m3fn)
    scale = torch.arange(2 * 16, dtype=torch.int8).view(torch.float8_e4m3fn)
    impl._compute_kv_only = MagicMock(return_value=(nope, rope, scale))
    hot = torch.zeros(1, 256, 1, 656, dtype=torch.int8)
    manager = SimpleNamespace(
        topk_buffers_k=[hot],
        offload_new_kv=MagicMock(),
        _get_offload_layer_id=MagicMock(return_value=0),
    )
    metadata = SimpleNamespace(fused_copy_sfa_enabled=True, copy_sfa_device_slots=torch.tensor([5, 6]))
    with patch("vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager", return_value=manager):
        impl.exec_kv(None, None, None, (None,), torch.tensor([5, 6]), metadata)

    expected = impl._pack_c8_kv(rope, nope, scale)
    torch.testing.assert_close(hot.view(-1, 656)[5:7], expected)
    manager.offload_new_kv.assert_called_once()
    torch.testing.assert_close(manager.offload_new_kv.call_args.kwargs["k"], expected)
    assert manager.offload_new_kv.call_args.kwargs["has_prefill"] is False


def test_reshape_sparse_offload_packed_c8_cache():
    spec = SimpleNamespace(
        cache_sparse_sfa_c8=True,
        page_size_bytes=128 * 656,
        block_size=128,
        num_kv_heads=1,
        head_size=656,
    )
    raw_npu = torch.zeros(2 * 128 * 656, dtype=torch.int8)
    raw_host = torch.zeros_like(raw_npu)
    hot = torch.zeros(1, 256, 1, 656, dtype=torch.int8)
    with patch(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager."
        "allocate_kv_offload_topk_buffer_pair",
        return_value=(hot, None),
    ):
        result = reshape_kv_cache_tensors_for_sparse_kv_offload(
            (raw_npu, None, raw_host, None, raw_npu.numel()),
            spec,
            MagicMock(),
            0,
            SimpleNamespace(),
            SimpleNamespace(),
        )

    assert result[0].shape == (2, 128, 1, 656)
    assert result[2].shape == (2, 128, 1, 656)
    assert result[0].dtype == result[2].dtype == torch.int8
    assert result[1] is None and result[3] is None and result[5] is None
    assert result[4] is hot


def test_packed_c8_d2h_uses_one_656_byte_descriptor_per_token():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.packed_c8 = True
    manager.sparse_kv_offload_config = SimpleNamespace(keep_device_kv_cache=False)
    manager.max_num_tokens = 2
    manager.token_size_bytes_k = 656
    manager.token_size_bytes_v = 0
    manager.d2h_src_ptrs_npu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_dst_ptrs_npu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_lengths_npu = torch.zeros(4, dtype=torch.int32)
    manager.d2h_size_npu = torch.zeros(1, dtype=torch.int32)
    manager.d2h_token_indices_npu = torch.arange(2, dtype=torch.int64)
    host = torch.zeros(4 * 128 * 656, dtype=torch.int8)
    current = torch.zeros(2, 656, dtype=torch.int8)
    slots = torch.tensor([0, 127], dtype=torch.int64)
    calls = []

    def sparse_copy(src, dst, lengths, count, device):
        calls.append((src.clone(), dst.clone(), lengths.clone(), count.clone(), device))
        return 0

    with patch(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager.offload",
        SimpleNamespace(sparse_copy=sparse_copy),
    ):
        manager._offload_new_kv_on_current_stream(
            slots,
            host,
            None,
            None,
            None,
            current,
            None,
            False,
        )

    assert len(calls) == 1
    _, dst, lengths, count, _ = calls[0]
    assert int(count.item()) == 2
    torch.testing.assert_close(lengths[:2], torch.tensor([656, 656], dtype=torch.int32))
    torch.testing.assert_close(dst[:2], torch.tensor([host.data_ptr(), host.data_ptr() + 127 * 656]))


def test_packed_c8_d2h_batches_65_descriptors_below_memfabric_limit():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.packed_c8 = True
    manager.sparse_kv_offload_config = SimpleNamespace(keep_device_kv_cache=True)
    manager.max_num_tokens = 65
    manager.token_size_bytes_k = 656
    manager.token_size_bytes_v = 0
    manager.d2h_src_ptrs_npu = torch.zeros(130, dtype=torch.int64)
    manager.d2h_dst_ptrs_npu = torch.zeros(130, dtype=torch.int64)
    manager.d2h_lengths_npu = torch.zeros(130, dtype=torch.int32)
    manager.d2h_size_npu = torch.zeros(1, dtype=torch.int32)
    manager.d2h_token_indices_npu = torch.arange(65, dtype=torch.int64)
    host = torch.zeros(65 * 128 * 656, dtype=torch.int8)
    current = torch.zeros(65, 656, dtype=torch.int8)  # noqa: F841
    calls = []

    def sparse_copy(src, dst, lengths, count, device):
        calls.append(int(count.item()))
        return 0

    with patch(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager.offload",
        SimpleNamespace(sparse_copy=sparse_copy),
    ):
        manager._offload_new_kv_on_current_stream(
            torch.arange(65),
            host,
            None,
            host,
            None,
            None,
            None,
            True,
        )

    assert calls == [32, 32, 1]
