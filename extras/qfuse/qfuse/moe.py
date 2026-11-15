"""Kernel 'moe': decode-size NVFP4 W4A4 expert MoE (qf_moe.cu) in place of the FlashInfer CUTLASS runner for
1..8 rows. Same quantization recipe as CUTLASS; results differ at rounding level (relative L2 about 0.005),
so each layer is compared once against the original before it turns fast.
"""
import torch

from . import MODE, STATE, check_contract, count, dump, ext, log
from .selfcheck import record, rel_err

TOL = 0.05
_layers = {}  # data_ptr of w13 -> True (fast) / False (disabled)
_warm = [False]


def _eligible(dispatch_output, quant_info, runner_config, output, enable_alltoall):
    x = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    return (
        output is None
        and not enable_alltoall
        and quant_info.quant_type == "fp4"
        and x.dim() == 2
        and 1 <= x.shape[0] <= 8
        and x.shape[1] == 2560
        and x.dtype == torch.bfloat16
        and getattr(dispatch_output, "hidden_states_scale", None) is None
        and quant_info.moe_ep_size == 1
        and quant_info.moe_tp_size == 1
        and runner_config.is_gated
        and runner_config.activation == "silu"
        and not runner_config.apply_router_weight_on_input
        and runner_config.gemm1_alpha is None
        and runner_config.gemm1_beta is None
        and runner_config.gemm1_clamp_limit is None
        and runner_config.swiglu_limit is None
        and not runner_config.num_fused_shared_experts
        and topk.topk_ids.dim() == 2
        and topk.topk_ids.shape[-1] == 10
        and quant_info.w13_weight.dtype == torch.uint8
        and quant_info.w13_weight.shape[0] == 512
        and quant_info.quant_scales is not None
        and len(quant_info.quant_scales) == 6
        and not torch.compiler.is_compiling()
    )


def _call(x, topk, quant_info):
    qs = quant_info.quant_scales
    return ext().nvfp4_moe_decode(
        x.contiguous(),
        topk.topk_ids.to(torch.int32).contiguous(),
        topk.topk_weights.to(torch.float32).contiguous(),
        quant_info.w13_weight,
        qs[1],
        qs[2],
        qs[0],
        quant_info.w2_weight,
        qs[4],
        qs[5],
        qs[3],
    )


def _install(module):
    if not check_contract("moe", module, ["_run_flashinfer_cutlass", "_prepare_input", "FlashInferCutlassMoeQuantInfo"]):
        STATE["kernels"]["moe"] = "off: contract mismatch"
        return
    orig = module._run_flashinfer_cutlass

    def _run_flashinfer_cutlass(*, dispatch_output, quant_info, runner_config, output=None, enable_alltoall=False):
        def stock():
            return orig(
                dispatch_output=dispatch_output,
                quant_info=quant_info,
                runner_config=runner_config,
                output=output,
                enable_alltoall=enable_alltoall,
            )

        capturing = torch.cuda.is_current_stream_capturing()
        x = dispatch_output.hidden_states
        if not _warm[0] and not capturing and isinstance(x, torch.Tensor) and x.is_cuda:
            ext().nvfp4_moe_prealloc(x)
            _warm[0] = True
            log("moe scratch allocated")
        if not _warm[0] or not _eligible(dispatch_output, quant_info, runner_config, output, enable_alltoall):
            return stock()
        topk = dispatch_output.topk_output
        key = quant_info.w13_weight.data_ptr()
        if MODE == "selftest":
            ref = stock()
            if not capturing:
                y1 = _call(x, topk, quant_info)
                y2 = _call(x, topk, quant_info)
                record("moe", x.shape[0], y1, y2, ref)
            return ref
        ok = _layers.get(key)
        if ok is False:
            return stock()
        if ok is None:
            if capturing:
                count("moe.capture_unverified")
                return stock()
            ref = stock()
            y = _call(x, topk, quant_info)
            r, m = rel_err(y, ref)
            good = ref.shape == y.shape and ref.dtype == y.dtype and (r <= TOL or m <= 1e-6)
            _layers[key] = bool(good)
            count("moe.verified" if good else "moe.rejected")
            if not good:
                log("moe layer rejected: rel=%g max_abs=%g" % (r, m))
            dump()
            return ref
        count("moe.fast")
        return _call(x, topk, quant_info)

    module._run_flashinfer_cutlass = _run_flashinfer_cutlass
    STATE["kernels"]["moe"] = MODE
    log("moe installed")


TARGETS = [("sglang.srt.layers.moe.moe_runner.flashinfer_cutlass", _install)]
