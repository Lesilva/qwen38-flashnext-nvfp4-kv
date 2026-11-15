"""Kernels 'ba' and 'gate': small BF16 projections served by qfuse_ext.gemv_bf16 instead of a tensor-core GEMM.

Off by default: measured no gain in the served model (see extras/qfuse/README.md). Kept as examples of a
per-instance wrapper with a one-time comparison against the original.

  ba    Qwen3_5GatedDeltaNet.in_proj_ba   [2 * num_v_heads, hidden]   (runs beside in_proj_qkvz on the alt stream)
  gate  Qwen2MoeSparseMoeBlock.gate       [num_experts, hidden]       (router logits)

The class-level wrappers only install a per-instance forward on first use; the surrounding control flow
(streams, top-k, experts) is the original code. An instance turns fast only after one comparison against the
original result outside CUDA-graph capture; inside a capture an unverified instance keeps the original path.
"""
import torch

from . import MODE, STATE, check_contract, count, dump, ext, log
from .selfcheck import record, rel_err

MAX_ROWS = 8
TOL = 2e-3


def _eligible(mod, x):
    w = mod.weight
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and w.dtype == torch.bfloat16
        and w.dim() == 2
        and w.shape[1] == x.shape[1]
        and w.shape[1] % 256 == 0
        and w.is_contiguous()
        and w.device == x.device
        and mod.bias is None
        and (x.data_ptr() & 15) == 0
        and (w.data_ptr() & 15) == 0
        and not torch.compiler.is_compiling()
    )


def _wrap_instance(mod, kernel):
    if getattr(mod, "_qfuse_wrapped", False):
        return
    orig = mod.forward
    st = {"ok": None}  # None: not compared yet, True: fast, False: disabled

    def forward(x):
        if st["ok"] is False or not _eligible(mod, x):
            return orig(x)
        capturing = torch.cuda.is_current_stream_capturing()
        if MODE == "selftest":
            ref = orig(x)
            if not capturing:
                y1 = ext().gemv_bf16(x, mod.weight)
                y2 = ext().gemv_bf16(x, mod.weight)
                record(kernel, x.shape[0], y1, y2, ref[0])
            return ref
        if st["ok"] is None:
            if capturing:
                count(kernel + ".capture_unverified")
                return orig(x)
            ref = orig(x)
            y = ext().gemv_bf16(x, mod.weight)
            r, m = rel_err(y, ref[0])
            good = (
                isinstance(ref, tuple)
                and ref[1] is None
                and ref[0].shape == y.shape
                and ref[0].dtype == y.dtype
                and (r <= TOL or m <= 1e-6)
            )
            st["ok"] = bool(good)
            count(kernel + (".verified" if good else ".rejected"))
            if not good:
                log("%s instance rejected: rel=%g max_abs=%g" % (kernel, r, m))
            dump()
            return ref
        count(kernel + ".fast")
        return ext().gemv_bf16(x, mod.weight), None

    mod.forward = forward
    object.__setattr__(mod, "_qfuse_wrapped", True)
    count(kernel + ".instances")


def _install_ba(module):
    from sglang.srt.layers import linear as L

    if not (
        check_contract("ba", module, ["Qwen3_5GatedDeltaNet._forward_input_proj"])
        and check_contract("ba", L, ["ColumnParallelLinear.forward"])
    ):
        STATE["kernels"]["ba"] = "off: contract mismatch"
        return
    cls = module.Qwen3_5GatedDeltaNet
    orig = cls._forward_input_proj

    def _forward_input_proj(self, hidden_states):
        if not getattr(self, "_qfuse_ba", False):
            _wrap_instance(self.in_proj_ba, "ba")
            object.__setattr__(self, "_qfuse_ba", True)
        return orig(self, hidden_states)

    cls._forward_input_proj = _forward_input_proj
    STATE["kernels"]["ba"] = MODE
    log("ba installed")


def _install_gate(module):
    from sglang.srt.layers import linear as L

    if not (
        check_contract("gate", module, ["Qwen2MoeSparseMoeBlock._forward_router_experts"])
        and check_contract("gate", L, ["ReplicatedLinear.forward"])
    ):
        STATE["kernels"]["gate"] = "off: contract mismatch"
        return
    cls = module.Qwen2MoeSparseMoeBlock
    orig = cls._forward_router_experts

    def _forward_router_experts(self, hidden_states):
        if not getattr(self, "_qfuse_gate", False):
            _wrap_instance(self.gate, "gate")
            object.__setattr__(self, "_qfuse_gate", True)
        return orig(self, hidden_states)

    cls._forward_router_experts = _forward_router_experts
    STATE["kernels"]["gate"] = MODE
    log("gate installed")


TARGETS_BA = [("sglang.srt.models.qwen3_5", _install_ba)]
TARGETS_GATE = [("sglang.srt.models.qwen2_moe", _install_gate)]
