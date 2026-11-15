"""Contract check / update. Run inside the serving image (no GPU needed):
    python tools/check.py            compare the installed sglang against qfuse/contracts.json
    python tools/check.py --update   rewrite contracts.json (and source snapshots) from the installed sglang
Exit status 0 when every contract matches. On a mismatch the unified diff against the snapshot is printed.
"""
import difflib
import importlib
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["QFUSE"] = "off"
import qfuse  # noqa: E402

TARGETS = {
    "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass": ["_run_flashinfer_cutlass", "_prepare_input", "FlashInferCutlassMoeQuantInfo"],
    "sglang.srt.models.qwen3_5": ["Qwen3_5GatedDeltaNet._forward_input_proj"],
    "sglang.srt.models.qwen2_moe": ["Qwen2MoeSparseMoeBlock._forward_router_experts"],
    "sglang.srt.layers.linear": ["ColumnParallelLinear.forward", "ReplicatedLinear.forward"],
}
SNAP = os.path.join(ROOT, "qfuse", "snapshots")


def main():
    update = "--update" in sys.argv
    try:
        old = json.load(open(qfuse.CONTRACTS))
    except Exception:
        old = {"targets": {}}
    new, bad = {}, 0
    for modname, names in TARGETS.items():
        mod = importlib.import_module(modname)
        for q in names:
            key = modname + ":" + q
            obj = qfuse.resolve(mod, q)
            src = inspect.getsource(obj)
            new[key] = qfuse.source_hash(obj)
            snap = os.path.join(SNAP, key.replace(":", "__") + ".py.txt")
            if update:
                os.makedirs(SNAP, exist_ok=True)
                open(snap, "w").write(src)
                continue
            if old["targets"].get(key) == new[key]:
                print("OK      ", key)
                continue
            bad += 1
            print("CHANGED ", key)
            if os.path.exists(snap):
                sys.stdout.writelines(difflib.unified_diff(open(snap).read().splitlines(True), src.splitlines(True), "snapshot", "installed"))
    if update:
        import sglang

        json.dump({"sglang_version": getattr(sglang, "__version__", "?"), "targets": new}, open(qfuse.CONTRACTS, "w"), indent=1, sort_keys=True)
        print("contracts.json updated:", len(new), "targets, sglang", getattr(sglang, "__version__", "?"))
        return 0
    print("mismatches:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
