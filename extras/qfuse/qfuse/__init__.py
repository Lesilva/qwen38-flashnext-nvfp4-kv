"""qfuse: out-of-tree fused decode kernels for Qwen3.8-Flash-Next on Pennyroyal/SGLang (RTX PRO 6000, SM120).

Nothing in the serving image is replaced on disk. A post-import hook swaps a few functions for wrappers that
take a fused path only for decode-size calls and fall back to the original otherwise. Every wrapper is bound
to a source-hash contract of the code it replaces: if an upgrade changed that code, the kernel stays off and
serving is stock.

  QFUSE          off (default) | on | selftest   (selftest: always return the original result; run the fused
                 path alongside outside CUDA-graph capture and record error / determinism)
  QFUSE_KERNELS  comma list out of: moe, ba, gate   (default: moe; ba and gate measured no gain, kept as examples)
  QFUSE_OUT      directory for qfuse.log and stats (default /qfuse-out; optional)
  QFUSE_STRICT   1: a contract mismatch raises instead of disabling the kernel
"""

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE = os.environ.get("QFUSE", "off").strip().lower()
MODE = {"1": "on", "true": "on"}.get(MODE, MODE)
KERNELS = [k.strip() for k in os.environ.get("QFUSE_KERNELS", "moe").split(",") if k.strip()]
OUT = os.environ.get("QFUSE_OUT", "/qfuse-out")
STRICT = os.environ.get("QFUSE_STRICT", "") == "1"
SO = os.environ.get("QFUSE_SO") or os.path.join(ROOT, "build", "qfuse_ext.so")
CONTRACTS = os.path.join(ROOT, "qfuse", "contracts.json")

STATE = {"mode": MODE, "pid": os.getpid(), "kernels": {}, "contracts": {}, "counters": {}, "selftest": {}}
_lock = threading.Lock()
_ext = None
_last_dump = [0.0]


def log(msg):
    try:
        with open(os.path.join(OUT, "qfuse.log"), "a") as f:
            f.write("%.3f pid=%d %s\n" % (time.time(), os.getpid(), msg))
    except Exception:
        pass


def count(key, n=1):
    c = STATE["counters"]
    c[key] = c.get(key, 0) + n


def dump(force=False):
    now = time.time()
    if not force and now - _last_dump[0] < 2.0:
        return
    _last_dump[0] = now
    try:
        p = os.path.join(OUT, "state-%d.json" % os.getpid())
        with open(p + ".tmp", "w") as f:
            json.dump(STATE, f, indent=1, default=str)
        os.replace(p + ".tmp", p)
    except Exception:
        pass


def ext():
    global _ext
    if _ext is None:
        loader = importlib.machinery.ExtensionFileLoader("qfuse_ext", SO)
        spec = importlib.util.spec_from_loader("qfuse_ext", loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _ext = mod
        log("loaded %s" % SO)
    return _ext


def source_hash(obj):
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def resolve(module, qualname):
    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def check_contract(kernel, module, names):
    """names: qualnames inside module. True when every source hash matches contracts.json."""
    try:
        want = json.load(open(CONTRACTS))["targets"]
    except Exception as e:
        want = {}
        log("contracts.json unreadable: %r" % (e,))
    ok = True
    for q in names:
        key = module.__name__ + ":" + q
        try:
            got = source_hash(resolve(module, q))
        except Exception as e:
            got = "error: %r" % (e,)
        match = want.get(key) == got
        STATE["contracts"][key] = {"match": match, "got": got, "want": want.get(key)}
        ok = ok and match
    if not ok:
        log("contract mismatch for kernel %s in %s: stays off" % (kernel, module.__name__))
        if STRICT:
            raise RuntimeError("qfuse contract mismatch for " + kernel)
    return ok


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, hooks):
        self.hooks = hooks
        self.busy = set()

    def find_spec(self, name, path, target=None):
        if name not in self.hooks or name in self.busy:
            return None
        self.busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.busy.discard(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        orig_exec = spec.loader.exec_module
        hooks = self.hooks.pop(name)

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            for kernel, fn in hooks:
                try:
                    fn(module)
                except Exception as e:
                    STATE["kernels"][kernel] = "install error: %r" % (e,)
                    log("install of %s failed: %r" % (kernel, e))
                    if STRICT:
                        raise
            dump(force=True)

        spec.loader.exec_module = exec_module
        return spec


def install():
    if MODE not in ("on", "selftest"):
        return
    from . import linear, moe

    table = {"moe": moe.TARGETS, "ba": linear.TARGETS_BA, "gate": linear.TARGETS_GATE}
    hooks = {}
    for k in KERNELS:
        if k not in table:
            log("unknown kernel %r ignored" % (k,))
            continue
        STATE["kernels"][k] = "pending import"
        for modname, fn in table[k]:
            hooks.setdefault(modname, []).append((k, fn))
    sys.meta_path.insert(0, _Finder(hooks))
    log("install mode=%s kernels=%s root=%s" % (MODE, KERNELS, ROOT))
    dump(force=True)
