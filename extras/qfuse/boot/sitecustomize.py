# qfuse boot: put this directory on PYTHONPATH. Inert unless QFUSE is on/selftest. Chains to any other
# sitecustomize that this one shadows.
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))


def _chain():
    try:
        import importlib.machinery
        import importlib.util

        paths = [p for p in sys.path if p and os.path.abspath(p) != _here]
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", paths)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    except Exception as e:  # never break interpreter start-up
        sys.stderr.write("qfuse: chained sitecustomize failed: %r\n" % (e,))


_chain()
if os.environ.get("QFUSE", "off").strip().lower() in ("on", "selftest", "1", "true"):
    try:
        root = os.environ.get("QFUSE_ROOT") or os.path.dirname(_here)
        if root not in sys.path:
            sys.path.insert(0, root)
        import qfuse

        qfuse.install()
    except Exception as e:  # never break interpreter start-up
        sys.stderr.write("qfuse: install failed, running stock: %r\n" % (e,))
