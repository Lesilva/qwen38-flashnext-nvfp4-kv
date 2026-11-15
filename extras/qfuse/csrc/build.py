# AOT build of qfuse_ext inside the serving image (no GPU needed). Output: <pkg>/build/qfuse_ext*.so
import glob, os, time
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
from torch.utils.cpp_extension import load
d = os.path.dirname(os.path.abspath(__file__)); bd = os.path.join(os.path.dirname(d), "build"); os.makedirs(bd, exist_ok=True)
t = time.time()
load(name="qfuse_ext", sources=[d + "/bind.cpp", d + "/qf_moe.cu", d + "/qf_gemv.cu"], build_directory=bd, verbose=True,
     extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math", "-lineinfo"])
print("built in", round(time.time() - t, 1), "s:", glob.glob(bd + "/*.so"))
