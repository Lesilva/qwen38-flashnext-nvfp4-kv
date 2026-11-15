// qfuse extension bindings.
#include <torch/extension.h>
at::Tensor nvfp4_moe_decode(const at::Tensor & x, const at::Tensor & topk_ids, const at::Tensor & topk_w,
                            const at::Tensor & w13, const at::Tensor & w13_sf, const at::Tensor & g1_alpha,
                            const at::Tensor & a1_gs, const at::Tensor & w2, const at::Tensor & w2_sf,
                            const at::Tensor & g2_alpha, const at::Tensor & a2_gs, bool pdl);
bool nvfp4_moe_prealloc(const at::Tensor & like);
at::Tensor gemv_bf16(const at::Tensor & x, const at::Tensor & w);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_moe_decode", &nvfp4_moe_decode, "decode-size NVFP4 W4A4 MoE (1..8 tokens, top-10)",
          py::arg("x"), py::arg("topk_ids"), py::arg("topk_w"), py::arg("w13"), py::arg("w13_sf"), py::arg("g1_alpha"),
          py::arg("a1_gs"), py::arg("w2"), py::arg("w2_sf"), py::arg("g2_alpha"), py::arg("a2_gs"), py::arg("pdl") = true);
    m.def("nvfp4_moe_prealloc", &nvfp4_moe_prealloc, "allocate the persistent MoE scratch on like.device");
    m.def("gemv_bf16", &gemv_bf16, "y = x @ W^T for 1..8 bf16 rows");
}
