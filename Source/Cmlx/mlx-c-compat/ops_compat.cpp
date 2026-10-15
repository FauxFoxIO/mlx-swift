// mlx-c v0.7.0 predates gather_qmm's `global_scale` (mlx v0.32.3): build its
// ops.cpp with that binding renamed, then bind it with the v0.32.4 signature.

#include "mlx/ops.h"

namespace mlx::core {

// The v0.32.2 overload that mlx-c's renamed binding still calls.
inline array gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    std::optional<array> lhs_indices,
    std::optional<array> rhs_indices,
    bool transpose,
    std::optional<int> group_size,
    std::optional<int> bits,
    const std::string& mode,
    bool sorted_indices,
    StreamOrDevice s) {
  return gather_qmm(
      x,
      w,
      scales,
      biases,
      std::move(lhs_indices),
      std::move(rhs_indices),
      transpose,
      group_size,
      bits,
      mode,
      std::nullopt,
      sorted_indices,
      s);
}

} // namespace mlx::core

#define mlx_gather_qmm mlx_gather_qmm_mlx_c_v0_7_0_
#include "../mlx-c/mlx/c/ops.h"
#include "../mlx-c/mlx/c/ops.cpp"
#undef mlx_gather_qmm

extern "C" int mlx_gather_qmm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    const mlx_array global_scale /* may be null */,
    bool sorted_indices,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::gather_qmm(
            mlx_array_get_(x),
            mlx_array_get_(w),
            mlx_array_get_(scales),
            (biases.ctx ? std::make_optional(mlx_array_get_(biases))
                        : std::nullopt),
            (lhs_indices.ctx ? std::make_optional(mlx_array_get_(lhs_indices))
                             : std::nullopt),
            (rhs_indices.ctx ? std::make_optional(mlx_array_get_(rhs_indices))
                             : std::nullopt),
            transpose,
            (group_size.has_value ? std::make_optional<int>(group_size.value)
                                  : std::nullopt),
            (bits.has_value ? std::make_optional<int>(bits.value)
                            : std::nullopt),
            std::string(mode),
            (global_scale.ctx ? std::make_optional(mlx_array_get_(global_scale))
                              : std::nullopt),
            sorted_indices,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
