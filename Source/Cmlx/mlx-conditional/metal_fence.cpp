// Copyright © 2026 Faux Fox, Inc.

// Builds mlx/backend/metal/fence.cpp with its kernels compiled at run time.
//
// MLX_METAL_FAST_SYNCH=1 needs the fence_* kernels, which use Metal 3.2
// (coherent(system)). SwiftPM compiles default.metallib for the package
// deployment target (macOS 14, Metal 3.1), so the kernels cannot live there.
// fence.cpp only takes the fast path on macOS 15 / iOS 18 or later, where the
// device compiles source at Metal 3.2 or newer.

#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::metal {
namespace {

// Keep in sync with mlx/backend/metal/kernels/fence.metal.
constexpr const char* fence_source = R"metal(
#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>

[[kernel]] void input_coherent(
    volatile coherent(system) device uint* input [[buffer(0)]],
    const constant uint& size [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < size) {
    input[index] = input[index];
  }
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

[[kernel]] void fence_update(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  timestamp[0] = value;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

[[kernel]] void fence_wait(
    volatile coherent(system) device uint* timestamp [[buffer(0)]],
    constant uint& value [[buffer(1)]]) {
  while (1) {
    metal::atomic_thread_fence(
        metal::mem_flags::mem_device,
        metal::memory_order_seq_cst,
        metal::thread_scope_system);
    if (timestamp[0] >= value) {
      break;
    }
  }
}
)metal";

MTL::Library* fence_library(Device& d) {
  return d.get_library(
      "mlx_swift_fence", [] { return std::string(fence_source); });
}

} // namespace
} // namespace mlx::core::metal

// fence.cpp calls d.get_kernel("fence_wait") and d.get_kernel("fence_update"),
// which look in default.metallib. Point both at the run-time library instead.
#define get_kernel(name) \
  get_kernel(name, ::mlx::core::metal::fence_library(d))
#include "../mlx/mlx/backend/metal/fence.cpp"
#undef get_kernel
